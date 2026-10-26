"""Reproducible strong- and weak-scaling benchmark for Network DtN solves.

The benchmark uses a manufactured piecewise-linear solution on a deterministic
graph.  It separately times problem setup, local matrix assembly, hierarchical
DtN factorization, the first solve, and repeated solves with cached factors.
Every reported phase is the maximum elapsed time over all MPI ranks.

Only rank zero emits output.  Each invocation writes one final JSON object to
stdout and, when requested, appends the same object to a JSON Lines file.
Optional phase checkpoints are emitted on stderr and can be persisted to a
separate JSON Lines file for post-timeout diagnosis.

Examples
--------
Strong scaling with a fixed graph::

    OMP_NUM_THREADS=1 PYTHONPATH=. mpiexec -n 4 python3 \
        benchmarks/network_dtn_scaling.py --scaling strong \
        --graph binary_tree --edges 4096 --resolution 32 --repeats 5

Weak scaling with a fixed number of edges per rank::

    OMP_NUM_THREADS=1 PYTHONPATH=. mpiexec -n 4 python3 \
        benchmarks/network_dtn_scaling.py --scaling weak \
        --graph chain --edges-per-rank 512 --resolution 32 --repeats 5
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import sys
import time

import dedalus
import dedalus.public as d3
from mpi4py import MPI
import mpi4py
import numpy as np
import scipy

from jcp.optimization_diagnostics import PhaseDiagnostics, solver_record
from jcp.network_graphs import GRAPH_FAMILIES
from jcp.network_graphs import METRICS
from jcp.network_graphs import build_network as build_synthetic_network


SCHEMA = "dedalus.network_dtn_scaling.v1"
PROGRESS_SCHEMA = "dedalus.network_dtn_scaling_progress.v1"


class BenchmarkHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve command examples while displaying meaningful defaults."""


def build_network(
    graph_family,
    num_edges,
    *,
    metric="uniform",
    seed=0,
):
    """Build a deterministic connected graph with exactly ``num_edges``."""
    return build_synthetic_network(
        graph_family,
        num_edges,
        metric=metric,
        seed=seed,
    )


def manufactured_vertex_values(network, dtype):
    """Return deterministic, bounded, nonconstant values at every vertex."""
    vertices = np.arange(network.num_vertices, dtype=np.float64)
    phase = 2 * np.pi * (vertices + 0.5) / network.num_vertices
    values = 1 + 0.25 * np.sin(phase) + 0.125 * np.cos(2 * phase)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 0.125j * np.cos(3 * phase)
    return values.astype(dtype)


def manufactured_vertex_rhs(network, vertex_values):
    """Build ``g = U + R.T @ ntrace(u)`` for linear edge solutions."""
    flux = np.zeros(network.num_vertices, dtype=vertex_values.dtype)
    for edge, (tail, head) in enumerate(network.endpoints):
        slope = (
            vertex_values[head] - vertex_values[tail]
        ) / network.lengths[edge]
        flux[tail] -= slope
        flux[head] += slope
    return vertex_values + flux


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark strong or weak scaling of whole-edge distributed "
            "Network DtN solves."
        ),
        epilog=(
            "examples:\n"
            "  strong: --scaling strong --graph binary_tree --edges 4096\n"
            "  weak:   --scaling weak --graph chain --edges-per-rank 512"
        ),
        formatter_class=BenchmarkHelpFormatter,
    )
    parser.add_argument(
        "--scaling",
        choices=("strong", "weak"),
        default="strong",
        help="Keep total edges fixed (strong) or edges per rank fixed (weak).",
    )
    parser.add_argument(
        "--graph",
        choices=GRAPH_FAMILIES,
        default="chain",
        help="Deterministic graph family.",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="uniform",
        help="Uniform or deterministic heterogeneous edge lengths.",
    )
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=0,
        help="Seed for deterministic irregular topology and metric lengths.",
    )
    parser.add_argument(
        "--edges",
        type=int,
        default=argparse.SUPPRESS,
        help="Total edge count for strong scaling (default: 128).",
    )
    parser.add_argument(
        "--edges-per-rank",
        type=int,
        default=argparse.SUPPRESS,
        help="Local target edge count for weak scaling (default: 128).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="Chebyshev coefficients per edge.",
    )
    parser.add_argument(
        "--partition",
        choices=("connected", "balanced_edges"),
        default="connected",
        help="Whole-edge partition strategy.",
    )
    parser.add_argument(
        "--vertex-storage",
        choices=("owned_ghost", "replicated", "auto"),
        default="owned_ghost",
        help="Distributed vertex-field storage policy.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float64", "complex128"),
        default="float64",
        help="Scalar dtype used by fields and matrices.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Untimed cached-factor solves before repeated timings.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed cached-factor solves.",
    )
    parser.add_argument(
        "--check-tolerance",
        type=float,
        default=1e-9,
        help=(
            "Maximum accepted relative solution and normwise backward "
            "error."
        ),
    )
    parser.add_argument(
        "--checksum-decimals",
        type=int,
        default=10,
        help="Decimal rounding used by the partition-independent checksum.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Append the rank-zero record to this JSON Lines file.",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        help=(
            "Append rank-zero phase checkpoints to this JSON Lines file. "
            "Checkpoints are flushed after every phase so a scheduler "
            "timeout still leaves useful diagnostics."
        ),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional label copied verbatim into the output record.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Retain Dedalus informational logging before the JSON record.",
    )
    parser.add_argument("--profile-directory", type=Path,
                        help="Opt-in per-rank phase cProfile traces; diagnostic timings only.")
    args = parser.parse_args(argv)
    if not hasattr(args, "edges"):
        args.edges = None
    if not hasattr(args, "edges_per_rank"):
        args.edges_per_rank = None
    return args


def resolve_size(args, ranks):
    if args.scaling == "strong":
        if args.edges_per_rank is not None:
            raise ValueError(
                "--edges-per-rank is only valid with --scaling weak."
            )
        edges = 128 if args.edges is None else args.edges
        edges_per_rank = edges / ranks
    else:
        if args.edges is not None:
            raise ValueError("--edges is only valid with --scaling strong.")
        requested = 128 if args.edges_per_rank is None else args.edges_per_rank
        edges = requested * ranks
        edges_per_rank = requested
    if edges <= 0:
        raise ValueError("The benchmark requires a positive edge count.")
    return int(edges), float(edges_per_rank)


def timed_phase(comm, operation, diagnostics=None, phase=None):
    """Run a collective phase and return its result and rank-time summary."""
    profiler = None if diagnostics is None else diagnostics.profiler()
    comm.Barrier()
    start = MPI.Wtime()
    result = operation() if profiler is None else profiler.runcall(operation)
    elapsed = MPI.Wtime() - start
    timings = np.asarray(comm.allgather(elapsed), dtype=np.float64)
    summary = {
        "min": float(np.min(timings)),
        "mean": float(np.mean(timings)),
        "max": float(np.max(timings)),
    }
    if diagnostics is not None:
        diagnostics.finish(phase, profiler)
    return result, summary


def emit_progress(comm, args, phase, state, **details):
    """Persist a rank-zero phase checkpoint without changing JSON stdout."""
    if comm.rank != 0:
        return
    record = {
        "schema": PROGRESS_SCHEMA,
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "run_label": args.run_label,
        "ranks": comm.size,
        "phase": phase,
        "state": state,
        **details,
    }
    line = json.dumps(record, sort_keys=True)
    print(f"PROGRESS {line}", file=sys.stderr, flush=True)
    if args.progress_output is not None:
        try:
            args.progress_output.parent.mkdir(parents=True, exist_ok=True)
            with args.progress_output.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            print(
                f"WARNING could not persist progress checkpoint: {error}",
                file=sys.stderr,
                flush=True,
            )


def timed_named_phase(comm, args, phase, operation):
    """Time a collective operation while recording durable boundaries."""
    emit_progress(comm, args, phase, "STARTED")
    try:
        result, timing = timed_phase(comm, operation, args.diagnostics, phase)
    except BaseException as error:
        emit_progress(
            comm,
            args,
            phase,
            "FAILED",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    emit_progress(comm, args, phase, "COMPLETED", timing_seconds=timing)
    return result, timing


def summarize(values):
    values = np.asarray(values)
    return {
        "min": values.min().item(),
        "mean": float(values.mean()),
        "max": values.max().item(),
        "sum": values.sum().item(),
    }


def canonical_numeric_bytes(values, decimals):
    """Serialize rounded real or complex data independently of native endian."""
    values = np.asarray(values)
    real = np.round(values.real, decimals=decimals)
    real[real == 0] = 0
    if np.iscomplexobj(values):
        imag = np.round(values.imag, decimals=decimals)
        imag[imag == 0] = 0
        values = np.stack((real, imag), axis=-1)
    else:
        values = real
    return np.ascontiguousarray(values, dtype="<f8").tobytes()


def solution_checksum(
    comm,
    edge_ids,
    edge_values,
    vertex_values,
    decimals,
):
    """Return a canonical checksum independent of the edge partition."""
    local_words = np.zeros(4, dtype=np.uint64)
    for edge, values in zip(edge_ids, edge_values):
        payload = (
            b"edge"
            + struct.pack("<q", int(edge))
            + canonical_numeric_bytes(values, decimals)
        )
        digest = hashlib.sha256(payload).digest()
        local_words ^= np.frombuffer(digest, dtype="<u8").astype(np.uint64)
    if comm.rank == 0:
        payload = (
            b"vertices"
            + canonical_numeric_bytes(vertex_values, decimals)
        )
        digest = hashlib.sha256(payload).digest()
        local_words ^= np.frombuffer(digest, dtype="<u8").astype(np.uint64)
    global_words = np.empty_like(local_words)
    comm.Allreduce(local_words, global_words, op=MPI.BXOR)
    return global_words.astype("<u8", copy=False).tobytes().hex()


def build_problem(args, num_edges, dtype):
    network = build_network(
        args.graph,
        num_edges,
        metric=args.metric,
        seed=args.graph_seed,
    )
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=args.partition,
        network_vertex_storage=args.vertex_storage,
    )
    basis = d3.NetworkBasis(network, size=args.resolution)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)

    exact_vertices = manufactured_vertex_values(network, dtype)
    g_values = manufactured_vertex_rhs(network, exact_vertices)
    g.change_layout("g")
    g.set_global_data(g_values[:, None])

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U + R.T @ ntrace(u) = g")
    solver = problem.build_solver(network_solver="dtn")
    return {
        "network": network,
        "dist": dist,
        "basis": basis,
        "u": u,
        "U": U,
        "tau_1": tau_1,
        "tau_2": tau_2,
        "g": g,
        "exact_vertices": exact_vertices,
        "R": R,
        "dx": dx,
        "lift": lift,
        "solver": solver,
    }


def correctness_record(context, args, comm):
    network = context["network"]
    dist = context["dist"]
    basis = context["basis"]
    u = context["u"]
    U = context["U"]
    tau_1 = context["tau_1"]
    tau_2 = context["tau_2"]
    g = context["g"]
    exact_vertices = context["exact_vertices"]
    R = context["R"]
    dx = context["dx"]
    lift = context["lift"]

    edge_grid, s = dist.local_grids(basis)
    edge_ids = np.asarray(edge_grid, dtype=np.int64).ravel()
    endpoints = network.endpoints[edge_ids]
    tails = exact_vertices[endpoints[:, 0], None]
    heads = exact_vertices[endpoints[:, 1], None]
    exact_edges = tails + (heads - tails) * s

    edge_error = np.max(np.abs(u["g"] - exact_edges), initial=0)
    global_vertices = U.allgather_data("g").ravel()
    vertex_error = np.max(
        np.abs(global_vertices - exact_vertices),
        initial=0,
    )
    solution_error = comm.allreduce(
        float(max(edge_error, vertex_error)),
        op=MPI.MAX,
    )
    local_solution_scale = max(
        1,
        np.max(np.abs(exact_edges), initial=0),
        np.max(np.abs(exact_vertices), initial=0),
    )
    solution_scale = comm.allreduce(
        float(local_solution_scale),
        op=MPI.MAX,
    )
    relative_solution_error = solution_error / solution_scale

    continuity = (d3.dtrace(u) - R @ U).evaluate()
    vertex_residual = (U + R.T @ d3.ntrace(u) - g).evaluate()
    edge_residual = (
        -dx(dx(u)) + lift(tau_1, -1) + lift(tau_2, -2)
    ).evaluate()
    local_residuals = np.array(
        (
            np.max(np.abs(edge_residual["g"]), initial=0),
            np.max(np.abs(continuity["g"]), initial=0),
            np.max(np.abs(vertex_residual["g"]), initial=0),
        ),
        dtype=np.float64,
    )
    residuals = np.empty_like(local_residuals)
    comm.Allreduce(local_residuals, residuals, op=MPI.MAX)
    residual = float(np.max(residuals))
    residual_scales = np.array(
        (
            max(1, np.max(np.abs(exact_edges), initial=0)),
            max(1, np.max(np.abs(exact_vertices), initial=0)),
            max(1, np.max(np.abs(g["g"]), initial=0)),
        ),
        dtype=np.float64,
    )
    global_residual_scales = np.empty_like(residual_scales)
    comm.Allreduce(residual_scales, global_residual_scales, op=MPI.MAX)
    scaled_residuals = residuals / global_residual_scales
    scaled_residual = float(np.max(scaled_residuals))

    actual_checksum = solution_checksum(
        comm,
        edge_ids,
        u["g"],
        global_vertices,
        args.checksum_decimals,
    )
    expected_checksum = solution_checksum(
        comm,
        edge_ids,
        exact_edges,
        exact_vertices,
        args.checksum_decimals,
    )

    # Measure the normwise backward error of the actual assembled system.
    # This remains meaningful as graph degree and spectral resolution change,
    # unlike a single absolute threshold for heterogeneous grid equations.
    solver = context["solver"]
    subproblem = solver.subproblems[0]
    for field in solver.state:
        field.change_layout("c")
    for field in solver.F:
        field.change_layout("c")
    state_vector = subproblem.gather_inputs(solver.state).copy()
    if subproblem.size:
        rhs_vector = subproblem.gather_outputs(solver.F).copy()
    else:
        # Resident Network placeholders keep idle ranks in collective matrix
        # application and DtN solves but have no local equation data to
        # gather through the subsystem's Cython scatter plan.
        rhs_vector = np.empty((0, 0), dtype=dist.dtype)
    matrix_state = solver.apply_subproblem_matrix(
        subproblem.L_min,
        state_vector,
        subproblem=subproblem,
    )
    absolute_row_sums = solver.apply_subproblem_matrix(
        abs(subproblem.L_min),
        np.ones(state_vector.shape, dtype=np.float64),
        subproblem=subproblem,
    )
    local_backward_residual = np.max(
        np.abs(matrix_state - rhs_vector),
        initial=0,
    )
    local_matrix_norm = np.max(np.abs(absolute_row_sums), initial=0)
    local_state_norm = np.max(np.abs(state_vector), initial=0)
    local_rhs_norm = np.max(np.abs(rhs_vector), initial=0)
    backward_residual, matrix_norm, state_norm, rhs_norm = (
        comm.allreduce(float(value), op=MPI.MAX)
        for value in (
            local_backward_residual,
            local_matrix_norm,
            local_state_norm,
            local_rhs_norm,
        )
    )
    backward_scale = matrix_norm * state_norm + rhs_norm
    backward_error = backward_residual / max(
        backward_scale,
        np.finfo(np.float64).tiny,
    )
    passed = (
        relative_solution_error <= args.check_tolerance
        and backward_error <= args.check_tolerance
    )
    return {
        "passed": bool(passed),
        "tolerance": args.check_tolerance,
        "solution_error_max": solution_error,
        "solution_scale_max": solution_scale,
        "solution_error_relative_max": relative_solution_error,
        "backward_residual_norm_coefficient_inf": backward_residual,
        "backward_denominator_coefficient_inf": backward_scale,
        "backward_matrix_norm_coefficient_inf": matrix_norm,
        "backward_state_norm_coefficient_inf": state_norm,
        "backward_rhs_norm_coefficient_inf": rhs_norm,
        "normwise_backward_error_coefficient_inf": backward_error,
        "residual_max": residual,
        "residual_scaled_max": scaled_residual,
        "residual_components": {
            "edge_equation": float(residuals[0]),
            "continuity": float(residuals[1]),
            "vertex_equation": float(residuals[2]),
        },
        "residual_scales": {
            "edge_equation": float(global_residual_scales[0]),
            "continuity": float(global_residual_scales[1]),
            "vertex_equation": float(global_residual_scales[2]),
        },
        "checksum_decimals": args.checksum_decimals,
        "solution_checksum": actual_checksum,
        "manufactured_checksum": expected_checksum,
        "checksum_matches_manufactured": actual_checksum == expected_checksum,
    }


def environment_record(comm):
    hostnames = comm.gather(platform.node(), root=0)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    rank_affinities = comm.gather(affinity, root=0)
    if comm.rank != 0:
        return None
    host_counts = {
        hostname: hostnames.count(hostname)
        for hostname in sorted(set(hostnames))
    }
    build_dependencies = getattr(np.__config__, "CONFIG", {}).get(
        "Build Dependencies",
        {},
    )

    def dependency_record(name):
        dependency = build_dependencies.get(name, {})
        return {
            key: dependency.get(key)
            for key in ("name", "version", "openblas configuration")
            if key in dependency
        }

    return {
        "python": platform.python_version(),
        "dedalus": dedalus.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "linear_algebra": {
            "blas": dependency_record("blas"),
            "lapack": dependency_record("lapack"),
        },
        "mpi4py": mpi4py.__version__,
        "mpi_library": MPI.Get_library_version().rstrip("\x00").strip(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "affinity_cpus": (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "nodes": len(host_counts),
        "rank_host_counts": host_counts,
        "rank_bindings": [
            {"hostname": hostname, "cpus": cpus}
            for hostname, cpus in zip(hostnames, rank_affinities)
        ],
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_MAX_THREADS",
            )
        },
        "scheduler_and_binding": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_NTASKS",
                "SLURM_TASKS_PER_NODE",
                "SLURM_CPUS_PER_TASK",
                "SLURM_CPU_BIND",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "OMPI_MCA_hwloc_base_binding_policy",
                "MPICH_RANK_REORDER_METHOD",
            )
        },
    }


def source_record():
    """Identify the exact source tree and benchmark script when available."""
    source_root = Path(dedalus.__file__).resolve().parent.parent

    def git(*arguments):
        try:
            result = subprocess.run(
                ("git",) + arguments,
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "benchmark_script_sha256": script_digest,
    }


def main(argv=None):
    if argv is None:
        parse_argv = None
        command_argv = list(sys.argv)
    else:
        parse_argv = list(argv)
        command_argv = [sys.argv[0], *parse_argv]
    args = parse_args(parse_argv)
    if not args.verbose:
        logging.root.setLevel(logging.WARNING)
    if args.resolution <= 2:
        raise ValueError("--resolution must be greater than two.")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("Require --warmups >= 0 and --repeats > 0.")
    if not np.isfinite(args.check_tolerance) or args.check_tolerance <= 0:
        raise ValueError("--check-tolerance must be finite and positive.")
    if not 0 <= args.checksum_decimals <= 15:
        raise ValueError("--checksum-decimals must lie in [0, 15].")

    comm = MPI.COMM_WORLD
    args.diagnostics = PhaseDiagnostics(comm, args.profile_directory)
    args.diagnostics.finish("before_setup", None)
    num_edges, requested_edges_per_rank = resolve_size(args, comm.size)
    dtype = np.dtype(args.dtype)
    emit_progress(
        comm,
        args,
        "benchmark",
        "STARTED",
        scaling=args.scaling,
        graph=args.graph,
        edges=num_edges,
        resolution=args.resolution,
    )

    context, setup_timing = timed_named_phase(
        comm,
        args,
        "setup",
        lambda: build_problem(args, num_edges, dtype),
    )
    solver = context["solver"]

    _, assembly_timing = timed_named_phase(
        comm,
        args,
        "matrix_assembly",
        lambda: solver.build_matrices(
            solver.subproblems,
            matrices=("L",),
        ),
    )

    def factorize():
        for subproblem in solver.subproblems:
            solver.subproblem_matsolvers[subproblem] = (
                solver.build_subproblem_solver(
                    subproblem,
                    subproblem.L_min,
                )
            )

    _, factorization_timing = timed_named_phase(
        comm,
        args,
        "factorization",
        factorize,
    )
    if args.profile_directory is not None:
        for factorization in solver.subproblem_matsolvers.values():
            args.diagnostics.instrument_separator(
                factorization.subgraph_solver.separator_solver,
            )
    _, first_solve_timing = timed_named_phase(
        comm,
        args,
        "first_solve",
        solver.solve,
    )

    emit_progress(
        comm,
        args,
        "warmup_solves",
        "STARTED",
        count=args.warmups,
    )
    warmup_started = MPI.Wtime()
    if args.warmups:
        comm.Barrier()
        for _ in range(args.warmups):
            solver.solve()
        comm.Barrier()
    warmup_elapsed = MPI.Wtime() - warmup_started
    warmup_elapsed_max = comm.allreduce(warmup_elapsed, op=MPI.MAX)
    emit_progress(
        comm,
        args,
        "warmup_solves",
        "COMPLETED",
        count=args.warmups,
        elapsed_seconds_max=float(warmup_elapsed_max),
    )

    emit_progress(
        comm,
        args,
        "repeated_solves",
        "STARTED",
        count=args.repeats,
    )
    repeated_timings = []
    repeated_rank_timings = []
    for repeat_index in range(args.repeats):
        _, timing = timed_phase(comm, solver.solve, args.diagnostics, f"cached_solve_{repeat_index + 1}")
        repeated_timings.append(timing["max"])
        repeated_rank_timings.append(timing)
        emit_progress(
            comm,
            args,
            "repeated_solve",
            "COMPLETED",
            repeat=repeat_index + 1,
            count=args.repeats,
            timing_seconds=timing,
        )
    emit_progress(
        comm,
        args,
        "repeated_solves",
        "COMPLETED",
        count=args.repeats,
    )

    emit_progress(comm, args, "correctness", "STARTED")
    correctness_started = MPI.Wtime()
    correctness = correctness_record(context, args, comm)
    correctness_elapsed = MPI.Wtime() - correctness_started
    correctness_elapsed_max = comm.allreduce(correctness_elapsed, op=MPI.MAX)
    args.diagnostics.finish("after_correctness", None)
    emit_progress(
        comm,
        args,
        "correctness",
        "COMPLETED",
        passed=bool(correctness["passed"]),
        elapsed_seconds_max=float(correctness_elapsed_max),
    )
    factorization = next(iter(solver.subproblem_matsolvers.values()))
    dist = context["dist"]
    partition = dist.network_partition
    network = context["network"]
    subproblem = solver.subproblems[0]

    owned_edge_counts = np.asarray(
        comm.allgather(factorization.owned_edges.size),
        dtype=np.int64,
    )
    interior_vertex_counts = np.asarray(
        comm.allgather(partition.interior_vertices[comm.rank].size),
        dtype=np.int64,
    )
    local_matrix_rows = np.asarray(
        comm.allgather(subproblem.L_min.shape[0]),
        dtype=np.int64,
    )
    local_matrix_nnz = np.asarray(
        comm.allgather(subproblem.L_min.nnz),
        dtype=np.int64,
    )
    owner_hash = hashlib.sha256(
        np.asarray(partition.edge_owners, dtype="<i8").tobytes()
    ).hexdigest()

    repeated_summary = {
        "min": float(np.min(repeated_timings)),
        "median": float(np.median(repeated_timings)),
        "mean": float(np.mean(repeated_timings)),
        "max": float(np.max(repeated_timings)),
    }
    sum_of_phase_maxima = (
        assembly_timing["max"]
        + factorization_timing["max"]
        + first_solve_timing["max"]
    )
    environment = environment_record(comm)
    detailed_solver = solver_record(factorization, dist, comm)
    args.diagnostics.finish("after_solver_telemetry", None)

    if comm.rank == 0:
        separator_solver = factorization.subgraph_solver.separator_solver
        separator_matrix = factorization.schur_matrix
        record = {
            "schema": SCHEMA,
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "run_label": args.run_label,
            "command": " ".join(shlex.quote(arg) for arg in command_argv),
            "scaling": args.scaling,
            "ranks": comm.size,
            "graph": {
                "family": args.graph,
                "metric": args.metric,
                "seed": args.graph_seed,
                "edges": network.num_edges,
                "vertices": network.num_vertices,
                "degree_mean": float(np.mean(network.degree)),
                "degree_max": int(np.max(network.degree)),
                "topology_geometry_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        network.endpoints,
                        dtype="<i8",
                    ).tobytes()
                    + np.ascontiguousarray(
                        network.lengths,
                        dtype="<f8",
                    ).tobytes()
                ).hexdigest(),
            },
            "discretization": {
                "dtype": dtype.name,
                "resolution_per_edge": args.resolution,
                "global_edge_grid_points": int(
                    network.num_edges * args.resolution
                ),
                "global_state_unknowns": int(
                    network.num_edges * (args.resolution + 2)
                    + network.num_vertices
                ),
                "requested_edges_per_rank": requested_edges_per_rank,
            },
            "partition": {
                "strategy": args.partition,
                "vertex_storage": dist.network_vertex_storage,
                "edge_owner_sha256": owner_hash,
                "separator_vertices": int(
                    partition.separator_vertices.size
                ),
                "separator_fraction": float(
                    partition.separator_vertices.size
                    / network.num_vertices
                ),
                "owned_edges": summarize(owned_edge_counts),
                "edge_imbalance_max_over_mean": float(
                    owned_edge_counts.max() / owned_edge_counts.mean()
                ),
                "interior_vertices": summarize(interior_vertex_counts),
                "idle_ranks": int(np.count_nonzero(owned_edge_counts == 0)),
            },
            "matrices": {
                "local_rows": summarize(local_matrix_rows),
                "local_nnz": summarize(local_matrix_nnz),
                "separator_rows": int(separator_matrix.shape[0]),
                "separator_nnz": int(separator_matrix.nnz),
                "separator_nnz_scope": "rank_zero_additive_local_contribution; see solver_by_rank",
                "separator_factorized": separator_solver is not None,
                "solver_backend": (
                    f"{solver.matsolver.__module__}."
                    f"{solver.matsolver.__qualname__}"
                ),
                "settings": {
                    "ncc_cutoff": solver.ncc_cutoff,
                    "max_ncc_terms": solver.max_ncc_terms,
                    "entry_cutoff": solver.entry_cutoff,
                    "bc_top": solver.bc_top,
                    "tau_left": solver.tau_left,
                    "interleave_components": solver.interleave_components,
                    "store_expanded_matrices": solver.store_expanded_matrices,
                },
            },
            "timings": {
                "setup_seconds": setup_timing,
                "matrix_assembly_seconds": assembly_timing,
                "factorization_seconds": factorization_timing,
                "first_solve_seconds": first_solve_timing,
                "sum_of_phase_maxima_seconds": sum_of_phase_maxima,
                "warmup_solves": args.warmups,
                "repeated_solve_max_seconds": repeated_timings,
                "repeated_solve_rank_seconds": repeated_rank_timings,
                "repeated_solve_summary_seconds": repeated_summary,
                "repeated_edge_grid_points_per_second": float(
                    network.num_edges
                    * args.resolution
                    / repeated_summary["median"]
                ),
            },
            "diagnostics": args.diagnostics.record(),
            "solver_by_rank": detailed_solver,
            "correctness": correctness,
            "environment": environment,
            "source": source_record(),
        }
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if args.output is not None:
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

    emit_progress(
        comm,
        args,
        "benchmark",
        "COMPLETED",
        correctness_passed=bool(correctness["passed"]),
    )

    if not correctness["passed"]:
        raise RuntimeError(
            "Network DtN benchmark failed its manufactured-solution check."
        )


if __name__ == "__main__":
    main()
