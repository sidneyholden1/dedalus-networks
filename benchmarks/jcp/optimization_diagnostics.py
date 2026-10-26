"""Opt-in per-phase cProfile traces and explicitly scoped memory/solver data.

Profiling never runs in primary timing cases. Trace serialization, RSS reads,
and telemetry collectives run outside measured operations. cProfile cumulative
times overlap: use caller edges/self times, never sum nested cumulative times.
"""

import cProfile
from dataclasses import asdict
import json
from pathlib import Path
import platform
import pstats
import resource
import sys
import time


class SeparatorCollectiveDiagnostics:
    """Opt-in MPI-call elapsed time, including rank waiting, for one solver.

    Payload bytes describe local logical inputs, not network traffic. Object
    collectives without an ndarray/scalar nbytes property have unknown size.
    No extra collective is introduced inside a timed operation.
    """

    def __init__(self, comm):
        self._comm = comm
        self.calls = {}

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def _call(self, method, caller, value, *args, **kwargs):
        start = time.perf_counter()
        try:
            return getattr(self._comm, method)(value, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            row = self.calls.setdefault(f"{method}/{caller}", {
                "calls": 0, "seconds": 0.0, "input_bytes": 0,
                "unknown_payload_calls": 0,
            })
            row["calls"] += 1
            row["seconds"] += elapsed
            payload = getattr(value, "nbytes", None)
            if payload is None:
                row["unknown_payload_calls"] += 1
            else:
                row["input_bytes"] += int(payload)

    def Allreduce(self, value, *args, **kwargs):
        return self._call("Allreduce", sys._getframe(1).f_code.co_name,
                          value, *args, **kwargs)

    def allreduce(self, value, *args, **kwargs):
        return self._call("allreduce", sys._getframe(1).f_code.co_name,
                          value, *args, **kwargs)

    def allgather(self, value, *args, **kwargs):
        return self._call("allgather", sys._getframe(1).f_code.co_name,
                          value, *args, **kwargs)


def memory_record():
    """Linux process RSS; peaks are lifetime peaks, not simultaneous node use."""
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, unit = line.split()
            values[key.rstrip(":") + "_bytes"] = int(value) * 1024
    values["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    values["hostname"] = platform.node()
    return values


class PhaseDiagnostics:
    def __init__(self, comm, directory=None):
        self.comm = comm
        self.directory = directory
        self.memory = {}
        self.profiles = {}
        self.separator_meters = []
        self.separator_collectives = {}

    def instrument_separator(self, solver):
        if self.directory is not None and solver is not None:
            meter = SeparatorCollectiveDiagnostics(solver.comm)
            solver.comm = meter
            self.separator_meters.append(meter)

    def profiler(self):
        # Warmup solves between phases are intentionally excluded.
        for meter in self.separator_meters:
            meter.calls = {}
        return cProfile.Profile() if self.directory is not None else None

    def finish(self, phase, profiler):
        self.memory[phase] = self.comm.gather(memory_record(), root=0)
        if profiler is None:
            return
        self.separator_collectives[phase] = self.comm.gather(
            [meter.calls for meter in self.separator_meters], root=0,
        )
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"{phase}.rank{self.comm.rank:04d}"
        profiler.dump_stats(str(stem) + ".prof")
        stats = pstats.Stats(profiler)
        rows = []
        for key, (primitive, calls, own, cumulative, callers) in stats.stats.items():
            rows.append({
                "function": list(key), "primitive_calls": primitive,
                "calls": calls, "self_seconds": own,
                "cumulative_seconds": cumulative,
                "callers": [{"function": list(caller), "stats": value}
                            for caller, value in callers.items()],
            })
        rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
        Path(str(stem) + ".json").write_text(json.dumps(rows, indent=2) + "\n")
        self.profiles[phase] = self.comm.gather(str(stem) + ".prof", root=0)

    def record(self):
        return {
            "profiled": self.directory is not None,
            "profile_files_by_rank": self.profiles,
            "memory_by_phase_and_rank": self.memory,
            "memory_scope": "Linux process current RSS and lifetime high-water RSS in bytes; rank-ordered samples after each phase, not synchronized node peaks",
            "profile_scope": "cProfile per rank inside timed operation; trace writing and telemetry excluded; cumulative times overlap",
            "separator_collectives_by_phase_and_rank": self.separator_collectives,
            "separator_collective_scope": "Opt-in local MPI-call elapsed seconds, including waiting; logical local input bytes, not network traffic; reset before each profiled phase; warmups excluded",
        }


def sparse_record(matrix):
    return {"shape": list(matrix.shape), "nnz": int(matrix.nnz),
            "storage_bytes": int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)}


def lu_record(factor):
    lu = getattr(factor, "LU", None)
    if lu is None:
        return {"exposed": False}
    return {"exposed": True, "L": sparse_record(lu.L), "U": sparse_record(lu.U)}


def solver_record(factorization, dist, comm):
    """Local sparse storage and actual separator policy, without densifying."""
    subgraph = factorization.subgraph_solver
    separator = subgraph.separator_solver
    edge_lu = [lu_record(factor) for _, _, factor in factorization.edge_solver.blocks.values()]
    exposed = [record for record in edge_lu if record["exposed"]]
    local = {
        "edge_blocks": len(edge_lu), "edge_lu_exposed_blocks": len(exposed),
        "edge_lu_nnz": sum(record[side]["nnz"] for record in exposed for side in ("L", "U")),
        "edge_lu_storage_bytes": sum(record[side]["storage_bytes"] for record in exposed for side in ("L", "U")),
        "edge_response": sparse_record(factorization.edge_response),
        "interior_matrix": sparse_record(subgraph.interior_matrix),
        "interior_response": sparse_record(subgraph.interior_response),
        "interior_lu": lu_record(subgraph.interior_solver),
        "separator_additive_local_matrix": sparse_record(factorization.schur_matrix),
    }
    if separator is not None:
        local["separator"] = {
            "method": separator.method, "rtol": separator.rtol, "atol": separator.atol,
            "restart": separator.restart, "maxiter": separator.maxiter,
            "preconditioner": separator.preconditioner,
            "last_solve_info": [asdict(info) for info in separator.last_solve_info],
            "replicated_dense_lu_bytes": 0 if separator._lu is None else sum(a.nbytes for a in separator._lu),
            "replicated_global_matrix": None if separator.global_matrix is None else sparse_record(separator.global_matrix),
        }
    plan = getattr(dist, "network_vertex_exchange_plan", None)
    if plan is not None:
        local["halo"] = {
            "resident_vertices": int(plan.resident.size),
            "owned_vertices": int(plan.owned.size), "ghost_vertices": int(plan.ghost.size),
            "forward_send_vertices_per_peer": [int(p.size) for p in plan.forward_send_positions],
            "forward_receive_vertices_per_peer": [int(p.size) for p in plan.forward_recv_positions],
        }
    return comm.gather(local, root=0)
