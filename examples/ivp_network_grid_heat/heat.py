"""Heat diffusion on a random triangulation of a disc, or a square network.

Each rod solves dt(u) = diffusivity * dx(dx(u)). Temperature is continuous
and the sum of outward heat fluxes is zero at every junction, including the
outer boundary. A smooth central hot spot spreads through the insulated network;
total heat is conserved while temperature differences decay.

Run ``python heat.py`` to save HDF5 snapshots and a colour animation in
``network_heat/``. Use ``--output`` to select another output directory.
The solve also supports MPI. ``--no-render`` saves data without plotting;
``--render-only`` regenerates the animation from existing snapshots.
"""

import argparse
import json
import logging
import math
from pathlib import Path
import time

from mpi4py import MPI
import numpy as np

import dedalus.public as d3

logger = logging.getLogger(__name__)


def square_grid():
    xy = np.array([(x, y) for y in range(5) for x in range(5)], dtype=float)
    endpoints = []
    for y in range(5):
        for x in range(5):
            vertex = 5 * y + x
            if x < 4:
                endpoints.append((vertex, vertex + 1))
            if y < 4:
                endpoints.append((vertex, vertex + 5))
    return xy, np.asarray(endpoints)


def initial_temperature(xy):
    return np.prod(np.sin(np.pi * xy / 4) ** 8, axis=-1)


def disc_triangulation(seed):
    """100 vertices: 20 irregular boundary points and 80 random interior points."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * (np.arange(20) + rng.uniform(-0.25, 0.25, 20)) / 20
    points = list(np.column_stack((np.cos(angles), np.sin(angles))))
    for attempt in range(10000):
        radius = 0.92 * np.sqrt(rng.random())
        angle = rng.uniform(0, 2 * np.pi)
        point = radius * np.array([np.cos(angle), np.sin(angle)])
        if np.min(np.linalg.norm(np.asarray(points) - point, axis=1)) > 0.075:
            points.append(point)
        if len(points) == 100:
            break
    else:
        raise RuntimeError("Could not place 100 separated disc vertices")
    xy = np.asarray(points)
    triangles = Delaunay(xy).simplices
    sides = np.concatenate([triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))])
    endpoints = np.unique(np.sort(sides, axis=1), axis=0)
    return xy, endpoints, triangles


def solve(args):
    comm = MPI.COMM_WORLD
    if args.geometry == "disc":
        xy, endpoints, triangles = disc_triangulation(args.seed)
    else:
        xy, endpoints = square_grid()
        triangles = np.empty((0, 3), dtype=int)
    lengths = np.linalg.norm(xy[endpoints[:, 1]] - xy[endpoints[:, 0]], axis=1)
    network = d3.Network(endpoints=endpoints, lengths=lengths)
    dist = d3.Distributor(
        network, comm=comm, dtype=np.float64,
        network_partition="connected", network_vertex_storage="auto",
    )
    basis = d3.NetworkBasis(network, size=args.modes)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    dx = basis.derivative
    dt = d3.TimeDerivative
    R = basis.restriction
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda value, n: d3.Lift(value, lift_basis, n)
    diffusivity = 1.0

    problem = d3.IVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "dt(u) - diffusivity*dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = 0")

    edge, s = basis.local_grids(dist, scales=(1, 1))
    indices = edge.astype(int).ravel()
    tail = xy[endpoints[indices, 0]][:, None, :]
    head = xy[endpoints[indices, 1]][:, None, :]
    if args.geometry == "disc":
        vertex_temperature = np.exp(-np.sum(xy**2, axis=1) / (2 * 0.2**2))
        vertex_temperature /= vertex_temperature.max()
        # A cubic interpolation has zero derivative at both ends of each rod,
        # so the initial pulse already satisfies Kirchhoff at irregular nodes.
        blend = 3 * s**2 - 2 * s**3
        left = vertex_temperature[endpoints[indices, 0]][:, None]
        right = vertex_temperature[endpoints[indices, 1]][:, None]
        u["g"] = left * (1 - blend) + right * blend
        initial_condition = "Gaussian vertex pulse (sigma=0.2), cubic smoothstep on rods"
    else:
        points = tail + s.reshape(1, -1, 1) * (head - tail)
        u["g"] = initial_temperature(points)
        vertex_temperature = initial_temperature(xy)
        initial_condition = "sin(pi*x/4)^8 * sin(pi*y/4)^8"
    U.change_layout("g")
    U.set_global_data(vertex_temperature[:, None])

    solver = problem.build_solver(d3.SBDF2, network_solver="dtn")
    steps_per_frame = max(
        1, math.ceil(args.stop_time / args.timestep / (args.frames - 1))
    )
    steps = steps_per_frame * (args.frames - 1)
    timestep = args.stop_time / steps
    snapshots = solver.evaluator.add_file_handler(
        args.output / "snapshots", mode="overwrite", max_writes=args.frames + 1
    )
    snapshots.add_task(u, name="u")
    snapshots.add_task(U, name="U")
    solver.evaluate_handlers([snapshots], dt=timestep)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        solver.step(timestep)
        if step % steps_per_frame == 0:
            solver.evaluate_handlers([snapshots], dt=timestep)
    elapsed = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)

    if comm.rank == 0:
        metadata = {
            "vertices": xy.tolist(), "endpoints": endpoints.tolist(),
            "geometry": args.geometry, "seed": args.seed,
            "triangles": triangles.tolist(), "lengths": lengths.tolist(),
            "diffusivity": diffusivity, "modes_per_rod": args.modes,
            "timestep": timestep, "steps": steps, "frames": args.frames,
            "final_time": solver.sim_time, "mpi_ranks": comm.size,
            "solve_seconds": elapsed, "boundary": "insulated; Kirchhoff at all vertices",
            "initial_condition": initial_condition,
        }
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info("Saved %d states in %.2f seconds", args.frames, elapsed)


def verify(output):
    """Check saved fields using polynomial reconstruction and physical quadrature."""
    import h5py
    from scipy.interpolate import BarycentricInterpolator

    metadata = json.loads((output / "run.json").read_text())
    endpoints = np.asarray(metadata["endpoints"])
    lengths = np.asarray(metadata.get("lengths", np.ones(len(endpoints))))
    with h5py.File(output / "snapshots/snapshots_s1.h5", "r") as file:
        task = file["tasks/u"]
        values = task[:]
        vertices = file["tasks/U"][:, :, 0]
        s = task.dims[2][0][:]
        times = file["scales/sim_time"][:]
    assert len(times) == metadata["frames"] and np.all(np.diff(times) > 0)
    assert np.isfinite(values).all() and np.isfinite(vertices).all()
    q, weights = np.polynomial.legendre.leggauss(max(40, 2 * len(s)))
    interpolation = BarycentricInterpolator(s, np.eye(len(s)))
    samples = values @ interpolation((q + 1) / 2).T
    mass = np.einsum("teq,e,q->t", samples, lengths, weights / 2)
    energy = np.einsum("teq,e,q->t", samples**2, lengths, weights / 2)
    coefficients = np.polynomial.chebyshev.chebfit(
        2 * s - 1, values.reshape(-1, len(s)).T, len(s) - 1
    )
    derivative = 2 * np.polynomial.chebyshev.chebder(coefficients)
    left = np.polynomial.chebyshev.chebval(-1, coefficients).reshape(values.shape[:2])
    right = np.polynomial.chebyshev.chebval(1, coefficients).reshape(values.shape[:2])
    flux_left = -np.polynomial.chebyshev.chebval(-1, derivative).reshape(values.shape[:2])
    flux_right = np.polynomial.chebyshev.chebval(1, derivative).reshape(values.shape[:2])
    flux_left /= lengths
    flux_right /= lengths
    flux = np.zeros_like(vertices)
    for edge, (tail, head) in enumerate(endpoints):
        flux[:, tail] += flux_left[:, edge]
        flux[:, head] += flux_right[:, edge]
    continuity = max(
        np.max(np.abs(left - vertices[:, endpoints[:, 0]])),
        np.max(np.abs(right - vertices[:, endpoints[:, 1]])),
    )
    checks = {
        "mass_relative_drift": float(np.max(np.abs(mass / mass[0] - 1))),
        "maximum_energy_increase": float(np.max(np.diff(energy))),
        "minimum_temperature": float(min(values.min(), vertices.min())),
        "maximum_temperature": float(max(values.max(), vertices.max())),
        "continuity_residual": float(continuity),
        "kirchhoff_residual": float(np.max(np.abs(flux))),
        "initial_heat": float(mass[0]), "final_peak": float(vertices[-1].max()),
    }
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if checks["mass_relative_drift"] > 1e-7:
        raise RuntimeError(f"Heat conservation failed: {checks}")
    if checks["maximum_energy_increase"] > 1e-8:
        raise RuntimeError(f"Energy decay failed: {checks}")
    if checks["minimum_temperature"] < -2e-6 or checks["maximum_temperature"] > 1 + 2e-6:
        raise RuntimeError(f"Temperature bounds failed: {checks}")
    if continuity > 1e-8 or checks["kirchhoff_residual"] > 1e-6:
        raise RuntimeError(f"Junction conditions failed: {checks}")
    logger.info("Heat balance and junction checks passed: %s", checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("network_heat"))
    parser.add_argument("--geometry", choices=("disc", "square"), default="disc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--stop-time", type=float)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.timestep is None:
        args.timestep = 1 / 1200 if args.geometry == "disc" else 0.005
    if args.stop_time is None:
        args.stop_time = 0.4 if args.geometry == "disc" else 2.0
    if args.frames < 2 or args.modes < 8 or args.stop_time <= 0 or args.timestep <= 0:
        parser.error("Use at least two frames, eight modes and positive times.")
    if not args.render_only:
        solve(args)
    if MPI.COMM_WORLD.rank == 0:
        verify(args.output)
        if not args.no_render:
            from plot_heat import render
            render(args.output)


if __name__ == "__main__":
    main()
