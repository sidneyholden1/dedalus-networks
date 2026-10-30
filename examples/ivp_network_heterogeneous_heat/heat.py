"""
Dedalus script solving the heat equation on a heterogeneous metric network.
This script demonstrates mixed edge resolutions and interval families,
whole-edge MPI storage, a shared vertex unknown, continuity equations, and a
matrix-style Dirichlet/Kirchhoff vertex law. It should take just a few seconds
to run.

The four oriented edges form one chain. Every edge solves
    dt(u) - d_x(d_x(u)) = 0,
with homogeneous Dirichlet conditions at the two leaves and continuity plus
summed outward-flux balance at the three interior vertices. The initial state
is the global eigenmode sin(pi*x/L), so
    u(x, t) = exp(-(pi/L)^2*t) * sin(pi*x/L).

To run in serial or MPI:
    $ python3 heat.py
    $ mpiexec -n 4 python3 heat.py

Both commands save network_heat.png and HDF5 time snapshots under network_heat/
in the current working directory. The plot compares the final edge profiles
and vertex-temperature decay with the analytic solution.
"""

import logging
from pathlib import Path

from mpi4py import MPI
import numpy as np
from scipy import sparse

import dedalus.public as d3

logger = logging.getLogger(__name__)


# Parameters
dtype = np.float64
lengths = np.array((0.5, 1.0, 0.75, 1.25))
stop_sim_time = 0.02
timestep = 2e-4

# Network and heterogeneous bases
network = d3.Network(
    vertex_ids=("left", "v1", "v2", "v3", "right"),
    edge_ids=("e0", "e1", "e2", "e3"),
    endpoints=((0, 1), (1, 2), (2, 3), (3, 4)),
    lengths=lengths,
)
bases = d3.NetworkBasisSet(
    network,
    {
        "coarse": {
            "edge_indices": (0, 2),
            "size": 12,
        },
        "fine": {
            "edge_indices": (1, 3),
            "size": 18,
            "interval_basis": d3.Legendre,
        },
    },
)

# The cubic resolution estimate balances unequal spectral work while retaining
# whole edges. This also permits idle ranks when MPI has more ranks than edges.
comm = MPI.COMM_WORLD
partition = d3.NetworkPartition.connected(
    network,
    comm.size,
    weights=bases.edge_costs,
)
dist = d3.Distributor(
    network,
    comm=comm,
    dtype=dtype,
    network_partition=partition,
    network_vertex_storage="auto",
)
coarse = bases["coarse"]
fine = bases["fine"]

# Fields: one edge field and two edgewise taus per bucket, plus one shared U
u_coarse = dist.Field(name="u_coarse", bases=coarse)
u_fine = dist.Field(name="u_fine", bases=fine)
tau_c1 = dist.Field(name="tau_c1", bases=coarse.edges)
tau_c2 = dist.Field(name="tau_c2", bases=coarse.edges)
tau_f1 = dist.Field(name="tau_f1", bases=fine.edges)
tau_f2 = dist.Field(name="tau_f2", bases=fine.edges)
U = dist.Field(name="U", bases=bases.vertices)

# Substitutions
dt = d3.TimeDerivative
lift_c_basis = coarse.interval.derivative_basis(2)
lift_f_basis = fine.interval.derivative_basis(2)
lift_c = lambda A, n: d3.Lift(A, lift_c_basis, n)
lift_f = lambda A, n: d3.Lift(A, lift_f_basis, n)

# D selects Dirichlet vertices; K selects Kirchhoff vertices.
D = d3.CardinalMap(
    bases.vertices,
    bases.vertices,
    sparse.diags((1.0, 0, 0, 0, 1)),
    name="D",
)
K = d3.CardinalMap(
    bases.vertices,
    bases.vertices,
    sparse.diags((0.0, 1, 1, 1, 0)),
    name="K",
)

# Problem
problem = d3.IVP(
    [u_coarse, u_fine, U, tau_c1, tau_c2, tau_f1, tau_f2],
    network_interfaces=[(u_coarse, U), (u_fine, U)],
    namespace=locals(),
)
problem.add_equation(
    "dt(u_coarse) - coarse.dx(coarse.dx(u_coarse)) "
    "+ lift_c(tau_c1,-1) + lift_c(tau_c2,-2) = 0"
)
problem.add_equation(
    "dt(u_fine) - fine.dx(fine.dx(u_fine)) "
    "+ lift_f(tau_f1,-1) + lift_f(tau_f2,-2) = 0"
)
problem.add_equation("continuity(u_coarse) = 0")
problem.add_equation("continuity(u_fine) = 0")
problem.add_equation(
    "D @ U + K @ (coarse.R.T @ coarse.ntrace(u_coarse) "
    "+ fine.R.T @ fine.ntrace(u_fine)) = 0"
)

# Initial condition
total_length = lengths.sum()
offsets = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
wavenumber = np.pi / total_length
for field, basis in ((u_coarse, coarse), (u_fine, fine)):
    edge_index, _ = basis.local_grids(dist, scales=(1, 1))
    x = offsets[edge_index.astype(int)] + basis.local_physical_grid(dist)
    field["g"] = np.sin(wavenumber * x)
vertex_x = np.concatenate(([0.0], np.cumsum(lengths)))
U.change_layout("g")
U.set_global_data(np.sin(wavenumber * vertex_x)[:, None])

# Solver
solver = problem.build_solver(d3.SBDF2, network_solver="dtn")
solver.stop_sim_time = stop_sim_time

# Each heterogeneous bucket has its own dense task and grid metadata. The
# gather-mode handler writes all ranks' data to one file on rank zero.
snapshots = solver.evaluator.add_file_handler(
    "network_heat",
    sim_dt=stop_sim_time / 10,
    max_writes=100,
    mode="overwrite",
)
snapshots.add_task(u_coarse, name="u_coarse")
snapshots.add_task(u_fine, name="u_fine")
snapshots.add_task(U, name="U")
while solver.proceed:
    solver.step(timestep)

# Scheduled output is evaluated before a step; explicitly save the final state.
solver.evaluate_handlers([snapshots], dt=timestep)

# Analytic validation. Edge errors use owned rows; U is gathered from owners.
amplitude = np.exp(-(wavenumber**2) * solver.sim_time)
local_error = 0.0
for field, basis in ((u_coarse, coarse), (u_fine, fine)):
    edge_index, _ = basis.local_grids(dist, scales=(1, 1))
    x = offsets[edge_index.astype(int)] + basis.local_physical_grid(dist)
    if field["g"].size:
        local_error = max(
            local_error,
            np.max(np.abs(field["g"] - amplitude * np.sin(wavenumber * x))),
        )
edge_error = comm.allreduce(local_error, op=MPI.MAX)
vertex_error = np.max(
    np.abs(
        U.allgather_data("g").ravel()
        - amplitude * np.sin(wavenumber * vertex_x)
    )
)
error = max(edge_error, vertex_error)
if error > 2e-6:
    raise RuntimeError(
        f"Heterogeneous Network heat solution failed its analytic check: {error:.3e}"
    )
if comm.rank == 0:
    import h5py
    import matplotlib.pyplot as plt

    logger.info(
        "Heterogeneous Network heat solve passed: t=%.3f, max error=%.3e",
        solver.sim_time,
        error,
    )

    # Read the saved data, so the figure also demonstrates how to use the output.
    with h5py.File(snapshots.current_file, "r") as file:
        times = file["scales/sim_time"][:]
        vertex_values = file["tasks/U"][:, :, 0]
        profiles = [
            file[f"tasks/{name}"][-1] for name in ("u_coarse", "u_fine")
        ]

    figure, (profile_axis, decay_axis) = plt.subplots(2, 1, figsize=(8, 7))
    x_plot = np.linspace(0, total_length, 400)
    profile_axis.plot(
        x_plot, np.sin(wavenumber * x_plot), "--", color="0.6", label="Initial"
    )
    profile_axis.plot(
        x_plot, amplitude * np.sin(wavenumber * x_plot), "k-", label="Final analytic"
    )
    labels = (
        f"Chebyshev, {coarse.interval.size} modes",
        f"Legendre, {fine.interval.size} modes",
    )
    for values, basis, label, marker in zip(
        profiles, (coarse, fine), labels, ("o", "x")
    ):
        edge_index, s = basis.global_grids(dist, scales=(1, 1))
        edge_index = edge_index.astype(int)
        x = offsets[edge_index] + lengths[edge_index] * s
        profile_axis.plot(
            x.ravel(), values.ravel(), marker, markersize=4, label=label
        )
    for x_vertex in vertex_x[1:-1]:
        profile_axis.axvline(x_vertex, color="0.8", linewidth=0.8)
    profile_axis.set(
        xlabel="Distance along the four-edge chain", ylabel="Temperature u"
    )
    profile_axis.set_title(
        f"Heat equation at t = {solver.sim_time:.4f}; max error = {error:.2e}"
    )
    profile_axis.legend(fontsize="small", ncols=2)

    # v2 lies near the peak of the initial eigenmode. The short verification
    # run has modest decay; a separate time axis makes it visible.
    exact_vertex = np.exp(-wavenumber**2 * times) * np.sin(wavenumber * vertex_x[2])
    decay_axis.plot(times, exact_vertex, "k-", label="Analytic")
    decay_axis.plot(
        times, vertex_values[:, 2], "o", markersize=4, label="Saved solution"
    )
    decay_axis.set(xlabel="Time", ylabel="Temperature at vertex v2")
    decay_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    decay_axis.legend()
    figure.tight_layout()
    figure.savefig("network_heat.png", dpi=200)
    plt.close(figure)
    logger.info("Saved plot: %s", Path("network_heat.png").resolve())
    logger.info("Saved time snapshots: %s", snapshots.current_file)
