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
"""

import logging

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
while solver.proceed:
    solver.step(timestep)

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
    logger.info(
        "Heterogeneous Network heat solve passed: t=%.3f, max error=%.3e",
        solver.sim_time,
        error,
    )
