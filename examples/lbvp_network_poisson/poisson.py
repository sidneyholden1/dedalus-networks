"""
Dedalus script solving a Poisson equation on a three-edge metric network.
This script demonstrates the NetworkBasis interface, continuity and Kirchhoff
vertex conditions, whole-edge distributed storage, the hierarchical
Dirichlet-to-Neumann solver, and ordinary Dedalus HDF5 output. It should take
just a few seconds to run.

The network is a star with one central junction and three Dirichlet leaves.
Every edge solves
    -d_x(d_x(u)) = 0.
The solution is continuous at the junction and has zero summed outward flux.

To run and plot:
    $ python3 poisson.py
"""

import logging

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import dedalus.public as d3

logger = logging.getLogger(__name__)


# Parameters
N = 24
dtype = np.float64
vertices = ("left", "junction", "upper", "lower")
edges = ("inlet", "upper_branch", "lower_branch")
endpoints = ((0, 1), (1, 2), (1, 3))
lengths = (1, 1.5, 2)

# Network and basis
network = d3.Network(
    vertex_ids=vertices,
    edge_ids=edges,
    endpoints=endpoints,
    lengths=lengths,
)
dist = d3.Distributor(
    network,
    dtype=dtype,
    network_partition="connected",
    network_vertex_storage="auto",
)
basis = d3.NetworkBasis(network, size=N)

# Coordinate grids: s is the shared reference coordinate, while x is local
# physical arclength and therefore has one row for each locally owned edge.
edge_index, s = dist.local_grids(basis)
x = basis.local_physical_grid(dist)

# Fields
u = dist.Field(name="u", bases=basis)
U = dist.Field(name="U", bases=basis.vertices)
tau_1 = dist.Field(name="tau_1", bases=basis.edges)
tau_2 = dist.Field(name="tau_2", bases=basis.edges)
alpha = dist.Field(name="alpha", bases=basis.vertices)
beta = dist.Field(name="beta", bases=basis.vertices)
g = dist.Field(name="g", bases=basis.vertices)

# Dirichlet conditions on the leaves and Kirchhoff flux balance at the junction
for field, values in (
    (alpha, (1, 0, 1, 1)),
    (beta, (0, 1, 0, 0)),
    (g, (1, 0, 0, 0)),
):
    field.change_layout("g")
    field.set_global_data(np.asarray(values, dtype=dtype)[:, None])

# Substitutions
R = basis.restriction
dx = basis.derivative
lift_basis = basis.interval.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# Problem
problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
problem.add_equation(
    "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
)
problem.add_equation("dtrace(u) - R @ U = 0")
problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = g")

# Solver
solver = problem.build_solver(network_solver="dtn")
solver.solve()

# The weighted average of the leaf values gives U_junction = 6/13.
expected_vertices = np.array([1, 6 / 13, 0, 0])
if not np.allclose(U.allgather_data("g").ravel(), expected_vertices):
    raise RuntimeError("Network Poisson solution failed its analytic check.")

# Analysis
analysis = solver.evaluator.add_file_handler(
    "network_poisson",
    max_writes=1,
)
analysis.add_task(u, name="u")
analysis.add_task(U, name="U")
solver.evaluate_handlers([analysis])

# Gather once for plotting. The distributed solve itself retains local edges.
values = u.gather_data(root=0, layout="g")
if dist.comm.rank == 0:
    vertex_xy = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.75],
            [1.0, -0.75],
        ]
    )
    s = dist.local_grid(basis.interval).ravel()
    segments = []
    colors = []
    for edge, (tail, head) in enumerate(network.endpoints):
        xy = (
            (1 - s[:, None]) * vertex_xy[tail]
            + s[:, None] * vertex_xy[head]
        )
        segments.extend(np.stack((xy[:-1], xy[1:]), axis=1))
        colors.extend((values[edge, :-1] + values[edge, 1:]) / 2)

    figure, axis = plt.subplots(figsize=(6, 4))
    collection = LineCollection(
        segments,
        array=np.asarray(colors),
        cmap="viridis",
        linewidths=7,
    )
    axis.add_collection(collection)
    axis.scatter(
        vertex_xy[:, 0],
        vertex_xy[:, 1],
        c=expected_vertices,
        cmap="viridis",
        edgecolor="black",
        s=75,
        zorder=3,
    )
    for label, (x, y) in zip(vertices, vertex_xy):
        axis.annotate(label, (x, y), xytext=(0, 10), textcoords="offset points", ha="center")
    figure.colorbar(collection, ax=axis, label="u")
    axis.set_aspect("equal")
    axis.autoscale()
    axis.margins(0.2)
    axis.set_axis_off()
    axis.set_title("Poisson equation on a metric network")
    figure.tight_layout()
    figure.savefig("network_poisson.png", dpi=200)
