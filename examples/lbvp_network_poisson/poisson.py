"""
Dedalus script solving a Poisson equation on a three-edge metric network.
This script demonstrates the NetworkBasis interface, continuity and Kirchhoff
vertex conditions, whole-edge distributed storage, the hierarchical
Dirichlet-to-Neumann solver, and ordinary Dedalus HDF5 output. It should take
just a few seconds to run.

The network is a star with junction v_0 and Dirichlet leaves v_1 (left),
v_2 (upper), and v_3 (lower). The leaf values are u(v_1)=1 and
u(v_2)=u(v_3)=0. Vertex arrays and saved metadata use the order v_0,...,v_3.
Every edge solves
    -d_x(d_x(u)) = 0.
The solution is continuous at v_0 and has zero summed outward flux there.

To run and plot:
    $ python3 poisson.py
"""

import logging

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

import dedalus.public as d3

logger = logging.getLogger(__name__)


# Parameters
N = 24
dtype = np.float64
vertices = ("v_0", "v_1", "v_2", "v_3")
edges = ("v_1_to_v_0", "v_0_to_v_2", "v_0_to_v_3")
endpoints = ((1, 0), (0, 2), (0, 3))
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

# Kirchhoff flux at v_0; Dirichlet values at v_1, v_2, v_3.
# Each coefficient array follows the vertex order (v_0, v_1, v_2, v_3).
for field, values in (
    (alpha, (0, 1, 1, 1)),
    (beta, (1, 0, 0, 0)),
    (g, (0, 1, 0, 0)),
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

# The length-weighted flux balance gives U(v_0) = 6/13.
# Expected values follow (v_0, v_1, v_2, v_3).
expected_vertices = np.array([6 / 13, 1, 0, 0])
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
            [0.0, 0.0],
            [-1.0, 0.0],
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

    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_axes((0.05, 0.06, 0.76, 0.88))
    norm = Normalize(vmin=0, vmax=1)
    collection = LineCollection(
        segments,
        array=np.asarray(colors),
        cmap="viridis",
        norm=norm,
        linewidths=7,
        capstyle="round",
    )
    axis.add_collection(collection)
    for vertex_id, (x, y) in zip(vertices, vertex_xy):
        below = vertex_id == "v_3"
        axis.annotate(
            f"${vertex_id}$", (x, y), xytext=(0, -12 if below else 12),
            textcoords="offset points", ha="center",
            va="top" if below else "bottom", fontsize=20, math_fontfamily="cm",
        )
    axis.set(xlim=(-1.2, 1.2), ylim=(-0.95, 0.95), aspect="equal")
    axis.set_axis_off()

    # Data coordinates align the colour bar with the network's vertical span,
    # independently of the figure margins and equal-aspect axes adjustment.
    ymin, ymax = vertex_xy[:, 1].min(), vertex_xy[:, 1].max()
    colorbar_axis = axis.inset_axes(
        (1.27, ymin, 0.07, ymax - ymin), transform=axis.transData
    )
    colorbar = figure.colorbar(
        collection, cax=colorbar_axis, ticks=(0, 0.25, 0.5, 0.75, 1)
    )
    colorbar.ax.set_yticklabels(("0", "0.25", "0.5", "0.75", "1"))
    colorbar.ax.set_title("$u$", fontsize=20, math_fontfamily="cm", pad=8)
    colorbar.ax.tick_params(labelsize=12, length=3, width=0.8, pad=6)
    colorbar.outline.set_linewidth(0.8)
    figure.savefig("network_poisson.png", dpi=200)
