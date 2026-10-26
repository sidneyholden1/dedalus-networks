"""Serial acceptance contracts for owned/ghost Network solvers."""

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="Serial owned/ghost solver contracts are covered here.",
)


def _set_vertex_data(field, values):
    values = np.asarray(values, dtype=field.dtype)
    field.set_global_data(values[:, None])


def _assert_vertex_solution(field, expected, atol):
    field.change_layout("g")
    dist = field.dist
    canonical = dist.network_vertex_canonical_elements(field.domain)
    resident = dist.network_partition.resident_vertices[0]
    local = field["g"].ravel()
    gathered = field.allgather_data("g").ravel()
    assert np.array_equal(canonical, resident)
    assert field["g"].shape == (resident.size, 1)
    assert np.allclose(local, gathered[canonical], atol=atol)
    assert np.allclose(gathered, expected, atol=atol)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_serial_owned_ghost_explicit_u_non_diagonal_lbvp(dtype):
    """Solve the mixed-interface reference problem through resident storage."""
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=(0, 0),
        network_vertex_storage="owned_ghost",
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    w = dist.Field(name="w", bases=basis.ports)
    g = dist.Field(name="g", bases=basis.vertices)
    w["g"] = 1
    _set_vertex_data(g, (4, 13, 20))

    R = basis.R
    A = d3.CardinalMap(
        basis.vertices,
        basis.vertices,
        sparse.csr_matrix(
            [[2, 1, 0], [0, 3, 1], [0, -1, 4]],
            dtype=dtype,
        ),
        name="A",
    )
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "A @ U + R.T @ ntrace(u) + R.T @ (w*(R @ U)) = g"
    )
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    s = basis.interval.global_grid(dist, scale=1).ravel()
    expected_u = np.vstack((1 + s, 2 + 2 * s)).astype(dtype)
    expected_U = np.array((1, 2, 4), dtype=dtype)
    assert dist.network_vertex_storage == "owned_ghost"
    assert np.allclose(u.allgather_data("g"), expected_u, atol=3e-12)
    _assert_vertex_solution(U, expected_U, atol=3e-12)


def test_serial_owned_ghost_hidden_u_lbvp():
    """Lower a problem-managed interface field into the resident solver."""
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1.7),
    )
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(0, 0),
        network_vertex_storage="owned_ghost",
    )
    basis = d3.NetworkBasis(network, size=10)
    u = dist.Field(name="u", bases=basis)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    forcing = dist.Field(name="forcing", bases=basis)
    boundary = dist.Field(name="boundary", bases=basis.vertices)
    target = 1.25
    forcing["g"] = target
    _set_vertex_data(boundary, np.full(network.num_vertices, target))

    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.LBVP(
        [u, tau_1, tau_2],
        network_interfaces=[u],
        namespace=locals(),
    )
    U = problem.vertex_field(u)
    problem.add_equation(
        "-dx(dx(u)) + u + lift(tau_1,-1) + lift(tau_2,-2) "
        "= forcing"
    )
    problem.add_equation("continuity(u) = 0")
    problem.add_equation("vertex_value(u) = boundary")
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    assert U is problem.variables[-1]
    assert np.allclose(u.allgather_data("g"), target, atol=2e-11)
    _assert_vertex_solution(
        U,
        np.full(network.num_vertices, target),
        atol=2e-11,
    )
