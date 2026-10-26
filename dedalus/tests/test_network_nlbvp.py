"""Nonlinear boundary-value problems on metric networks."""

import numpy as np
import pytest

import dedalus.public as d3


def _solve_constant_cubic(dtype, network_solver):
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=10)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    edge_forcing = dist.Field(name="edge_forcing", bases=basis)
    vertex_forcing = dist.Field(
        name="vertex_forcing",
        bases=basis.vertices,
    )

    target = 1.0 if dtype == np.float64 else 1.0 + 0.25j
    edge_forcing["g"] = target**3
    vertex_forcing["g"] = target**3
    u["g"] = 0.7 * target
    U["g"] = 0.7 * target

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.NLBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + u*u*u + lift(tau_1,-1) + lift(tau_2,-2) "
        "= edge_forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U*U*U = vertex_forcing")
    solver = problem.build_solver(network_solver=network_solver)
    for _ in range(7):
        solver.newton_iteration()
    return solver, u["g"].copy(), U["g"].copy(), target


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_network_nlbvp_direct_matches_dtn(dtype):
    """Condense each Newton Jacobian without changing the nonlinear solve."""
    direct, direct_u, direct_U, target = _solve_constant_cubic(dtype, None)
    dtn, dtn_u, dtn_U, _ = _solve_constant_cubic(dtype, "dtn")

    assert direct.network_solver is None
    assert dtn.network_solver == "dtn"
    assert np.allclose(direct_u, target, atol=2e-11)
    assert np.allclose(direct_U, target, atol=2e-11)
    assert np.allclose(dtn_u, direct_u, atol=2e-11)
    assert np.allclose(dtn_U, direct_U, atol=2e-11)
