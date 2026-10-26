"""Analytic tests for network trace signs, forcing, and orientation."""

import numpy as np
import pytest

import dedalus.public as d3


def build_scalar_problem(network, size=16, dtype=np.float64):
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=size)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    dx = basis.derivative
    R = basis.restriction
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    return dist, basis, u, U, tau_1, tau_2, dx, R, lift


@pytest.mark.parametrize("network_solver", [None, "dtn"])
@pytest.mark.parametrize("neumann_vertex", [0, 1])
def test_nonzero_neumann_trace_sign(network_solver, neumann_vertex):
    """ntrace is the geometric outward derivative at both endpoints."""
    network = d3.Network(endpoints=((0, 1),), lengths=(2,))
    (
        dist,
        basis,
        u,
        U,
        tau_1,
        tau_2,
        dx,
        R,
        lift,
    ) = build_scalar_problem(network)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)

    if neumann_vertex == 0:
        alpha["g"] = np.array([0, 1]).reshape(alpha["g"].shape)
        beta["g"] = np.array([1, 0]).reshape(beta["g"].shape)
        g["g"] = np.array([-5, 13]).reshape(g["g"].shape)
    else:
        alpha["g"] = np.array([1, 0]).reshape(alpha["g"].shape)
        beta["g"] = np.array([0, 1]).reshape(beta["g"].shape)
        g["g"] = np.array([3, 5]).reshape(g["g"].shape)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = g")
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    s = dist.local_grid(basis.interval)
    assert np.allclose(u["g"], 3 + 10*s)
    assert np.allclose(U["g"].ravel(), (3, 13))
    assert np.allclose(d3.ntrace(u).evaluate()["g"].ravel(), (-5, 5))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_reaction_dtn_matrix(dtype, network_solver):
    """Match the analytic positive DtN matrix for -kappa*u_xx + q*u."""
    length = 1.7
    kappa = 2.0
    reaction = 3.0
    network = d3.Network(endpoints=((0, 1),), lengths=(length,))
    (
        dist,
        basis,
        u,
        U,
        tau_1,
        tau_2,
        dx,
        R,
        lift,
    ) = build_scalar_problem(network, size=24, dtype=dtype)
    boundary_data = dist.Field(name="boundary_data", bases=basis.vertices)
    boundary_values = np.array([1.25, -0.4], dtype=dtype)
    boundary_data["g"] = boundary_values.reshape(boundary_data["g"].shape)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-kappa*dx(dx(u)) + reaction*u "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = boundary_data")
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    mu = np.sqrt(reaction / kappa)
    scale = kappa * mu / np.sinh(mu * length)
    expected_matrix = scale * np.array(
        [[np.cosh(mu * length), -1], [-1, np.cosh(mu * length)]]
    )
    flux = kappa * d3.ntrace(u)
    assert np.allclose(
        flux.evaluate()["g"].ravel(),
        expected_matrix @ boundary_values,
        atol=2e-11,
    )


@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_forced_chain_with_reversed_edge(network_solver):
    """Affine edge DtN forcing is invariant to an edge orientation reversal."""
    network = d3.Network(
        endpoints=((0, 1), (2, 1)),
        lengths=(1, 2),
    )
    (
        dist,
        basis,
        u,
        U,
        tau_1,
        tau_2,
        dx,
        R,
        lift,
    ) = build_scalar_problem(network, size=16)
    forcing = dist.Field(name="forcing", bases=basis)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    forcing["g"] = 1
    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = 0")
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    s = dist.local_grid(basis.interval)
    physical_x = np.concatenate((s, 3 - 2*s), axis=0)
    expected = physical_x * (3 - physical_x) / 2
    assert np.allclose(u["g"], expected)
    assert np.allclose(U["g"].ravel(), (0, 1, 0))
    assert np.allclose(
        (R.T @ d3.ntrace(u)).evaluate()["g"].ravel()[1],
        0,
        atol=2e-11,
    )


@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_dirichlet_resonance_is_diagnosed(network_solver):
    """A globally regular Robin problem can have a singular edge DtN map."""
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    (
        dist,
        basis,
        u,
        U,
        tau_1,
        tau_2,
        dx,
        R,
        lift,
    ) = build_scalar_problem(network, size=20)
    forcing = dist.Field(name="forcing", bases=basis)
    robin_data = dist.Field(name="robin_data", bases=basis.vertices)
    s = dist.local_grid(basis.interval)
    expected = 1 + s
    wavenumber = np.pi
    forcing["g"] = -(wavenumber**2) * expected
    robin_data["g"] = np.array([0, 3]).reshape(robin_data["g"].shape)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) - wavenumber**2*u "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U + R.T @ ntrace(u) = robin_data")

    if network_solver == "dtn":
        solver = problem.build_solver(network_solver="dtn")
        with pytest.raises(
            RuntimeError,
            match="Dirichlet block is numerically singular",
        ):
            solver.solve()
    else:
        solver = problem.build_solver(network_solver=None)
        solver.solve()
        assert np.allclose(u["g"], expected, atol=2e-10)
