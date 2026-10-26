"""Global scalar constraints for metric-network problems."""

import numpy as np
import pytest

import dedalus.public as d3


def _solve_neumann_poisson(dtype, network_solver, gauge="integ(u)"):
    """Solve a compatible pure-Neumann problem with a mean-zero gauge."""
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=8)

    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    tau_gauge = dist.Field(name="tau_gauge")
    forcing = dist.Field(name="forcing", bases=basis)

    s = dist.local_grid(basis.interval)
    forcing["g"] = s

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP(
        [u, U, tau_1, tau_2, tau_gauge],
        namespace=locals(),
    )
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "+ tau_gauge = forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = 0")
    problem.add_equation(f"{gauge} = 0")
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()
    return solver, s, u, U, tau_gauge


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_network_neumann_poisson_global_gauge_direct_matches_dtn(dtype):
    """Condense a basis-free tau and full-network integral exactly."""
    direct = _solve_neumann_poisson(dtype, None)
    dtn = _solve_neumann_poisson(dtype, "dtn")

    for solver, s, u, U, tau_gauge in (direct, dtn):
        expected = s**2 / 4 - s**3 / 6 - 1 / 24
        assert np.allclose(u["g"], expected, atol=2e-12)
        assert np.allclose(U["g"].ravel(), (-1 / 24, 1 / 24), atol=2e-12)
        assert np.allclose(tau_gauge["g"], 1 / 2, atol=2e-12)
        assert np.allclose(d3.Integrate(u).evaluate()["g"], 0, atol=2e-12)

    assert np.allclose(dtn[2]["g"], direct[2]["g"], atol=2e-12)
    assert np.allclose(dtn[3]["g"], direct[3]["g"], atol=2e-12)
    assert np.allclose(dtn[4]["g"], direct[4]["g"], atol=2e-12)
    factorization = next(iter(dtn[0].subproblem_matsolvers.values()))
    assert factorization.schur_matrix.shape == (3, 3)


def test_network_average_global_gauge_reinitializes():
    """Rebuild an Average expression while linearizing the gauge row."""
    _, _, u, U, tau_gauge = _solve_neumann_poisson(
        np.float64,
        "dtn",
        gauge="ave(u)",
    )
    assert np.allclose(d3.Average(u).evaluate()["g"], 0, atol=2e-12)
    assert np.allclose(U["g"].ravel(), (-1 / 24, 1 / 24), atol=2e-12)
    assert np.allclose(tau_gauge["g"], 1 / 2, atol=2e-12)


def _build_neumann_gauge_fields(dtype, size=24):
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=size)
    fields = {
        "u": dist.Field(name="u", bases=basis),
        "U": dist.Field(name="U", bases=basis.vertices),
        "tau_1": dist.Field(name="tau_1", bases=basis.edges),
        "tau_2": dist.Field(name="tau_2", bases=basis.edges),
        "tau_gauge": dist.Field(name="tau_gauge"),
    }
    return dist, basis, fields


def _add_neumann_gauge_equations(problem, edge_equation):
    problem.add_equation(edge_equation)
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = 0")
    problem.add_equation("integ(u) = 0")


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_network_heat_ivp_global_gauge_direct_matches_dtn(dtype):
    """Keep a pure-Neumann heat mode mean-zero through an implicit stage."""
    results = []
    timestep = 0.01
    for network_solver in (None, "dtn"):
        dist, basis, fields = _build_neumann_gauge_fields(dtype)
        u = fields["u"]
        U = fields["U"]
        tau_1 = fields["tau_1"]
        tau_2 = fields["tau_2"]
        tau_gauge = fields["tau_gauge"]
        s = dist.local_grid(basis.interval)
        initial = np.cos(np.pi * s).astype(dtype)
        u["g"] = initial
        U["g"] = np.array((1, -1), dtype=dtype).reshape(U["g"].shape)

        R = basis.restriction
        dx = basis.derivative
        dt = d3.TimeDerivative
        lift_basis = basis.interval.derivative_basis(2)
        lift = lambda A, n: d3.Lift(A, lift_basis, n)
        problem = d3.IVP(
            [u, U, tau_1, tau_2, tau_gauge],
            namespace=locals(),
        )
        _add_neumann_gauge_equations(
            problem,
            "dt(u) - dx(dx(u)) + lift(tau_1,-1) "
            "+ lift(tau_2,-2) + tau_gauge = 0",
        )
        solver = problem.build_solver(
            d3.SBDF1,
            network_solver=network_solver,
        )
        solver.step(timestep)
        amplification = 1 / (1 + timestep * np.pi**2)
        assert np.allclose(u["g"], amplification * initial, atol=2e-11)
        assert np.allclose(d3.Integrate(u).evaluate()["g"], 0, atol=2e-12)
        assert np.allclose(tau_gauge["g"], 0, atol=2e-11)
        results.append((u["g"].copy(), U["g"].copy()))

    assert np.allclose(results[1][0], results[0][0], atol=2e-11)
    assert np.allclose(results[1][1], results[0][1], atol=2e-11)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_network_nlbvp_global_gauge_direct_matches_dtn(dtype):
    """Condense the global row and column in every Newton Jacobian."""
    results = []
    amplitude = dtype(1) if dtype == np.float64 else dtype(1 + 0.2j)
    for network_solver in (None, "dtn"):
        dist, basis, fields = _build_neumann_gauge_fields(dtype)
        u = fields["u"]
        U = fields["U"]
        tau_1 = fields["tau_1"]
        tau_2 = fields["tau_2"]
        tau_gauge = fields["tau_gauge"]
        forcing = dist.Field(name="forcing", bases=basis)
        s = dist.local_grid(basis.interval)
        expected = amplitude * np.cos(np.pi * s)
        u["g"] = 0.8 * expected
        U["g"] = 0.8 * np.array(
            (amplitude, -amplitude),
            dtype=dtype,
        ).reshape(U["g"].shape)
        forcing["g"] = np.pi**2 * expected + expected**3

        R = basis.restriction
        dx = basis.derivative
        lift_basis = basis.interval.derivative_basis(2)
        lift = lambda A, n: d3.Lift(A, lift_basis, n)
        problem = d3.NLBVP(
            [u, U, tau_1, tau_2, tau_gauge],
            namespace=locals(),
        )
        _add_neumann_gauge_equations(
            problem,
            "-dx(dx(u)) + u*u*u + lift(tau_1,-1) "
            "+ lift(tau_2,-2) + tau_gauge = forcing",
        )
        solver = problem.build_solver(network_solver=network_solver)
        for _ in range(6):
            solver.newton_iteration()
        assert np.allclose(u["g"], expected, atol=2e-11)
        assert np.allclose(d3.Integrate(u).evaluate()["g"], 0, atol=2e-12)
        assert np.allclose(tau_gauge["g"], 0, atol=2e-11)
        results.append((u["g"].copy(), U["g"].copy()))

    assert np.allclose(results[1][0], results[0][0], atol=2e-11)
    assert np.allclose(results[1][1], results[0][1], atol=2e-11)
