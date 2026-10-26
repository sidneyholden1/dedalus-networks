"""Regression tests for degenerate metric-network topologies."""

import numpy as np
import pytest

import dedalus.public as d3


def _solve_active_self_loop(dtype, network_solver):
    """Solve a reacting loop whose two endpoints represent one vertex."""
    length = 1.6
    kappa = 1.7
    reaction = 2.3
    alpha = 0.8
    beta = 1.1
    target = 1.25
    if np.issubdtype(dtype, np.complexfloating):
        target -= 0.4j
    target = np.asarray(target, dtype=dtype).item()

    network = d3.Network(
        vertex_ids=("junction",),
        edge_ids=("loop",),
        endpoints=((0, 0),),
        lengths=(length,),
    )
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=24)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)

    mu = np.sqrt(reaction / kappa)
    one_port_flux = kappa * mu * np.tanh(mu * length / 2) * target
    g["g"] = np.asarray(
        [alpha * target + beta * 2 * one_port_flux],
        dtype=dtype,
    ).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-kappa*dx(dx(u)) + reaction*u "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*kappa*(R.T @ ntrace(u)) = g"
    )
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    s = dist.local_grid(basis.interval)
    expected_u = (
        target
        * np.cosh(mu * (length * s - length / 2))
        / np.cosh(mu * length / 2)
    )
    port_values = d3.dtrace(u).evaluate()["g"].ravel().copy()
    port_fluxes = (kappa * d3.ntrace(u)).evaluate()["g"].ravel().copy()
    vertex_flux = (
        kappa * (R.T @ d3.ntrace(u))
    ).evaluate()["g"].ravel().copy()
    return {
        "u": u["g"].copy(),
        "U": U["g"].ravel().copy(),
        "expected_u": expected_u,
        "target": target,
        "one_port_flux": one_port_flux,
        "port_values": port_values,
        "port_fluxes": port_fluxes,
        "vertex_flux": vertex_flux,
    }


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_active_reacting_self_loop_direct_matches_dtn(dtype):
    """Count both ports of a nonconstant loop in the vertex flux balance."""
    direct = _solve_active_self_loop(dtype, None)
    dtn = _solve_active_self_loop(dtype, "dtn")

    for result in (direct, dtn):
        assert np.allclose(result["u"], result["expected_u"], atol=2e-11)
        assert np.allclose(result["U"], result["target"], atol=2e-11)
        assert np.allclose(
            result["port_values"],
            (result["target"], result["target"]),
            atol=2e-11,
        )
        assert np.allclose(
            result["port_fluxes"],
            (result["one_port_flux"], result["one_port_flux"]),
            atol=2e-11,
        )
        assert np.allclose(
            result["vertex_flux"],
            2 * result["one_port_flux"],
            atol=2e-11,
        )
        midpoint = result["u"][0, result["u"].shape[1] // 2]
        assert not np.isclose(midpoint, result["target"], atol=1e-4)

    assert np.allclose(dtn["u"], direct["u"], atol=2e-11)
    assert np.allclose(dtn["U"], direct["U"], atol=2e-11)
    assert np.allclose(dtn["port_fluxes"], direct["port_fluxes"], atol=2e-11)
