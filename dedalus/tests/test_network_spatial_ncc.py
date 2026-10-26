"""Manufactured network problems with edge- and position-dependent NCCs."""

import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_network_spatially_varying_diffusion(dtype, network_solver):
    """Solve a polynomial diffusion problem across unequal edge lengths."""
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
    kappa = dist.Field(name="kappa", bases=basis)
    forcing = dist.Field(name="forcing", bases=basis)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    boundary_data = dist.Field(name="boundary_data", bases=basis.vertices)

    s = dist.local_grid(basis.interval)
    physical_x = np.concatenate((s, 1 + 2*s), axis=0)
    expected_u = physical_x**2
    kappa["g"] = 1 + physical_x
    forcing["g"] = -2 - 4*physical_x
    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)
    boundary_data["g"] = np.array([0, 0, 9]).reshape(
        boundary_data["g"].shape
    )

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(kappa*dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "= forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ (dtrace(kappa)*ntrace(u))) "
        "= boundary_data"
    )
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    flux = R.T @ (d3.dtrace(kappa) * d3.ntrace(u))
    assert np.allclose(u["g"], expected_u)
    assert np.allclose(U["g"].ravel(), [0, 1, 9])
    assert np.allclose(flux.evaluate()["g"].ravel(), [0, 0, 24])
