"""Serial regression for dynamic Network vertex conditions."""

import numpy as np
import pytest

import dedalus.public as d3
from dedalus.tests import network_dynamic_vertex_manufactured as manufactured


def _step_dynamic_vertex_mode(dtype, network_solver):
    network = manufactured.build_network()
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=18)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    mass = dist.Field(name="mass", bases=basis.vertices)
    damping = dist.Field(name="damping", bases=basis.vertices)

    s = dist.local_grid(basis.interval)
    initial = manufactured.mode_data(network, s, dtype)
    u["g"] = initial["edge"]
    U["g"] = initial["vertex"].reshape(U["g"].shape)
    mass["g"] = manufactured.VERTEX_MASS.reshape(mass["g"].shape)
    damping["g"] = manufactured.VERTEX_DAMPING.reshape(
        damping["g"].shape
    )

    kappa = manufactured.KAPPA
    R = basis.restriction
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.IVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "dt(u) - kappa*dx(dx(u)) "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "mass*dt(U) + damping*U "
        "+ kappa*(R.T @ ntrace(u)) = 0"
    )
    assert problem.eqs[2]["M"]
    solver = problem.build_solver(
        d3.SBDF1,
        network_solver=network_solver,
    )
    timestep = 0.025
    solver.step(timestep)
    amplification = 1 / (1 + timestep * manufactured.DECAY_RATE)

    return {
        "u": u["g"].copy(),
        "U": U["g"].ravel().copy(),
        "port_flux": (
            kappa * d3.ntrace(u)
        ).evaluate()["g"].ravel().copy(),
        "vertex_flux": (
            kappa * (R.T @ d3.ntrace(u))
        ).evaluate()["g"].ravel().copy(),
        "initial": initial,
        "amplification": amplification,
        "solver": solver,
    }


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_dynamic_vertex_mode_direct_matches_dtn(dtype):
    """Advance a heat PDE coupled to inertial ODEs at every vertex."""
    initial = manufactured.mode_data(
        manufactured.build_network(),
        np.array(((0.0,),)),
        dtype,
    )
    assert np.allclose(
        manufactured.VERTEX_DAMPING * initial["vertex"]
        + initial["vertex_flux"],
        manufactured.DECAY_RATE
        * manufactured.VERTEX_MASS
        * initial["vertex"],
    )

    direct = _step_dynamic_vertex_mode(dtype, None)
    dtn = _step_dynamic_vertex_mode(dtype, "dtn")
    for result in (direct, dtn):
        amplification = result["amplification"]
        assert np.allclose(
            result["u"],
            amplification * result["initial"]["edge"],
            atol=3e-10,
        )
        assert np.allclose(
            result["U"],
            amplification * result["initial"]["vertex"],
            atol=3e-10,
        )
        assert np.allclose(
            result["port_flux"],
            amplification * result["initial"]["port_flux"],
            atol=3e-10,
        )
        assert np.allclose(
            result["vertex_flux"],
            amplification * result["initial"]["vertex_flux"],
            atol=3e-10,
        )

    assert direct["solver"].network_solver is None
    assert dtn["solver"].network_solver == "dtn"
    factorization = dtn["solver"].subproblems[0].LHS_solver
    assert factorization.schur_matrix.shape == (3, 3)
    for key in ("u", "U", "port_flux", "vertex_flux"):
        assert np.allclose(dtn[key], direct[key], atol=3e-10)
