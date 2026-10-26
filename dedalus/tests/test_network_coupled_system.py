"""Serial regression for a genuinely coupled Network boundary system."""

import numpy as np

import dedalus.public as d3
from dedalus.tests import network_coupled_manufactured as manufactured


def _solve_coupled_system(network_solver):
    network = manufactured.build_network()
    dist = d3.Distributor(network, dtype=np.complex128)
    basis = d3.NetworkBasis(network, size=14)
    u = dist.Field(name="u", bases=basis)
    v = dist.Field(name="v", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    V = dist.Field(name="V", bases=basis.vertices)
    u_tau_1 = dist.Field(name="u_tau_1", bases=basis.edges)
    u_tau_2 = dist.Field(name="u_tau_2", bases=basis.edges)
    v_tau_1 = dist.Field(name="v_tau_1", bases=basis.edges)
    v_tau_2 = dist.Field(name="v_tau_2", bases=basis.edges)
    forcing_u = dist.Field(name="forcing_u", bases=basis)
    forcing_v = dist.Field(name="forcing_v", bases=basis)
    boundary_u = dist.Field(name="boundary_u", bases=basis.vertices)
    boundary_v = dist.Field(name="boundary_v", bases=basis.vertices)

    s = dist.local_grid(basis.interval)
    expected = manufactured.manufactured_data(network, s)
    forcing_u["g"] = expected["forcing_u"]
    forcing_v["g"] = expected["forcing_v"]
    boundary_u["g"] = expected["boundary_u"].reshape(boundary_u["g"].shape)
    boundary_v["g"] = expected["boundary_v"].reshape(boundary_v["g"].shape)

    ku = manufactured.KU
    kv = manufactured.KV
    edge_uu, edge_uv = manufactured.EDGE_MATRIX[0]
    edge_vu, edge_vv = manufactured.EDGE_MATRIX[1]
    value_uu, value_uv = manufactured.VALUE_MATRIX[0]
    value_vu, value_vv = manufactured.VALUE_MATRIX[1]
    flux_uu, flux_uv = manufactured.FLUX_MATRIX[0]
    flux_vu, flux_vv = manufactured.FLUX_MATRIX[1]
    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    variables = (
        u,
        v,
        U,
        V,
        u_tau_1,
        u_tau_2,
        v_tau_1,
        v_tau_2,
    )
    problem = d3.LBVP(variables, namespace=locals())
    problem.add_equation(
        "-ku*dx(dx(u)) + edge_uu*u + edge_uv*v "
        "+ lift(u_tau_1,-1) + lift(u_tau_2,-2) = forcing_u"
    )
    problem.add_equation(
        "-kv*dx(dx(v)) + edge_vu*u + edge_vv*v "
        "+ lift(v_tau_1,-1) + lift(v_tau_2,-2) = forcing_v"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("dtrace(v) - R @ V = 0")
    problem.add_equation(
        "value_uu*U + value_uv*V "
        "+ flux_uu*ku*(R.T @ ntrace(u)) "
        "+ flux_uv*kv*(R.T @ ntrace(v)) = boundary_u"
    )
    problem.add_equation(
        "value_vu*U + value_vv*V "
        "+ flux_vu*ku*(R.T @ ntrace(u)) "
        "+ flux_vv*kv*(R.T @ ntrace(v)) = boundary_v"
    )
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    return {
        "u": u["g"].copy(),
        "v": v["g"].copy(),
        "U": U["g"].ravel().copy(),
        "V": V["g"].ravel().copy(),
        "u_port_flux": (ku * d3.ntrace(u)).evaluate()["g"].ravel().copy(),
        "v_port_flux": (kv * d3.ntrace(v)).evaluate()["g"].ravel().copy(),
        "u_vertex_flux": (
            ku * (R.T @ d3.ntrace(u))
        ).evaluate()["g"].ravel().copy(),
        "v_vertex_flux": (
            kv * (R.T @ d3.ntrace(v))
        ).evaluate()["g"].ravel().copy(),
        "expected": expected,
    }


def test_complex_coupled_system_direct_matches_dtn():
    """Condense cross-coupled edges and two mixed interface fields."""
    direct = _solve_coupled_system(None)
    dtn = _solve_coupled_system("dtn")

    for result in (direct, dtn):
        expected = result["expected"]
        assert np.allclose(result["u"], expected["u"], atol=3e-11)
        assert np.allclose(result["v"], expected["v"], atol=3e-11)
        assert np.allclose(result["U"], manufactured.U_VALUES, atol=3e-11)
        assert np.allclose(result["V"], manufactured.V_VALUES, atol=3e-11)
        assert np.allclose(
            result["u_port_flux"],
            expected["u_port_flux"],
            atol=3e-11,
        )
        assert np.allclose(
            result["v_port_flux"],
            expected["v_port_flux"],
            atol=3e-11,
        )
        assert np.allclose(
            result["u_vertex_flux"],
            expected["u_vertex_flux"],
            atol=3e-11,
        )
        assert np.allclose(
            result["v_vertex_flux"],
            expected["v_vertex_flux"],
            atol=3e-11,
        )

    for key in ("u", "v", "U", "V", "u_port_flux", "v_port_flux"):
        assert np.allclose(dtn[key], direct[key], atol=3e-11)
