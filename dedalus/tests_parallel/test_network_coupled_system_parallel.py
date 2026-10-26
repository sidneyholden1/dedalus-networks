"""MPI regression for a genuinely coupled Network boundary system."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.tests import network_coupled_manufactured as manufactured


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("vertex_storage", ["replicated", "owned_ghost"])
@pytest.mark.parametrize("elimination", [None, "two_stage"])
def test_complex_coupled_system_irregular_idle_ownership(vertex_storage, elimination):
    """Condense coupled fields when root and an interior rank own no edges."""
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip("regression uses exactly four MPI ranks")

    network = manufactured.build_network()
    owners = np.array((3, 1))
    dist = d3.Distributor(
        network,
        dtype=np.complex128,
        network_partition=owners,
        network_vertex_storage=vertex_storage,
    )
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
    owned_edges = dist.network_partition.edges[comm.rank]
    forcing_u["g"] = expected["forcing_u"][owned_edges]
    forcing_v["g"] = expected["forcing_v"][owned_edges]
    if vertex_storage == "owned_ghost":
        boundary_u.change_layout("g")
        boundary_v.change_layout("g")
        boundary_u.set_global_data(expected["boundary_u"][:, None])
        boundary_v.set_global_data(expected["boundary_v"][:, None])
    else:
        boundary_u["g"] = expected["boundary_u"].reshape(
            boundary_u["g"].shape
        )
        boundary_v["g"] = expected["boundary_v"].reshape(
            boundary_v["g"].shape
        )

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
    options = {} if elimination is None else {"network_interior_elimination": elimination}
    solver = problem.build_solver(**options)
    assert solver.network_solver == "dtn"
    solver.solve()
    # Reuse each factorization for multiple complex right-hand sides, including
    # ranks whose local RHS is the empty placeholder.
    for subproblem, factor in solver.subproblem_matsolvers.items():
        rhs = subproblem.gather_outputs(solver.F).copy()
        single = factor.solve(rhs)
        multiple = factor.solve(np.concatenate((rhs, 2j * rhs), axis=1))
        assert np.allclose(multiple, np.concatenate((single, 2j * single), axis=1), atol=5e-11)

    partition = dist.network_partition
    assert np.array_equal(partition.edge_owners, owners)
    assert np.array_equal(partition.separator_vertices, (1,))
    assert np.array_equal(owned_edges, np.flatnonzero(owners == comm.rank))
    if comm.rank in (0, 2):
        assert owned_edges.size == 0
        assert u["g"].shape == (0, basis.size)
        assert v["g"].shape == (0, basis.size)

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    coupled = vertex_storage == "owned_ghost" and elimination is None
    assert factorization.coupled == coupled
    if coupled:
        # The default factors edge/tau coefficients together with vertices.
        assert not hasattr(factorization, "edge_solver")
        subgraph = factorization.subgraph_solver
        assert subgraph.interior_matrix.shape[0] == (
            factorization.edge_rows.size + subgraph.interior_rows.size
        )
    assert np.array_equal(factorization.owned_edges, owned_edges)
    assert factorization.schur_matrix.shape == (2, 2)

    assert np.allclose(u["g"], expected["u"][owned_edges], atol=5e-11)
    assert np.allclose(v["g"], expected["v"][owned_edges], atol=5e-11)
    assert np.allclose(u.allgather_data("g"), expected["u"], atol=5e-11)
    assert np.allclose(v.allgather_data("g"), expected["v"], atol=5e-11)
    assert np.allclose(
        U.allgather_data("g").ravel(),
        manufactured.U_VALUES,
        atol=5e-11,
    )
    assert np.allclose(
        V.allgather_data("g").ravel(),
        manufactured.V_VALUES,
        atol=5e-11,
    )
    if vertex_storage == "owned_ghost":
        resident = partition.resident_vertices[comm.rank]
        assert np.allclose(
            U["g"].ravel(), manufactured.U_VALUES[resident], atol=5e-11
        )
        assert np.allclose(
            V["g"].ravel(), manufactured.V_VALUES[resident], atol=5e-11
        )

    owned_ports = np.flatnonzero(owners[network.port_edges] == comm.rank)
    u_port_flux = (ku * d3.ntrace(u)).evaluate()
    v_port_flux = (kv * d3.ntrace(v)).evaluate()
    assert np.allclose(
        u_port_flux["g"].ravel(),
        expected["u_port_flux"][owned_ports],
        atol=5e-11,
    )
    assert np.allclose(
        v_port_flux["g"].ravel(),
        expected["v_port_flux"][owned_ports],
        atol=5e-11,
    )
    assert np.allclose(
        u_port_flux.allgather_data("g").ravel(),
        expected["u_port_flux"],
        atol=5e-11,
    )
    assert np.allclose(
        v_port_flux.allgather_data("g").ravel(),
        expected["v_port_flux"],
        atol=5e-11,
    )
    u_vertex_flux = (ku * (R.T @ d3.ntrace(u))).evaluate()
    v_vertex_flux = (kv * (R.T @ d3.ntrace(v))).evaluate()
    resident = (
        partition.resident_vertices[comm.rank]
        if vertex_storage == "owned_ghost"
        else np.arange(network.num_vertices)
    )
    assert np.allclose(
        u_vertex_flux["g"].ravel(),
        expected["u_vertex_flux"][resident],
        atol=5e-11,
    )
    assert np.allclose(
        v_vertex_flux["g"].ravel(),
        expected["v_vertex_flux"][resident],
        atol=5e-11,
    )
