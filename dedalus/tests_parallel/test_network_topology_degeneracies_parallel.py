"""MPI regressions for degenerate metric-network topologies."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("vertex_storage", ["replicated", "owned_ghost"])
def test_disconnected_components_and_isolated_vertex_irregular_ownership(
    vertex_storage,
):
    """Condense disconnected local pieces with two edge-idle MPI ranks."""
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip("regression uses exactly four MPI ranks")

    # Each active rank owns one edge from each connected component.  Rank zero
    # and rank two own no edges; rank two nevertheless owns the isolated-
    # vertex equation under the partition's deterministic assignment rule.
    network = d3.Network(
        vertex_ids=("a0", "a1", "a2", "b0", "b1", "b2", "isolated"),
        edge_ids=("a-left", "a-right", "b-left", "b-right"),
        endpoints=((0, 1), (2, 1), (3, 4), (5, 4)),
        lengths=(1.0, 1.5, 0.75, 2.0),
    )
    owners = np.array((3, 1, 3, 1))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
        network_vertex_storage=vertex_storage,
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)

    expected_U = np.array((1.2, 0.4, -0.7, 2.1, -0.3, 0.8, 3.25))
    endpoint_values = expected_U[network.endpoints]
    physical_slopes = np.diff(endpoint_values, axis=1).ravel() / network.lengths
    expected_port_fluxes = np.column_stack(
        (-physical_slopes, physical_slopes)
    ).ravel()
    expected_vertex_fluxes = network.restriction_matrix.T @ expected_port_fluxes
    expected_g = expected_U + expected_vertex_fluxes
    if vertex_storage == "owned_ghost":
        g.change_layout("g")
        g.set_global_data(expected_g[:, None])
    elif comm.rank == 0:
        g["g"] = expected_g.reshape(g["g"].shape)
    else:
        # Replicated problem data are authoritative on root, even when root
        # owns no edge and another idle rank owns the isolated equation.
        g["g"] = (expected_g + 100 + comm.rank).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U + R.T @ ntrace(u) = g")
    solver = problem.build_solver()
    assert solver.network_solver == "dtn"
    solver.solve()

    partition = dist.network_partition
    assert np.array_equal(partition.edge_owners, owners)
    assert np.array_equal(partition.separator_vertices, (1, 4, 6))
    assert partition.vertex_equation_owners[6] == 2
    owned_edges = partition.edges[comm.rank]
    assert np.array_equal(
        owned_edges,
        np.flatnonzero(owners == comm.rank),
    )
    if comm.rank in (0, 2):
        assert owned_edges.size == 0
        assert u["g"].shape == (0, basis.size)

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    assert np.array_equal(factorization.owned_edges, owned_edges)
    assert factorization.schur_matrix.shape == (3, 3)

    s = dist.local_grid(basis.interval)
    expected_global_u = (
        endpoint_values[:, :1]
        + (endpoint_values[:, 1:] - endpoint_values[:, :1]) * s
    )
    assert np.allclose(u["g"], expected_global_u[owned_edges], atol=2e-12)
    assert np.allclose(u.allgather_data("g"), expected_global_u, atol=2e-12)
    assert np.allclose(
        U.allgather_data("g").ravel(),
        expected_U,
        atol=2e-12,
    )
    if vertex_storage == "owned_ghost":
        resident = partition.resident_vertices[comm.rank]
        assert np.allclose(U["g"].ravel(), expected_U[resident], atol=2e-12)
        assert 6 in partition.owned_vertices[2]
        assert 6 in partition.resident_vertices[2]
    else:
        assert np.allclose(U["g"].ravel(), expected_U, atol=2e-12)

    port_fluxes = d3.ntrace(u).evaluate()
    owned_ports = np.flatnonzero(owners[network.port_edges] == comm.rank)
    assert np.allclose(
        port_fluxes["g"].ravel(),
        expected_port_fluxes[owned_ports],
        atol=2e-12,
    )
    assert np.allclose(
        port_fluxes.allgather_data("g").ravel(),
        expected_port_fluxes,
        atol=2e-12,
    )
    vertex_fluxes = (R.T @ d3.ntrace(u)).evaluate()
    if vertex_storage == "owned_ghost":
        resident = partition.resident_vertices[comm.rank]
        assert np.allclose(
            vertex_fluxes["g"].ravel(),
            expected_vertex_fluxes[resident],
            atol=2e-12,
        )
    else:
        assert np.allclose(
            vertex_fluxes["g"].ravel(),
            expected_vertex_fluxes,
            atol=2e-12,
        )
    assert np.isclose(U.allgather_data("g").ravel()[6], expected_g[6])


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_owned_ghost_reacting_self_loop_with_idle_ranks(dtype):
    """Count both loop ports while every nonowner rank is fully idle."""
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip("regression uses exactly four MPI ranks")

    length = 1.6
    kappa = 1.7
    reaction = 2.3
    alpha = 0.8
    beta = 1.1
    target = 1.25 - (0.4j if np.issubdtype(dtype, np.complexfloating) else 0)
    network = d3.Network(endpoints=((0, 0),), lengths=(length,))
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=(3,),
        network_vertex_storage="owned_ghost",
    )
    basis = d3.NetworkBasis(network, size=24)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)

    mu = np.sqrt(reaction / kappa)
    one_port_flux = kappa * mu * np.tanh(mu * length / 2) * target
    g.change_layout("g")
    g.set_global_data(
        np.asarray(
            [[alpha * target + beta * 2 * one_port_flux]],
            dtype=dtype,
        )
    )

    R = basis.R
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-kappa*dx(dx(u)) + reaction*u "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*kappa*(R.T @ ntrace(u)) = g"
    )
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    s = basis.interval.global_grid(dist, scale=1).ravel()
    expected_u = (
        target
        * np.cosh(mu * (length * s - length / 2))
        / np.cosh(mu * length / 2)
    )[None, :]
    assert np.allclose(u.allgather_data("g"), expected_u, atol=2e-11)
    assert np.allclose(U.allgather_data("g").ravel(), target, atol=2e-11)
    assert np.allclose(
        (kappa * d3.ntrace(u)).evaluate().allgather_data("g").ravel(),
        (one_port_flux, one_port_flux),
        atol=2e-11,
    )
    assert np.allclose(
        (kappa * (R.T @ d3.ntrace(u))).evaluate().allgather_data("g"),
        2 * one_port_flux,
        atol=2e-11,
    )
    factorization = next(iter(solver.subproblem_matsolvers.values()))
    assert factorization.schur_matrix.shape == (0, 0)
    if comm.rank != 3:
        assert u["g"].shape == (0, basis.size)
        assert dist.network_partition.resident_vertices[comm.rank].size == 0
