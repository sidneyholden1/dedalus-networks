"""Parallel tests for replicated-symbolic network DtN solves."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core import network as network_core


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_packed_row_allgather(dtype):
    """Gather uneven multi-RHS row blocks, including an empty rank zero."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    local_rows = comm.rank
    values = np.arange(local_rows * 3, dtype=np.float64).reshape(
        local_rows,
        3,
    )
    values += 100 * comm.rank
    values = values.astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * (values + 1)
    row_counts = comm.allgather(local_rows)

    gathered = network_core._allgatherv_packed_rows(
        values,
        row_counts,
        comm,
    )
    expected = np.concatenate(comm.allgather(values), axis=0)
    assert gathered.shape == (sum(row_counts), 3)
    assert np.array_equal(gathered, expected)


@pytest.mark.mpi(min_size=2)
def test_network_edge_owned_away_from_root():
    """Solve when rank zero owns no edge and the separator is empty."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    partition = d3.NetworkPartition(
        network,
        (comm.size - 1,),
        num_parts=comm.size,
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)
    g["g"] = np.array((1, 0)).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = g")
    solver = problem.build_solver(network_partition=partition)
    solver.solve()

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    expected_edges = int(comm.rank == comm.size - 1)
    assert factorization.owned_edges.size == expected_edges
    assert factorization.edge_solver.local_columns.size == (
        expected_edges * (basis.size + 2)
    )
    if comm.rank == 0:
        assert factorization.edge_response.shape[0] == 0
    assert factorization.schur_matrix.shape == (0, 0)
    s = dist.local_grid(basis.interval)
    assert np.allclose(u["g"], 1 - s)
    assert np.allclose(U["g"].ravel(), (1, 0))


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_parallel_network_chain_poisson(dtype):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    network = d3.Network(
        vertex_ids=("left", "junction", "right"),
        edge_ids=("short", "long"),
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(network, dtype=dtype)
    assert dist.comm.rank == comm.rank
    assert dist.comm.size == comm.size
    assert dist.layout_comm == MPI.COMM_SELF
    basis = d3.NetworkBasis(network, size=6)

    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)

    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)
    if comm.rank == 0:
        g["g"] = np.array([1, 0, 0]).reshape(g["g"].shape)
    else:
        # Root owns replicated inputs; the solver broadcasts its assembled RHS.
        g["g"] = np.array([17, 19, 23]).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = g")
    # MPI network problems automatically select the DtN backend.
    solver = problem.build_solver()
    assert solver.network_solver == "dtn"
    solver.solve()

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    subproblem = solver.subproblems[0]
    if comm.rank == 0:
        assert subproblem.L_min.nnz > 0
    else:
        assert subproblem.L_min.nnz == 0
    expected_owned = factorization.partition.edges[comm.rank]
    assert np.array_equal(factorization.owned_edges, expected_owned)
    assert len(factorization.edge_solver.blocks) == expected_owned.size
    local_edge_columns = factorization.edge_solver.local_columns.size
    assert factorization.edge_response.shape[0] == local_edge_columns
    assert factorization.local_interface_to_edge.shape[1] == local_edge_columns
    assert factorization.edge_column_counts[comm.rank] == local_edge_columns
    assert (
        factorization.edge_column_counts.sum()
        == factorization.edge_columns.size
    )
    all_owned = comm.allgather(expected_owned)
    ownership = np.concatenate(all_owned)
    assert np.array_equal(np.sort(ownership), np.arange(network.num_edges))
    assert np.unique(ownership).size == ownership.size
    separator_size = factorization.partition.separator_vertices.size
    assert factorization.schur_matrix.shape == (separator_size,) * 2

    s = dist.local_grid(basis.interval)
    expected_u = np.concatenate((1 - s / 3, (2 / 3) * (1 - s)), axis=0)
    assert np.allclose(u["g"], expected_u)
    assert np.allclose(U["g"].ravel(), [1, 2 / 3, 0])

    # Every process receives the same reconstructed state.
    gathered_U = comm.allgather(U["g"].copy())
    assert all(np.array_equal(gathered_U[0], values) for values in gathered_U[1:])
    gathered_u = comm.allgather(u["g"].copy())
    assert all(np.array_equal(gathered_u[0], values) for values in gathered_u[1:])


@pytest.mark.mpi(min_size=2)
def test_connected_subgraph_condensation():
    """Eliminate multiple local vertices and retain only chain cut vertices."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    num_edges = 8
    network = d3.Network(
        endpoints=tuple((vertex, vertex + 1) for vertex in range(num_edges)),
        lengths=np.ones(num_edges),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=5)

    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)
    alpha_values = np.zeros(network.num_vertices)
    alpha_values[[0, -1]] = 1
    beta_values = 1 - alpha_values
    g_values = np.zeros(network.num_vertices)
    g_values[0] = 1
    alpha["g"] = alpha_values.reshape(alpha["g"].shape)
    beta["g"] = beta_values.reshape(beta["g"].shape)
    g["g"] = g_values.reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = g")
    solver = problem.build_solver(network_partition="connected")
    solver.solve()

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    expected_cuts = min(comm.size, num_edges) - 1
    assert factorization.partition.separator_vertices.size == expected_cuts
    assert factorization.schur_matrix.shape == (expected_cuts,) * 2

    s = dist.local_grid(basis.interval)
    edge_offsets = np.arange(num_edges)[:, None]
    assert np.allclose(u["g"], 1 - (edge_offsets + s) / num_edges)
    assert np.allclose(
        U["g"].ravel(),
        1 - np.arange(network.num_vertices) / num_edges,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("timestepper", [d3.SBDF1, d3.RK111])
def test_parallel_network_heat_step(timestepper):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=12)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)

    s = dist.local_grid(basis.interval)
    initial_u = np.concatenate(
        (
            np.sin(np.pi * s / 3),
            np.sin(np.pi * (1 + 2 * s) / 3),
        ),
        axis=0,
    )
    initial_U = np.array([0, np.sin(np.pi / 3), 0])
    if comm.rank == 0:
        u["g"] = initial_u
        U["g"] = initial_U.reshape(U["g"].shape)
    else:
        # Initial state is authoritative on root, as are matrices and RHSs.
        u["g"] = 3 * initial_u + 1
        U["g"] = np.array((4, 5, 6)).reshape(U["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.IVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "dt(u) - dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = 0")
    solver = problem.build_solver(timestepper)
    assert solver.network_solver == "dtn"
    if comm.rank == 0:
        assert solver.subproblems[0].M_min.nnz > 0
        assert solver.subproblems[0].L_min.nnz > 0
    else:
        assert solver.subproblems[0].M_min.nnz == 0
        assert solver.subproblems[0].L_min.nnz == 0
    timestep = 0.01
    solver.step(timestep)

    decay = 1 / (1 + timestep * (np.pi / 3) ** 2)
    assert np.allclose(u["g"], decay * initial_u, atol=2e-10)
    assert np.allclose(U["g"].ravel(), decay * initial_U, atol=2e-10)

    # A changed timestep rebuilds and redistributes the implicit matrix.
    second_timestep = 0.02
    solver.step(second_timestep)
    decay /= 1 + second_timestep * (np.pi / 3) ** 2
    assert np.allclose(u["g"], decay * initial_u, atol=3e-10)
    assert np.allclose(U["g"].ravel(), decay * initial_U, atol=3e-10)


@pytest.mark.mpi(min_size=2)
def test_parallel_topology_mismatch_fails_collectively():
    """Reject incompatible replicated topology before allocating fields."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    if comm.rank == 0:
        endpoints = ((0, 1), (1, 2))
    else:
        endpoints = ((0, 2), (2, 1))
    network = d3.Network(endpoints=endpoints, lengths=(1, 1))
    with pytest.raises(RuntimeError, match="Network topology validation"):
        d3.Distributor(network, dtype=np.float64)


@pytest.mark.mpi(min_size=2)
def test_parallel_root_authoritative_matrix_rebuild():
    """Rank zero supplies replicated NCC values when matrices are rebuilt."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    kappa = dist.Field(name="kappa", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)
    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)
    g["g"] = np.array([1, 0, 0]).reshape(g["g"].shape)
    if comm.rank == 0:
        kappa["g"] = np.array([1, 4]).reshape(kappa["g"].shape)
    else:
        kappa["g"] = np.array([9, 1]).reshape(kappa["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(kappa*dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ (dtrace(kappa)*ntrace(u))) = g"
    )
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(U["g"].ravel(), (1, 0.2, 0))

    if comm.rank == 0:
        kappa["g"] = np.array([4, 1]).reshape(kappa["g"].shape)
    else:
        kappa["g"] = np.array([1, 9]).reshape(kappa["g"].shape)
    solver.solve(rebuild_matrices=True)
    assert np.allclose(U["g"].ravel(), (1, 0.8, 0))
