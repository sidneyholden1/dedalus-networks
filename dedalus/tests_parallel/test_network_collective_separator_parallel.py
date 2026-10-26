"""End-to-end MPI regressions for root-free Network separator solves."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.mpi(min_size=4)
def test_network_separator_gmres_with_idle_root():
    """Force collective GMRES when rank zero owns no edge or separator row."""
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip("regression uses exactly four MPI ranks")

    # Two connected edges per non-root rank leave two separator vertices.
    num_edges = 2 * (comm.size - 1)
    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(num_edges)),
        lengths=np.ones(num_edges),
    )
    owners = np.repeat(np.arange(1, comm.size), 2)
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)
    alpha_values = np.zeros(network.num_vertices)
    alpha_values[[0, -1]] = 1
    alpha["g"] = alpha_values.reshape(alpha["g"].shape)
    beta["g"] = (1 - alpha_values).reshape(beta["g"].shape)
    g_values = np.zeros(network.num_vertices)
    g_values[0] = 1
    g["g"] = g_values.reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, n: d3.Lift(field, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = g")
    solver = problem.build_solver()
    # These attributes are public build options in SolverBase; assigning them
    # here also verifies that factorization consumes the solver configuration.
    solver.network_separator_method = "gmres"
    solver.network_separator_restart = 4
    solver.network_separator_maxiter = 40
    solver.network_separator_rtol = 2e-12
    solver.network_separator_atol = 0.0
    solver.network_separator_preconditioner = "jacobi"
    solver.solve()

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    separator = factorization.subgraph_solver
    assert factorization.owned_edges.size == (0 if comm.rank == 0 else 2)
    assert separator.separator_solver is not None
    assert separator.separator_solver.method == "gmres"
    assert separator.schur_matrix_is_local
    assert bool(separator.separator_owned_rows.any()) is (comm.rank in (1, 2))
    if comm.rank == 0:
        assert separator.local_schur_matrix.nnz == 0
    assert all(
        info.converged for info in separator.separator_solver.last_solve_info
    )

    owned_edges = dist.network_partition.edges[comm.rank]
    s = dist.local_grid(basis.interval)
    expected_local = 1 - (owned_edges[:, None] + s) / num_edges
    assert np.allclose(u["g"], expected_local, atol=4e-11, rtol=4e-11)
    assert np.allclose(
        U["g"].ravel(),
        1 - np.arange(network.num_vertices) / num_edges,
        atol=4e-11,
        rtol=4e-11,
    )
    gathered_vertices = comm.allgather(U["g"].copy())
    assert all(
        np.array_equal(gathered_vertices[0], values)
        for values in gathered_vertices[1:]
    )


@pytest.mark.mpi(min_size=4)
def test_network_separator_direct_matrix_is_replicated():
    """The small direct route exposes the same assembled Schur matrix."""
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip("regression uses exactly four MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1.0, 1.0),
    )
    owners = np.array((1, 3))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)
    g["g"] = np.array((1.0, 0.0, 0.0)).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, n: d3.Lift(field, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = g")
    solver = problem.build_solver()
    solver.network_separator_method = "replicated_direct"
    solver.solve()

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    separator = factorization.subgraph_solver
    assert separator.separator_solver.method == "replicated_direct"
    assert not separator.schur_matrix_is_local
    matrices = comm.allgather(separator.schur_matrix.toarray())
    assert all(np.array_equal(matrices[0], matrix) for matrix in matrices[1:])
