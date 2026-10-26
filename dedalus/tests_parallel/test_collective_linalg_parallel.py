"""MPI tests for root-free collective separator solves."""

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

from dedalus.tools.collective_linalg import CollectiveSparseSolver


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.complex64, np.complex128))
def test_collective_hermitian_inner_product_with_idle_root(dtype):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    left = np.arange(1, 6).astype(dtype)
    right = np.linspace(0.4, 1.6, 5).astype(dtype)
    if np.dtype(dtype).kind == "c":
        left += 0.25j * left[::-1]
        right += 0.125j * np.roll(right, 1)
    owners = np.arange(5) % (comm.size - 1) + 1
    solver = CollectiveSparseSolver(sparse.eye(5, dtype=dtype, format="csr") / comm.size,
                                    comm, owners == comm.rank, method="gmres")
    inner = solver._inner(left, right)
    tolerance = 10 * np.finfo(left.real.dtype).eps
    assert np.asarray(inner).dtype == np.dtype(dtype)
    assert np.allclose(inner, np.vdot(left, right), rtol=tolerance, atol=tolerance)
    assert np.isclose(solver._norm(left), np.linalg.norm(left), rtol=tolerance)
    assert all(value == inner for value in comm.allgather(inner))


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("method", ("replicated_direct", "gmres"))
def test_additive_collective_solver_with_idle_root(method):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    matrix = np.array(
        [
            [6.0, -1.0, 0.0, 0.5],
            [2.0, 7.0, -1.0, 0.0],
            [0.0, -2.0, 5.0, 1.0],
            [1.0, 0.0, -1.0, 4.0],
        ]
    )
    rows, columns = np.indices(matrix.shape)
    entry_owner = (2 * rows + columns + 1) % comm.size
    local_matrix = sparse.csr_matrix(
        np.where(entry_owner == comm.rank, matrix, 0.0)
    )

    # Rank zero deliberately owns no inner-product rows.  This exercises the
    # absence of a distinguished factorization or solve rank in both modes.
    row_owner = np.arange(matrix.shape[0]) % (comm.size - 1) + 1
    owned_rows = row_owner == comm.rank
    expected = np.array(
        [[1.0, -0.5], [2.0, 1.0], [-1.0, 0.25], [0.5, -2.0]]
    )
    rhs = matrix @ expected
    rhs_owner = (np.arange(matrix.shape[0]) + 1) % comm.size
    local_rhs = np.where(rhs_owner[:, None] == comm.rank, rhs, 0.0)

    solver = CollectiveSparseSolver(
        local_matrix,
        comm,
        owned_rows,
        method=method,
        restart=4,
        maxiter=20,
        rtol=5e-13,
    )
    solution = solver.solve(local_rhs)

    assert not owned_rows.any() if comm.rank == 0 else True
    assert np.allclose(solution, expected, atol=3e-12, rtol=3e-12)
    gathered = comm.allgather(solution)
    assert all(np.array_equal(other, gathered[0]) for other in gathered[1:])
    assert all(info.converged for info in solver.last_solve_info)


@pytest.mark.mpi(min_size=2)
def test_collective_solver_replicated_rhs_mode():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    diagonal = np.arange(2, 7, dtype=np.float64)
    local_diagonal = np.where(
        np.arange(diagonal.size) % comm.size == comm.rank,
        diagonal,
        0.0,
    )
    solver = CollectiveSparseSolver(
        sparse.diags(local_diagonal),
        comm,
        method="gmres",
        direct_max_size=0,
        rtol=1e-13,
    )
    replicated_rhs = diagonal * np.arange(1, 6)

    solution = solver.solve(replicated_rhs, sum_rhs=False)

    assert np.allclose(solution, np.arange(1, 6))


@pytest.mark.mpi(min_size=2)
def test_collective_solver_rejects_inconsistent_replicated_rhs():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    solver = CollectiveSparseSolver(
        sparse.eye(2, format="csr") / comm.size,
        comm,
        method="gmres",
    )
    inconsistent_rhs = np.array((1.0 + comm.rank, 2.0))

    with pytest.raises(ValueError, match="requires identical replicated values"):
        solver.solve(inconsistent_rhs, sum_rhs=False)


@pytest.mark.parametrize('complex_matrix', [False, True])
def test_owned_krylov_complex_rhs_and_empty_owner(complex_matrix):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('requires multiple ranks')
    matrix = np.diag(np.arange(4., 11.)) + np.diag(np.full(6, -.3), 1)
    if complex_matrix:
        matrix = matrix + .2j*np.eye(7)
    # Noncontiguous owners, including an idle root; matrix contributions have
    # different ownership, so reduce-scatter must sum overlapping rows.
    owners = np.arange(7) % (comm.size-1) + 1
    solver = CollectiveSparseSolver(sparse.csr_matrix(matrix/comm.size), comm,
                                    owners == comm.rank, method='gmres', restart=4,
                                    maxiter=100, rtol=1e-12)
    expected = np.column_stack((np.arange(7)+1j, np.arange(7)[::-1]-.5j))
    actual = solver.solve((matrix @ expected)/comm.size)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    assert solver._owned_krylov.size == np.count_nonzero(owners == comm.rank)
    assert all(i.converged for i in solver.last_solve_info)
