"""Independent matrix oracle for indexed resident distributed solutions."""
import numpy as np
import pytest
from mpi4py import MPI
from scipy import sparse
from dedalus.tools.collective_linalg import CollectiveSparseSolver


def solver_for(matrix):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('requires multiple ranks')
    owners = np.arange(len(matrix)) % (comm.size - 1) + 1
    return CollectiveSparseSolver(sparse.csr_matrix(matrix / comm.size), comm,
                                  owners == comm.rank, method='gmres', restart=7,
                                  maxiter=100, rtol=1e-12)


@pytest.mark.parametrize('complex_matrix', [False, True])
@pytest.mark.parametrize('complex_rhs', [False, True])
@pytest.mark.parametrize('nrhs', [1, 3])
def test_resident_additive_duplicate_queries(complex_matrix, complex_rhs, nrhs):
    comm = MPI.COMM_WORLD
    matrix = np.diag(np.arange(5., 12.)) + np.diag(np.full(6, -.4), 1)
    if complex_matrix:
        matrix = matrix + .2j*np.eye(7)
    solver = solver_for(matrix)
    # Idle root has no matrix rows, RHS contributions, or requested solutions.
    # Other ranks duplicate every RHS row, use an arbitrary order, and request
    # a subset with repeated columns. Duplicate contributions must be summed.
    rows = np.tile(np.arange(7)[::-1], 2) if comm.rank else np.array([], dtype=int)
    columns = np.array([6, 1, 6, 3]) if comm.rank else np.array([], dtype=int)
    solver.configure_resident(rows, columns)
    expected = np.arange(1, 7*nrhs+1, dtype=float).reshape(7, nrhs) / 13
    if complex_rhs:
        expected = expected + .1j*expected[::-1]
    rhs = (matrix @ expected)[rows] / (2*(comm.size-1))
    np.testing.assert_allclose(solver.solve_resident(rhs), expected[columns], rtol=2e-11, atol=2e-11)
    if nrhs == 1:
        np.testing.assert_allclose(solver.solve_resident(rhs[:, 0]), expected[columns, 0], rtol=2e-11, atol=2e-11)
    assert solver.solve_resident(np.empty((len(rows), 0), dtype=rhs.dtype)).shape == (len(columns), 0)


def test_resident_collective_validation_and_reuse():
    comm = MPI.COMM_WORLD
    solver = solver_for(np.diag(np.arange(3., 10.)))
    with pytest.raises(ValueError, match='Resident index validation'):
        solver.configure_resident([7] if comm.rank == 0 else [], [])
    rows = np.arange(7)
    solver.configure_resident(rows, rows)
    with pytest.raises(ValueError, match='Resident RHS validation'):
        solver.solve_resident(np.ones(6 if comm.rank == 0 else 7))
    with pytest.raises(ValueError, match='Resident RHS signature'):
        solver.solve_resident(np.ones((7, 2 if comm.rank == 0 else 1)))
    rhs = np.arange(3., 10.) / comm.size
    np.testing.assert_allclose(solver.solve_resident(rhs), np.ones(7), atol=1e-12)
