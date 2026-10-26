"""Runtime values must still be validated after RHS metadata agreement is cached."""
import numpy as np
import pytest
from mpi4py import MPI
from scipy import sparse
from dedalus.tools.collective_linalg import CollectiveSparseSolver


def test_warm_resident_nonfinite_failure_and_recovery():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('multiple ranks required')
    matrix = np.diag(np.arange(3., 10.))
    owned = np.arange(7) % comm.size == comm.rank
    solver = CollectiveSparseSolver(sparse.csr_matrix(matrix/comm.size), comm,
                                    owned, method='gmres', rtol=1e-12)
    solver.configure_resident(np.arange(7), np.arange(7))
    rhs = np.arange(3., 10.)/comm.size
    np.testing.assert_allclose(solver.solve_resident(rhs), np.ones(7), atol=1e-12)
    bad = rhs.copy()
    if comm.rank == 1:
        bad[2] = np.nan
    with pytest.raises(ValueError, match='Nonfinite resident RHS'):
        solver.solve_resident(bad)
    np.testing.assert_allclose(solver.solve_resident(2*rhs), 2*np.ones(7), atol=1e-12)
