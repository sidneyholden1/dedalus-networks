"""Unit tests for the collective separator linear algebra backend."""

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

from dedalus.tools.collective_linalg import (
    CollectiveLinearSolveError,
    CollectiveSparseSolver,
    contiguous_row_ownership,
)


def test_contiguous_row_ownership_self():
    owned = contiguous_row_ownership(5, MPI.COMM_SELF)
    assert np.array_equal(owned, np.ones(5, dtype=bool))


@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
def test_collective_gmres_multiple_rhs(preconditioner):
    matrix = sparse.csr_matrix(
        np.array(
            [
                [5.0, -1.0, 0.5, 0.0],
                [2.0, 6.0, -1.0, 0.0],
                [0.0, -1.0, 4.0, 1.0],
                [0.5, 0.0, -1.0, 3.0],
            ]
        )
    )
    expected = np.array(
        [
            [1.0, -2.0],
            [0.5, 1.0],
            [-1.0, 0.25],
            [2.0, 0.0],
        ]
    )
    rhs = matrix @ expected
    solver = CollectiveSparseSolver(
        matrix,
        MPI.COMM_SELF,
        method="gmres",
        restart=3,
        maxiter=40,
        rtol=2e-13,
        preconditioner=preconditioner,
    )

    solution = solver.solve(rhs)

    assert np.allclose(solution, expected, atol=2e-12, rtol=2e-12)
    assert len(solver.last_solve_info) == 2
    assert all(info.converged for info in solver.last_solve_info)
    assert all(info.iterations > 0 for info in solver.last_solve_info)
    assert all(info.restart_cycles >= 1 for info in solver.last_solve_info)


def test_replicated_direct_complex_multiple_rhs():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [3 + 1j, -1j, 0],
                [2, 4 - 0.5j, 1j],
                [0.5j, -1, 2 + 2j],
            ]
        )
    )
    expected = np.array(
        [[1 - 1j, 2j], [0.5 + 0.25j, -1], [2, 3 - 0.5j]]
    )
    solver = CollectiveSparseSolver(
        matrix,
        MPI.COMM_SELF,
        method="replicated_direct",
    )

    solution = solver.solve(matrix @ expected)

    assert np.allclose(solution, expected)
    assert all(
        info.method == "replicated_direct"
        for info in solver.last_solve_info
    )


def test_auto_method_threshold_and_replicated_rhs():
    matrix = sparse.eye(3, format="csr") * 2
    direct = CollectiveSparseSolver(
        matrix,
        MPI.COMM_SELF,
        method="auto",
        direct_max_size=3,
    )
    iterative = CollectiveSparseSolver(
        matrix,
        MPI.COMM_SELF,
        method="auto",
        direct_max_size=2,
    )

    assert direct.method == "replicated_direct"
    assert iterative.method == "gmres"
    assert np.allclose(
        iterative.solve(np.array((2.0, 4.0, 6.0)), sum_rhs=False),
        (1.0, 2.0, 3.0),
    )


def test_real_direct_factorization_accepts_complex_rhs():
    matrix = sparse.csr_matrix(np.array([[3.0, 1.0], [1.0, 2.0]]))
    expected = np.array((1 + 2j, -0.5j))
    solver = CollectiveSparseSolver(
        matrix,
        MPI.COMM_SELF,
        method="replicated_direct",
    )

    assert np.allclose(solver.solve(matrix @ expected), expected)


def test_gmres_nonconvergence_has_diagnostics():
    matrix = sparse.csr_matrix(
        np.array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    )
    solver = CollectiveSparseSolver(
        matrix,
        MPI.COMM_SELF,
        method="gmres",
        restart=1,
        maxiter=1,
        rtol=1e-15,
        preconditioner="none",
    )

    with pytest.raises(
        CollectiveLinearSolveError,
        match="failed to converge after 1 iterations",
    ) as raised:
        solver.solve(np.array((1.0, -2.0, 0.5)))

    assert not raised.value.info.converged
    assert raised.value.info.iterations == 1
    assert raised.value.info.residual_norm > 0
    assert solver.last_solve_info == (raised.value.info,)
    # Independent one-dimensional minimum-residual solution for this Krylov
    # space, checking that maxiter still forms and checks the final iterate.
    rhs = np.array((1.0, -2.0, 0.5))
    image = matrix @ rhs
    expected = np.vdot(image, rhs) / np.vdot(image, image) * rhs
    assert np.isclose(raised.value.info.residual_norm,
                      np.linalg.norm(rhs - matrix @ expected), rtol=1e-12)


@pytest.mark.parametrize("dtype", (np.float64, np.complex128))
@pytest.mark.parametrize("restart", (2, 7))
def test_gmres_iterate_at_convergence_and_restart_boundaries(dtype, restart):
    diagonal = np.arange(3, 10, dtype=np.float64).astype(dtype)
    if dtype == np.complex128:
        diagonal += 0.3j * np.arange(7)
    matrix = sparse.diags((np.full(6, 0.2), diagonal, np.full(6, -0.7)),
                          offsets=(-1, 0, 1), format="csr")
    expected = np.column_stack((np.sin(np.arange(7)), np.cos(np.arange(7)))).astype(dtype)
    if dtype == np.complex128:
        expected += 0.2j * expected[::-1]
    rhs = matrix @ expected
    solver = CollectiveSparseSolver(matrix, MPI.COMM_SELF, method="gmres",
                                    restart=restart, maxiter=100, rtol=2e-12,
                                    preconditioner="none")
    solution = solver.solve(rhs)
    assert np.allclose(solution, expected, atol=2e-10, rtol=2e-10)
    residuals = np.linalg.norm(rhs - matrix @ solution, axis=0)
    assert np.all(residuals <= 2.1e-12 * np.linalg.norm(rhs, axis=0))
    assert all(info.converged for info in solver.last_solve_info)
    if restart == 2:
        assert all(info.restart_cycles > 1 for info in solver.last_solve_info)


def test_invalid_row_ownership_is_rejected():
    with pytest.raises(ValueError, match="exactly one owner"):
        CollectiveSparseSolver(
            sparse.eye(2, format="csr"),
            MPI.COMM_SELF,
            owned_rows=np.array((True, False)),
        )


def test_zero_size_and_zero_rhs():
    empty = sparse.csr_matrix((0, 0), dtype=np.float64)
    direct = CollectiveSparseSolver(
        empty,
        MPI.COMM_SELF,
        method="replicated_direct",
    )
    iterative = CollectiveSparseSolver(
        sparse.eye(2, format="csr"),
        MPI.COMM_SELF,
        method="gmres",
    )

    assert direct.solve(np.empty((0, 2))).shape == (0, 2)
    assert np.array_equal(iterative.solve(np.zeros(2)), np.zeros(2))
    assert iterative.last_solve_info[0].iterations == 0
