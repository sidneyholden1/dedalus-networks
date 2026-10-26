"""Focused regressions for collective Network separator condensation."""

from mpi4py import MPI
import numpy as np
from scipy import sparse

import dedalus.public as d3
from dedalus.core import network as network_core


class _DenseFactorization:
    def __init__(self, matrix, solver):
        self.matrix = sparse.csr_matrix(matrix).toarray()

    def solve(self, rhs):
        return np.linalg.solve(self.matrix, rhs)


class _SeparatorOptions:
    entry_cutoff = 1e-12
    network_separator_method = "gmres"
    network_separator_restart = 8
    network_separator_maxiter = 40
    network_separator_rtol = 1e-13
    network_separator_atol = 0.0
    network_separator_preconditioner = "jacobi"


def test_collective_subgraph_zero_separator_multi_rhs():
    """A wholly interior serial subgraph bypasses an empty separator."""
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1.0, 1.0),
    )
    partition = d3.NetworkPartition(network, (0, 0), num_parts=1)
    matrix = sparse.csr_matrix(
        np.array(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 3.0, -0.5],
                [0.0, -0.5, 2.0],
            ]
        )
    )
    factorization = network_core._SubgraphDtNFactorization(
        matrix,
        sparse.csr_matrix(matrix.shape),
        np.arange(3),
        np.arange(3),
        partition,
        _DenseFactorization,
        _SeparatorOptions(),
        MPI.COMM_SELF,
    )
    rhs = np.array(
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0]]
    )
    solution = factorization.solve(rhs, np.zeros_like(rhs))

    assert factorization.separator_rows.size == 0
    assert factorization.separator_solver is None
    assert factorization.schur_matrix.shape == (0, 0)
    assert np.allclose(solution, np.linalg.solve(matrix.toarray(), rhs))


def test_collective_subgraph_promoted_global_row_multi_rhs():
    """A global row promotes touched vertices and is solved by GMRES."""
    network = d3.Network(endpoints=((0, 1),), lengths=(1.0,))
    partition = d3.NetworkPartition(network, (0,), num_parts=1)
    # Label 2 is a synthetic global-scalar interface degree of freedom.  Its
    # coupling to vertex 0 promotes that vertex from the rank interior.
    matrix = sparse.csr_matrix(
        np.array(
            [
                [5.0, -1.0, 0.75],
                [-1.0, 4.0, 0.0],
                [0.75, 0.0, 2.5],
            ]
        )
    )
    labels = np.arange(3)
    factorization = network_core._SubgraphDtNFactorization(
        matrix,
        sparse.csr_matrix(matrix.shape),
        labels,
        labels,
        partition,
        _DenseFactorization,
        _SeparatorOptions(),
        MPI.COMM_SELF,
    )
    rhs = np.array(
        [[1.0, -2.0], [3.0, 0.5], [-1.5, 4.0]]
    )
    solution = factorization.solve(rhs, np.zeros_like(rhs))

    assert np.array_equal(factorization.separator_rows, (0, 2))
    assert factorization.separator_solver.method == "gmres"
    assert factorization.schur_matrix_is_local
    assert len(factorization.separator_solver.last_solve_info) == 2
    assert all(
        info.converged
        for info in factorization.separator_solver.last_solve_info
    )
    assert np.allclose(
        solution,
        np.linalg.solve(matrix.toarray(), rhs),
        atol=2e-12,
        rtol=2e-12,
    )


def test_serial_network_poisson_with_separator_options():
    """Public serial DtN solves remain unchanged by separator configuration."""
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1.0, 2.0),
    )
    dist = d3.Distributor(network, comm=MPI.COMM_SELF, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)
    alpha["g"] = np.array((1.0, 0.0, 1.0)).reshape(alpha["g"].shape)
    beta["g"] = np.array((0.0, 1.0, 0.0)).reshape(beta["g"].shape)
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
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = g")
    solver = problem.build_solver(network_solver="dtn")
    solver.network_separator_method = "gmres"
    solver.solve()

    s = dist.local_grid(basis.interval)
    expected_u = np.concatenate(
        (1 - s / 3, (2 / 3) * (1 - s)),
        axis=0,
    )
    assert np.allclose(u["g"], expected_u)
    assert np.allclose(U["g"].ravel(), (1.0, 2 / 3, 0.0))
