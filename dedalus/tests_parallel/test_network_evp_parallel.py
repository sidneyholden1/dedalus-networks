"""MPI regression tests for Network eigenvalue problems."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.tools.collective_eigs import collective_arnoldi


def build_chain_evp(
    comm,
    *,
    size=14,
    distributed=True,
    dtype=np.complex128,
):
    """Build a Dirichlet Laplacian on two edges spanning [0, pi]."""
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(np.pi / 2, np.pi / 2),
    )
    owners = np.array((comm.size - 1, 1), dtype=int)
    dist_kw = {"network_partition": owners} if distributed else {}
    dist = d3.Distributor(network, dtype=dtype, **dist_kw)
    basis = d3.NetworkBasis(network, size=size)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    eigenvalue = dist.Field(name="eigenvalue")

    alpha.set_global_data(np.array((1, 0, 1))[:, None])
    beta.set_global_data(np.array((0, 1, 0))[:, None])
    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.EVP(
        [u, U, tau_1, tau_2],
        eigenvalue,
        namespace=locals(),
    )
    problem.add_equation(
        "eigenvalue*u + dx(dx(u)) "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ ntrace(u)) = 0"
    )
    return problem, u, U, owners


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", (np.float64, np.complex128))
def test_whole_edge_distributed_network_evp_arnoldi(dtype):
    """Recover targeted modes with idle ranks and an edge-idle root."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    problem, u, U, _ = build_chain_evp(comm, dtype=dtype)
    # Exercise independent equation-row and unknown-column permutations.  In
    # particular, this catches accidental use of column ownership to norm the
    # original-pencil residuals.
    solver = problem.build_solver(bc_top=True, tau_left=True)
    solver.solve_sparse(
        solver.subproblems[0],
        N=3,
        target=0.8,
        ncv=26,
        tol=2e-7,
    )

    assert solver.network_solver == "dtn"
    assert solver.dist.network_partition.edges[0].size == 0
    assert np.allclose(solver.eigenvalues.real, (1, 4, 9), atol=2e-6)
    assert np.all(np.abs(solver.eigenvalues.imag) < 2e-7)
    assert np.all(solver.eigenvalue_residuals < 2e-7)

    solver.set_state(0)
    assert u.allreduce_L2_norm() > 0
    # Vertex data are part of the distributed eigenvector, not reconstructed
    # later from a rank-zero-only result.
    gathered_U = U.allgather_data(layout="g").ravel()
    assert np.all(np.isfinite(gathered_U))
    assert np.max(np.abs(gathered_U[[0, 2]])) < (
        2e-6 * np.max(np.abs(gathered_U))
    )


@pytest.mark.mpi(min_size=2)
def test_distributed_network_evp_left_modes_are_explicitly_rejected():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    problem, _, _, _ = build_chain_evp(comm, size=8)
    solver = problem.build_solver()
    with pytest.raises(NotImplementedError, match="Dense eigenvalue solves"):
        solver.solve_dense(solver.subproblems[0])
    with pytest.raises(NotImplementedError, match="Left eigenvectors"):
        solver.solve_sparse(
            solver.subproblems[0],
            N=1,
            target=0.8,
            left=True,
        )


@pytest.mark.mpi(min_size=2)
def test_replicated_compatibility_network_evp_arnoldi():
    """Use the same collective Arnoldi path with replicated field storage."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    problem, _, _, owners = build_chain_evp(
        comm,
        size=12,
        distributed=False,
    )
    solver = problem.build_solver(network_partition=owners)
    solver.solve_sparse(
        solver.subproblems[0],
        N=2,
        target=0.8,
        ncv=22,
        tol=2e-7,
    )

    assert not solver.dist.distributed_network
    assert solver.network_solver == "dtn"
    assert np.allclose(solver.eigenvalues.real, (1, 4), atol=2e-6)
    assert np.all(solver.eigenvalue_residuals < 2e-7)


@pytest.mark.mpi(min_size=2)
def test_collective_arnoldi_diagonalizes_projected_problem_on_root(
    monkeypatch,
):
    """Never splice rank-local LAPACK choices for clustered Ritz values."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    if comm.rank != 0:
        def forbidden_rank_local_eig(*args, **kw):
            raise AssertionError("projected eigensolve ran away from rank zero")

        monkeypatch.setattr(np.linalg, "eig", forbidden_rank_local_eig)

    diagonal = np.array((3.0, 3.0 + 2e-10, 0.75))
    initial = np.array((1.0, -0.7, 0.25))
    owned = np.full(initial.shape, comm.rank == 0)
    values, vectors, estimates, dimension = collective_arnoldi(
        lambda vector: diagonal * vector,
        initial,
        comm,
        owned,
        num_eigenvalues=2,
        ncv=3,
    )

    gathered_values = comm.allgather(values)
    assert all(
        np.array_equal(rank_values, gathered_values[0])
        for rank_values in gathered_values[1:]
    )
    assert dimension == 3
    assert vectors.shape == (3, 2)
    assert estimates.shape == (2,)
    assert np.allclose(np.sort(values.real), np.sort(diagonal[:2]))


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "case",
    (
        "left_mismatch",
        "N_conversion",
        "target_mismatch",
        "rebuild_mismatch",
        "normalize_left_mismatch",
        "raise_on_mismatch_mismatch",
        "ncv_conversion",
        "tol_mismatch",
        "seed_mismatch",
        "breakdown_invalid",
        "raise_mismatch",
        "which_invalid",
        "unknown_keyword",
    ),
)
def test_distributed_network_evp_option_errors_are_collective(case):
    """A bad or inconsistent option must raise on every rank before assembly."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    problem, _, _, _ = build_chain_evp(comm, size=8)
    solver = problem.build_solver()
    solve_kw = {
        "N": 1,
        "target": 0.8,
        "ncv": 8,
        "raise_on_no_convergence": False,
    }
    if case == "left_mismatch":
        solve_kw["left"] = comm.rank == 0
    elif case == "N_conversion" and comm.rank == 0:
        solve_kw["N"] = "not-an-integer"
    elif case == "target_mismatch" and comm.rank != 0:
        solve_kw["target"] = 0.9
    elif case == "rebuild_mismatch":
        solve_kw["rebuild_matrices"] = comm.rank == 0
    elif case == "normalize_left_mismatch":
        solve_kw["normalize_left"] = comm.rank == 0
    elif case == "raise_on_mismatch_mismatch":
        solve_kw["raise_on_mismatch"] = comm.rank == 0
    elif case == "ncv_conversion" and comm.rank == 0:
        solve_kw["ncv"] = "not-an-integer"
    elif case == "tol_mismatch" and comm.rank != 0:
        solve_kw["tol"] = 1e-7
    elif case == "seed_mismatch" and comm.rank != 0:
        solve_kw["seed"] = 17
    elif case == "breakdown_invalid" and comm.rank == 0:
        solve_kw["breakdown_tolerance"] = -1
    elif case == "raise_mismatch" and comm.rank != 0:
        solve_kw["raise_on_no_convergence"] = True
    elif case == "which_invalid" and comm.rank == 0:
        solve_kw["which"] = "SM"
    elif case == "unknown_keyword" and comm.rank == 0:
        solve_kw["unsupported_option"] = True

    with pytest.raises(RuntimeError) as raised:
        solver.solve_sparse(solver.subproblems[0], **solve_kw)

    messages = comm.allgather(str(raised.value))
    assert len(set(messages)) == 1
    assert (
        "Distributed Network EVP option parsing" in messages[0]
        or "Distributed Network EVP option validation" in messages[0]
    )


@pytest.mark.mpi(min_size=2)
def test_distributed_network_evp_v0_shape_error_is_collective():
    """Rank-local expanded-vector validation cannot strand peer ranks."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    problem, _, _, _ = build_chain_evp(comm, size=8)
    solver = problem.build_solver()
    subproblem = solver.subproblems[0]
    solver.build_matrices([subproblem], ["M", "L"])
    expanded_size = subproblem.pre_right.shape[0]
    if comm.rank == 0:
        v0 = np.ones(expanded_size + 1)
    else:
        v0 = np.ones(expanded_size)

    with pytest.raises(
        RuntimeError,
        match="Distributed Network EVP initial-vector construction",
    ) as raised:
        solver.solve_sparse(
            subproblem,
            N=1,
            target=0.8,
            ncv=8,
            v0=v0,
        )

    messages = comm.allgather(str(raised.value))
    assert len(set(messages)) == 1
