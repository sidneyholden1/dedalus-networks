"""Small dependency-free eigensolver kernels for distributed vectors."""

import numpy as np
from mpi4py import MPI


def collective_inner(comm, owned, left, right):
    """Return an inner product while counting every global degree once."""
    owned = np.asarray(owned, dtype=bool)
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.shape != owned.shape:
        raise ValueError("Collective inner-product operands must have one shape.")
    local = np.vdot(left[owned], right[owned])
    return comm.allreduce(local, op=MPI.SUM)


def collective_norm(comm, owned, vector):
    """Return the Euclidean norm of a distributed vector."""
    norm_sq = collective_inner(comm, owned, vector, vector).real
    # Roundoff in a collective complex dot product can make an exact zero
    # infinitesimally negative.
    return float(np.sqrt(max(0.0, norm_sq)))


def collective_arnoldi(
    matvec,
    initial,
    comm,
    owned,
    *,
    num_eigenvalues,
    ncv,
    breakdown_tolerance=None,
):
    """Build one collectively orthogonalized Arnoldi factorization.

    Parameters
    ----------
    matvec : callable
        Collective application of the distributed linear operator.  The input
        and output arrays have rank-local shapes.
    initial : ndarray
        Rank-local portion of the initial vector.  Ghost or replicated entries
        must already agree between ranks.
    comm : MPI communicator
        Communicator over the distributed vector.
    owned : ndarray of bool
        Mask selecting the rank-local entries that count in global inner
        products.  Replicated entries must be selected on exactly one rank.
    num_eigenvalues : int
        Number of Ritz pairs to return.
    ncv : int
        Maximum Arnoldi-space dimension.
    breakdown_tolerance : float, optional
        Absolute threshold for terminating on an invariant subspace.

    Returns
    -------
    values : ndarray
        Ritz values, ordered from largest to smallest magnitude.
    vectors : ndarray
        Rank-local Ritz vectors in columns.
    estimates : ndarray
        Arnoldi residual estimates for the returned Ritz pairs.
    dimension : int
        Dimension reached by the Arnoldi factorization.

    Notes
    -----
    This is deliberately an unrestarted kernel.  It is used for the first
    dependency-free distributed Network EVP backend, where ``ncv`` provides a
    clear memory/convergence tradeoff.  A future thick-restart layer can reuse
    the same collective vector operations without changing the PDE reduction.
    """
    initial = np.asarray(initial)
    owned = np.asarray(owned, dtype=bool)
    if initial.ndim != 1 or owned.shape != initial.shape:
        raise ValueError("Arnoldi vectors and ownership masks must be one-dimensional.")
    num_eigenvalues = int(num_eigenvalues)
    ncv = int(ncv)
    if num_eigenvalues <= 0:
        raise ValueError("num_eigenvalues must be positive.")
    if ncv <= num_eigenvalues:
        raise ValueError("ncv must be larger than num_eigenvalues.")

    dtype = np.result_type(initial.dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    if breakdown_tolerance is None:
        breakdown_tolerance = 100 * np.finfo(real_dtype).eps
    breakdown_tolerance = float(breakdown_tolerance)
    if not np.isfinite(breakdown_tolerance) or breakdown_tolerance < 0:
        raise ValueError("breakdown_tolerance must be finite and nonnegative.")

    initial_norm = collective_norm(comm, owned, initial)
    if not np.isfinite(initial_norm) or initial_norm == 0:
        raise ValueError("The Arnoldi initial vector has zero or non-finite norm.")

    basis = np.zeros((initial.size, ncv + 1), dtype=dtype)
    hessenberg = np.zeros((ncv + 1, ncv), dtype=dtype)
    basis[:, 0] = initial / initial_norm
    dimension = 0
    has_residual_vector = False

    for column in range(ncv):
        work = np.asarray(matvec(basis[:, column]), dtype=dtype)
        if work.shape != initial.shape:
            raise ValueError("Arnoldi matvec returned an unexpected local shape.")

        # Two-pass modified Gram--Schmidt.  Each pass uses one vector-valued
        # reduction, rather than one collective per existing basis vector.
        for _ in range(2):
            local_projection = (
                basis[owned, : column + 1].conj().T @ work[owned]
            )
            projection = np.empty_like(local_projection)
            comm.Allreduce(local_projection, projection, op=MPI.SUM)
            hessenberg[: column + 1, column] += projection
            work -= basis[:, : column + 1] @ projection

        next_norm = collective_norm(comm, owned, work)
        if not np.isfinite(next_norm):
            raise FloatingPointError("Arnoldi iteration produced a non-finite norm.")
        hessenberg[column + 1, column] = next_norm
        dimension = column + 1
        if next_norm <= breakdown_tolerance:
            break
        basis[:, column + 1] = work / next_norm
        has_residual_vector = True

    if dimension < num_eigenvalues:
        raise RuntimeError(
            "Arnoldi reached an invariant subspace before producing the "
            "requested number of Ritz pairs."
        )

    projected = hessenberg[:dimension, :dimension]
    # LAPACK is free to choose different bases and orderings inside repeated
    # or tightly clustered invariant subspaces.  Calling it independently on
    # every rank can therefore splice different projected eigenvectors into a
    # nominally collective Ritz vector.  Diagonalize once and make rank zero's
    # result authoritative before lifting back into the distributed basis.
    eigensolve_error = None
    eigensolve_payload = None
    if comm.rank == 0:
        try:
            eigensolve_payload = np.linalg.eig(projected)
        except Exception as error:
            eigensolve_error = f"{type(error).__name__}: {error}"
    eigensolve_error = comm.bcast(eigensolve_error, root=0)
    if eigensolve_error is not None:
        raise RuntimeError(
            "Projected Arnoldi eigensolve failed on network rank 0: "
            + eigensolve_error
        )
    values, projected_vectors = comm.bcast(eigensolve_payload, root=0)
    order = np.argsort(-np.abs(values), kind="stable")[:num_eigenvalues]
    values = values[order]
    projected_vectors = projected_vectors[:, order]
    vectors = basis[:, :dimension] @ projected_vectors

    if has_residual_vector and dimension < hessenberg.shape[0]:
        estimates = np.abs(
            hessenberg[dimension, dimension - 1]
            * projected_vectors[-1]
        )
    else:
        estimates = np.zeros(values.shape, dtype=real_dtype)
    return values, vectors, estimates, dimension
