"""Dependency-free collective linear algebra for replicated vectors.

The sparse matrices handled here are *additive rank-local contributions* to a
global matrix.  Vectors are replicated, while a deterministic ownership mask
ensures that collective inner products count every global row exactly once.
This layout is useful for small graph separators: it avoids a distinguished
solver rank without requiring a distributed sparse-direct dependency.
"""

from dataclasses import dataclass
import warnings

from mpi4py import MPI
import numpy as np
from scipy import linalg
from scipy import sparse
from .collective_metadata import CollectiveMetadata


__all__ = [
    "CollectiveLinearSolveError",
    "CollectiveSolveInfo",
    "CollectiveSparseSolver",
    "contiguous_row_ownership",
]


@dataclass(frozen=True)
class CollectiveSolveInfo:
    """Convergence information for one right-hand side."""

    method: str
    converged: bool
    iterations: int
    restart_cycles: int
    residual_norm: float
    relative_residual: float
    reason: str


class CollectiveLinearSolveError(RuntimeError):
    """Raised when a collective iterative solve does not converge."""

    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


def contiguous_row_ownership(size, comm):
    """Return a deterministic contiguous ownership mask for replicated rows."""
    size = int(size)
    if size < 0:
        raise ValueError("The row count must be nonnegative.")
    counts = np.full(comm.size, size // comm.size, dtype=np.int64)
    counts[: size % comm.size] += 1
    offsets = np.concatenate(([0], np.cumsum(counts)))
    owned = np.zeros(size, dtype=bool)
    owned[offsets[comm.rank] : offsets[comm.rank + 1]] = True
    return owned


def _collective_validation(comm, error, context):
    """Raise the same validation failure on every rank."""
    failed = np.asarray(error is not None, dtype=np.int32)
    comm.Allreduce(MPI.IN_PLACE, failed, op=MPI.MAX)
    if not failed:
        return
    errors = comm.allgather(error)
    failures = [
        f"rank {rank}: {message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if failures:
        raise ValueError(f"{context} failed ({'; '.join(failures)}).")


class CollectiveSparseSolver:
    """Solve an additively assembled sparse system without a root rank.

    Parameters
    ----------
    local_matrix : sparse matrix or array_like
        This rank's additive contribution to the global square matrix.  Every
        rank must use the same shape and dtype, but entries may differ.
    comm : MPI communicator
        Communicator participating in the solve.
    owned_rows : ndarray of bool, optional
        Rows counted by this rank in global inner products.  Each row must be
        selected on exactly one rank.  A deterministic contiguous partition is
        used by default.
    method : {"auto", "replicated_direct", "gmres"}, optional
        ``auto`` uses replicated direct factorization for systems no larger
        than ``direct_max_size`` and collective restarted GMRES otherwise.
    direct_max_size : int, optional
        Largest system assembled densely and factored on every rank in
        ``auto`` mode.
    restart : int, optional
        Maximum Krylov dimension before a GMRES restart.
    maxiter : int, optional
        Maximum total GMRES iterations.  Defaults to at least five Krylov
        spaces or five times the system size, whichever is larger.
    rtol, atol : float, optional
        The solve converges when ``||b-Ax|| <= max(atol, rtol*||b||)``.
    preconditioner : {"jacobi", "none"}, optional
        Replicated right preconditioner.  Jacobi entries are collectively
        assembled from the local matrix diagonals.

    Notes
    -----
    ``solve`` also accepts additive rank-local right-hand-side contributions
    by default and returns a replicated solution.  Multi-column right-hand
    sides are solved one at a time, reusing the direct factorization or Jacobi
    preconditioner. Multi-rank GMRES keeps Krylov vectors on their row owners;
    only the public input/output interface is replicated. The Network subgraph
    integration uses resident right-hand sides and solutions throughout. The
    GMRES path never assembles the global sparse matrix on one rank.
    """

    _METHODS = frozenset(("auto", "replicated_direct", "gmres"))
    _PRECONDITIONERS = frozenset(("jacobi", "none"))

    def __init__(
        self,
        local_matrix,
        comm,
        owned_rows=None,
        *,
        method="auto",
        direct_max_size=256,
        restart=40,
        maxiter=None,
        rtol=1e-10,
        atol=0.0,
        preconditioner="jacobi",
    ):
        self.comm = comm
        matrix = None
        matrix_error = None
        try:
            matrix = sparse.csr_matrix(local_matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(
                    f"The local matrix must be square, got {matrix.shape}."
                )
            if matrix.dtype.kind not in "fc":
                matrix = matrix.astype(np.float64)
            if matrix.data.size and not np.all(np.isfinite(matrix.data)):
                raise ValueError("The local matrix contains non-finite entries.")
        except Exception as error:
            matrix_error = f"{type(error).__name__}: {error}"
        _collective_validation(comm, matrix_error, "Collective matrix validation")
        self.local_matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype
        self.size = matrix.shape[0]

        signatures = comm.allgather((self.shape, self.dtype.str))
        signature_error = None
        if any(signature != signatures[0] for signature in signatures[1:]):
            signature_error = (
                "matrix shapes and dtypes differ between communicator ranks"
            )
        _collective_validation(
            comm,
            signature_error,
            "Collective matrix signature validation",
        )

        option_error = None
        try:
            method = str(method).lower()
            preconditioner = str(preconditioner).lower()
            direct_max_size = int(direct_max_size)
            restart = int(restart)
            if maxiter is not None:
                maxiter = int(maxiter)
            rtol = float(rtol)
            atol = float(atol)
            if method not in self._METHODS:
                raise ValueError(
                    f"method must be one of {sorted(self._METHODS)}, got {method!r}"
                )
            if preconditioner not in self._PRECONDITIONERS:
                raise ValueError(
                    "preconditioner must be one of "
                    f"{sorted(self._PRECONDITIONERS)}, got {preconditioner!r}"
                )
            if direct_max_size < 0:
                raise ValueError("direct_max_size must be nonnegative")
            if restart <= 0:
                raise ValueError("restart must be positive")
            if maxiter is not None and maxiter <= 0:
                raise ValueError("maxiter must be positive")
            if not np.isfinite(rtol) or rtol < 0:
                raise ValueError("rtol must be finite and nonnegative")
            if not np.isfinite(atol) or atol < 0:
                raise ValueError("atol must be finite and nonnegative")
        except Exception as error:
            option_error = f"{type(error).__name__}: {error}"
        _collective_validation(comm, option_error, "Collective solver options")

        options = (
            method,
            direct_max_size,
            restart,
            maxiter,
            rtol,
            atol,
            preconditioner,
        )
        all_options = comm.allgather(options)
        options_error = None
        if any(value != all_options[0] for value in all_options[1:]):
            options_error = "solver options differ between communicator ranks"
        _collective_validation(
            comm,
            options_error,
            "Collective solver option agreement",
        )

        if owned_rows is None:
            owned_rows = contiguous_row_ownership(self.size, comm)
        ownership_error = None
        try:
            owned_rows = np.asarray(owned_rows, dtype=bool)
            if owned_rows.shape != (self.size,):
                raise ValueError(
                    "owned_rows must be a one-dimensional mask with one "
                    "entry per matrix row"
                )
        except Exception as error:
            ownership_error = f"{type(error).__name__}: {error}"
        _collective_validation(
            comm,
            ownership_error,
            "Collective row-ownership validation",
        )
        ownership_counts = np.empty(self.size, dtype=np.int64)
        comm.Allreduce(
            owned_rows.astype(np.int64),
            ownership_counts,
            op=MPI.SUM,
        )
        ownership_error = None
        if np.any(ownership_counts != 1):
            bad_rows = np.flatnonzero(ownership_counts != 1)
            ownership_error = (
                "every row must have exactly one owner; invalid rows are "
                f"{bad_rows.tolist()}"
            )
        _collective_validation(
            comm,
            ownership_error,
            "Collective row-ownership validation",
        )

        self.owned_rows = owned_rows
        self.direct_max_size = direct_max_size
        self.restart = min(restart, max(1, self.size))
        if maxiter is None:
            maxiter = max(5 * max(1, self.size), 5 * self.restart)
        self.maxiter = maxiter
        self.rtol = rtol
        self.atol = atol
        self.preconditioner = preconditioner
        if method == "auto":
            method = (
                "replicated_direct"
                if self.size <= direct_max_size
                else "gmres"
            )
        self.method = method
        self.last_solve_info = ()

        self._lu = None
        self._inverse_diagonal = None
        self.global_matrix = None
        self._owned_krylov = None
        if self.method == "replicated_direct":
            self._build_replicated_factorization()
        else:
            self._build_right_preconditioner()
            if comm.size > 1:
                self._owned_krylov = _OwnedKrylov(self)

    def configure_resident(self, rows, columns):
        """Prepare local additive RHS and requested-solution transport."""
        if self._owned_krylov is None:
            raise ValueError('Resident transport requires multi-rank GMRES')
        self._resident_metadata = CollectiveMetadata()
        self._resident_rhs = _OwnedIndexExchange(self._owned_krylov, rows)
        self._resident_solution = _OwnedIndexExchange(self._owned_krylov, columns)

    def solve_resident(self, local_rhs):
        """Solve additive resident RHS rows, returning requested columns only."""
        error = None
        try:
            rhs = np.asarray(local_rhs)
            squeeze = rhs.ndim == 1
            if squeeze:
                rhs = rhs[:, None]
            if rhs.ndim != 2 or rhs.shape[0] != self._resident_rhs.query_size:
                raise ValueError('Resident RHS shape does not match configured rows')
            if rhs.dtype.kind not in 'fc':
                rhs = rhs.astype(np.float64)
            if not np.isfinite(rhs).all():
                raise ValueError('Nonfinite resident RHS')
        except Exception as exc:
            error = str(exc)
        signature = None if error else (rhs.shape[1], rhs.dtype.str, squeeze)
        self._resident_metadata.check(
            self.comm, signature, 'Resident RHS validation; Resident RHS signature',
            error=error, exception=ValueError,
        )
        dtype = np.result_type(self.dtype, rhs.dtype)
        owned_rhs = self._resident_rhs.reduce(np.asarray(rhs, dtype=dtype))
        owned_solution = np.empty_like(owned_rhs)
        infos = []
        for column in range(rhs.shape[1]):
            try:
                solution, info = self._owned_krylov._gmres(owned_rhs[:, column])
            except CollectiveLinearSolveError as exc:
                self.last_solve_info = tuple(infos+[exc.info])
                raise
            owned_solution[:, column] = solution
            infos.append(info)
        self.last_solve_info = tuple(infos)
        result = self._resident_solution.forward(owned_solution)
        return result[:, 0] if squeeze else result

    def _build_replicated_factorization(self):
        """Assemble and factor the small global matrix on every rank."""
        local_dense = np.ascontiguousarray(self.local_matrix.toarray())
        global_dense = np.empty_like(local_dense)
        self.comm.Allreduce(local_dense, global_dense, op=MPI.SUM)
        self.global_matrix = sparse.csr_matrix(global_dense)
        factor_error = None
        try:
            if self.size:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", linalg.LinAlgWarning)
                    self._lu = linalg.lu_factor(global_dense)
            else:
                self._lu = None
        except Exception as error:
            factor_error = f"{type(error).__name__}: {error}"
        _collective_validation(
            self.comm,
            factor_error,
            "Replicated separator factorization",
        )

    def _build_right_preconditioner(self):
        if self.preconditioner == "none":
            self._inverse_diagonal = None
            return
        local_diagonal = np.asarray(self.local_matrix.diagonal())
        diagonal = np.empty_like(local_diagonal)
        self.comm.Allreduce(local_diagonal, diagonal, op=MPI.SUM)
        inverse = np.ones_like(diagonal)
        real_dtype = np.empty((), dtype=self.dtype).real.dtype
        scale = np.max(np.abs(diagonal), initial=0.0)
        threshold = 100 * np.finfo(real_dtype).eps * max(1.0, scale)
        usable = np.abs(diagonal) > threshold
        inverse[usable] = 1 / diagonal[usable]
        self._inverse_diagonal = inverse

    def _inner(self, left, right):
        local = np.asarray(np.vdot(left[self.owned_rows], right[self.owned_rows]))
        result = np.empty_like(local)
        # A one-element numerical reduction avoids Python object serialization
        # while preserving the real/complex dtype, including on idle ranks.
        self.comm.Allreduce(local, result, op=MPI.SUM)
        return result[()]

    def _norm(self, vector):
        norm_sq = self._inner(vector, vector).real
        return float(np.sqrt(max(0.0, norm_sq)))

    def apply(self, vector):
        """Collectively apply the summed matrix to a replicated vector."""
        vector = np.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError(
                f"Expected a vector of shape {(self.size,)}, got {vector.shape}."
            )
        local = np.asarray(self.local_matrix @ vector)
        result = np.empty_like(local)
        self.comm.Allreduce(local, result, op=MPI.SUM)
        return result

    def _apply_right_preconditioner(self, vector):
        if self._inverse_diagonal is None:
            return np.array(vector, copy=True)
        return self._inverse_diagonal * vector

    def _direct_solve(self, rhs):
        if self.size == 0:
            return np.empty_like(rhs)
        if self.dtype.kind != "c" and np.iscomplexobj(rhs):
            return (
                linalg.lu_solve(self._lu, rhs.real)
                + 1j * linalg.lu_solve(self._lu, rhs.imag)
            )
        return linalg.lu_solve(self._lu, rhs)

    def _combine_gmres_basis(self, base, preconditioned, coefficients):
        """Form an iterate from the current right-preconditioned basis."""
        return base + preconditioned[:, :coefficients.size] @ coefficients

    def _gmres(self, rhs):
        dtype = np.result_type(self.dtype, rhs.dtype)
        rhs = np.asarray(rhs, dtype=dtype)
        x = np.zeros(self.size, dtype=dtype)
        rhs_norm = self._norm(rhs)
        tolerance = max(self.atol, self.rtol * rhs_norm)
        if rhs_norm == 0:
            return x, CollectiveSolveInfo(
                method="gmres",
                converged=True,
                iterations=0,
                restart_cycles=0,
                residual_norm=0.0,
                relative_residual=0.0,
                reason="zero right-hand side",
            )

        real_dtype = np.empty((), dtype=dtype).real.dtype
        breakdown_scale = 100 * np.finfo(real_dtype).eps
        iterations = 0
        cycles = 0
        residual = np.array(rhs, copy=True)
        residual_norm = rhs_norm
        reason = "maximum iterations reached"

        while iterations < self.maxiter:
            cycles += 1
            cycle_size = min(self.restart, self.maxiter - iterations)
            # Arnoldi writes columns and repeatedly multiplies leading-column
            # prefixes. Column-major storage makes columns contiguous and keeps
            # each active prefix contiguous as the Krylov basis grows.
            basis = np.zeros((self.size, cycle_size + 1), dtype=dtype, order="F")
            preconditioned = np.zeros((self.size, cycle_size), dtype=dtype, order="F")
            hessenberg = np.zeros((cycle_size + 1, cycle_size), dtype=dtype)
            target = np.zeros(cycle_size + 1, dtype=dtype)
            target[0] = residual_norm
            basis[:, 0] = residual / residual_norm
            x_base = np.array(x, copy=True)
            candidate = x_base
            happy_breakdown = False

            for column in range(cycle_size):
                preconditioned[:, column] = self._apply_right_preconditioner(
                    basis[:, column]
                )
                work = self.apply(preconditioned[:, column])
                work_norm = self._norm(work)

                # Two-pass block modified Gram--Schmidt.  Each pass requires
                # one vector-valued reduction rather than one per basis vector.
                for _ in range(2):
                    local_projection = (
                        basis[self.owned_rows, : column + 1].conj().T
                        @ work[self.owned_rows]
                    )
                    projection = np.empty_like(local_projection)
                    self.comm.Allreduce(
                        local_projection,
                        projection,
                        op=MPI.SUM,
                    )
                    hessenberg[: column + 1, column] += projection
                    work -= basis[:, : column + 1] @ projection

                next_norm = self._norm(work)
                if not np.isfinite(next_norm):
                    reason = "non-finite Arnoldi norm"
                    iterations += 1
                    break
                hessenberg[column + 1, column] = next_norm
                iterations += 1
                if next_norm > breakdown_scale * max(1.0, work_norm):
                    basis[:, column + 1] = work / next_norm
                else:
                    happy_breakdown = True

                projected = hessenberg[: column + 2, : column + 1]
                coefficients = np.linalg.lstsq(
                    projected,
                    target[: column + 2],
                    rcond=None,
                )[0]
                estimate = np.linalg.norm(
                    target[: column + 2] - projected @ coefficients
                )

                # The projected residual does not use the full iterate. Form
                # that replicated vector only for an explicit residual check
                # or the end of this restart cycle, including a maxiter cut.
                if estimate <= tolerance or happy_breakdown or column + 1 == cycle_size:
                    candidate = self._combine_gmres_basis(
                        x_base, preconditioned, coefficients,
                    )
                if estimate <= tolerance or happy_breakdown:
                    actual_residual = rhs - self.apply(candidate)
                    actual_norm = self._norm(actual_residual)
                    if actual_norm <= tolerance:
                        relative = actual_norm / rhs_norm
                        return candidate, CollectiveSolveInfo(
                            method="gmres",
                            converged=True,
                            iterations=iterations,
                            restart_cycles=cycles,
                            residual_norm=actual_norm,
                            relative_residual=relative,
                            reason=(
                                "Arnoldi breakdown at a converged solution"
                                if happy_breakdown
                                else "residual tolerance reached"
                            ),
                        )
                    # A residual-gap false positive is safest to handle as an
                    # immediate restart from the explicitly computed residual.
                    x = candidate
                    residual = actual_residual
                    residual_norm = actual_norm
                    reason = (
                        "Arnoldi breakdown before the requested tolerance"
                        if happy_breakdown
                        else "residual estimate did not pass explicit check"
                    )
                    break
            else:
                x = candidate
                residual = rhs - self.apply(x)
                residual_norm = self._norm(residual)
                if residual_norm <= tolerance:
                    return x, CollectiveSolveInfo(
                        method="gmres",
                        converged=True,
                        iterations=iterations,
                        restart_cycles=cycles,
                        residual_norm=residual_norm,
                        relative_residual=residual_norm / rhs_norm,
                        reason="residual tolerance reached after restart",
                    )
                continue

            if not np.isfinite(residual_norm) or happy_breakdown:
                break

        relative = residual_norm / rhs_norm
        info = CollectiveSolveInfo(
            method="gmres",
            converged=False,
            iterations=iterations,
            restart_cycles=cycles,
            residual_norm=residual_norm,
            relative_residual=relative,
            reason=reason,
        )
        message = (
            "Collective GMRES failed to converge after "
            f"{iterations} iterations ({reason}); residual={residual_norm:.6e}, "
            f"relative_residual={relative:.6e}, rtol={self.rtol:.6e}, "
            f"atol={self.atol:.6e}."
        )
        raise CollectiveLinearSolveError(message, info)

    def solve(self, local_rhs, *, sum_rhs=True):
        """Solve for one or more additive rank-local right-hand sides.

        Parameters
        ----------
        local_rhs : ndarray
            Shape ``(n,)`` or ``(n, nrhs)``.  By default, values are additive
            rank-local contributions and are summed before the solve.
        sum_rhs : bool, optional
            Set to false only when an already-replicated right-hand side is
            supplied identically on every rank.
        """
        rhs_error = None
        rhs = None
        squeeze = False
        try:
            rhs = np.asarray(local_rhs)
            squeeze = rhs.ndim == 1
            if squeeze:
                rhs = rhs[:, None]
            if rhs.ndim != 2 or rhs.shape[0] != self.size:
                raise ValueError(
                    "The right-hand side must have shape (n,) or (n, nrhs); "
                    f"got {np.asarray(local_rhs).shape}."
                )
            if rhs.dtype.kind not in "fc":
                rhs = rhs.astype(np.float64)
            if not np.all(np.isfinite(rhs)):
                raise ValueError("The right-hand side contains non-finite entries.")
        except Exception as error:
            rhs_error = f"{type(error).__name__}: {error}"
        _collective_validation(
            self.comm,
            rhs_error,
            "Collective right-hand-side validation",
        )

        signature = (rhs.shape, rhs.dtype.str, bool(sum_rhs), squeeze)
        signatures = self.comm.allgather(signature)
        signature_error = None
        if any(value != signatures[0] for value in signatures[1:]):
            signature_error = (
                "right-hand-side shapes, dtypes, or assembly modes differ "
                "between communicator ranks"
            )
        _collective_validation(
            self.comm,
            signature_error,
            "Collective right-hand-side agreement",
        )

        if sum_rhs:
            assembled_rhs = np.empty_like(rhs)
            self.comm.Allreduce(rhs, assembled_rhs, op=MPI.SUM)
        else:
            local_real = np.ascontiguousarray(rhs.real)
            minimum_real = np.empty_like(local_real)
            maximum_real = np.empty_like(local_real)
            self.comm.Allreduce(local_real, minimum_real, op=MPI.MIN)
            self.comm.Allreduce(local_real, maximum_real, op=MPI.MAX)
            replicas_agree = np.array_equal(minimum_real, maximum_real)
            if np.iscomplexobj(rhs):
                local_imag = np.ascontiguousarray(rhs.imag)
                minimum_imag = np.empty_like(local_imag)
                maximum_imag = np.empty_like(local_imag)
                self.comm.Allreduce(local_imag, minimum_imag, op=MPI.MIN)
                self.comm.Allreduce(local_imag, maximum_imag, op=MPI.MAX)
                replicas_agree &= np.array_equal(minimum_imag, maximum_imag)
            replica_error = None
            if not replicas_agree:
                replica_error = (
                    "sum_rhs=False requires identical replicated values on "
                    "every communicator rank"
                )
            _collective_validation(
                self.comm,
                replica_error,
                "Collective replicated right-hand-side validation",
            )
            assembled_rhs = rhs

        result_dtype = np.result_type(self.dtype, assembled_rhs.dtype)
        solution = np.empty(assembled_rhs.shape, dtype=result_dtype)
        infos = []
        for column in range(assembled_rhs.shape[1]):
            column_rhs = assembled_rhs[:, column]
            if self.method == "replicated_direct":
                column_solution = self._direct_solve(column_rhs)
                residual_norm = self._norm(
                    column_rhs - self.apply(column_solution)
                )
                rhs_norm = self._norm(column_rhs)
                relative = residual_norm / rhs_norm if rhs_norm else 0.0
                info = CollectiveSolveInfo(
                    method="replicated_direct",
                    converged=True,
                    iterations=0,
                    restart_cycles=0,
                    residual_norm=residual_norm,
                    relative_residual=relative,
                    reason="replicated direct factorization",
                )
            else:
                try:
                    if self._owned_krylov is None:
                        column_solution, info = self._gmres(column_rhs)
                    else:
                        column_solution, info = self._owned_krylov.solve_column(column_rhs)
                except CollectiveLinearSolveError as error:
                    self.last_solve_info = tuple(infos + [error.info])
                    raise
            solution[:, column] = column_solution
            infos.append(info)
        self.last_solve_info = tuple(infos)
        if squeeze:
            return solution[:, 0]
        return solution


class _OwnedKrylov(CollectiveSparseSolver):
    """Keep Arnoldi vectors on their owners, preserving the replicated API.

    Additive matrix rows are assembled once on their owners. Matvecs exchange
    requested halo columns only. Krylov arithmetic and storage use owned rows;
    the existing public right-hand-side and returned-solution API is replicated.
    """

    def __init__(self, parent):
        self.comm = parent.comm
        self.local_matrix = parent.local_matrix
        self.dtype = parent.dtype
        self.global_size = parent.size
        self.local_indices = np.flatnonzero(parent.owned_rows)
        self.size = self.local_indices.size
        self.owned_rows = np.ones(self.size, dtype=bool)
        self.counts = np.asarray(self.comm.allgather(self.size), dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(self.counts[:-1])]
        self.indices = np.empty(self.global_size, dtype=np.int64)
        self.comm.Allgatherv(self.local_indices,
                            [self.indices, self.counts, self.offsets, MPI.INT64_T])
        self.restart, self.maxiter = parent.restart, parent.maxiter
        self.rtol, self.atol = parent.rtol, parent.atol
        self._inverse_diagonal = (None if parent._inverse_diagonal is None else
                                  parent._inverse_diagonal[self.local_indices])

        self._build_owned_matrix()

    def _build_owned_matrix(self):
        # Assemble additive matrix entries once on their row owners. Krylov
        # matvecs then exchange only columns requested from neighboring owners.
        owners = np.empty(self.global_size, dtype=np.int64)
        owners[self.indices] = np.repeat(np.arange(self.comm.size), self.counts)
        entries = self.local_matrix.tocoo()
        destinations = owners[entries.row]
        order = np.argsort(destinations, kind='stable')
        send_counts = np.bincount(destinations, minlength=self.comm.size).astype(np.int64)
        recv_counts = np.empty_like(send_counts)
        self.comm.Alltoall(send_counts, recv_counts)
        send_offsets = np.r_[0, np.cumsum(send_counts[:-1])]
        recv_offsets = np.r_[0, np.cumsum(recv_counts[:-1])]
        send_keys = np.column_stack((entries.row[order], entries.col[order])).astype(np.int64)
        keys = np.empty((int(recv_counts.sum()), 2), dtype=np.int64)
        self.comm.Alltoallv([send_keys, 2*send_counts, 2*send_offsets, MPI.INT64_T],
                           [keys, 2*recv_counts, 2*recv_offsets, MPI.INT64_T])
        values = np.empty(len(keys), dtype=entries.data.dtype)
        self.comm.Alltoallv([entries.data[order], (send_counts, send_offsets)],
                           [values, (recv_counts, recv_offsets)])
        columns, inverse = np.unique(keys[:, 1], return_inverse=True)
        rows = np.searchsorted(self.local_indices, keys[:, 0])
        self.owned_matrix = sparse.csr_matrix((values, (rows, inverse)),
                                             shape=(self.size, columns.size))
        self.owned_matrix.sum_duplicates()
        self.column_owned = owners[columns] == self.comm.rank
        self.column_owned_positions = np.searchsorted(self.local_indices, columns[self.column_owned])
        ghosts = columns[~self.column_owned]
        ghost_owners = owners[ghosts]
        request_order = np.argsort(ghost_owners, kind='stable')
        self.ghost_order = request_order
        request_counts = np.bincount(ghost_owners, minlength=self.comm.size).astype(np.int64)
        incoming_counts = np.empty_like(request_counts)
        self.comm.Alltoall(request_counts, incoming_counts)
        request_offsets = np.r_[0, np.cumsum(request_counts[:-1])]
        incoming_offsets = np.r_[0, np.cumsum(incoming_counts[:-1])]
        requested = np.empty(int(incoming_counts.sum()), dtype=np.int64)
        self.comm.Alltoallv([ghosts[request_order], request_counts, request_offsets, MPI.INT64_T],
                           [requested, incoming_counts, incoming_offsets, MPI.INT64_T])
        self.send_positions = np.searchsorted(self.local_indices, requested)
        self.halo_send_layout = (incoming_counts, incoming_offsets)
        self.halo_receive_layout = (request_counts, request_offsets)
        self.ghost_count = ghosts.size

    def _replicate(self, local):
        packed = np.empty(self.global_size, dtype=local.dtype)
        self.comm.Allgatherv(np.ascontiguousarray(local),
                            [packed, (self.counts, self.offsets)])
        result = np.empty_like(packed)
        result[self.indices] = packed
        return result

    def apply(self, vector):
        send = np.ascontiguousarray(vector[self.send_positions])
        received = np.empty(self.ghost_count, dtype=vector.dtype)
        self.comm.Alltoallv([send, self.halo_send_layout],
                           [received, self.halo_receive_layout])
        values = np.empty(self.owned_matrix.shape[1], dtype=vector.dtype)
        values[self.column_owned] = vector[self.column_owned_positions]
        ghosts = np.empty_like(received)
        ghosts[self.ghost_order] = received
        values[~self.column_owned] = ghosts
        return np.asarray(self.owned_matrix @ values)

    def solve_column(self, rhs):
        local, info = self._gmres(rhs[self.local_indices])
        return self._replicate(local), info


class _OwnedIndexExchange:
    """Static indexed request/reply transport for additive resident vectors."""

    def __init__(self, krylov, query):
        self.comm = krylov.comm
        error = None
        try:
            query = np.asarray(query)
            if query.ndim != 1 or (query.size and query.dtype.kind not in 'iu'):
                raise ValueError('Resident indices must be a one-dimensional integer array')
            if np.any(query < 0) or np.any(query >= krylov.global_size):
                raise ValueError('Resident index out of range')
            query = np.asarray(query, dtype=np.int64)
        except Exception as exc:
            error = str(exc)
        _collective_validation(self.comm, error, 'Resident index validation')
        owners = np.empty(krylov.global_size, dtype=np.int64)
        owners[krylov.indices] = np.repeat(np.arange(self.comm.size), krylov.counts)
        destinations = owners[query]
        self.order = np.argsort(destinations, kind='stable')
        counts = np.bincount(destinations, minlength=self.comm.size).astype(np.int64)
        incoming = np.empty_like(counts)
        self.comm.Alltoall(counts, incoming)
        offsets = np.r_[0, np.cumsum(counts[:-1])]
        incoming_offsets = np.r_[0, np.cumsum(incoming[:-1])]
        received = np.empty(int(incoming.sum()), dtype=np.int64)
        self.comm.Alltoallv([query[self.order], counts, offsets, MPI.INT64_T],
                           [received, incoming, incoming_offsets, MPI.INT64_T])
        self.query_size, self.owned_size = query.size, krylov.size
        self.positions = np.searchsorted(krylov.local_indices, received)
        self.query_layout = (counts, offsets)
        self.owner_layout = (incoming, incoming_offsets)
        self.payload_layouts = {}

    def _exchange(self, send, receive_size, *, reverse):
        nrhs = send.shape[1]
        layouts = self.payload_layouts.get(nrhs)
        if layouts is None:
            layouts = tuple(tuple(a*nrhs for a in layout)
                            for layout in (self.query_layout, self.owner_layout))
            if len(self.payload_layouts) >= 8:
                self.payload_layouts.pop(next(iter(self.payload_layouts)))
            self.payload_layouts[nrhs] = layouts
        send_layout, receive_layout = layouts[::-1] if reverse else layouts
        receive = np.empty((receive_size, nrhs), dtype=send.dtype)
        self.comm.Alltoallv([send, send_layout], [receive, receive_layout])
        return receive

    def reduce(self, values):
        send = np.ascontiguousarray(values[self.order])
        receive = self._exchange(send, self.positions.size, reverse=False)
        result = np.zeros((self.owned_size, values.shape[1]), dtype=values.dtype)
        np.add.at(result, self.positions, receive)
        return result

    def forward(self, values):
        send = np.ascontiguousarray(values[self.positions])
        receive = self._exchange(send, self.query_size, reverse=True)
        result = np.empty_like(receive)
        result[self.order] = receive
        return result
