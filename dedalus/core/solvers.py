"""Solver classes."""

from mpi4py import MPI
from .network_storage import elements_signature

import numpy as np
import time
import h5py
import pathlib
import scipy.linalg
import cProfile
import pstats
import numbers
from math import prod
from collections import defaultdict
import pickle

from . import subsystems
from . import timesteppers
from .evaluator import Evaluator
from ..libraries.matsolvers import matsolvers
from ..tools.config import config
from ..tools.array import apply_sparse, scipy_sparse_eigs
from ..tools.parallel import ProfileWrapper, parallel_mkdir

PROFILE_DEFAULT = config['profiling'].getboolean('PROFILE_DEFAULT')
PARALLEL_PROFILE_DEFAULT = config['profiling'].getboolean('PARALLEL_PROFILE_DEFAULT')
PROFILE_DIRECTORY = pathlib.Path(config['profiling'].get('PROFILE_DIRECTORY'))

import logging
logger = logging.getLogger(__name__.split('.')[-1])


def _resolve_network_solver(problem, network_solver):
    if network_solver not in (None, "auto", "dtn"):
        raise ValueError("network_solver must be None, 'auto', or 'dtn'.")
    network_comm = getattr(problem.dist, "network_comm", None)
    if network_solver != "auto":
        if (
            network_solver is None
            and problem.dist.distributed_network
            and network_comm is not None
            and network_comm.size > 1
        ):
            raise ValueError(
                "MPI Network problems with distributed field storage require "
                "network_solver='dtn'."
            )
        return network_solver
    if network_comm is None or network_comm.size == 1:
        return None
    has_vertex_interface = any(
        any(
            getattr(basis, "_network_role", None) == "vertices"
            for basis in variable.domain.bases
        )
        for variable in problem.LHS_variables
    )
    if not has_vertex_interface:
        raise ValueError(
            "Parallel Network problems require an explicit vertex field for "
            "automatic DtN condensation."
        )
    return "dtn"


def _resolve_network_partition_option(problem, network_partition):
    """Bind solver ownership to storage ownership when fields are sharded."""
    if not problem.dist.distributed_network:
        return network_partition
    if network_partition is not None:
        raise ValueError(
            "Distributed Network ownership is fixed when constructing the "
            "Distributor; omit the solver-time network_partition option."
        )
    return problem.dist.network_partition


def _network_matrix_assembly_root(problem, network_solver):
    """Select the authoritative matrix rank for an MPI network problem."""
    network_comm = getattr(problem.dist, "network_comm", None)
    if (
        network_solver == "dtn"
        and network_comm is not None
        and network_comm.size > 1
        and not problem.dist.distributed_network
    ):
        return 0
    return None


_NETWORK_SEPARATOR_METHODS = frozenset(
    ("auto", "replicated_direct", "gmres")
)
_NETWORK_SEPARATOR_PRECONDITIONERS = frozenset(("jacobi", "none"))


def _validate_network_separator_options(
    method,
    direct_max_size,
    restart,
    maxiter,
    rtol,
    atol,
    preconditioner,
):
    """Validate and normalize public Network separator solver options."""
    if not isinstance(method, str):
        raise TypeError("network_separator_method must be a string.")
    if method not in _NETWORK_SEPARATOR_METHODS:
        raise ValueError(
            "network_separator_method must be 'auto', "
            "'replicated_direct', or 'gmres'."
        )

    integer_options = (
        ("network_separator_direct_max_size", direct_max_size, True),
        ("network_separator_restart", restart, False),
        ("network_separator_maxiter", maxiter, False),
    )
    normalized_integers = {}
    for name, value, allow_zero in integer_options:
        if name == "network_separator_maxiter" and value is None:
            normalized_integers[name] = None
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            numbers.Integral,
        ):
            raise TypeError(f"{name} must be an integer.")
        value = int(value)
        invalid_range = value < 0 if allow_zero else value <= 0
        if invalid_range:
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}.")
        normalized_integers[name] = value

    normalized_tolerances = {}
    for name, value in (
        ("network_separator_rtol", rtol),
        ("network_separator_atol", atol),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            numbers.Real,
        ):
            raise TypeError(f"{name} must be a real number.")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        normalized_tolerances[name] = value

    if not isinstance(preconditioner, str):
        raise TypeError("network_separator_preconditioner must be a string.")
    if preconditioner not in _NETWORK_SEPARATOR_PRECONDITIONERS:
        raise ValueError(
            "network_separator_preconditioner must be 'jacobi' or 'none'."
        )

    return (
        method,
        normalized_integers["network_separator_direct_max_size"],
        normalized_integers["network_separator_restart"],
        normalized_integers["network_separator_maxiter"],
        normalized_tolerances["network_separator_rtol"],
        normalized_tolerances["network_separator_atol"],
        preconditioner,
    )


def _resolve_network_separator_options(problem, *options):
    """Validate separator options without stranding MPI Network peers."""
    comm = getattr(problem.dist, "network_comm", None)
    collective = comm is not None and comm.size > 1
    normalized = None
    option_error = None
    try:
        normalized = _validate_network_separator_options(*options)
    except Exception as error:
        if not collective:
            raise
        option_error = f"{type(error).__name__}: {error}"

    if collective:
        from .network import (
            _raise_collective_error,
            _validate_collective_signature,
        )

        _raise_collective_error(
            comm,
            option_error,
            "Network separator option validation",
        )
        _validate_collective_signature(
            comm,
            normalized,
            "Network separator option agreement",
        )
    return normalized


def _resolve_network_interior_elimination(problem, method):
    """Agree on the interior algorithm before entering collective setup."""
    comm = getattr(problem.dist, "network_comm", None)
    collective = comm is not None and comm.size > 1
    option_error = None
    try:
        if not isinstance(method, str):
            raise TypeError("network_interior_elimination must be a string.")
        if method not in ("coupled", "two_stage"):
            raise ValueError(
                "network_interior_elimination must be 'coupled' or 'two_stage'."
            )
    except Exception as error:
        if not collective:
            raise
        option_error = f"{type(error).__name__}: {error}"
    if collective:
        from .network import _raise_collective_error, _validate_collective_signature

        _raise_collective_error(comm, option_error, "Network interior option validation")
        _validate_collective_signature(comm, method, "Network interior option agreement")
    return method


def _network_domain_signature(domain):
    """Describe the Network discretization represented by a domain."""
    from .network import _array_signature

    def interval_signature(interval):
        parameters = tuple(
            (
                name,
                repr(getattr(interval, name, None)),
            )
            for name in (
                "size",
                "bounds",
                "dealias",
                "library",
                "a",
                "b",
                "a0",
                "b0",
            )
        )
        return (
            type(interval).__module__,
            type(interval).__qualname__,
            parameters,
        )

    signature = []
    network = getattr(domain.dist, "network", None)
    for basis in domain.bases:
        network_basis = getattr(basis, "_network_basis", None)
        basis_network = getattr(basis, "_network", None)
        if basis_network is None and network_basis is not None:
            basis_network = network_basis.network
        role = getattr(basis, "_network_role", None)
        if basis_network is network and role in (
            "edges",
            "vertices",
            "ports",
        ):
            # Keep canonical ordering/subset checks without replicating a
            # Python integer tuple for every atom on every receiving rank.
            canonical_elements = elements_signature(getattr(basis, "_network_elements", range(basis.size)))
            signature.append(
                (
                    "cardinal",
                    role,
                    basis.size,
                    canonical_elements,
                )
            )
        basis_coords = getattr(
            getattr(basis, "coordsys", None),
            "coords",
            (),
        )
        if network is not None and any(
            coord is network.position_coord for coord in basis_coords
        ):
            # Fingerprint the basis actually present in this domain.  This is
            # essential for differentiated Jacobi domains, whose (a, b)
            # parameters differ from the parent NetworkBasis interval.
            signature.append(
                (
                    "position",
                    interval_signature(basis),
                )
            )
    return tuple(signature)


def _validate_network_discretization(problem):
    """Require every MPI rank to construct the same Network bases."""
    comm = getattr(problem.dist, "network_comm", None)
    if comm is None or comm.size == 1:
        return

    variable_signature = tuple(
        (
            np.dtype(variable.dtype).str,
            tuple(coordsys.dim for coordsys in variable.tensorsig),
            _network_domain_signature(variable.domain),
        )
        for variable in problem.LHS_variables
    )
    equation_signature = []
    for equation in problem.equations:
        expression_signatures = []
        for name in ("M", "L", "dF", "F"):
            expression = equation.get(name)
            if not expression:
                expression_signatures.append((name, ()))
                continue
            atom_signatures = tuple(
                _network_domain_signature(atom.domain)
                for atom in expression.atoms()
                if hasattr(atom, "domain")
            )
            expression_signatures.append((name, atom_signatures))
        equation_signature.append(
            (
                _network_domain_signature(equation["domain"]),
                tuple(expression_signatures),
            )
        )

    from .network import _validate_collective_signature

    _validate_collective_signature(
        comm,
        (
            np.dtype(problem.dist.dtype).str,
            np.dtype(problem.dtype).str,
            variable_signature,
            tuple(equation_signature),
        ),
        "Network discretization validation",
    )


def _default_subproblem_solver_factory(matrix, solver, subproblem):
    """Build the configured matrix solver for a subproblem matrix."""
    return solver.matsolver(matrix, solver)


def _network_subproblem_solver_factory(matrix, solver, subproblem):
    """Build a network DtN solver for a subproblem matrix."""
    from .network import (
        DistributedNetworkDtNFactorization,
        NetworkDtNFactorization,
    )
    network_comm = getattr(solver.dist, "network_comm", None)
    if network_comm is None:
        network_comm = solver.dist.comm
    if solver.dist.distributed_network:
        factorization_class = DistributedNetworkDtNFactorization
    else:
        factorization_class = NetworkDtNFactorization
    return factorization_class(
        matrix,
        solver=solver,
        subproblem=subproblem,
        matsolver=solver.matsolver,
        partition=solver.network_partition,
        comm=network_comm,
    )


def _validate_network_subproblems(solver):
    """Require ranks to construct a compatible subproblem sequence."""
    if solver.network_solver != "dtn":
        return
    comm = getattr(solver.dist, "network_comm", None)
    if comm is None or comm.size == 1:
        return
    resident = (
        solver.dist.distributed_network
        and solver.dist.network_vertex_storage == "owned_ghost"
    )
    if resident:
        # Whole-edge and resident-vertex layouts can legitimately give idle
        # ranks different numbers of local subsystems.  The collective DtN
        # path requires the same ordered matrix groups, not equal local work.
        signature = (
            len(solver.subproblems),
            tuple(repr(subproblem.group) for subproblem in solver.subproblems),
        )
    else:
        signature = (
            len(solver.subproblems),
            tuple(
                (repr(subproblem.group), len(subproblem.subsystems))
                for subproblem in solver.subproblems
            ),
        )
    signatures = comm.allgather(signature)
    reference = signatures[0]
    for rank, rank_signature in enumerate(signatures):
        if rank_signature != reference:
            raise RuntimeError(
                "Network subproblem validation failed on network rank "
                f"{rank}: subproblem sequence does not match rank 0."
            )


class SolverBase:
    """
    Base class for PDE solvers.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    ncc_cutoff : float, optional
        Mode amplitude cutoff for LHS NCC expansions (default: 1e-6)
    max_ncc_terms : int, optional
        Maximum terms to include in LHS NCC expansions (default: None (no limit))
    entry_cutoff : float, optional
        Matrix entry cutoff to avoid fill-in from cancellation errors (default: 1e-12)
    matrix_coupling : tuple of bool, optional
        Matrix coupling override.
    matsolver : str or Matsolver class, optional
        Matrix solver. Default taken from config.
    subproblem_solver_factory : callable, optional
        Factory called as ``factory(matrix, solver, subproblem)`` to build an
        object implementing ``solve`` for each assembled subproblem matrix.
        By default, the configured matrix solver is instantiated directly.
    network_interior_elimination : {"coupled", "two_stage"}, optional
        Interior elimination for distributed Network DtN solves with
        ``owned_ghost`` vertex storage. The default, ``"coupled"``, factors
        edge coefficients, tau variables and internal vertex values together.
        ``"two_stage"`` eliminates edges before internal vertices. Serial and
        replicated-storage solvers retain their existing elimination paths.
    network_separator_method : {"auto", "replicated_direct", "gmres"}, optional
        Solver for the partition-separator system in a distributed Network
        DtN solve.  ``"auto"`` uses replicated direct factorization for small
        separators and collective restarted GMRES for larger ones.
    network_separator_direct_max_size : int, optional
        Largest separator solved with replicated direct factorization in
        ``"auto"`` mode (default: 256).  Set to zero to always select GMRES
        for a nonempty separator.
    network_separator_restart : int, optional
        Krylov dimension before a collective GMRES restart (default: 40).
    network_separator_maxiter : int or None, optional
        Maximum total collective GMRES iterations.  ``None`` selects a size-
        dependent default.
    network_separator_rtol, network_separator_atol : float, optional
        Relative and absolute collective GMRES tolerances (defaults: 1e-10
        and 0.0).
    network_separator_preconditioner : {"jacobi", "none"}, optional
        Collective GMRES right preconditioner (default: ``"jacobi"``).
    bc_top : bool, optional
        Whether to place boundary conditions at top of matrices. Default taken from config.
    tau_left : bool, optional
        Whether to place tau columns at left of matrices. Default taken from config.
    interleave_components : bool, optional
        Whether to interleave components before variables. Default taken from config.
    store_expanded_matrices : bool, optional
        Whether to store right-preconditioned matrices. Default taken from config.
    """

    def __init__(self, problem, ncc_cutoff=1e-6, max_ncc_terms=None, entry_cutoff=1e-12, matrix_coupling=None, matsolver=None,
                 bc_top=None, tau_left=None, interleave_components=None, store_expanded_matrices=None,
                 subproblem_solver_factory=None, network_separator_method="auto",
                 network_separator_direct_max_size=256, network_separator_restart=40,
                 network_separator_maxiter=None, network_separator_rtol=1e-10,
                 network_separator_atol=0.0, network_separator_preconditioner="jacobi",
                 network_interior_elimination="coupled"):
        # Take attributes from problem
        self.problem = problem
        self.dist = problem.dist
        self.network_interior_elimination = _resolve_network_interior_elimination(
            problem, network_interior_elimination
        )
        separator_options = _resolve_network_separator_options(
            problem,
            network_separator_method,
            network_separator_direct_max_size,
            network_separator_restart,
            network_separator_maxiter,
            network_separator_rtol,
            network_separator_atol,
            network_separator_preconditioner,
        )
        _validate_network_discretization(problem)
        self.dtype = problem.dtype
        self.state = problem.variables
        # Process options
        self.ncc_cutoff = ncc_cutoff
        self.max_ncc_terms = max_ncc_terms
        self.entry_cutoff = entry_cutoff
        (
            self.network_separator_method,
            self.network_separator_direct_max_size,
            self.network_separator_restart,
            self.network_separator_maxiter,
            self.network_separator_rtol,
            self.network_separator_atol,
            self.network_separator_preconditioner,
        ) = separator_options
        if matrix_coupling is None:
            matrix_coupling = np.array(problem.matrix_coupling)
            # Couple fully separable problems along last axis by default for efficiency
            if not np.any(matrix_coupling):
                matrix_coupling[-1] = True
        else:
            # Check specified coupling for compatibility
            problem_coupling = np.array(problem.matrix_coupling)
            matrix_coupling = np.array(matrix_coupling)
            if np.any(~matrix_coupling & problem_coupling):
                raise ValueError(f"Specified solver coupling is incompatible with problem coupling: {problem_coupling}")
        # Check that coupled dimensions are local
        coeff_layout = self.dist.coeff_layout
        coupled_nonlocal = matrix_coupling & ~coeff_layout.local
        if self.dist.distributed_network:
            coupled_nonlocal[0] = False
        if np.any(coupled_nonlocal):
            raise ValueError(f"Problem is coupled along distributed dimensions: {tuple(np.where(coupled_nonlocal)[0])}")
        self.matrix_coupling = matrix_coupling
        # Determine matrix dependence based on specified coupling
        self.matrix_dependence = np.array(problem.matrix_dependence)
        for eq in problem.eqs:
            for basis in eq['domain'].bases:
                first_axis = self.dist.get_basis_axis(basis)
                slices = slice(first_axis, first_axis+basis.dim)
                self.matrix_dependence[slices] = self.matrix_dependence[slices] | basis.matrix_dependence(matrix_coupling[slices])
        # Process config options
        if matsolver is None:
            matsolver = config['linear algebra'][self.matsolver_default]
        if isinstance(matsolver, str):
            matsolver = matsolvers[matsolver.lower()]
        self.matsolver = matsolver
        if subproblem_solver_factory is None:
            subproblem_solver_factory = _default_subproblem_solver_factory
        elif not callable(subproblem_solver_factory):
            raise TypeError("subproblem_solver_factory must be callable.")
        self.subproblem_solver_factory = subproblem_solver_factory
        if bc_top is None:
            bc_top = config['matrix construction'].getboolean('BC_TOP')
        self.bc_top = bc_top
        if tau_left is None:
            tau_left = config['matrix construction'].getboolean('TAU_LEFT')
        self.tau_left = tau_left
        if interleave_components is None:
            interleave_components = config['matrix construction'].getboolean('INTERLEAVE_COMPONENTS')
        self.interleave_components = interleave_components
        if store_expanded_matrices is None:
            store_expanded_matrices = config['matrix construction'].getboolean('STORE_EXPANDED_MATRICES')
        self.store_expanded_matrices = store_expanded_matrices
        # Process option overrides from matsolver
        for key, value in matsolver.config.items():
            if getattr(self, key, None) is not value:
                logger.info("matsolver overriding solver option '%i' with '%s'" %(key, value))
                setattr(self, key, value)
        # Setup subsystems and subproblems without building matrices
        self.subsystems = subsystems.build_subsystems(self)
        self.subproblems = subsystems.build_subproblems(self, self.subsystems)
        self.subproblems_by_group = {sp.group: sp for sp in self.subproblems}
        # Build evaluator
        namespace = {}
        self.evaluator = Evaluator(self.dist, namespace)

    def build_subproblem_solver(self, subproblem, matrix):
        """Build a solver for an assembled subproblem matrix."""
        return self.subproblem_solver_factory(matrix, self, subproblem)

    def apply_subproblem_matrix(
        self,
        matrix,
        array,
        axis=0,
        out=None,
        subproblem=None,
    ):
        """Apply an assembled matrix, respecting root-only storage."""
        if self.dist.distributed_network:
            if subproblem is None:
                raise ValueError(
                    "Distributed Network matrix application requires its "
                    "subproblem."
                )
            from .network import apply_distributed_network_subproblem_matrix

            return apply_distributed_network_subproblem_matrix(
                matrix,
                array,
                self,
                subproblem,
                axis=axis,
                out=out,
            )
        assembly_root = getattr(self, "matrix_assembly_root", None)
        if assembly_root is None:
            return apply_sparse(matrix, array, axis=axis, out=out)
        comm = getattr(self.dist, "network_comm", None)
        if comm is None:
            raise ValueError(
                "Root-only matrix application requires a network communicator."
            )
        if out is None:
            out_shape = list(array.shape)
            out_shape[axis] = matrix.shape[0]
            out = np.empty(out_shape, dtype=array.dtype)
        apply_error = None
        if comm.rank == assembly_root:
            try:
                apply_sparse(matrix, array, axis=axis, out=out)
            except Exception as error:
                apply_error = f"{type(error).__name__}: {error}"
        apply_error = comm.bcast(apply_error, root=assembly_root)
        if apply_error is not None:
            raise RuntimeError(
                "Root-only matrix application failed on network rank "
                f"{assembly_root}: {apply_error}"
            )
        comm.Bcast(out, root=assembly_root)
        return out

    def build_matrices(self, subproblems=None, matrices=None):
        """Build matrices for selected subproblems."""
        if subproblems is None:
            subproblems = self.subproblems
        if matrices is None:
            matrices = self.matrices
        subsystems.build_subproblem_matrices(self, subproblems, matrices)


class EigenvalueSolver(SolverBase):
    """
    Linear eigenvalue problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus eigenvalue problem.
    network_solver : {"auto", "dtn", None}, optional
        Network linear-solve backend used by collective targeted EVPs.
        ``"auto"`` selects DtN for MPI Network problems and keeps the direct
        SciPy path in serial.
    network_partition : optional
        Edge ownership for DtN in replicated compatibility mode.  Whole-edge
        distributed storage binds ownership on the Distributor instead.
    **kw :
        Other options passed to :class:`SolverBase`.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution (after set_state method is called).
    eigenvalues : ndarray
        Vector of eigenvalues.
    eigenvectors : ndarray
        Array of eigenvectors. The eigenvector corresponding to the i-th
        eigenvalues is eigenvectors[:, i].
    eigenvalue_subproblem : Subproblem object
        The subproblem for which the EVP has been solved.
    """

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'
    matrices = ['M', 'L']

    def __init__(
        self,
        problem,
        network_solver="auto",
        network_partition=None,
        **kw,
    ):
        logger.debug('Beginning EVP instantiation')
        self.network_solver = _resolve_network_solver(problem, network_solver)
        self.network_partition = _resolve_network_partition_option(
            problem,
            network_partition,
        )
        self.matrix_assembly_root = _network_matrix_assembly_root(
            problem,
            self.network_solver,
        )
        if self.network_solver == "dtn":
            if kw.get("subproblem_solver_factory") is not None:
                raise ValueError(
                    "network_solver='dtn' cannot be combined with a custom "
                    "subproblem_solver_factory."
                )
            kw["subproblem_solver_factory"] = _network_subproblem_solver_factory
        super().__init__(problem, **kw)
        _validate_network_subproblems(self)
        logger.debug('Finished EVP instantiation')

    def print_subproblem_ranks(self, subproblems=None, target=0):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            if not hasattr(sp, 'L_min'):
                continue
            L = sp.L_min.toarray()
            M = sp.M_min.toarray()
            A = L + target * M
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}, cond: {np.linalg.cond(A):.1e}")

    def solve_dense(self, subproblem, rebuild_matrices=False, left=False, normalize_left=True, **kw):
        """
        Perform dense eigenvector search for selected subproblem.
        This routine finds all eigenvectors but is computationally expensive.

        Parameters
        ----------
        subproblem : Subproblem object
            Subproblem for which to solve the EVP.
        rebuild_matrices : bool, optional
            Rebuild LHS matrices if coefficients have changed (default: False).
        left : bool, optional
            Solve for the left eigenvectors of the system in addition to the
            right eigenvectors. The left eigenvectors are the right eigenvectors
            of the conjugate-transposed problem. Follows same definition described
            in scipy.linalg.eig documentation (default: False).
        normalize_left : bool, optional
            Normalize the left eigenvectors such that the modified left eigenvectors
            form a biorthonormal (not just biorthogonal) set with respect to the right
            eigenvectors (default: True).
        **kw :
            Other keyword options passed to scipy.linalg.eig.
        """
        network_comm = getattr(self.dist, "network_comm", None)
        if (
            self.network_solver == "dtn"
            and network_comm is not None
            and network_comm.size > 1
        ):
            raise NotImplementedError(
                "Dense eigenvalue solves are not available for collective "
                "Network DtN problems; use solve_sparse() for targeted right "
                "eigenmodes or run the direct EVP on one MPI rank."
            )
        self.eigenvalue_subproblem = sp = subproblem
        # Build matrices if directed or not yet built
        if rebuild_matrices or not hasattr(sp, 'L_min'):
            self.build_matrices([sp], ['M', 'L'])
        # Solve as dense general eigenvalue problem
        A = sp.L_min.toarray()
        B = - sp.M_min.toarray()
        eig_output = scipy.linalg.eig(A, b=B, left=left, **kw)
        # Unpack output
        if left:
            self.eigenvalues, pre_left_evecs, pre_right_evecs = eig_output
            self.right_eigenvectors = self.eigenvectors = sp.pre_right @ pre_right_evecs
            self.left_eigenvectors = sp.pre_left.conj().T @ pre_left_evecs
            self.modified_left_eigenvectors = (sp.M_min @ sp.pre_right_pinv).conj().T @ pre_left_evecs
            if normalize_left:
                norms = np.diag(pre_left_evecs.T.conj() @ sp.M_min @ pre_right_evecs)
                self.left_eigenvectors /= np.conj(norms)
                self.modified_left_eigenvectors /= np.conj(norms)
        else:
            self.eigenvalues, pre_eigenvectors = eig_output
            self.eigenvectors = sp.pre_right @ pre_eigenvectors

    def solve_sparse(self, subproblem, N, target, rebuild_matrices=False, left=False, normalize_left=True, raise_on_mismatch=True, v0=None, **kw):
        """
        Perform targeted sparse eigenvector search for selected subproblem.
        This routine finds a subset of eigenvectors near the specified target.

        Parameters
        ----------
        subproblem : Subproblem object
            Subproblem for which to solve the EVP.
        N : int
            Number of eigenvectors to solve for. Note: the dense method may
            be more efficient for finding large numbers of eigenvectors.
        target : complex
            Target eigenvalue for search.
        rebuild_matrices : bool, optional
            Rebuild LHS matrices if coefficients have changed (default: False).
        left : bool, optional
            Solve for the left eigenvectors of the system in addition to the
            right eigenvectors. The left eigenvectors are the right eigenvectors
            of the conjugate-transposed problem. Follows same definition described
            in scipy.linalg.eig documentation (default: False).
        normalize_left : bool, optional
            Normalize the left eigenvectors such that the modified left eigenvectors
            form a biorthonormal (not just biorthogonal) set with respect to the right
            eigenvectors (default: True).
        raise_on_mismatch : bool, optional
            Raise a RuntimeError if the left and right eigenvalues do not match (default: True).
        v0 : ndarray, optional
            Initial guess for eigenvector, e.g. from subsystem.gather (default: None).
        **kw :
            Other keyword options passed to scipy.sparse.linalg.eig.
        """
        self.eigenvalue_subproblem = sp = subproblem
        network_comm = getattr(self.dist, "network_comm", None)
        if (
            self.network_solver == "dtn"
            and network_comm is not None
            and network_comm.size > 1
        ):
            return self._solve_sparse_distributed_network(
                sp,
                N,
                target,
                rebuild_matrices=rebuild_matrices,
                left=left,
                normalize_left=normalize_left,
                raise_on_mismatch=raise_on_mismatch,
                v0=v0,
                **kw,
            )
        # Build matrices if directed or not yet built
        if rebuild_matrices or not hasattr(sp, 'L_min'):
            self.build_matrices([sp], ['M', 'L'])
        # Solve as sparse general eigenvalue problem
        A = sp.L_min
        B = - sp.M_min
        # Precondition starting guess if provided
        if v0 is not None:
            v0 = sp.pre_right_pinv @ v0
        # Solve for the right (and optionally left) eigenvectors
        eig_output = scipy_sparse_eigs(A=A, B=B, left=left, N=N, target=target, matsolver=self.matsolver, v0=v0, **kw)
        if left:
            # Note: this definition of "left eigenvectors" is consistent with the documentation for scipy.linalg.eig
            self.eigenvalues, pre_right_evecs, self.left_eigenvalues, pre_left_evecs = eig_output
            self.right_eigenvectors = self.eigenvectors = sp.pre_right @ pre_right_evecs
            self.left_eigenvectors = sp.pre_left.conj().T @ pre_left_evecs
            self.modified_left_eigenvectors = (sp.M_min @ sp.pre_right_pinv).conj().T @ pre_left_evecs
            # Check that eigenvalues match
            if not np.allclose(self.eigenvalues, np.conjugate(self.left_eigenvalues)):
                if raise_on_mismatch:
                    raise RuntimeError("Conjugate of left eigenvalues does not match right eigenvalues. "
                                       "The full sets of left and right vectors won't form a biorthogonal set. "
                                       "This error can be disabled by passing raise_on_mismatch=False to "
                                       "solve_sparse().")
                else:
                    logger.warning("Conjugate of left eigenvalues does not match right eigenvalues.")
                    if normalize_left:
                        logger.warning("Cannot normalize left eigenvectors.")
                        normalize_left = False
            # Normalize
            if normalize_left:
                norms = np.diag(pre_left_evecs.T.conj() @ sp.M_min @ pre_right_evecs)
                self.left_eigenvectors /= np.conj(norms)
                self.modified_left_eigenvectors /= np.conj(norms)
        else:
            self.eigenvalues, pre_right_evecs = eig_output
            self.right_eigenvectors = self.eigenvectors = sp.pre_right @ pre_right_evecs

    def _solve_sparse_distributed_network(
        self,
        subproblem,
        N,
        target,
        *,
        rebuild_matrices=False,
        left=False,
        normalize_left=True,
        raise_on_mismatch=True,
        v0=None,
        **kw,
    ):
        """Solve a distributed Network EVP with fixed-shift DtN Arnoldi."""
        comm = self.dist.network_comm
        from .network import (
            _raise_collective_error,
            _validate_collective_signature,
        )

        # Parse every user-controlled option before entering matrix assembly.
        # A conversion or validation failure on only one rank must be shared
        # before its peers advance to the next collective operation.  Parsed
        # values are then fingerprinted so individually valid but inconsistent
        # rank-local calls fail collectively too.
        option_error = None
        try:
            if self.network_solver != "dtn":
                raise ValueError(
                    "Distributed Network EVPs require network_solver='dtn'."
                )

            left = bool(left)
            normalize_left = bool(normalize_left)
            raise_on_mismatch = bool(raise_on_mismatch)
            N = int(N)
            if N <= 0:
                raise ValueError("N must be positive.")

            target_array = np.asarray(target)
            if target_array.ndim != 0:
                raise ValueError("target must be a scalar.")
            target_dtype = target_array.dtype.str
            target_signature = complex(target_array.item())
            if not (
                np.isfinite(target_signature.real)
                and np.isfinite(target_signature.imag)
            ):
                raise ValueError("target must be finite.")
            if target_signature.imag == 0:
                target = target_signature.real
            else:
                target = target_signature

            rebuild_matrices = bool(rebuild_matrices)
            has_v0 = v0 is not None

            options = dict(kw)
            requested_ncv = options.pop("ncv", None)
            if requested_ncv is not None:
                requested_ncv = int(requested_ncv)
                if requested_ncv <= 0:
                    raise ValueError("ncv must be positive.")
                if requested_ncv <= N:
                    raise ValueError("ncv must be larger than N.")

            tol = float(options.pop("tol", 1e-8))
            if tol < 0 or not np.isfinite(tol):
                raise ValueError("tol must be finite and nonnegative.")

            seed = int(options.pop("seed", 0))
            breakdown_tolerance = options.pop(
                "breakdown_tolerance",
                None,
            )
            if breakdown_tolerance is not None:
                breakdown_tolerance = float(breakdown_tolerance)
                if (
                    breakdown_tolerance < 0
                    or not np.isfinite(breakdown_tolerance)
                ):
                    raise ValueError(
                        "breakdown_tolerance must be finite and nonnegative."
                    )

            raise_on_no_convergence = bool(
                options.pop("raise_on_no_convergence", True)
            )
            which = options.pop("which", "LM")
            if not isinstance(which, str):
                raise TypeError("which must be a string.")
            if which != "LM":
                raise NotImplementedError(
                    "Distributed Network shift-invert Arnoldi currently "
                    "supports only which='LM', selecting eigenvalues nearest "
                    "target."
                )
            if "maxiter" in options:
                raise TypeError(
                    "The unrestarted distributed Network Arnoldi backend uses "
                    "ncv as its iteration and memory limit; maxiter is not "
                    "supported."
                )
            if options:
                names = ", ".join(sorted(options))
                raise TypeError(
                    "Unsupported distributed Network Arnoldi option(s): "
                    + names
                )
        except Exception as caught:
            option_error = f"{type(caught).__name__}: {caught}"
        _raise_collective_error(
            comm,
            option_error,
            "Distributed Network EVP option parsing",
        )
        _validate_collective_signature(
            comm,
            (
                left,
                normalize_left,
                raise_on_mismatch,
                N,
                target_dtype,
                target_signature,
                rebuild_matrices,
                has_v0,
                requested_ncv,
                tol,
                seed,
                breakdown_tolerance,
                raise_on_no_convergence,
                which,
            ),
            "Distributed Network EVP option validation",
        )
        if left:
            raise NotImplementedError(
                "Left eigenvectors are not yet implemented for distributed "
                "Network EVPs."
            )
        if rebuild_matrices or not hasattr(subproblem, 'L_min'):
            self.build_matrices([subproblem], ['M', 'L'])

        # Dedalus writes the pencil as L x = lambda (-M) x.  Condensing the
        # fixed shifted matrix L + sigma M therefore applies the ordinary
        # shift-invert operator (L + sigma M)^-1 (-M), without constructing a
        # lambda-dependent DtN map.
        A = subproblem.L_min
        B = -subproblem.M_min
        shifted = A + target * subproblem.M_min
        factorization = self.build_subproblem_solver(subproblem, shifted)

        edge_columns = np.asarray(factorization.edge_columns, dtype=int)
        interface_columns = np.asarray(
            factorization.interface_columns,
            dtype=int,
        )
        interface_rows = np.asarray(
            factorization.interface_rows,
            dtype=int,
        )
        local_size = shifted.shape[1]
        local_row_size = shifted.shape[0]
        if self.dist.distributed_network:
            owned = np.ones(local_size, dtype=bool)
            row_owned = np.ones(local_row_size, dtype=bool)
            if factorization.resident:
                owned[interface_columns] = (
                    factorization.interface_column_owned
                )
                row_owned[interface_rows] = (
                    factorization.interface_row_owned
                )
                global_size = comm.allreduce(
                    int(np.count_nonzero(owned)),
                    op=MPI.SUM,
                )
            else:
                if comm.rank != factorization.root:
                    owned[interface_columns] = False
                    row_owned[interface_rows] = False
                global_size = comm.allreduce(
                    edge_columns.size,
                    op=MPI.SUM,
                )
                global_size += interface_columns.size
        else:
            # Compatibility storage replicates complete vectors.  Its DtN
            # factorization still distributes edge work, but global products
            # must count the authoritative root copy exactly once.
            owned = np.full(local_size, comm.rank == factorization.root)
            row_owned = np.full(
                local_row_size,
                comm.rank == factorization.root,
            )
            global_size = local_size
        if N >= global_size:
            raise ValueError(
                f"N={N} must be smaller than the global EVP size "
                f"{global_size}."
            )

        ncv = requested_ncv
        if ncv is None:
            ncv = min(global_size, max(4 * N + 16, 32))
        ncv = min(ncv, global_size)
        if ncv <= N:
            raise ValueError("ncv must be larger than N.")

        vector_dtype = np.result_type(shifted.dtype, B.dtype)
        initial_error = None
        try:
            if v0 is None:
                rng = np.random.default_rng(seed + comm.rank)
                initial = rng.standard_normal(local_size).astype(vector_dtype)
                if np.issubdtype(vector_dtype, np.complexfloating):
                    initial += 1j * rng.standard_normal(local_size)
            else:
                v0 = np.asarray(v0)
                expanded_size = subproblem.pre_right.shape[0]
                if v0.shape != (expanded_size,):
                    raise ValueError(
                        "v0 must match the rank-local expanded subproblem size "
                        f"{expanded_size}."
                    )
                initial = np.asarray(
                    subproblem.pre_right_pinv @ v0,
                    dtype=vector_dtype,
                )
        except Exception as caught:
            initial = None
            initial_error = f"{type(caught).__name__}: {caught}"
        _raise_collective_error(
            comm,
            initial_error,
            "Distributed Network EVP initial-vector construction",
        )

        if self.dist.distributed_network:
            # Every owned edge keeps a local start vector.  Replicated vertex
            # storage uses rank zero as before; resident storage refreshes
            # only its owner/ghost interface entries.
            interface_initial = np.ascontiguousarray(initial[interface_columns])
            if factorization.resident:
                factorization.interface_column_dofs.forward(
                    interface_initial
                )
            else:
                comm.Bcast(interface_initial, root=factorization.root)
            initial[interface_columns] = interface_initial
        else:
            initial = np.ascontiguousarray(initial)
            comm.Bcast(initial, root=factorization.root)

        from ..tools.collective_eigs import (
            collective_arnoldi,
            collective_norm,
        )

        def apply_collective(matrix, vector, stage):
            error = None
            try:
                result = self.apply_subproblem_matrix(
                    matrix,
                    vector,
                    subproblem=subproblem,
                )
            except Exception as caught:
                result = None
                error = f"{type(caught).__name__}: {caught}"
            _raise_collective_error(comm, error, stage)
            return result

        def shift_invert_matvec(vector):
            mass_vector = apply_collective(
                B,
                vector,
                "Distributed Network EVP mass-matrix application",
            )
            error = None
            try:
                result = factorization.solve(mass_vector)
            except Exception as caught:
                result = None
                error = f"{type(caught).__name__}: {caught}"
            _raise_collective_error(
                comm,
                error,
                "Distributed Network EVP shifted solve",
            )
            return result

        ritz_values, pre_vectors, arnoldi_estimates, arnoldi_dimension = (
            collective_arnoldi(
                shift_invert_matvec,
                initial,
                comm,
                owned,
                num_eigenvalues=N,
                ncv=ncv,
                breakdown_tolerance=breakdown_tolerance,
            )
        )
        real_dtype = np.empty((), dtype=vector_dtype).real.dtype
        zero_cutoff = 100 * np.finfo(real_dtype).eps
        finite_transform = np.abs(ritz_values) > zero_cutoff
        eigenvalues = np.full(
            ritz_values.shape,
            np.inf,
            dtype=np.result_type(ritz_values.dtype, target),
        )
        eigenvalues[finite_transform] = (
            target + 1 / ritz_values[finite_transform]
        )

        residuals = np.empty(eigenvalues.size, dtype=real_dtype)
        for index, (eigenvalue, vector) in enumerate(
            zip(eigenvalues, pre_vectors.T)
        ):
            if not np.isfinite(eigenvalue):
                residuals[index] = np.inf
                continue
            Av = apply_collective(
                A,
                vector,
                "Distributed Network EVP stiffness-matrix application",
            )
            Bv = apply_collective(
                B,
                vector,
                "Distributed Network EVP residual mass-matrix application",
            )
            residual = Av - eigenvalue * Bv
            # Arnoldi vectors live in column space, but A v, B v, and the
            # pencil residual live in equation-row space.  Matrix ordering
            # options can permute those independently, so residual norms must
            # use the factorization's interface-row ownership mask.
            numerator = collective_norm(comm, row_owned, residual)
            denominator = (
                collective_norm(comm, row_owned, Av)
                + abs(eigenvalue) * collective_norm(comm, row_owned, Bv)
            )
            residuals[index] = numerator / max(
                denominator,
                np.finfo(real_dtype).tiny,
            )

        order = np.argsort(np.abs(eigenvalues - target), kind="stable")
        self.eigenvalues = eigenvalues[order]
        pre_vectors = pre_vectors[:, order]
        self.eigenvalue_residuals = residuals[order]
        self.arnoldi_residual_estimates = arnoldi_estimates[order]
        self.arnoldi_dimension = arnoldi_dimension
        self.right_eigenvectors = self.eigenvectors = (
            subproblem.pre_right @ pre_vectors
        )
        self._collective_network_evp = True

        unconverged = self.eigenvalue_residuals > tol
        if np.any(unconverged):
            message = (
                "Distributed Network Arnoldi did not converge all requested "
                f"modes to tol={tol:.3e}; maximum relative pencil residual "
                f"is {self.eigenvalue_residuals.max():.3e}. Increase ncv or "
                "move the target away from a difficult shift."
            )
            if raise_on_no_convergence:
                raise RuntimeError(message)
            logger.warning(message)

    def set_state(self, index, subsystem=0):
        """
        Set state vector to the specified eigenmode.

        Parameters
        ----------
        index : int
            Index of desired eigenmode.
        subsystem : Subsystem object or int, optional
            Subsystem that will be set to the corresponding eigenmode.
            If an integer, the corresponding subsystem of the last specified
            eigenvalue_subproblem will be used. Default: 0.
        """
        # TODO: allow setting left modified eigenvectors?
        subproblem = self.eigenvalue_subproblem
        if isinstance(subsystem, int):
            subsystem = subproblem.subsystems[subsystem]
        # Check selection
        if subsystem not in subproblem.subsystems:
            raise ValueError("subsystem must be in eigenvalue_subproblem")
        state_vector = self.eigenvectors[:, index]
        eigenvalue = self.eigenvalues[index]
        if getattr(self, "_collective_network_evp", False):
            real_state = not np.issubdtype(
                self.problem.dtype,
                np.complexfloating,
            )
            if real_state and np.iscomplexobj(state_vector):
                comm = self.dist.network_comm
                max_imag = comm.allreduce(
                    np.max(np.abs(state_vector.imag), initial=0),
                    op=MPI.MAX,
                )
                scale = comm.allreduce(
                    np.max(np.abs(state_vector), initial=0),
                    op=MPI.MAX,
                )
                tolerance = (
                    1000
                    * np.finfo(self.problem.dtype).eps
                    * max(1, scale)
                )
                if (
                    max_imag > tolerance
                    or abs(np.imag(eigenvalue)) > tolerance
                ):
                    raise ValueError(
                        "This distributed Network eigenmode is complex but "
                        "the problem uses a real dtype; rebuild the "
                        "Distributor with dtype=np.complex128 before calling "
                        "set_state()."
                    )
                state_vector = state_vector.real
                eigenvalue = np.real(eigenvalue)
        # Set coefficients
        for var in self.state:
            var['c'] = 0
        subsystem.scatter(state_vector, self.state)
        if (
            self.dist.distributed_network
            and self.dist.network_vertex_storage == "owned_ghost"
        ):
            from .field import synchronize_network_vertex_inputs

            synchronize_network_vertex_inputs(self.dist, self.state)
        # Set eigenvalue
        self.problem.eigenvalue['g'] = eigenvalue


class LinearBoundaryValueSolver(SolverBase):
    """
    Linear boundary value problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    network_solver : {"auto", "dtn", None}, optional
        Network linear-solve backend.  ``"auto"`` uses the direct backend in
        serial and DtN condensation for parallel Network problems; ``"dtn"``
        requests DtN explicitly; ``None`` requests the direct backend.
    network_partition : optional
        Edge ownership for DtN in replicated compatibility mode.  For
        whole-edge distributed storage, bind ownership on the
        :class:`~dedalus.core.distributor.Distributor` and omit this option.
    **kw :
        Other options passed to :class:`SolverBase`.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution (after solve method is called).
    """

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'
    matrices = ['L']

    def __init__(
        self,
        problem,
        network_solver="auto",
        network_partition=None,
        **kw,
    ):
        logger.debug('Beginning LBVP instantiation')
        self.network_solver = _resolve_network_solver(problem, network_solver)
        self.network_partition = _resolve_network_partition_option(
            problem,
            network_partition,
        )
        self.matrix_assembly_root = _network_matrix_assembly_root(
            problem,
            self.network_solver,
        )
        if self.network_solver == "dtn":
            if kw.get("subproblem_solver_factory") is not None:
                raise ValueError(
                    "network_solver='dtn' cannot be combined with a custom "
                    "subproblem_solver_factory."
                )
            kw["subproblem_solver_factory"] = _network_subproblem_solver_factory
        super().__init__(problem, **kw)
        _validate_network_subproblems(self)
        self.subproblem_matsolvers = {}
        self.iteration = 0
        # Create RHS handler
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields
        logger.debug('Finished LBVP instantiation')

    def print_subproblem_ranks(self, subproblems=None):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            if not hasattr(sp, 'L_min'):
                continue
            L = sp.L_min.toarray()
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(L)}/{L.shape[0]}, cond: {np.linalg.cond(L):.1e}")

    def solve(self, subproblems=None, rebuild_matrices=False):
        """
        Solve BVP over selected subproblems.

        Parameters
        ----------
        subproblems : Subproblem object or list of Subproblem objects, optional
            Subproblems for which to solve the BVP (default: None (all)).
        rebuild_matrices : bool, optional
            Rebuild LHS matrices if coefficients have changed (default: False).
        """
        # Resolve subproblems
        if subproblems is None:
            subproblems = self.subproblems
        if isinstance(subproblems, subsystems.Subproblem):
            subproblems = [subproblems]
        # Build matrices and matsolvers if directed or not yet built
        if rebuild_matrices:
            sp_to_build = subproblems
        else:
            sp_to_build = [sp for sp in subproblems if sp not in self.subproblem_matsolvers]
        if sp_to_build:
            self.build_matrices(sp_to_build, ['L'])
            for sp in sp_to_build:
                factorization = self.build_subproblem_solver(sp, sp.L_min)
                self.subproblem_matsolvers[sp] = factorization
        # Compute RHS
        self.evaluator.evaluate_scheduled(iteration=self.iteration)
        # Ensure coeff space before subsystem gathers/scatters
        for field in self.F:
            field.change_layout('c')
        for field in self.state:
            field.preset_layout('c')
        # Solve system for each subproblem
        for sp in subproblems:
            # Gather RHS
            spF = sp.gather_outputs(self.F)
            # Solve
            spX = self.subproblem_matsolvers[sp].solve(spF)  # CREATES TEMPORARY
            # Scatter solution
            sp.scatter_inputs(spX, self.state)
        self.iteration += 1

    def evaluate_handlers(self, handlers=None):
        """Evaluate specified list of handlers (all by default)."""
        if handlers is None:
            handlers = self.evaluator.handlers
        self.evaluator.evaluate_handlers(handlers, iteration=self.iteration)


class NonlinearBoundaryValueSolver(SolverBase):
    """
    Nonlinear boundary value problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    network_solver : {"auto", "dtn", None}, optional
        Network linear-solve backend.  ``"auto"`` uses the direct backend in
        serial and DtN condensation for parallel Network problems; ``"dtn"``
        requests DtN explicitly; ``None`` requests the direct backend.
    network_partition : optional
        Edge ownership for DtN in replicated compatibility mode.  For
        whole-edge distributed storage, bind ownership on the
        :class:`~dedalus.core.distributor.Distributor` and omit this option.
    **kw :
        Other options passed to :class:`SolverBase`.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution.
    perturbations : list of Field objects
        Perturbations to problem variables from each Newton iteration.
    iteration : int
        Current iteration.
    """

    # Default to solver since every iteration sees a new matrix
    matsolver_default = 'MATRIX_SOLVER'
    matrices = ['dF']

    def __init__(
        self,
        problem,
        network_solver="auto",
        network_partition=None,
        **kw,
    ):
        logger.debug('Beginning NLBVP instantiation')
        self.network_solver = _resolve_network_solver(problem, network_solver)
        self.network_partition = _resolve_network_partition_option(
            problem,
            network_partition,
        )
        self.matrix_assembly_root = _network_matrix_assembly_root(
            problem,
            self.network_solver,
        )
        if self.network_solver == "dtn":
            if kw.get("subproblem_solver_factory") is not None:
                raise ValueError(
                    "network_solver='dtn' cannot be combined with a custom "
                    "subproblem_solver_factory."
                )
            kw["subproblem_solver_factory"] = _network_subproblem_solver_factory
        super().__init__(problem, **kw)
        _validate_network_subproblems(self)
        self.perturbations = problem.perturbations
        # Copy valid modes from variables to perturbations (may have been changed after problem instantiation)
        for pert, var in zip(problem.perturbations, problem.variables):
            pert.valid_modes[:] = var.valid_modes
        self.iteration = 0
        # Create RHS handler
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields
        logger.debug('Finished NLBVP instantiation')

    def print_subproblem_ranks(self, subproblems=None):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            if not hasattr(sp, 'dF_min'):
                continue
            dF = sp.dF_min.toarray()
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(dF)}/{dF.shape[0]}, cond: {np.linalg.cond(dF):.1e}")

    def newton_iteration(self, damping=1):
        """Update solution with a Newton iteration."""
        # Compute RHS
        self.evaluator.evaluate_scheduled(iteration=self.iteration)
        # Recompute Jacobian
        # TODO: split out linear part for faster recomputation?
        self.build_matrices(self.subproblems, ['dF'])
        # Ensure coeff space before subsystem gathers/scatters
        for field in self.F:
            field.change_layout('c')
        for field in self.perturbations:
            field.preset_layout('c')
        # Solve system for each subproblem, updating perturbations
        for sp in self.subproblems:
            # Gather RHS
            spF = sp.gather_outputs(self.F)
            # Solve
            sp_matsolver = self.build_subproblem_solver(sp, sp.dF_min)
            spX = sp_matsolver.solve(spF)  # CREATES TEMPORARY
            # Scatter solution to perturbations
            sp.scatter_inputs(spX, self.perturbations)
        # Update state
        for var, pert in zip(self.state, self.perturbations):
            var['c'] -= damping * pert['c']
        self.iteration += 1

    def evaluate_handlers(self, handlers=None):
        """Evaluate specified list of handlers (all by default)."""
        if handlers is None:
            handlers = self.evaluator.handlers
        self.evaluator.evaluate_handlers(handlers, iteration=self.iteration)


class InitialValueSolver(SolverBase):
    """
    Initial value problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    timestepper : Timestepper class or str
        Timestepper to use in evolving initial conditions.
    enforce_real_cadence : int, optional
        Iteration cadence for enforcing Hermitian symmetry on real variables (default: 100).
    warmup_iterations : int, optional
        Number of warmup iterations to disregard when computing runtime statistics (default: 10).
    profile : bool, optional
        Save accumulated profiles with cProfile (default: False).
    parallel_profile : bool, optional
        Save per-process and accumulated profiles with cProfile (default: False).
    network_solver : {"auto", "dtn", None}, optional
        Network linear-solve backend.  ``"auto"`` uses the direct backend in
        serial and DtN condensation for parallel Network problems; ``"dtn"``
        requests DtN explicitly; ``None`` requests the direct backend.
    network_partition : optional
        Edge ownership for DtN in replicated compatibility mode.  For
        whole-edge distributed storage, bind ownership on the
        :class:`~dedalus.core.distributor.Distributor` and omit this option.
    **kw :
        Other options passed to :class:`SolverBase`.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution.
    stop_sim_time : float
        Simulation stop time, in simulation units.
    stop_wall_time : float
        Wall stop time, in seconds from instantiation.
    stop_iteration : int
        Stop iteration.
    sim_time : float
        Current simulation time.
    iteration : int
        Current iteration.
    dt : float
        Last timestep.
    """

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'
    matrices = ['M', 'L']

    def build_matrices(self, subproblems=None, matrices=None):
        """Rebuild IVP operators and invalidate their timestep factors."""
        subproblems = tuple(self.subproblems if subproblems is None else subproblems)
        matrices = tuple(self.matrices if matrices is None else matrices)
        super().build_matrices(subproblems, matrices)
        if self.store_expanded_matrices and set(matrices) != set(self.matrices):
            # A partial rebuild must recombine the fresh operator with the
            # unchanged one using a common sparse pattern.
            for sp in subproblems:
                sp.expand_matrices(self.matrices)
        timestepper = getattr(self, "timestepper", None)
        if timestepper is not None:
            # Rebuild even if dt is unchanged. This also clears old factors
            # through the timestepper's normal collective construction path.
            timestepper._LHS_params = None


    def __init__(
        self,
        problem,
        timestepper,
        enforce_real_cadence=100,
        warmup_iterations=10,
        profile=PROFILE_DEFAULT,
        parallel_profile=PARALLEL_PROFILE_DEFAULT,
        network_solver="auto",
        network_partition=None,
        **kw,
    ):
        logger.debug('Beginning IVP instantiation')
        self.network_solver = _resolve_network_solver(problem, network_solver)
        self.network_partition = _resolve_network_partition_option(
            problem,
            network_partition,
        )
        self.matrix_assembly_root = _network_matrix_assembly_root(
            problem,
            self.network_solver,
        )
        if self.network_solver == "dtn":
            if kw.get("subproblem_solver_factory") is not None:
                raise ValueError(
                    "network_solver='dtn' cannot be combined with a custom "
                    "subproblem_solver_factory."
                )
            kw["subproblem_solver_factory"] = _network_subproblem_solver_factory
        # Setup timing and profiling
        self.dist = problem.dist
        self._bcast_array = np.zeros(1, dtype=float)
        self.init_time = self.world_time
        if profile or parallel_profile:
            parallel_mkdir(PROFILE_DIRECTORY, comm=self.dist.comm)
            self.profile = True
            self.parallel_profile = parallel_profile
            self.setup_profiler = cProfile.Profile()
            self.warmup_profiler = cProfile.Profile()
            self.run_profiler = cProfile.Profile()
            self.setup_profiler.enable()
        else:
            self.profile = False
        # Build subsystems and subproblems
        super().__init__(problem, **kw)
        _validate_network_subproblems(self)
        # Build LHS matrices
        self.build_matrices(self.subproblems, ['M', 'L'])
        # Compute total modes
        if (
            self.dist.distributed_network
            and self.dist.network_vertex_storage == "owned_ghost"
        ):
            from .network import _distributed_network_matrix_layout

            local_modes = 0
            for subproblem in self.subproblems:
                layout = _distributed_network_matrix_layout(self, subproblem)
                local_modes += layout.edge_columns.size
                local_modes += np.count_nonzero(
                    layout.interface_column_owned
                )
        else:
            local_modes = sum(prod(sp.shape) for sp in self.subproblems)
        self.total_modes = self.dist.layout_comm.allreduce(local_modes, op=MPI.SUM)
        # Create RHS handler
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields
        # Initialize timestepper
        if isinstance(timestepper, str):
            timestepper = timesteppers.schemes[timestepper]
        self.timestepper = timestepper(self)
        # Attributes
        self.sim_time = self.initial_sim_time = problem.time.allreduce_data_max(layout='g')
        self.iteration = self.initial_iteration = 0
        self.warmup_iterations = warmup_iterations
        if np.isrealobj(self.dtype.type()):
            self.enforce_real_cadence = enforce_real_cadence
        else:
            self.enforce_real_cadence = None
        # Default integration parameters
        self.stop_sim_time = np.inf
        self.stop_wall_time = np.inf
        self.stop_iteration = np.inf
        logger.debug('Finished IVP instantiation')

    @property
    def sim_time(self):
        return self._sim_time

    @sim_time.setter
    def sim_time(self, t):
        self._sim_time = t
        self.problem.time['g'] = t

    @property
    def world_time(self):
        if self.dist.comm.size == 1:
            return time.time()
        else:
            # Broadcast time from root process
            self._bcast_array[0] = time.time()
            self.dist.comm.Bcast(self._bcast_array, root=0)
            return self._bcast_array[0]

    @property
    def wall_time(self):
        """Seconds ellapsed since instantiation."""
        return self.world_time - self.init_time

    @property
    def proceed(self):
        """Check that current time and iteration pass stop conditions."""
        if self.sim_time >= self.stop_sim_time:
            logger.info('Simulation stop time reached.')
            return False
        elif self.wall_time >= self.stop_wall_time:
            logger.info('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            logger.info('Stop iteration reached.')
            return False
        else:
            return True

    def load_state(self, path, index=-1, allow_missing=False):
        """
        Load state from a grid- or coefficient-space HDF5 task.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to Dedalus HDF5 savefile
        index : int, optional
            Local write index (within file) to load (default: -1)
        allow_missing : bool, optional
            Do not raise an error if state variables are missing from the savefile (default: False).

        Returns
        -------
        write : int
            Global write number of loaded write
        dt : float
            Timestep at loaded write
        """
        path = pathlib.Path(path)
        logger.info("Loading solver state from: {}".format(path))
        with h5py.File(str(path), mode='r') as file:
            # Load solver attributes
            write = file['scales']['write_number'][index]
            dt = file['scales']['timestep'][index]
            self.iteration = self.initial_iteration = file['scales']['iteration'][index]
            self.sim_time = self.initial_sim_time = file['scales']['sim_time'][index]
            # Log restart info
            logger.info("Loading iteration: {}".format(self.iteration))
            logger.info("Loading write: {}".format(write))
            logger.info("Loading sim time: {}".format(self.sim_time))
            logger.info("Loading timestep: {}".format(dt))
            # Load fields
            for field in self.state:
                if field.name in file['tasks']:
                    field.load_from_hdf5(file, index)
                elif allow_missing:
                    logger.warning(f"Field '{field.name}' not found in savefile.")
                else:
                    raise IOError(f"Field '{field.name}' not found in savefile. Set allow_missing=True to ignore this error.")
        return write, dt

    def enforce_hermitian_symmetry(self, fields):
        """Transform fields to grid and back."""
        # TODO: maybe this should be on scales=1?
        for f in fields:
            f.change_scales(f.domain.dealias)
        self.evaluator.require_grid_space(fields)
        self.evaluator.require_coeff_space(fields)

    def step(self, dt):
        """Advance system by one iteration/timestep."""
        # Assert finite timestep
        if not np.isfinite(dt):
            raise ValueError("Invalid timestep")
        # Record times
        wall_time = self.wall_time
        if self.iteration == self.initial_iteration:
            self.start_time_end = wall_time
            if self.profile:
                self.dump_profiles(self.setup_profiler, "setup")
                self.warmup_profiler.enable()
            self.warmup_time_start = self.wall_time
        if self.iteration == self.initial_iteration + self.warmup_iterations:
            self.warmup_time_end = self.wall_time
            if self.profile:
                self.dump_profiles(self.warmup_profiler, "warmup")
                self.run_profiler.enable()
            self.run_time_start = self.wall_time
        # Advance using timestepper
        self.timestepper.step(dt, wall_time)
        # Enforce Hermitian symmetry for real variables
        if self.enforce_real_cadence:
            # Enforce for as many iterations as timestepper uses internally
            if self.iteration % self.enforce_real_cadence < self.timestepper.steps:
                self.enforce_hermitian_symmetry(self.state)
        # Update iteration
        self.iteration += 1
        self.dt = dt

    def evolve(self, timestep_function, log_cadence=100):
        """Advance system until stopping criterion is reached."""
        # Check for a stopping criterion
        if np.isinf(self.stop_sim_time):
            if np.isinf(self.stop_wall_time):
                if np.isinf(self.stop_iteration):
                    raise ValueError("No stopping criterion specified.")
        # Evolve
        try:
            logger.info("Starting main loop")
            while self.proceed:
                timestep = timestep_function()
                self.step(timestep)
                if (self .iteration-1) % log_cadence == 0:
                    logger.info(f"Iteration={self.iteration}, Time={self.sim_time:e}, Step={timestep:e}")
        except:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.log_stats()

    def print_subproblem_ranks(self, subproblems=None, dt=1):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            M = sp.M_min
            L = sp.L_min
            A = (M + dt*L).toarray()
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}, cond: {np.linalg.cond(A):.1e}")

    def evaluate_handlers_now(self, dt, handlers=None):
        logger.warning("Deprecation warning: evaluate_handlers_now -> evaluate_handlers")
        self.evaluate_handlers(handlers=handlers, dt=dt)

    def evaluate_handlers(self, handlers=None, dt=0):
        """Evaluate specified list of handlers (all by default)."""
        if handlers is None:
            handlers = self.evaluator.handlers
        self.evaluator.evaluate_handlers(handlers, iteration=self.iteration, wall_time=self.wall_time, sim_time=self.sim_time, timestep=dt)

    def log_stats(self, format=".4g"):
        """Log timing statistics with specified string formatting (optional)."""
        self.run_time_end = self.wall_time
        start_time = self.start_time_end
        logger.info(f"Final iteration: {self.iteration}")
        logger.info(f"Final sim time: {self.sim_time}")
        logger.info(f"Setup time (init - iter 0): {start_time:{format}} sec")
        if self.profile:
            self.dump_profiles(self.run_profiler, "runtime")
        if self.iteration >= self.initial_iteration + self.warmup_iterations:
            warmup_time = self.warmup_time_end - self.warmup_time_start
            run_time = self.run_time_end - self.run_time_start
            profile_output_time = (self.warmup_time_start-self.start_time_end) + (self.run_time_start-self.warmup_time_end)
            cpus = self.dist.comm.size
            modes = self.total_modes
            stages = (self.iteration - self.warmup_iterations - self.initial_iteration) * self.timestepper.stages
            logger.info(f"Warmup time (iter 0-{self.warmup_iterations}): {warmup_time:{format}} sec")
            if self.profile:
                logger.info(f"Profile output time: {profile_output_time:{format}} sec")
            logger.info(f"Run time (iter {self.warmup_iterations}-end): {run_time:{format}} sec")
            logger.info(f"CPU time (iter {self.warmup_iterations}-end): {run_time*cpus/3600:{format}} cpu-hr")
            logger.info(f"Speed: {(modes*stages/cpus/run_time):{format}} mode-stages/cpu-sec")
        else:
            logger.info(f"Timings unavailable because warmup did not complete.")

    def dump_profiles(self, profiler, name):
        "Save profiling data to disk."
        comm = self.dist.comm
        # Disable and create stats on each process
        profiler.create_stats()
        p = pstats.Stats(profiler)
        p.strip_dirs()
        # Gather using wrapper class to avoid pickling issues
        profiles = comm.gather(ProfileWrapper(p.stats), root=0)
        # Sum stats on root process
        if comm.rank == 0:
            if self.parallel_profile:
                stats = {'primcalls': defaultdict(list),
                         'totcalls': defaultdict(list),
                         'tottime': defaultdict(list),
                         'cumtime': defaultdict(list)}
                for profile in profiles:
                    for func, (primcalls, totcalls, tottime, cumtime, callers) in profile.stats.items():
                        stats['primcalls'][func].append(primcalls)
                        stats['totcalls'][func].append(totcalls)
                        stats['tottime'][func].append(tottime)
                        stats['cumtime'][func].append(cumtime)
                pickle.dump(stats, open(PROFILE_DIRECTORY / f"{name}_parallel.pickle", 'wb'))
            # Creation of joint_stats destroys profiles, so do this second
            joint_stats = pstats.Stats(*profiles)
            joint_stats.dump_stats(PROFILE_DIRECTORY / f"{name}.prof")
        comm.Barrier()
