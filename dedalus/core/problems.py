"""Classes for representing systems of equations."""

import hashlib
import numpy as np
from collections import ChainMap

from .field import Operand, Field, LockedField
from . import arithmetic
from . import operators
from . import solvers
from ..tools import parsing
from ..tools.general import unify_attributes
from ..tools.exceptions import UnsupportedEquationError

import logging
logger = logging.getLogger(__name__.split('.')[-1])

# Public interface
__all__ = ['EigenvalueProblem',
           'LinearBoundaryValueProblem',
           'NonlinearBoundaryValueProblem',
           'InitialValueProblem',
           'EVP',
           'LBVP',
           'NLBVP',
           'IVP']


# Build basic parsing namespace
parseables = {}
parseables.update({name: getattr(operators, name) for name in operators.__all__})
parseables.update({name: getattr(arithmetic, name) for name in arithmetic.__all__})


def _network_hidden_vertex_name(edge_field):
    """Build a restart-stable reserved name from a semantic field name."""
    name = edge_field.name
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Automatically managed Network interface fields require a "
            "nonempty edge-field name. Name the edge field or bind an "
            "explicit vertex field with network_interfaces=[(u, U)]."
        )
    label = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in name
    ).strip("_")
    if not label:
        label = "field"
    digest = hashlib.blake2b(
        name.encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"__network_vertex_{label[:32]}_{digest}"


class ProblemBase:
    """Base class for all problem types."""

    def __init__(self, variables, namespace=None, network_interfaces=None):
        user_variables = tuple(variables)
        variables = list(user_variables)
        interface_bindings = []
        hidden_interface_fields = []
        automatic_interface_names = set()

        if network_interfaces is None:
            interface_specs = ()
        elif isinstance(network_interfaces, Field):
            interface_specs = (network_interfaces,)
        else:
            try:
                interface_specs = tuple(network_interfaces)
            except TypeError as error:
                raise TypeError(
                    "network_interfaces must be a Field or an iterable of "
                    "Fields and (edge_field, vertex_field) pairs."
                ) from error

        for interface_spec in interface_specs:
            if isinstance(interface_spec, Field):
                edge_field = interface_spec
                vertex_field = None
            elif (
                isinstance(interface_spec, (tuple, list))
                and len(interface_spec) == 2
            ):
                edge_field, vertex_field = interface_spec
            else:
                raise TypeError(
                    "Each network interface must be an edge Field or an "
                    "(edge_field, vertex_field) pair."
                )

            if not isinstance(edge_field, Field):
                raise TypeError("Network interface edge variables must be Fields.")
            if not any(edge_field is variable for variable in user_variables):
                raise ValueError(
                    "Network interface edge fields must be included in the "
                    "problem variable list."
                )
            if any(
                edge_field is bound_edge
                for bound_edge, _, _ in interface_bindings
            ):
                raise ValueError(
                    "Each Network edge field may be registered as an "
                    "interface only once."
                )
            if edge_field.tensorsig:
                raise UnsupportedEquationError(
                    "Network interface fields must be scalar."
                )

            edge_bases = [
                basis
                for basis in edge_field.domain.bases
                if getattr(basis, "_network_role", None) == "edges"
            ]
            if len(edge_bases) != 1:
                raise ValueError(
                    "Network interface fields must use the edge basis of "
                    "exactly one NetworkBasis."
                )
            edge_basis = edge_bases[0]
            network_basis = getattr(edge_basis, "_network_basis", None)
            if network_basis is None or edge_basis is not network_basis.edges:
                raise ValueError(
                    "Network interface fields must use a canonical "
                    "NetworkBasis edge basis."
                )
            if (
                edge_field.domain.get_basis(
                    network_basis.network.position_coord
                )
                is None
            ):
                raise ValueError(
                    "Network interface fields must vary along the Network "
                    "interval coordinate."
                )

            heterogeneous_bindings = [
                (bound_vertex, bound_basis)
                for _, bound_vertex, bound_basis in interface_bindings
                if (
                    bound_basis is not network_basis
                    and bound_basis.network is network_basis.network
                )
            ]
            if heterogeneous_bindings and (
                vertex_field is None
                or any(
                    any(
                        bound_vertex is hidden
                        for hidden in hidden_interface_fields
                    )
                    for bound_vertex, _ in heterogeneous_bindings
                )
            ):
                raise ValueError(
                    "Automatic Network interfaces cannot infer one physical "
                    "vertex field across heterogeneous NetworkBasis buckets. "
                    "Create a shared U on basis_set.vertices and register "
                    "each bucket explicitly, for example "
                    "network_interfaces=[(u_coarse, U), (u_fine, U)]."
                )

            if vertex_field is None:
                if edge_field.name in automatic_interface_names:
                    raise ValueError(
                        "Automatically managed Network interface edge fields "
                        "must have unique names."
                    )
                automatic_interface_names.add(edge_field.name)
                hidden_name = _network_hidden_vertex_name(edge_field)
                if any(
                    isinstance(variable.name, str)
                    and variable.name.startswith("__network_vertex_")
                    for variable in user_variables
                ):
                    raise ValueError(
                        "Problem variable names beginning with "
                        "'__network_vertex_' are reserved for automatic "
                        "Network interface fields."
                    )
                if any(
                    variable.name == hidden_name
                    for variable in hidden_interface_fields
                ):
                    raise ValueError(
                        "Automatically managed Network interface edge-field "
                        "names produced the same reserved vertex-field name."
                    )
                vertex_field = Field(
                    dist=edge_field.dist,
                    bases=network_basis.vertices,
                    name=hidden_name,
                    dtype=edge_field.dtype,
                )
                variables.append(vertex_field)
                hidden_interface_fields.append(vertex_field)
            else:
                if not isinstance(vertex_field, Field):
                    raise TypeError(
                        "Explicit Network interface variables must be Fields."
                    )
                if not any(
                    vertex_field is variable for variable in user_variables
                ):
                    raise ValueError(
                        "Explicit Network interface fields must be included "
                        "in the problem variable list."
                    )
                if vertex_field.dist is not edge_field.dist:
                    raise ValueError(
                        "A Network edge field and its vertex interface field "
                        "must use the same Distributor."
                    )
                if vertex_field.tensorsig:
                    raise UnsupportedEquationError(
                        "Network interface fields must be scalar."
                    )
                if (
                    vertex_field.domain.bases
                    != (network_basis.vertices,)
                ):
                    raise ValueError(
                        "Explicit Network interface fields must use the "
                        "matching NetworkBasis vertex basis."
                    )
            interface_bindings.append(
                (edge_field, vertex_field, network_basis)
            )

        self.network_interfaces = tuple(
            (edge_field, vertex_field)
            for edge_field, vertex_field, _ in interface_bindings
        )
        self._network_interface_bindings = tuple(interface_bindings)
        self._hidden_network_interface_fields = tuple(hidden_interface_fields)
        self.variables = variables
        self.LHS_variables = variables
        self.dist = unify_attributes(variables, 'dist')
        if getattr(self.dist, "network", None) is not None:
            tensor_variables = [
                variable
                for variable in variables
                if variable.tensorsig
            ]
            if tensor_variables:
                names = ", ".join(
                    str(variable.name) if variable.name else "<unnamed>"
                    for variable in tensor_variables
                )
                raise UnsupportedEquationError(
                    "Tensor-valued Network PDE variables are not supported "
                    f"yet: {names}. Network currently describes the (edge, s) "
                    "storage axes, not a physical tensor coordinate system. "
                    "Use scalar Fields, one per physical or system component. "
                    "VectorField and TensorField remain available for data "
                    "storage and I/O."
                )
        self.equations = self.eqs = []
        # Build namespace via chainmap to react to upstream changes
        # Priority: local namespace, external namespace, parseables
        self.local_namespace = {}
        for var in variables:
            if var.name:
                self.local_namespace[var.name] = var
        if interface_bindings:
            for helper_name in ("continuity", "vertex_value"):
                if any(variable.name == helper_name for variable in variables):
                    raise ValueError(
                        f"Problem variable name {helper_name!r} is reserved "
                        "when using network_interfaces."
                    )
            self.local_namespace["continuity"] = self.continuity
            self.local_namespace["vertex_value"] = self.vertex_value
        if namespace is None:
            self.namespace = ChainMap(self.local_namespace, parseables)
        else:
            self.namespace = ChainMap(self.local_namespace, namespace, parseables)

    def _network_interface_binding(self, edge_field):
        """Return the registered vertex field and basis for an edge field."""
        if not isinstance(edge_field, Field):
            raise TypeError(
                "Network interface helpers require a registered edge Field."
            )
        for bound_edge, vertex_field, network_basis in (
            self._network_interface_bindings
        ):
            if edge_field is bound_edge:
                return vertex_field, network_basis
        name = edge_field.name or "<unnamed>"
        raise ValueError(
            f"Field {name!r} is not registered in network_interfaces."
        )

    def vertex_field(self, edge_field):
        """Return the vertex interface field registered for ``edge_field``."""
        vertex_field, _ = self._network_interface_binding(edge_field)
        return vertex_field

    def vertex_value(self, edge_field):
        """Return the common vertex-value unknown for ``edge_field``."""
        return self.vertex_field(edge_field)

    def continuity(self, edge_field):
        """Return the port equation enforcing continuity of ``edge_field``."""
        vertex_field, network_basis = self._network_interface_binding(edge_field)
        return (
            network_basis.dtrace(edge_field)
            - network_basis.restriction @ vertex_field
        )

    @property
    def matrix_dependence(self):
        return np.logical_or.reduce([eqn['matrix_dependence'] for eqn in self.equations])

    @property
    def matrix_coupling(self):
        return np.logical_or.reduce([eqn['matrix_coupling'] for eqn in self.equations])

    @property
    def dtype(self):
        return np.result_type(*[eqn['dtype'] for eqn in self.equations])

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug(f"Adding equation {len(self.eqs)}")
        # Split equation into LHS and RHS expressions
        if isinstance(equation, str):
            # Parse string-valued equations
            namespace = dict(self.namespace)
            LHS_str, RHS_str = parsing.split_equation(equation)
            LHS = eval(LHS_str, namespace)
            RHS = eval(RHS_str, namespace)
        else:
            # Split operator tuples
            LHS, RHS = equation
        logger.debug(f"  LHS: {LHS}")
        logger.debug(f"  RHS: {RHS}")
        logger.debug(f"  condition: {condition}")
        # Build basic equation dictionary
        # Note: domain determined after NCC reinitialization
        expr = LHS - RHS
        eqn = {'eqn': expr,
               'LHS': LHS,
               'RHS': RHS,
               'condition': condition,
               'tensorsig': expr.tensorsig,
               'dtype': expr.dtype,
               'valid_modes': expr.valid_modes.copy()}
        self._check_equation_conditions(eqn)
        self._build_matrix_expressions(eqn)
        self.equations.append(eqn)
        return eqn

    def build_solver(self, *args, **kw):
        """Build corresponding solver class."""
        return self.solver_class(self, *args, **kw)

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        raise NotImplementedError("Subclassses must implement.")

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        raise NotImplementedError("Subclassses must implement.")

    @staticmethod
    def _check_domain_containment(subexpr, supexpr, subname, supname):
        """Require one domain contain another."""
        if np.any(np.logical_and(subexpr.domain.nonconstant, supexpr.domain.constant)):
            raise UnsupportedEquationError("{} domain cannot be larger than {} domain.".format(subname, supname))


class LinearBoundaryValueProblem(ProblemBase):
    """
    Linear boundary value problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.
    network_interfaces : iterable, optional
        Network edge fields whose common vertex-value fields should be managed
        by the problem.  Entries may also be ``(edge_field, vertex_field)``
        pairs binding an existing explicit vertex field.

    Notes
    -----
    This class supports linear boundary value problems of the form:
        L.X = F
    The LHS terms must be linear in the problem variables, and the RHS can be
    inhomogeneous but must be independent of the problem variables.

    Solution procedure:
        - Form L
        - Evaluate F
        - Solve X from L.X = F
    """

    solver_class = solvers.LinearBoundaryValueSolver

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # Cast LHS and RHS to operands
        LHS = Operand.cast(eqn['LHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        RHS = Operand.cast(eqn['RHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        # Check conditions
        LHS.require_linearity(*self.variables, allow_affine=False,
            self_name='LBVP LHS', vars_name='problem variables', error=UnsupportedEquationError)
        RHS.require_independent(*self.variables,
            self_name='LBVP RHS', vars_name='problem variables', error=UnsupportedEquationError)
        self._check_domain_containment(RHS, LHS, 'RHS', 'LHS')

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        dist = self.dist
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        # Extract matrix expressions
        L = eqn['LHS']
        F = eqn['RHS']
        # Reinitialize and prep NCCs
        L = L.reinitialize(ncc=True, ncc_vars=vars)
        L.prep_nccs(vars=vars)
        # Convert to same domain
        domain = (L - F).domain
        L = operators.convert(L, domain.bases)
        if F:
            # Cast to match LHS
            F = Operand.cast(F, dist, tensorsig=tensorsig, dtype=dtype)
            F = operators.convert(F, domain.bases)
        else:
            # Allocate zero field
            F = Field(dist=dist, bases=domain.bases, tensorsig=tensorsig, dtype=dtype)
            F['c'] = 0
        # Save expressions and metadata
        eqn['L'] = L
        eqn['F'] = F
        eqn['domain'] = domain
        eqn['matrix_dependence'] = L.matrix_dependence(*vars)
        eqn['matrix_coupling'] = L.matrix_coupling(*vars)
        # Debug logging
        logger.debug(f"  L: {L}")
        logger.debug(f"  F: {F}")


class NonlinearBoundaryValueProblem(ProblemBase):
    """
    Nonlinear boundary value problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.
    network_interfaces : iterable, optional
        Network edge fields whose common vertex-value fields should be managed
        by the problem.  Entries may also be ``(edge_field, vertex_field)``
        pairs binding an existing explicit vertex field.

    Notes
    -----
    This class supports nonlinear boundary value problems of the form:
        G(X) = H(X)
    which are recombined to form the root-finding problem:
        F(X) = G(X) - H(X) = 0

    The problem is reduced into a linear BVP for an update to the solution
    using the Newton-Kantorovich method and the symbolically-computed Frechet
    differential of the equation:
        F(X[n+1]) = 0
        F(X[n] + dX) = 0
        F(X[n]) + dF(X[n]).dX = 0
        dF(X[n]).dX = - F(X[n])

    Iteration procedure:
        - Form dF(X[n])
        - Evaluate F(X[n])
        - Solve dF(X[n]).dX = -F(X[n])
        - Update X[n+1] = X[n] + dX
    """

    solver_class = solvers.NonlinearBoundaryValueSolver

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Build perturbation variables
        self.perturbations = [var.copy() for var in self.variables]
        for pert, var in zip(self.perturbations, self.variables):
            pert.preset_scales(1)
            pert['c'] = 0
            if var.name:
                pert.name = 'δ'+var.name
        self.LHS_variables = self.perturbations

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # No conditions
        pass

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        perts = self.perturbations
        # Extract matrix expressions
        F = eqn['LHS'] - eqn['RHS']
        dF = F.frechet_differential(vars, perts)
        # Remove any field locks
        dF = dF.replace(operators.Lock, lambda x: x)
        for field in dF.atoms(LockedField):
            dF = dF.replace(field, field.unlock())
        # Reinitialize and prep NCCs
        dF = dF.reinitialize(ncc=True, ncc_vars=perts)
        dF.prep_nccs(vars=perts)
        # Convert to same domain
        domain = (dF + F).domain
        F = operators.convert(F, domain.bases)
        dF = operators.convert(dF, domain.bases)
        # Save expressions and metadata
        eqn['F'] = F
        eqn['dF'] = dF
        eqn['domain'] = domain
        eqn['matrix_dependence'] = dF.matrix_dependence(*perts)
        eqn['matrix_coupling'] = dF.matrix_coupling(*perts)
        # Debug logging
        logger.debug(f"  F: {F}")
        logger.debug(f"  dF: {dF}")


class InitialValueProblem(ProblemBase):
    """
    Initial value problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    time : str or Field object, optional
        Name (if str) or field for time variable (default: 't').
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.
    network_interfaces : iterable, optional
        Network edge fields whose common vertex-value fields should be managed
        by the problem.  Entries may also be ``(edge_field, vertex_field)``
        pairs binding an existing explicit vertex field.

    Notes
    -----
    This class supports non-linear initial value problems of the form:
        M.dt(X) + L.X = F(X, t)
    The LHS terms must be linear in the problem variables, time independent,
    and first-order in time derivatives. The RHS terms must not contain any
    explicit time derivatives.
    """

    solver_class = solvers.InitialValueSolver

    def __init__(self, variables, time='t', **kw):
        super().__init__(variables, **kw)
        if isinstance(time, str):
            self.time = Field(dist=self.dist, name=time, dtype=np.float64)
        elif isinstance(time, Field):
            if any(time.domain.nonconstant):
                raise ValueError("Time field cannot have any bases.")
            self.time = time
        else:
            raise ValueError("Time must be str or Field object.")

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # Cast LHS and RHS to operands
        LHS = Operand.cast(eqn['LHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        RHS = Operand.cast(eqn['RHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        # Check conditions
        LHS.require_linearity(*self.variables, allow_affine=False,
            self_name='IVP LHS', vars_name='problem variables', error=UnsupportedEquationError)
        LHS.require_independent(self.time,
            self_name='IVP LHS', vars_name='time', error=UnsupportedEquationError)
        LHS.require_first_order(operators.TimeDerivative,
            self_name='IVP LHS', ops_name='time derivatives', error=UnsupportedEquationError)
        RHS.require_independent(operators.TimeDerivative,
            self_name='IVP RHS', vars_name='time derivatives', error=UnsupportedEquationError)
        self._check_domain_containment(RHS, LHS, 'RHS', 'LHS')

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        dist = self.dist
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        # Extract matrix expressions
        M, L = eqn['LHS'].split(operators.TimeDerivative)
        F = eqn['RHS']
        # Drop time derivatives
        if M:
            M = M.replace(operators.TimeDerivative, lambda x: x)
        # Reinitialize and prep NCCs
        if M:
            M = M.reinitialize(ncc=True, ncc_vars=vars)
            M.prep_nccs(vars=vars)
        if L:
            L = L.reinitialize(ncc=True, ncc_vars=vars)
            L.prep_nccs(vars=vars)
        # Convert to same domain
        domain = (M + L - F).domain
        if M:
            M = operators.convert(M, domain.bases)
        if L:
            L = operators.convert(L, domain.bases)
        if F:
            # Cast to match LHS
            F = Operand.cast(F, dist, tensorsig=tensorsig, dtype=dtype)
            F = operators.convert(F, domain.bases)
        else:
            # Allocate zero field
            F = Field(dist=dist, bases=domain.bases, tensorsig=tensorsig, dtype=dtype)
            F['c'] = 0
        # Save expressions and metadata
        eqn['M'] = M
        eqn['L'] = L
        eqn['F'] = F
        eqn['domain'] = domain
        eqn['matrix_dependence'] = (M + L).matrix_dependence(*vars)
        eqn['matrix_coupling'] = (M + L).matrix_coupling(*vars)
        # Debug logging
        logger.debug(f"  M: {M}")
        logger.debug(f"  L: {L}")
        logger.debug(f"  F: {F}")

    def build_EVP(self, eigenvalue=None, backgrounds=None, perturbations=None, **kw):
        """
        Create an eigenvalue problem from an initial value problem.

        Parameters
        ----------
        eigenvalue : Field, optional
            Eigenvalue field.
        backgrounds : list of Fields, optional
            Background fields for linearizing the RHS. Default: the IVP variables.
        perturbations : list of Fields, optional
            Perturbation fields for the EVP. Default: copies of IVP variables.
            When the IVP manages hidden Network vertex fields, provide only
            the exposed perturbation fields; the hidden vertex perturbations
            and their interface bindings are recreated automatically.

        Notes
        -----
        This method converts time-independent IVP equations of the form
            M.dt(X) + L.X = F(X)
        to EVP equations as
            λ*M.X1 + L.X1 - F'(X0).X1 = 0.
        If backgrounds (X0) are not specified, the IVP variables (X) are used.
        """
        # Create eigenvalue problem for perturbations.  Problem-managed vertex
        # fields are implementation details of a hidden Network interface, so
        # recreate them through the EVP constructor instead of turning their
        # copies into unrelated explicit variables.  This preserves the
        # interface bindings, helper namespace, and hidden-field metadata.
        variables = tuple(self.variables)
        hidden_variable_ids = {
            id(variable)
            for variable in self._hidden_network_interface_fields
        }
        exposed_variables = tuple(
            variable
            for variable in variables
            if id(variable) not in hidden_variable_ids
        )
        if eigenvalue is None:
            eigenvalue = self.dist.Field(name='λ')
        if perturbations is None:
            exposed_perturbations = [
                var.copy()
                for var in exposed_variables
            ]
            for pert, var in zip(exposed_perturbations, exposed_variables):
                if var.name:
                    pert.name = 'δ'+var.name
        else:
            exposed_perturbations = list(perturbations)
            if len(exposed_perturbations) != len(exposed_variables):
                if hidden_variable_ids:
                    detail = (
                        "; omit problem-managed hidden vertex fields because "
                        "build_EVP recreates them automatically"
                    )
                else:
                    detail = ""
                raise ValueError(
                    "perturbations must provide one Field for each exposed "
                    f"IVP variable ({len(exposed_variables)} expected{detail})."
                )

        perturbation_by_variable_id = {
            id(var): pert
            for var, pert in zip(exposed_variables, exposed_perturbations)
        }
        interface_specs = []
        for edge_field, vertex_field in self.network_interfaces:
            edge_perturbation = perturbation_by_variable_id[id(edge_field)]
            if id(vertex_field) in hidden_variable_ids:
                interface_specs.append(edge_perturbation)
            else:
                interface_specs.append(
                    (
                        edge_perturbation,
                        perturbation_by_variable_id[id(vertex_field)],
                    )
                )

        if self.network_interfaces:
            if "network_interfaces" in kw:
                raise ValueError(
                    "build_EVP preserves the IVP network_interfaces "
                    "automatically; do not pass a replacement binding."
                )
            kw["network_interfaces"] = interface_specs
        EVP = EigenvalueProblem(exposed_perturbations, eigenvalue, **kw)

        for edge_field, vertex_field in self.network_interfaces:
            if id(vertex_field) in hidden_variable_ids:
                edge_perturbation = perturbation_by_variable_id[id(edge_field)]
                perturbation_by_variable_id[id(vertex_field)] = (
                    EVP.vertex_field(edge_perturbation)
                )
        perturbations = [
            perturbation_by_variable_id[id(variable)]
            for variable in variables
        ]
        for pert, var in zip(perturbations, variables):
            pert.valid_modes[:] = var.valid_modes

        if backgrounds is None:
            backgrounds = variables
        else:
            backgrounds = tuple(backgrounds)
            if len(backgrounds) != len(variables):
                raise ValueError(
                    "backgrounds must provide one Field for each IVP "
                    f"variable ({len(variables)} expected)."
                )
        # Convert equations from IVP
        for eqn in self.equations:
            # Extract IVP expressions
            M, L = eqn['LHS'].split(operators.TimeDerivative)
            F = eqn['RHS']
            # Convert M@dt(X) to λ*M@Y
            if M:
                M = M.replace(operators.TimeDerivative, lambda x: eigenvalue*x)
                for var, pert in zip(variables, perturbations):
                    M = M.replace(var, pert)
            # Convert L@X to L@Y
            if L:
                for var, pert in zip(variables, perturbations):
                    L = L.replace(var, pert)
            # Take Frechet differential of F(X)
            if F:
                if F.has(self.time):
                    raise UnsupportedEquationError("Cannot convert time-dependent IVP to EVP.")
                dF = F.frechet_differential(variables=variables, perturbations=perturbations, backgrounds=backgrounds)
            else:
                dF = 0
            # Add linearized equation and copy valid modes
            evp_eqn = EVP.add_equation((M + L - dF, 0))
            evp_eqn['valid_modes'][:] = eqn['valid_modes']
        # Add backgrounds to EVP namespace
        for var in backgrounds:
            if var.name:
                EVP.local_namespace[var.name] = var
        return EVP


class EigenvalueProblem(ProblemBase):
    """
    Linear eigenvalue problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    eigenvalue : Field object
        Field object representing the eigenvalue.
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.
    network_interfaces : iterable, optional
        Network edge fields whose common vertex-value fields should be managed
        by the problem.  Entries may also be ``(edge_field, vertex_field)``
        pairs binding an existing explicit vertex field.

    Notes
    -----
    This class supports linear eigenvalue problems of the form:
        λ*M.X + L.X = 0
    The LHS terms must be linear in the specified variables and affine in the eigenvalue.
    The RHS must be zero.
    """

    solver_class = solvers.EigenvalueSolver

    def __init__(self, variables, eigenvalue, **kw):
        super().__init__(variables, **kw)
        if any(eigenvalue.domain.nonconstant):
            raise ValueError("Eigenvalue field cannot have any bases.")
        self.eigenvalue = eigenvalue

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # Cast LHS to operand
        LHS = Operand.cast(eqn['LHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        # Check conditions
        LHS.require_linearity(*self.variables, allow_affine=False,
            self_name='EVP LHS', vars_name='problem variables', error=UnsupportedEquationError)
        LHS.require_linearity(self.eigenvalue, allow_affine=True,
            self_name='EVP LHS', vars_name='the eigenvalue', error=UnsupportedEquationError)
        if eqn['RHS'] != 0:
            raise UnsupportedEquationError("EVP RHS must be identically zero.")

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        # Extract matrix expressions
        M, L = eqn['LHS'].split(self.eigenvalue)
        # Drop eigenvalue
        if M:
            M = M.replace(self.eigenvalue, 1)
        # Reinitialize and prep NCCs
        if M:
            M = M.reinitialize(ncc=True, ncc_vars=vars)
            M.prep_nccs(vars=vars)
        if L:
            L = L.reinitialize(ncc=True, ncc_vars=vars)
            L.prep_nccs(vars=vars)
        # Convert to same domain
        domain = (M + L).domain
        if M:
            M = operators.convert(M, domain.bases)
        if L:
            L = operators.convert(L, domain.bases)
        # Save expressions and metadata
        eqn['M'] = M
        eqn['L'] = L
        eqn['domain'] = domain
        eqn['matrix_dependence'] = (M + L).matrix_dependence(*vars)
        eqn['matrix_coupling'] = (M + L).matrix_coupling(*vars)
        # Debug logging
        logger.debug(f"  M: {M}")
        logger.debug(f"  L: {L}")


# Aliases
IVP = InitialValueProblem
LBVP = LinearBoundaryValueProblem
NLBVP = NonlinearBoundaryValueProblem
EVP = EigenvalueProblem
