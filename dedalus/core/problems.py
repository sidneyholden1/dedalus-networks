"""Classes for representing systems of equations."""

import hashlib
import numpy as np
from collections import ChainMap

from .field import Operand, Field, LockedField, network_algebra_required
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

        # A regularity declaration belongs to the Field, and supplies the
        # matching equations through every ordinary problem class.
        continuous = tuple(v for v in user_variables
                           if getattr(v, 'regularity', None) == 'continuous')
        registered = tuple(spec if isinstance(spec, Field) else spec[0]
                           for spec in interface_specs
                           if isinstance(spec, Field)
                           or (isinstance(spec, (tuple, list)) and len(spec) == 2))
        interface_specs += tuple(v for v in continuous if v not in registered)
        self._continuous_network_fields = continuous
        self._network_matching_added = False
        self.network_lowered_equations = []
        self._network_constraints = []
        self._network_constraints_compiled = False
        self._hidden_network_constraint_fields = ()
        self._hidden_network_tau_fields = ()
        self._network_galerkin_balance = None
        self._network_galerkin_compiling = False

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
            if edge_field in continuous:
                edge_field._network_vertex = vertex_field

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
        if getattr(self.dist, 'network', None) is not None:
            from .network_operators import (Hessian, HessianAdjoint, endpoint, TraceAdjoint,
                                            DirichletTrace, NeumannTrace)
            # Mathematical aliases, just like the standard dt/grad/div names.
            self.local_namespace.update(H=Hessian, Hstar=HessianAdjoint,
                                        endpoint=self._network_endpoint, trace_adjoint=TraceAdjoint,
                                        dtrace=DirichletTrace, ntrace=NeumannTrace)

    def _network_endpoint(self, operand, **kwargs):
        """Resolve vertex-only endpoint expressions from the problem's bucket."""
        from .network_operators import endpoint, _find_basis
        if 'basis' not in kwargs:
            bases = getattr(getattr(operand, 'domain', None), 'bases', ())
            if len(bases) == 1 and getattr(bases[0], '_network_role', None) == 'vertices':
                kwargs['basis'] = _find_basis(*self.variables)
        return endpoint(operand, **kwargs)

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

    @property
    def tau_fields(self):
        """Automatically generated network tau Fields, in equation/mode order.

        These are ordinary mutable Fields for diagnostics and output. They are
        created by add_equation, before solver construction. Explicit user taus
        remain the caller's own fields and are not included in this tuple.
        """
        return self._hidden_network_tau_fields

    def add_vertex_constraint(self, equation, condition="True", *, test=None):
        """Impose ``C u = g`` together with its unknown adjoint reaction.

        At solver build, add the constraint rows, create reaction unknowns
        ``lambda``, and add ``C* lambda`` to the corresponding vertex/endpoint
        balances. This is exactly equivalent to
        ``add_equation(equation, vertex_constraint=True)``.

        Supports LBVPs and IVPs with fixed direct endpoint constraints on a
        continuous unknown. Prescribed values may be constants or scalar
        Fields on the full vertex basis. Coupled evolution balances identify
        their field from a unique time derivative and consistent flux; use
        ``test=u`` for ambiguous balances. Supports may be outer ends or
        internal junctions.

        With ``tau='adjoint'`` on a scalar balance, static homogeneous linear
        vertex constraints use physical-L2 reactions.
        """
        return self.add_equation(equation, condition=condition, test=test,
                                 vertex_constraint=True)

    def add_equation(self, equation, condition="True", *, test=None,
                     vertex_constraint=None, tau="auto", constraint=None):
        """Add an equation, optionally imposing a vertex constraint and reaction.

        ``vertex_constraint=True`` is equivalent to ``add_vertex_constraint``:
        it supplies both ``C u = g`` and the unknown reaction ``C* lambda`` in
        the associated balance. ``None`` or ``False`` leaves the equation as
        an ordinary equation. The old ``constraint`` keyword is retained only
        as a compatibility alias; new scripts should use the vertex spelling.

        Native scalar graph equations automatically receive edge tau lifts at
        their differential order. Existing unknown-bearing lifts are respected.
        Use ``tau=False`` to supply a custom discretization; generated Fields
        are accessible through ``problem.tau_fields`` and the lowered edge row's
        ``network_tau_fields`` entry. This does not choose physical conditions.

        Coupled network balances infer their test Field from a unique
        time-differentiated continuous unknown, provided the distributional
        flux involves only that unknown. Lower-order edge coupling is allowed.
        Use ``test=field`` explicitly for mixed time derivatives, cross-fluxes
        or coupled static balances; field names and equation order never select
        the association.

        ``tau='adjoint'`` constructs correction shapes from the physical-L2
        adjoints of continuity and vertex constraints, retaining every Legendre
        mode and projecting regular RHS products in physical L2. This opt-in
        path currently supports one continuous scalar on a regular-degree
        graph, static homogeneous constraints, and serial LBVP/IVP assembly.
        The previous ``tau='galerkin'`` spelling remains a compatibility alias.
        """
        for name, value in (('vertex_constraint', vertex_constraint), ('constraint', constraint)):
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise TypeError(f'{name} must be a Boolean.')
        if (vertex_constraint is not None and constraint is not None
                and vertex_constraint != constraint):
            raise ValueError('vertex_constraint and the compatibility alias constraint disagree.')
        if vertex_constraint is None:
            vertex_constraint = constraint
        if tau == 'galerkin':
            tau = 'adjoint'
        if tau is not False and tau not in ('auto', 'adjoint'):
            raise ValueError("tau must be 'auto', 'adjoint', or False "
                             "('galerkin' is a compatibility alias).")
        if self._network_galerkin_balance is not None and not self._network_galerkin_compiling:
            if not vertex_constraint:
                raise ValueError('Adjoint tau assembly accepts one balance and its vertex constraints.')
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
        if vertex_constraint:
            if self._network_galerkin_balance is not None:
                from .network_galerkin import register_constraint
                return register_constraint(self, LHS, RHS, condition, test)
            return self._add_network_constraint(LHS, RHS, condition, test)
        if tau == 'adjoint':
            from .network_galerkin import register_balance
            return register_balance(self, LHS, RHS, condition, test)
        if network_algebra_required(LHS) or network_algebra_required(RHS):
            return self._add_network_equation(LHS, RHS, condition, test, tau)
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
        if self._network_galerkin_balance is not None:
            from .network_galerkin import compile_balance
            compile_balance(self)
        self._ensure_network_constraints()
        self._ensure_network_matching()
        initialized_vertices = set()
        for field in self._continuous_network_fields:
            vertex = self.vertex_field(field)
            if vertex not in initialized_vertices:
                self._initialize_network_vertex(field)
                initialized_vertices.add(vertex)
        if self._continuous_network_fields:
            # Traverse expression order, not an identity-hashed set: every
            # MPI rank must validate coefficients in the same collective order.
            coefficients, seen = [], set()
            def visit(value):
                if isinstance(value, Field):
                    if value not in seen:
                        seen.add(value)
                        coefficients.append(value)
                else:
                    for arg in getattr(value, 'original_args', getattr(value, 'args', ())):
                        visit(arg)
            for equation in self.equations:
                visit(equation['eqn'])
            for field in coefficients:
                if field.regularity == 'continuous' and field not in self.variables:
                    self._initialize_network_vertex(field)
        return self.solver_class(self, *args, **kw)

    def _add_network_constraint(self, lhs, rhs, condition, test):
        from numbers import Number
        from .network_operators import _SelectedTrace
        if not isinstance(self, (LinearBoundaryValueProblem, InitialValueProblem)):
            raise NotImplementedError('Automatic vertex constraints currently support LBVPs and IVPs.')
        if self._network_constraints_compiled:
            raise ValueError('Add vertex constraints before building the solver.')
        if condition != 'True':
            raise NotImplementedError('Automatic vertex constraints must be unconditional.')
        if not isinstance(lhs, _SelectedTrace) or not isinstance(lhs.trace_operand, Field):
            raise TypeError('A vertex constraint requires a direct endpoint(u, deriv=k).')
        if lhs.side is not None:
            raise NotImplementedError('Vertex constraints use on= selections, not per-edge side selections.')
        if lhs.vertices is None:
            raise NotImplementedError('Automatic vertex constraints require replicated graph topology.')
        field = lhs.trace_operand
        if field not in self._continuous_network_fields:
            raise ValueError('Automatic vertex constraints require a continuous problem unknown.')
        if test is not None and test is not field:
            raise ValueError('A vertex constraint must be paired with its own unknown.')
        if not isinstance(rhs, Number):
            basis = lhs.network_basis
            if (not isinstance(rhs, Field) or rhs in self.variables or rhs.tensorsig
                    or rhs.domain.bases != (basis.vertices,)):
                raise NotImplementedError('A vertex constraint RHS must be a constant or prescribed vertex Field.')
        for previous in self._network_constraints:
            other = previous['LHS']
            same_field = (other.trace_operand is field or
                          (lhs.order == 0 and self.vertex_field(other.trace_operand) is self.vertex_field(field)))
            if (same_field and other.order == lhs.order
                    and np.intersect1d(other.vertices, lhs.vertices).size):
                raise ValueError('Overlapping vertex constraints would duplicate endpoint conditions.')
        record = {'LHS': lhs, 'RHS': rhs, 'condition': condition}
        self._network_constraints.append(record)
        return record

    def _ensure_network_constraints(self):
        """Pair kinematic rows and reaction columns using the same trace map."""
        from .network_operators import _trace_selection, Trace, TraceAdjoint, _NetworkLowerer
        if self._network_constraints_compiled or not self._network_constraints:
            return
        plans = []
        for index, record in enumerate(self._network_constraints):
            trace = record['LHS']
            field, basis = trace.trace_operand, trace.network_basis
            role = f'endpoint_{trace.order}' if trace.order else 'vertex'
            balances = [eq for eq in self.equations
                        if eq.get('network_role') == role
                        and (eq.get('network_test') is field
                             or field in eq.get('network_tests', ()))]
            if len(balances) != 1:
                raise ValueError('A vertex constraint needs exactly one corresponding '
                                 f'distributional balance; specify test={field.name} on it.')
            mapping = trace.mapping
            if trace.order == 0 and not trace.summed:
                # Continuity already identifies endpoint values. One row per
                # vertex suffices; averaging is exact on the declared space.
                _, mapping = _trace_selection(basis, trace.vertices, True, average=True)
            elif mapping is None:
                _, mapping = _trace_selection(basis, trace.vertices, False)
            reaction_name = f'__network_reaction_{field.name}_{index}'
            if any(v.name == reaction_name for v in self.variables):
                raise ValueError(f'Variable name {reaction_name!r} is reserved for reactions.')
            multiplier = self.dist.Field(name=reaction_name, bases=mapping.target,
                                         dtype=field.dtype)
            multiplier['c'] = 0
            graph = _NetworkLowerer(self.dist, basis, self._network_interface_bindings)
            reaction = graph.scalar(TraceAdjoint(mapping.H @ multiplier, trace.order))
            reaction = (reaction.endpoints[trace.order] if trace.order
                        else graph.vertex_balance(reaction))
            lhs = mapping @ Trace(field, trace.order)
            rhs = record['RHS']
            if isinstance(rhs, Field):
                rhs_mapping = (_trace_selection(basis, trace.vertices, True, average=True)[1]
                               if trace.summed else mapping)
                rhs = rhs_mapping @ (basis.restriction @ rhs)
            plans.append((record, balances[0], multiplier, reaction, lhs, rhs))

        # Compilation is transactional: a malformed constraint must not leave
        # hidden unknowns or partly changed balances in a retryable problem.
        original_variables = len(self.variables)
        original_equations = [(eq, eq.copy()) for eq in self.equations]
        try:
            self.variables.extend(plan[2] for plan in plans)
            for record, balance, multiplier, reaction, lhs, rhs in plans:
                balance['LHS'] = balance['LHS'] + reaction
                equation = self.add_equation((lhs, rhs))
                equation['network_role'] = 'constraint'
            # Refresh all matrix expressions with the complete variable list.
            for equation in self.equations:
                expr = equation['LHS'] - equation['RHS']
                equation.update(eqn=expr, tensorsig=expr.tensorsig, dtype=expr.dtype,
                                valid_modes=expr.valid_modes.copy())
                self._check_equation_conditions(equation)
                self._build_matrix_expressions(equation)
        except Exception:
            del self.variables[original_variables:]
            self.equations[:] = [eq for eq, saved in original_equations]
            for equation, saved in original_equations:
                equation.clear()
                equation.update(saved)
            raise
        self._hidden_network_constraint_fields = tuple(plan[2] for plan in plans)
        self._network_constraints_compiled = True

    def _ensure_network_matching(self):
        if not self._network_matching_added:
            for field in self._continuous_network_fields:
                equation = self.add_equation((self.continuity(field), 0))
                equation['network_role'] = 'continuity'
            self._network_matching_added = True

    def _initialize_network_vertex(self, field):
        """Derive the common initial value from compatible endpoint traces."""
        from mpi4py import MPI
        if field in self._continuous_network_fields:
            vertex, basis = self._network_interface_binding(field)
            group = [(edge, bundle) for edge, common, bundle in self._network_interface_bindings
                     if common is vertex and edge in self._continuous_network_fields]
        else:
            from .network_operators import _find_basis
            basis = _find_basis(field)
            vertex = field.dist.Field(bases=basis.vertices, dtype=field.dtype)
            group = [(field, basis)]
        states = [(edge, edge.scales, edge.layout) for edge, _ in group]
        try:
            counts, totals = 0, 0
            for edge, bundle in group:
                ones = self.dist.Field(bases=bundle.ports, dtype=edge.dtype)
                ones['g'] = 1
                counts = counts + (bundle.restriction.T @ ones).evaluate()['g']
                totals = totals + (bundle.restriction.T @ bundle.dtrace(edge)).evaluate()['g']
            if self.dist.comm.allreduce(bool(np.any(counts == 0)), op=MPI.LOR):
                raise ValueError('A continuous field needs incident edges at every vertex.')
            vertex['g'] = totals / counts
            error = 0.
            for edge, bundle in group:
                residual = (bundle.dtrace(edge) - bundle.restriction @ vertex).evaluate()['g']
                error = max(error, float(np.max(np.abs(residual), initial=0)))
            error = self.dist.comm.allreduce(error, op=MPI.MAX)
            scale = self.dist.comm.allreduce(float(np.max(np.abs(vertex['g']), initial=0)), op=MPI.MAX)
        finally:
            for edge, scales, layout in states:
                edge.change_scales(scales)
                edge.change_layout(layout)
        if error > 1e-9 * max(1., scale):
            raise ValueError(f'Initial values of continuous field {field.name!r} have '
                             f'incompatible endpoint traces (maximum mismatch {error:.3g}).')

    def _add_network_equation(self, lhs, rhs, condition, test, tau):
        from .network_operators import (_find_basis, _NetworkLowerer,
                                        _network_tau_order, _network_evolution_test)
        from .network_calculus import _zero
        basis = _find_basis(lhs, rhs)
        graph = _NetworkLowerer(self.dist, basis, self._network_interface_bindings)
        left, right = graph.scalar(lhs), graph.scalar(rhs)
        singular = bool(left.endpoints or right.endpoints
                        or not _zero(left.point) or not _zero(right.point))
        if test is None:
            choices = [edge for edge, _, bundle in self._network_interface_bindings
                       if bundle is basis and edge in self._continuous_network_fields]
            if singular and not choices:
                raise ValueError('A distributional graph equation needs an explicit test Field '
                                 'or a continuous Field declaration.')
            continuous = True
            test_field = choices[0] if len(choices) == 1 else None
            if len(choices) > 1:
                test_field = _network_evolution_test(
                    lhs, rhs, left, right, choices, self._network_interface_bindings,
                    basis, self.variables)
        else:
            if not isinstance(test, Field) or _find_basis(test) is not basis:
                raise ValueError('The test Field must use the equation\'s NetworkBasis.')
            continuous = test.regularity == 'continuous'
            test_field = test
        if left.rank or right.rank:
            raise TypeError('Graph balance equations currently require scalar tests.')
        order = (_network_tau_order(left.edge, right.edge, self.variables, basis)
                 if tau == 'auto' else 0)
        if order >= basis.interval.size:
            raise ValueError('Automatic network taus need more interval modes than '
                             'the differential order.')
        taus = tuple(self.dist.Field(
            name=f'__network_tau_{len(self.network_lowered_equations)}_{j}',
            bases=basis.edges, dtype=np.result_type(
                getattr(left.edge, 'dtype', type(left.edge)),
                getattr(right.edge, 'dtype', type(right.edge)), self.variables[0].dtype))
            for j in range(order))
        for field in taus:
            if any(v.name == field.name for v in self.variables):
                raise ValueError(f'Variable name {field.name!r} is reserved for automatic taus.')
            field['c'] = 0
        edge_lhs = left.edge
        if taus:
            lift_basis = basis.interval.derivative_basis(order)
            for j, field in enumerate(taus):
                edge_lhs = edge_lhs + operators.Lift(field, lift_basis, -j-1)
        rows = [('edge', edge_lhs, right.edge)]
        if continuous:
            rows.append(('vertex', graph.vertex_balance(left), graph.vertex_balance(right)))
        else:
            if not _zero(left.point) or not _zero(right.point):
                raise TypeError('A common vertex measure requires continuous tests.')
            rows.append(('endpoint_0', left.endpoints.get(0, 0), right.endpoints.get(0, 0)))
        orders = sorted((left.endpoints.keys() | right.endpoints.keys()) - {0})
        rows.extend((f'endpoint_{k}', left.endpoints.get(k, 0), right.endpoints.get(k, 0))
                    for k in orders)
        start = len(self.equations)
        original_equations = [(eq, eq.copy()) for eq in self.equations]
        original_variables = len(self.variables)
        original_perturbations = len(getattr(self, 'perturbations', ()))
        generated = []
        try:
            self.variables.extend(taus)
            if isinstance(self, NonlinearBoundaryValueProblem):
                for field in taus:
                    perturbation = field.copy()
                    perturbation.name = 'δ' + field.name
                    perturbation['c'] = 0
                    self.perturbations.append(perturbation)
            for role, a, b in rows:
                if _zero(a) and _zero(b):
                    continue
                common = (self.vertex_field(test_field) if role == 'vertex'
                          and test_field in self._continuous_network_fields else None)
                existing = [eq for eq in self.equations if common is not None
                            and eq.get('network_common_vertex') is common]
                if existing:
                    if len(existing) != 1 or test_field in existing[0]['network_tests']:
                        raise ValueError('Each continuous resolution bucket needs one vertex balance.')
                    equation = existing[0]
                    equation['LHS'] = equation['LHS'] + a
                    equation['RHS'] = equation['RHS'] + b
                    expr = equation['LHS'] - equation['RHS']
                    equation.update(eqn=expr, tensorsig=expr.tensorsig, dtype=expr.dtype,
                                    valid_modes=expr.valid_modes.copy())
                    self._check_equation_conditions(equation)
                    self._build_matrix_expressions(equation)
                    equation['network_tests'] += (test_field,)
                    generated.append(equation)
                    continue
                equation = self.add_equation((a, b), condition=condition)
                equation['network_role'] = role
                equation['network_test'] = test_field
                if common is not None:
                    equation['network_common_vertex'] = common
                    equation['network_tests'] = (test_field,)
                if role == 'edge':
                    equation['network_tau_fields'] = taus
                    equation['network_tau_order'] = order
                generated.append(equation)
        except Exception:
            del self.equations[start:]
            for equation, saved in original_equations:
                equation.clear()
                equation.update(saved)
            del self.variables[original_variables:]
            if isinstance(self, NonlinearBoundaryValueProblem):
                del self.perturbations[original_perturbations:]
            raise
        self._hidden_network_tau_fields += taus
        self.network_lowered_equations.append(tuple(generated))
        return tuple(generated)

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
            the exposed perturbation fields; hidden vertex and tau perturbations
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
        self._ensure_network_matching()
        variables = tuple(self.variables)
        hidden_variable_ids = {
            id(variable)
            for variable in (*self._hidden_network_interface_fields, *self.tau_fields)
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
                        "; omit problem-managed hidden vertex and tau fields because "
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

        # The equations below are already lowered, including their tau lifts.
        # Copy these internal unknowns without asking the caller to supply
        # perturbations for them or running automatic discretization a second time.
        tau_perturbations = []
        for field in self.tau_fields:
            perturbation = field.copy()
            perturbation.name = 'δ' + field.name
            perturbation_by_variable_id[id(field)] = perturbation
            tau_perturbations.append(perturbation)
        EVP.variables.extend(tau_perturbations)
        EVP._hidden_network_tau_fields = tuple(tau_perturbations)

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
        # Matching rows have already been linearized above.
        EVP._network_matching_added = True
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
