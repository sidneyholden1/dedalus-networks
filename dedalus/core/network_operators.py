"""Native deferred network operators and distributional equation lowering.

Fields retain their normal Dedalus storage. A Dirac companion basis tags the
meaning of its coefficients, while graph expressions retain endpoint terms
until the problem chooses its test space. The compiler emits ordinary Dedalus
operators; it is not a second problem or solver API.
"""

from functools import reduce
from numbers import Number
import operator

import numpy as np
from scipy import sparse

from .field import Field, Operand, network_algebra_required
from .future import FutureField
from .domain import Domain
from . import operators, arithmetic
from .network_calculus import NetworkCalculus, _Expression, _zero

__all__ = ['Hessian', 'HessianAdjoint', 'EndpointTrace', 'EndpointTraceAdjoint',
           'endpoint', 'DirichletTrace', 'NeumannTrace', 'TraceConstraint']


class NetworkDiracBasis:
    """Coefficient space for vertex measures or incident-endpoint derivatives."""

    def __init__(self, basis, order=0, endpoints=False):
        if not isinstance(order, int) or order < 0:
            raise ValueError('Dirac derivative order must be a nonnegative integer.')
        if order and not endpoints:
            raise ValueError('Dirac derivatives require incident-endpoint labels.')
        self.basis, self.network = basis, basis.network
        self.order, self.endpoints = order, endpoints
        self._network_dirac_descriptor = self

    def __dedalus_bases__(self):
        return (self.basis.ports if self.endpoints else self.basis.vertices,)


def _find_basis(*values):
    found = set()
    def visit(value):
        if isinstance(value, NetworkExpression):
            found.add(value.basis)
            return
        descriptor = getattr(value, '_network_dirac', None)
        if descriptor is not None:
            found.add(descriptor.basis)
        for basis in getattr(getattr(value, 'domain', None), 'bases', ()):
            bundle = getattr(basis, '_network_basis', None)
            if bundle is not None:
                found.add(bundle)
        for arg in getattr(value, 'args', ()):
            visit(arg)
    for value in values:
        visit(value)
    if len(found) != 1:
        raise ValueError('A graph expression must use exactly one NetworkBasis bucket.')
    return found.pop()


def _network_evolution_test(lhs, rhs, left, right, choices, bindings, basis, variables):
    """Associate a coupled balance by its time derivative, without row ordering.

    Lower-order edge coupling does not select a test field. Singular terms
    must belong to the same unknown as the time derivative; mixed capacities
    or cross-diffusion need an explicit choice by the caller.
    """
    # Count every unknown, including piecewise flux fields. Otherwise an
    # additional time derivative or cross-flux could disappear from this audit.
    aliases = {id(field): field for field in variables}
    aliases.update({id(vertex): edge for edge, vertex, bundle in bindings
                    if bundle is basis and edge in choices})
    memo = {}

    def dependencies(value):
        key = id(value)
        if key not in memo:
            if key in aliases:
                memo[key] = {aliases[key]}
            else:
                memo[key] = set().union(*(dependencies(arg) for arg in getattr(
                    value, 'original_args', getattr(value, 'args', ()))))
        return memo[key]

    timed = set()
    visited = set()

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, operators.TimeDerivative):
            timed.update(dependencies(value))
        for arg in getattr(value, 'original_args', getattr(value, 'args', ())):
            visit(arg)

    visit(lhs)
    visit(rhs)
    if len(timed) != 1 or not timed.issubset(choices):
        raise ValueError('Cannot infer the test Field for a coupled network balance: '
                         'require one time-differentiated continuous unknown; '
                         'supply test=<field> explicitly.')
    field, = timed
    singular_fields = set().union(*(dependencies(value) for side in (left, right)
                                   for value in (side.point, *side.endpoints.values())))
    if singular_fields - {field}:
        raise ValueError('Cannot infer the test Field for a coupled network balance: '
                         'its time derivative and distributional flux involve '
                         'different unknowns; supply test=<field> explicitly.')
    return field


def _network_tau_order(lhs, rhs, variables, basis):
    """Infer scalar edge order, respecting an existing unknown-bearing lift.

    Inspect lowered expressions so native physical derivatives and the graph
    operators follow the same rule. Derivatives of prescribed coefficients do
    not raise the unknown's order. Original arguments survive prior evaluation.
    """
    variable_ids = {id(v) for v in variables}
    memo = {}

    def visit(value):
        key = id(value)
        if key in memo:
            return memo[key]
        if isinstance(value, Field):
            depends = key in variable_ids
            order = (0 if depends and value.domain.get_basis(
                basis.network.position_coord) is not None else -1)
            edge_unknown = depends and value.domain.bases == (basis.edges,)
            result = (order, depends, False,
                      hasattr(value, '_lift_polynomial'), edge_unknown)
        else:
            children = [visit(arg) for arg in getattr(
                value, 'original_args', getattr(value, 'args', ()))]
            order = max((child[0] for child in children), default=-1)
            depends = any(child[1] for child in children)
            manual = any(child[2] for child in children)
            lifted_polynomial = any(child[3] for child in children)
            edge_unknown = any(child[4] for child in children)
            if isinstance(value, operators.Lift) and depends:
                manual = True
            # Jacobi Lift becomes P_n * tau at dispatch, before graph lowering.
            if isinstance(value, arithmetic.Multiply) and lifted_polynomial and edge_unknown:
                manual = True
            if isinstance(value, operators.Differentiate) and order >= 0:
                if value.coord != basis.network.position_coord:
                    raise NotImplementedError('Automatic network taus require edge '
                                              'derivatives; supply explicit taus with tau=False.')
                order += 1
            result = (order, depends, manual, lifted_polynomial, edge_unknown)
        memo[key] = result
        return result

    results = (visit(lhs), visit(rhs))
    if any(result[2] for result in results):
        return 0
    return max(0, *(result[0] for result in results))


class NetworkExpression(FutureField):
    """Deferred graph expression, lowered before native matrix assembly."""

    _network_symbolic = True

    def __init__(self, operation, *args, order=0):
        super().__init__(*args)
        self.operation, self.name, self.order = operation, operation, order
        self.basis = _find_basis(*args)
        self.domain = Domain(self.dist, (self.basis,))
        self.dtype = np.result_type(*[getattr(a, 'dtype', type(a)) for a in args])
        ranks = [len(getattr(a, 'tensorsig', ())) for a in args]
        if operation == 'grad':
            if ranks[0]:
                raise TypeError('grad requires a regular scalar expression.')
            rank = 1
        elif operation == 'mul':
            rank = sum(ranks)
            if rank > 1:
                raise TypeError('Use @ to contract two tangent-vector expressions.')
        elif operation == 'add':
            nonzero_ranks = {r for a, r in zip(args, ranks) if not _zero(a)}
            if len(nonzero_ranks) > 1:
                raise TypeError('Cannot add scalar and tangent-vector expressions.')
            rank = max(nonzero_ranks, default=0)
        else:
            rank = 0
        self.tensorsig = (self.basis.network.tangent,) if rank else ()

    def new_operands(self, *args, **kw):
        return type(self)(self.operation, *args, order=self.order)

    def reinitialize(self, **kw):
        return self.new_operands(*[a.reinitialize(**kw) if isinstance(a, Operand) else a
                                   for a in self.args])

    def __pow__(self, exponent):
        return type(self)('pow', self, exponent)

    def evaluate(self, **kw):
        value = _NetworkLowerer(self.dist, self.basis).scalar(self)
        if value.distribution:
            raise TypeError('Distributions have coefficients, not pointwise grid values; '
                            'use them in a problem equation.')
        edge = value.edge
        if isinstance(edge, Number):
            result = self.dist.Field(bases=self.basis, dtype=self.dtype)
            result['g'] = edge
        else:
            result = edge if isinstance(edge, Field) else edge.evaluate(**kw)
        if not self.tensorsig:
            return result
        vector = self.dist.Field(bases=result.domain.bases, tensorsig=self.tensorsig,
                                 dtype=self.dtype)
        vector.change_scales(result.scales)
        vector.preset_layout(result.layout)
        vector.data[0] = result.data
        return vector


class _Coefficient(operators.LinearOperator):
    """Live coefficient view: strip measure semantics or one tangent axis."""

    name = 'NetworkCoefficient'
    _network_coefficient = True

    def __init__(self, operand):
        super().__init__(operand)
        self.operand = operand
        self.domain, self.dtype, self.tensorsig = operand.domain, operand.dtype, ()

    def new_operand(self, operand, **kw):
        return 0 if _zero(operand) else type(self)(operand)

    def matrix_dependence(self, *vars):
        return self.operand.matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.operand.matrix_coupling(*vars)

    def subproblem_matrix(self, subproblem):
        return sparse.identity(subproblem.field_size(self), format='csr', dtype=self.dtype)

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        value = self.args[0]
        out.preset_layout(value.layout)
        out.data[...] = value.data.reshape(out.data.shape)


class _NetworkLowerer(NetworkCalculus):
    """Internal algebra shared with the independently checked prototype."""

    def __init__(self, dist, basis, bindings=()):
        super().__init__(dist, basis)
        self.bindings = {edge: vertex for edge, vertex, bundle in bindings if bundle is basis}

    def scalar(self, value):
        if isinstance(value, _Expression):
            return super().scalar(value)
        if isinstance(value, NetworkExpression):
            if value.basis is not self.basis:
                raise ValueError('Cannot mix NetworkBasis buckets in a graph equation.')
            op, args = value.operation, value.args
            if op == 'add':
                return reduce(operator.add, map(self.scalar, args))
            if op == 'mul':
                return reduce(operator.mul, map(self.scalar, args))
            if op == 'dot':
                return self.scalar(args[0]) @ self.scalar(args[1])
            if op == 'pow':
                return self.scalar(args[0]) ** args[1]
            if op == 'trace_adjoint':
                return self.trace_adjoint(args[0], value.order)
            return getattr(self, op)(self.scalar(args[0]))
        descriptor = getattr(value, '_network_dirac', None)
        if descriptor is not None:
            coefficient = _Coefficient(value)
            if descriptor.endpoints:
                return self.endpoint(coefficient, descriptor.order)
            return self.point(coefficient)
        if isinstance(value, operators.TimeDerivative):
            return self.dt(self.scalar(value.operand))
        if isinstance(value, arithmetic.Add):
            return reduce(operator.add, map(self.scalar, value.args))
        if isinstance(value, arithmetic.Multiply):
            return reduce(operator.mul, map(self.scalar, value.args))
        if isinstance(value, arithmetic.DotProduct):
            return self.scalar(value.args[0]) @ self.scalar(value.args[1])
        if isinstance(value, operators.Power):
            exponent = value.args[1]
            if isinstance(exponent, Number) and int(exponent) == exponent:
                exponent = int(exponent)
            return self.scalar(value.args[0]) ** exponent
        if isinstance(value, operators.Convert):
            # Keep factor traces before truncation/conversion of edge products.
            inner = self.scalar(value.operand)
            if inner.distribution:
                raise TypeError('A distribution cannot be converted to an edge basis.')
            return _Expression(self, value, rank=inner.rank, vertex=inner.vertex,
                               trace_rule=lambda k: self.trace(inner, k))
        if isinstance(value, Operand) and value.tensorsig:
            if value.tensorsig != (self.network.tangent,):
                raise TypeError('Network calculus requires the intrinsic network.tangent.')
            return _Expression(self, _Coefficient(value), rank=1)
        if network_algebra_required(value):
            raise NotImplementedError(f'{type(value).__name__} cannot yet consume graph distributions.')
        result = super().scalar(value)
        if isinstance(value, Field) and value.regularity == 'continuous':
            vertex = self.bindings.get(value)
            if vertex is None:
                # Coefficient fields have a common value derived from their
                # traces, without adding a new independent unknown.
                R = self.basis.restriction
                vertex = (R.T @ self.basis.dtrace(value)) * (R.T @ self._port_one)**(-1)
            result.vertex = vertex
        return result


def Hessian(operand):
    """Regular edge curvature H; ``H.T`` is its distributional transpose."""
    return 0 if isinstance(operand, Number) else NetworkExpression('H', operand)


def HessianAdjoint(operand):
    """Distributional H*, including independent endpoint moments and forces."""
    return 0 if _zero(operand) else NetworkExpression('Hstar', operand)


# H is a callable in the equation namespace, so its transpose is callable too.
# Keep the full distributional endpoint terms, using the existing lowering.
Hessian.T = HessianAdjoint


def Trace(operand, order=0, *, basis=None):
    """Incident-endpoint outward derivative, retaining each end of a loop."""
    basis = _find_basis(operand) if basis is None else basis
    if getattr(operand, 'domain', None) is not None:
        if operand.domain.bases in ((basis.vertices,), (basis.ports,)):
            if order:
                raise ValueError('Derivatives require an edge field, not vertex/endpoint data.')
            return basis.restriction @ operand if operand.domain.bases == (basis.vertices,) else operand
    return _NetworkLowerer(operand.dist, basis).trace(operand, order)


def TraceAdjoint(operand, order=0):
    """Adjoint of Trace under the endpoint pairing."""
    return NetworkExpression('trace_adjoint', operand, order=order)


# Do not shadow Dedalus's existing tensor Trace operator in the public API.
EndpointTrace = Trace
EndpointTraceAdjoint = TraceAdjoint


def _trace_selection(basis, on, summed, *, average=False):
    """Build a compact port restriction or a separate sum at each vertex."""
    from .basis import CardinalBasis, CardinalMap
    from .network import _normalize_vertex_group
    network = basis.network
    if getattr(network, '_local_input', False):
        raise NotImplementedError('Selected traces require replicated graph topology.')
    if not isinstance(summed, (bool, np.bool_)):
        raise TypeError('The trace sum option must be Boolean.')
    vertices = _normalize_vertex_group(
        np.arange(network.num_vertices) if on is None else on,
        network.num_vertices, 'trace selection')
    if not vertices.size:
        raise ValueError('A trace selection must contain at least one vertex.')
    key = (tuple(vertices), bool(summed), bool(average))
    cache = basis.__dict__.setdefault('_trace_selection_cache', {})
    if key in cache:
        return cache[key]
    port_vertices = network.endpoints[np.asarray(basis.edge_indices)].ravel()
    ports = np.flatnonzero(np.isin(port_vertices, vertices))
    counts = np.bincount(port_vertices, minlength=network.num_vertices)
    if np.any(counts[vertices] == 0):
        raise ValueError('Every selected vertex must have an endpoint in this NetworkBasis.')
    if summed:
        rows = np.searchsorted(vertices, port_vertices[ports])
        elements, role = vertices, 'vertices'
        data = 1 / counts[port_vertices[ports]] if average else np.ones(ports.size)
    else:
        rows = np.arange(ports.size)
        elements, role = np.asarray(basis.port_indices)[ports], 'ports'
        data = np.ones(ports.size)
    if not summed and ports.size == basis.ports.size:
        target = basis.ports
    elif summed and vertices.size == network.num_vertices:
        target = basis.vertices
    elif summed and average:
        # A sum and an average have the same selected-vertex data space.
        target = _trace_selection(basis, vertices, True)[1].target
    else:
        target = CardinalBasis(network.edge_coord, size=elements.size)
        target._network = network
        target._network_basis = basis
        target._network_role = role
        target._network_elements = elements
    matrix = sparse.csr_matrix((data, (rows, ports)),
                               shape=(elements.size, basis.ports.size))
    mapping = CardinalMap(basis.ports, target, matrix, name='selected_trace')
    cache[key] = (vertices, mapping)
    return vertices, mapping


class _SelectedTrace(operators.LinearOperator):
    """Native trace expression retaining its selection for constraint assembly."""

    def __init__(self, value, order, on, summed, side=None, basis=None):
        self.network_basis = basis = _find_basis(value) if basis is None else basis
        if value.dist.network is not basis.network:
            raise ValueError('Endpoint data and the selected basis must use the same network.')
        for value_basis in value.domain.bases:
            value_network = getattr(value_basis, '_network', None)
            if value_network is not None and value_network is not basis.network:
                raise ValueError('Endpoint data and the selected basis must use the same network.')
        if not isinstance(summed, (bool, np.bool_)):
            raise TypeError('The trace sum option must be Boolean.')
        if side is None and on is None and not summed:
            # An unfiltered trace already uses the canonical port basis. It
            # needs neither a topology-wide selection nor an identity map.
            # Retain constraint metadata where the full topology is available.
            self.vertices = (None if getattr(basis.network, '_local_input', False)
                             else np.unique(basis.network.endpoints[basis.edge_indices]))
            self.mapping = None
        elif side is None:
            self.vertices, self.mapping = _trace_selection(basis, on, summed)
        else:
            from .basis import CardinalMap
            if on is not None or summed:
                raise ValueError('Endpoint side selection cannot be combined with on or sum.')
            if isinstance(side, str):
                if side not in ('left', 'right'):
                    raise ValueError("Endpoint side must be 'left', 'right', or 0/1 per edge.")
                sides = np.full(basis.edges.size, int(side == 'right'))
            else:
                sides = np.asarray(side)
                if sides.ndim == 0:
                    sides = np.full(basis.edges.size, sides)
                elif sides.shape == (basis.network.num_edges,):
                    sides = sides[np.asarray(basis.edge_indices)]
                if sides.shape != (basis.edges.size,) or np.any((sides != 0) & (sides != 1)):
                    raise ValueError('Endpoint side requires a fixed 0/1 selection per edge.')
                sides = sides.astype(int)
            rows = np.arange(basis.edges.size)
            matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, 2*rows+sides)),
                                       shape=(basis.edges.size, basis.ports.size))
            self.vertices = np.unique(basis.network.endpoints[basis.edge_indices])
            self.mapping = CardinalMap(basis.ports, basis.edges, matrix, name='endpoint_side')
        self.side = side
        self.trace_operand, self.order, self.summed = value, order, summed
        trace = Trace(value, order, basis=basis)
        if isinstance(trace, Number):
            constant = value.dist.Field(bases=basis.ports, dtype=value.dtype)
            constant['g'] = trace
            trace = constant
        operand = trace if self.mapping is None else self.mapping @ trace
        super().__init__(operand)
        self.operand = operand
        self.domain, self.dtype, self.tensorsig = operand.domain, operand.dtype, operand.tensorsig
        self.name = f'endpoint_deriv_{order}'

    def new_operand(self, operand, **kw):
        # Selection is already represented in the native expression. Metadata
        # is needed only when the original equation declares a constraint.
        return operand

    def matrix_dependence(self, *vars):
        return self.operand.matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.operand.matrix_coupling(*vars)

    def subproblem_matrix(self, subproblem):
        return sparse.identity(subproblem.field_size(self), format='csr', dtype=self.dtype)

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        value = self.args[0]
        out.preset_layout(value.layout)
        out.data[...] = value.data


def endpoint(operand, *, deriv=0, on=None, sum=False, side=None, basis=None):
    """Outward endpoint derivative of arbitrary nonnegative integer order.

    Order zero returns values, order one outward slopes, and order two edge
    curvature. In general the trace is ``normal**order * d_x**order(operand)``
    with physical arclength x and normals -1/+1 at tail/head, respectively.

    ``on`` contains vertex-table indices or a Boolean vertex mask. With
    ``sum=False``, retain every incident endpoint, including both loop ends.
    With ``sum=True``, return one sum per selected vertex. Output follows
    canonical port or vertex order, respectively.

    Zeroth-order vertex data are broadcast to incident endpoints; endpoint
    data can be reduced with sum=True. ``side='left'``/``'right'`` or a fixed
    0/1 array selects one endpoint per edge and returns edge-indexed values.
    This supports explicitly declared inflow/outflow laws without user maps.
    For vertex-only data, supply ``basis=`` when called outside an equation
    namespace. A problem with a single network bucket supplies this context.
    """
    if (isinstance(deriv, (bool, np.bool_))
            or not isinstance(deriv, (int, np.integer)) or deriv < 0):
        raise ValueError('Endpoint trace order must be a nonnegative integer.')
    return _SelectedTrace(operand, int(deriv), on, sum, side, basis)


def DirichletTrace(operand, *, on=None, sum=False):
    """Compatibility alias for ``endpoint(operand, deriv=0, on=on, sum=sum)``."""
    return endpoint(operand, deriv=0, on=on, sum=sum)


def NeumannTrace(operand, *, on=None, sum=False):
    """Outward physical derivatives with the same selection rules as dtrace.

    The derivative is negative at an edge tail and positive at its head.
    ``sum=True`` adds these signed values separately at each selected vertex.
    """
    return endpoint(operand, deriv=1, on=on, sum=sum)


class TraceConstraint:
    """C = sum A_k Trace_k, with C* generated from those same maps.

    Supply a dictionary ``{order: CardinalMap}``; every map takes basis.ports
    into the same independent constraint basis. Native multiplier Fields use
    that output basis. Do not include displacement matching already requested
    by ``regularity='continuous'``.
    """

    def __init__(self, basis, maps):
        self.network_basis, self.maps = basis, dict(maps)
        if not self.maps:
            raise ValueError('A trace constraint needs at least one map.')
        for order, mapping in self.maps.items():
            if not isinstance(order, int) or order < 0 or mapping.source is not basis.ports:
                raise ValueError('Constraint maps must take endpoint traces of nonnegative order.')
        targets = {mapping.target for mapping in self.maps.values()}
        if len(targets) != 1:
            raise ValueError('Constraint maps must share one output basis.')
        self.basis = targets.pop()

    def __call__(self, operand):
        return sum(mapping @ Trace(operand, order) for order, mapping in self.maps.items())

    def adjoint(self, multiplier):
        return sum(TraceAdjoint(mapping.H @ multiplier, order)
                   for order, mapping in self.maps.items())
