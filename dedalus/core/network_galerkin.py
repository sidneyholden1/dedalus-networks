"""Constrained physical-L2 assembly for native scalar network balances.

Selected with ``tau='adjoint'``, this discretization retains all Legendre modes
and represents the reactions by M**-1 C.H, rather than by highest-mode edge
tau lifts. It is currently restricted to one continuous scalar, a regular-degree
graph, static homogeneous vertex constraints, and serial LBVP/IVP assembly.
In particular it does not claim that transposing a local MPI matrix constructs
a global adjoint.
"""

from functools import reduce
from numbers import Number
import operator

import numpy as np
from scipy import sparse

from . import arithmetic, operators
from .basis import CardinalMap
from .field import Field
from .domain import Domain
from .network_calculus import _zero
from .network_operators import (
    NetworkExpression, _Coefficient, _NetworkLowerer, _find_basis,
    _network_tau_order,
)


def _legendre_values(value, basis):
    """Represent an explicit polynomial factor on the physical quadrature.

    Derivative Jacobi bases keep the parent's Legendre grid (a0=b0=0).
    Their values can therefore be used directly by a Legendre grid function,
    whose forward transform supplies the physical-L2 projection. Dedalus's
    sparse Convert supports increasing Jacobi parameters only.
    """
    if isinstance(value, Number):
        return value
    interval = value.domain.get_basis(basis.network.position_coord)
    if interval is None or interval is basis.interval:
        return value
    if interval.a0 != 0 or interval.b0 != 0:
        raise ValueError('Physical-L2 factors must share the Legendre quadrature grid.')
    domain = Domain(value.dist, tuple(basis.interval if b is interval else b
                                      for b in value.domain.bases))
    return operators.GeneralFunction(value.dist, domain, tensorsig=value.tensorsig,
        dtype=value.dtype, layout='g', func=lambda field: field.data, args=[value])


def _physical_expression(value, basis):
    """Evaluate regular RHS products on the physical Legendre quadrature.

    Convert differentiated factors before multiplying. Converting only the
    finished product from Jacobi(3,3) would project with the wrong weight.
    Graph lowering separately checks that no singular RHS is discarded.
    """
    if isinstance(value, Number):
        return value
    if isinstance(value, Field):
        return _legendre_values(_Coefficient(value) if value.tensorsig else value, basis)
    args = getattr(value, 'original_args', value.args)
    if isinstance(value, NetworkExpression):
        op = value.operation
    elif isinstance(value, arithmetic.Add):
        op = 'add'
    elif isinstance(value, (arithmetic.Multiply, arithmetic.DotProduct)):
        op = 'mul'
    elif isinstance(value, operators.Power):
        op = 'pow'
    elif isinstance(value, operators.Convert):
        return _legendre_values(_physical_expression(value.operand, basis), basis)
    else:
        # Ordinary Dedalus expressions (including prescribed GeneralFunctions)
        # already specify their evaluation domain.
        return _legendre_values(value, basis)
    factors = [_physical_expression(a, basis) for a in args]
    if op == 'add':
        return sum(factors)
    if op in ('mul', 'dot'):
        return reduce(operator.mul, factors)
    if op == 'pow':
        return factors[0] ** factors[1]
    if op in ('grad', 'div', 'H'):
        result = factors[0]
        for _ in range(2 if op == 'H' else 1):
            result = _legendre_values(basis.derivative(result), basis)
        return result
    raise NotImplementedError(f'Physical-L2 RHS lowering does not support {op!r}.')


class _PhysicalL2Adjoint(operators.LinearOperator):
    """Implicit coefficient map M**-1 C.H, with the physical edge mass M."""

    name = 'PhysicalL2Adjoint'

    def __init__(self, operand, constraint, field, basis):
        super().__init__(operand)
        self.operand = operand
        self.constraint, self.field, self.basis = constraint, field, basis
        self.domain, self.dtype, self.tensorsig = field.domain, field.dtype, ()

    def new_operand(self, operand, **kw):
        if _zero(operand):
            return 0
        return type(self)(operand, self.constraint, self.field, self.basis)

    def matrix_dependence(self, *vars):
        return np.ones(self.dist.dim, dtype=bool)

    def matrix_coupling(self, *vars):
        return np.ones(self.dist.dim, dtype=bool)

    def subproblem_matrix(self, subproblem):
        variables = [self.field]
        constraint = self.constraint.reinitialize(ncc=True, ncc_vars=variables)
        constraint.prep_nccs(vars=variables)
        constraint.gather_ncc_coeffs()
        matrix = constraint.expression_matrices(
            subproblem, variables, ncc_cutoff=0, max_ncc_terms=None)[self.field]
        # The stored Legendre modes are orthonormal on [-1,1]. Physical
        # arclength supplies L_e/2, independently for each edge and mode.
        lengths = self.basis.network.lengths[self.basis.edge_indices]
        inverse_mass = np.broadcast_to(2 / lengths[:, None],
                                       (len(lengths), self.basis.size))
        inverse_mass = inverse_mass[subproblem.coeff_slices(self.field.domain)].ravel()
        return sparse.diags(inverse_mass) @ matrix.conjugate().T

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        raise NotImplementedError('Physical-L2 reactions are implicit matrix operators.')


def register_balance(problem, lhs, rhs, condition, test):
    from .problems import InitialValueProblem, LinearBoundaryValueProblem
    if not isinstance(problem, (InitialValueProblem, LinearBoundaryValueProblem)):
        raise NotImplementedError("tau='adjoint' currently supports LBVPs and IVPs.")
    if problem.dist.comm.size != 1:
        raise NotImplementedError('Constrained physical-L2 assembly is currently serial.')
    if problem.equations or problem._network_galerkin_balance is not None:
        raise ValueError('Declare one adjoint-tau balance before its vertex constraints.')
    fields = [v for v in problem.variables if v not in problem._hidden_network_interface_fields]
    if len(fields) != 1 or fields[0].regularity != 'continuous' or fields[0].tensorsig:
        raise ValueError('Adjoint tau assembly needs one continuous scalar unknown.')
    field = fields[0]
    if condition != 'True' or (test is not None and test is not field):
        raise NotImplementedError('Adjoint tau assembly requires an unconditional scalar balance.')
    basis = _find_basis(lhs, rhs, field)
    if any(getattr(basis.interval, key, None) != 0 for key in ('a', 'b', 'a0', 'b0')):
        raise ValueError('Physical-L2 adjoint tau assembly requires a Legendre basis.')
    if len(basis.edge_indices) != basis.network.num_edges:
        raise NotImplementedError('Adjoint tau assembly requires a single full-network bucket.')
    degree = np.bincount(basis.network.endpoints.ravel(), minlength=basis.network.num_vertices)
    if not len(degree) or np.any(degree != degree[0]) or degree[0] < 2:
        raise NotImplementedError('Adjoint tau continuity currently requires uniform vertex degree >= 2.')
    graph = _NetworkLowerer(problem.dist, basis, problem._network_interface_bindings)
    left, right = graph.scalar(lhs), graph.scalar(rhs)
    if left.rank or right.rank or (left.endpoints.keys() | right.endpoints.keys()) - {0}:
        raise NotImplementedError('Adjoint tau balances support scalar zeroth-order vertex measures.')
    if not _zero(right.point) or right.endpoints:
        raise NotImplementedError('Adjoint tau RHS must be regular; retain nonzero vertex flux explicitly.')
    natural = graph.vertex_balance(left)
    if not _zero(natural):
        _check_constraint(problem, natural, 0, field, basis)
    record = dict(lhs=left.edge, rhs=_physical_expression(rhs, basis),
                  natural=natural, field=field, basis=basis, constraints=[], compiled=False)
    problem._network_galerkin_balance = record
    return record


def _check_constraint(problem, lhs, rhs, field, basis):
    if not _zero(rhs):
        raise NotImplementedError('Adjoint tau vertex constraints must be homogeneous.')
    if not hasattr(lhs, 'domain') or lhs.domain.bases != (basis.vertices,):
        raise NotImplementedError('Adjoint tau constraints must give one scalar row per vertex.')
    if lhs.tensorsig:
        raise TypeError('Adjoint tau constraints must be scalar.')
    lhs.require_linearity(field, allow_affine=False)
    if lhs.has(operators.TimeDerivative):
        raise NotImplementedError('Adjoint tau vertex constraints must be time independent.')
    if hasattr(problem, 'time') and lhs.has(problem.time):
        raise NotImplementedError('Adjoint tau vertex constraints must be time independent.')


def register_constraint(problem, lhs, rhs, condition, test):
    record = problem._network_galerkin_balance
    if record['compiled']:
        raise ValueError('Cannot add adjoint tau constraints after building the solver.')
    if condition != 'True' or (test is not None and test is not record['field']):
        raise NotImplementedError('Adjoint tau constraints must use the scalar balance test.')
    _check_constraint(problem, lhs, rhs, record['field'], record['basis'])
    record['constraints'].append(lhs)
    return lhs


def compile_balance(problem):
    record = problem._network_galerkin_balance
    if record is None or record['compiled']:
        return
    field, basis = record['field'], record['basis']
    # Validate initial continuity through the same public regularity contract.
    problem._initialize_network_vertex(field)
    ports = [np.flatnonzero(basis.network.endpoints.ravel() == v)
             for v in range(basis.network.num_vertices)]
    constraints = []
    for j in range(1, len(ports[0])):
        rows = np.repeat(np.arange(len(ports)), 2)
        columns = np.array([(p[0], p[j]) for p in ports]).ravel()
        data = np.tile([-1., 1.], len(ports))
        matrix = sparse.csr_matrix((data, (rows, columns)),
                                   shape=(len(ports), basis.ports.size))
        mapping = CardinalMap(basis.ports, basis.vertices, matrix,
                              name=f'continuous_difference_{j}')
        constraints.append(mapping @ basis.dtrace(field))
    constraints.extend(record['constraints'])
    if not _zero(record['natural']):
        constraints.append(record['natural'])
    scales = [(np.min(basis.network.lengths) / basis.size**2) **
              _network_tau_order(c, 0, [field], basis) for c in constraints]
    constraints = [scale*c for scale, c in zip(scales, constraints)]
    reactions = tuple(problem.dist.Field(name=f'__network_galerkin_reaction_{j}',
                      bases=basis.vertices, dtype=field.dtype) for j in range(len(constraints)))
    penalty = sum(_PhysicalL2Adjoint(lam, c, field, basis)
                  for lam, c in zip(reactions, constraints))
    # Continuity is represented by independent differences, with physical-L2
    # adjoints, so the usual redundant common-value unknown is unnecessary.
    attributes = ('_continuous_network_fields', '_network_interface_bindings',
                  '_hidden_network_interface_fields', 'network_interfaces',
                  '_network_matching_added')
    saved = {name: getattr(problem, name) for name in attributes}
    variables = problem.variables.copy()
    start = len(problem.equations)
    try:
        problem.variables[:] = [field, *reactions]
        problem._continuous_network_fields = ()
        problem._network_interface_bindings = ()
        problem._hidden_network_interface_fields = ()
        problem.network_interfaces = ()
        problem._network_matching_added = True
        problem._network_galerkin_compiling = True
        edge = problem.add_equation((record['lhs'] + penalty, record['rhs']), tau=False)
        edge['network_role'] = 'edge'
        edge['network_discretization'] = 'physical_l2_galerkin'
        zero = problem.dist.Field(name='__network_galerkin_zero', bases=basis.vertices,
                                  dtype=field.dtype)
        rows = [edge]
        for c in constraints:
            row = problem.add_equation((c, zero), tau=False)
            row['network_role'] = 'vertex_constraint'
            rows.append(row)
    except Exception:
        del problem.equations[start:]
        problem.variables[:] = variables
        for name, value in saved.items():
            setattr(problem, name, value)
        raise
    finally:
        problem._network_galerkin_compiling = False
    if getattr(field, '_network_vertex', None) in saved['_hidden_network_interface_fields']:
        del field._network_vertex
    for vertex in saved['_hidden_network_interface_fields']:
        problem.local_namespace.pop(vertex.name, None)
    problem.local_namespace.update({reaction.name: reaction for reaction in reactions})
    problem._hidden_network_constraint_fields += reactions
    problem.network_lowered_equations.append(tuple(rows))
    record.update(compiled=True, constraints_scaled=tuple(constraints), reactions=reactions)
