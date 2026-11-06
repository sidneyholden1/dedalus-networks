"""Opt-in graph expressions lowered to the existing scalar Network solver.

Regular edge expressions and endpoint distributions remain separate until an
equation is assembled. Endpoint coefficients are native port expressions,
indexed by Dirac derivative order; they are not samples of a delta function.
``point`` additionally represents a measure defined once per physical vertex,
which can only be multiplied by a coefficient with a defined common value.

The problem builder currently supplies the explicitly selected second-order
tau recipe for continuous scalar problems. Higher-order endpoint algebra is
available for inspection, but general trace constraints and their problem
assembly are not yet supplied by this builder.
"""

from collections import ChainMap
from dataclasses import dataclass, field
from math import comb
from numbers import Number

import numpy as np

from .field import Field, Operand
from . import operators, problems
from ..tools import parsing

__all__ = ["NetworkCalculus"]


def _zero(value):
    return isinstance(value, Number) and value == 0


def _sum(a, b):
    if _zero(a):
        return b
    if _zero(b):
        return a
    return a + b


def _product(a, b):
    if _zero(a) or _zero(b):
        return 0
    return a * b


def _time(value):
    return 0 if isinstance(value, Number) else operators.TimeDerivative(value)


@dataclass(eq=False)
class _Expression:
    graph: object
    edge: object = 0
    # 0 = scalar; 1 = the single component in the oriented edge tangent.
    rank: int = 0
    distribution: bool = False
    vertex: object = None
    point: object = 0
    endpoints: dict = field(default_factory=dict)
    # Preserve boundary algebra before truncating any edge product. A rule
    # takes a normal-derivative order and returns a native port expression.
    trace_rule: object = field(default=None, repr=False)

    __array_priority__ = 1000

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs or len(inputs) != 2:
            return NotImplemented
        a, b = map(self.graph.scalar, inputs)
        if ufunc is np.add:
            return a + b
        if ufunc is np.subtract:
            return a - b
        if ufunc is np.multiply:
            return a * b
        return NotImplemented

    def __add__(self, other):
        other = self.graph.scalar(other)
        # The scalar zero is an additive identity for either tensor character.
        if self.graph._is_zero(other):
            return self
        if self.graph._is_zero(self):
            return other
        if self.rank != other.rank:
            raise TypeError("Cannot add scalar and tangent-vector expressions.")
        if self.distribution or other.distribution:
            a, b = self.graph.density(self), self.graph.density(other)
            orders = a.endpoints.keys() | b.endpoints.keys()
            return _Expression(
                self.graph, _sum(a.edge, b.edge), rank=a.rank,
                distribution=True, point=_sum(a.point, b.point),
                endpoints={k: _sum(a.endpoints.get(k, 0),
                                   b.endpoints.get(k, 0)) for k in orders},
            )
        vertex = None
        if self.vertex is not None and other.vertex is not None:
            vertex = _sum(self.vertex, other.vertex)
        return _Expression(
            self.graph, _sum(self.edge, other.edge), rank=self.rank, vertex=vertex,
            trace_rule=lambda k: _sum(self.graph.trace(self, k), self.graph.trace(other, k)),
        )

    __radd__ = __add__

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-self.graph.scalar(other))

    def __rsub__(self, other):
        return self.graph.scalar(other) + (-self)

    def __mul__(self, other):
        other = self.graph.scalar(other)
        if self.distribution and other.distribution:
            raise TypeError("Products of two distributions are undefined.")
        if self.rank and other.rank:
            raise TypeError("Use @ to contract two tangent-vector expressions.")
        rank = self.rank + other.rank
        if self.distribution or other.distribution:
            measure, regular = ((self, other) if self.distribution
                                else (other, self))
            point = 0
            if not _zero(measure.point):
                if regular.vertex is None:
                    raise TypeError(
                        "A vertex point measure needs a multiplier with a "
                        "defined common vertex value. Use endpoint measures "
                        "for a discontinuous multiplier."
                    )
                point = _product(measure.point, regular.vertex)
            endpoints = {}
            for order, coefficient in measure.endpoints.items():
                for j in range(order + 1):
                    contribution = _product(
                        (-1)**j * comb(order, j),
                        _product(coefficient, self.graph.trace(regular, j)),
                    )
                    k = order - j
                    endpoints[k] = _sum(endpoints.get(k, 0), contribution)
            return _Expression(
                self.graph, _product(measure.edge, regular.edge), rank=rank,
                distribution=True, point=point, endpoints=endpoints,
            )
        vertex = None
        if self.vertex is not None and other.vertex is not None:
            vertex = _product(self.vertex, other.vertex)
        return _Expression(
            self.graph, _product(self.edge, other.edge), rank=rank, vertex=vertex,
            trace_rule=lambda k: self.graph._product_trace(self, other, k),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Graph-expression division requires a scalar number.")
        return self * (1 / other)

    def __pow__(self, exponent):
        if self.distribution or self.rank or not isinstance(exponent, int) or exponent < 0:
            raise TypeError("Only nonnegative integer powers of regular scalars are supported.")
        result = self.graph.scalar(1)
        for _ in range(exponent):
            result = result * self
        return result

    def __matmul__(self, other):
        other = self.graph.scalar(other)
        if self.rank != 1 or other.rank != 1:
            raise TypeError("The intrinsic dot product requires two tangent vectors.")
        if self.distribution or other.distribution:
            raise TypeError("Distributional dot products are not supplied by this interface.")
        return _Expression(
            self.graph, _product(self.edge, other.edge),
            trace_rule=lambda k: self.graph._product_trace(self, other, k),
        )


class _ScalarField(_Expression):
    """A regular graph scalar backed by ordinary edge and vertex Fields."""

    def __getitem__(self, layout):
        return self.edge[layout]

    def __setitem__(self, layout, data):
        self.edge[layout] = data

    def change_scales(self, scales):
        self.edge.change_scales(scales)


class NetworkCalculus:
    """Bind the graph calculus to a Distributor and a single NetworkBasis.

    ``network`` specifies the metric graph; ``basis`` supplies its spectral
    discretization. This object adds expression types and equation lowering,
    not a second geometry or solver. Material coefficients can be numbers or
    native scalar edge Fields wrapped with ``scalar``. Native Fields remain
    available as ``T.edge`` and ``T.vertex`` for output and diagnostics.
    """

    def __init__(self, dist, basis):
        if dist.network is not basis.network:
            raise ValueError("Distributor and basis must describe the same Network.")
        self.dist, self.basis, self.network = dist, basis, basis.network
        self._normal = dist.Field(name="__graph_endpoint_normal", bases=basis.ports)
        port = dist.local_grid(basis.ports)
        self._normal["g"] = 2 * (port.astype(int) % 2) - 1
        self._vertex_one = dist.Field(name="__graph_vertex_one", bases=basis.vertices)
        self._vertex_one["g"] = 1
        self._port_one = dist.Field(name="__graph_port_one", bases=basis.ports)
        self._port_one["g"] = 1

    def _check_native(self, value, role):
        if isinstance(value, Number):
            return
        if not isinstance(value, Operand) or value.dist is not self.dist:
            raise TypeError("Expected a number or a native expression on this Distributor.")
        if value.tensorsig:
            raise TypeError("Use intrinsic scalar/tangent expressions, not Network storage-axis tensors.")
        roles = {getattr(b, "_network_role", None) for b in value.domain.bases}
        roles.discard(None)
        allowed_roles = {role, "position"} if role == "edges" else {role}
        if not roles <= allowed_roles:
            raise TypeError(f"Expected a {role} expression, found roles {roles}.")
        expected = getattr(self.basis, role)
        for b in value.domain.bases:
            if getattr(b, "_network_role", None) == role and b is not expected:
                raise ValueError("Expression uses a different NetworkBasis bucket.")

    @staticmethod
    def _is_zero(value):
        return (_zero(value.edge) and _zero(value.point)
                and not value.endpoints
                and (value.vertex is None or _zero(value.vertex)))

    def scalar(self, value):
        """Wrap an edge coefficient, or preserve an existing typed expression."""
        if isinstance(value, _Expression):
            if value.graph is not self:
                raise ValueError("Cannot combine expressions from different graph contexts.")
            return value
        self._check_native(value, "edges")
        vertex = value if isinstance(value, Number) else None
        if isinstance(value, Operand) and not value.domain.bases:
            vertex = value
        return _Expression(self, value, vertex=vertex)

    def ScalarField(self, name, regularity="continuous"):
        """Create an edge unknown and, for continuity, its common vertex value."""
        if not isinstance(name, str) or not name:
            raise ValueError("A graph field needs a nonempty name.")
        if regularity not in ("continuous", "piecewise"):
            raise ValueError("regularity must be 'continuous' or 'piecewise'.")
        edge = self.dist.Field(name=name, bases=self.basis)
        vertex = None
        if regularity == "continuous":
            vertex = self.dist.Field(name=f"{name}_vertex", bases=self.basis.vertices)
        return _ScalarField(self, edge, vertex=vertex)

    def density(self, value):
        """Embed a regular edge expression as an edge-integrated distribution."""
        value = self.scalar(value)
        if value.distribution:
            return value
        return _Expression(self, value.edge, rank=value.rank, distribution=True)

    def point(self, value):
        """Embed one coefficient per vertex, counted once regardless of degree."""
        if isinstance(value, _Expression):
            value = self.scalar(value)
            if value.distribution or value.rank or value.vertex is None:
                raise TypeError("point requires a common scalar vertex value.")
            value = value.vertex
        self._check_native(value, "vertices")
        if isinstance(value, Number):
            value = _product(value, self._vertex_one)
        return _Expression(self, distribution=True, point=value)

    def endpoint(self, value, order=0):
        """Embed port coefficients multiplying delta^(order), before merging."""
        if not isinstance(order, int) or order < 0:
            raise ValueError("Dirac derivative order must be a nonnegative integer.")
        self._check_native(value, "ports")
        if isinstance(value, Number):
            value = _product(value, self._port_one)
        return _Expression(self, distribution=True, endpoints={order: value})

    def trace(self, value, order=0):
        """Return independent outward derivative traces, as a native port expression."""
        value = self.scalar(value)
        if value.distribution:
            raise TypeError("An endpoint trace requires a regular expression.")
        if not isinstance(order, int) or order < 0:
            raise ValueError("Trace order must be a nonnegative integer.")
        if value.trace_rule is not None:
            return value.trace_rule(order)
        edge = value.edge
        for _ in range(order):
            edge = self.basis.derivative(edge)
        if isinstance(edge, Number):
            result = edge
        elif not edge.domain.bases:
            result = _product(edge, self._port_one)
        else:
            result = self.basis.dtrace(edge)
        return _product(self._normal, result) if order % 2 else result

    def _product_trace(self, a, b, order):
        # The trace of a regular product is the product of its endpoint
        # values (and Leibniz for derivatives), not the trace of an already
        # truncated spectral product. Both factors keep their own traces.
        result = 0
        for j in range(order + 1):
            term = _product(comb(order, j),
                            _product(self.trace(a, j), self.trace(b, order-j)))
            result = _sum(result, term)
        return result

    def trace_adjoint(self, value, order=0):
        """T_k^* lambda = (-1)^k lambda delta^(k), with port pairing."""
        if not isinstance(order, int) or order < 0:
            raise ValueError("Trace order must be a nonnegative integer.")
        return self.endpoint(_product((-1)**order, value), order)

    def grad(self, value):
        value = self.scalar(value)
        if value.distribution or value.rank:
            raise TypeError("grad requires a regular scalar expression.")
        return _Expression(
            self, self.basis.derivative(value.edge), rank=1,
            trace_rule=lambda k: _product(self._normal, self.trace(value, k+1)),
        )

    def div(self, value):
        value = self.scalar(value)
        if value.distribution or value.rank != 1:
            raise TypeError("div requires a regular tangent-vector expression.")
        return _Expression(
            self, self.basis.derivative(value.edge), distribution=True,
            endpoints={0: -_product(self._normal, self.trace(value))},
        )

    def H(self, value):
        """Regular edge Hessian, using its one-dimensional scalar component."""
        value = self.scalar(value)
        if value.distribution or value.rank:
            raise TypeError("H requires a regular scalar expression.")
        return _Expression(
            self, self.basis.derivative(self.basis.derivative(value.edge)),
            trace_rule=lambda k: self.trace(value, k+2),
        )

    def Hstar(self, value):
        """Distributional Hessian adjoint before any endpoint combination."""
        value = self.scalar(value)
        if value.distribution or value.rank:
            raise TypeError("Hstar requires a regular scalar expression.")
        return _Expression(
            self, self.H(value).edge, distribution=True,
            endpoints={0: -self.trace(value, 1), 1: -self.trace(value)},
        )

    def dt(self, value):
        value = self.scalar(value)
        return _Expression(
            self, _time(value.edge), rank=value.rank,
            distribution=value.distribution,
            vertex=None if value.vertex is None else _time(value.vertex),
            point=_time(value.point),
            endpoints={k: _time(v) for k, v in value.endpoints.items()},
            trace_rule=None if value.distribution else lambda k: _time(self.trace(value, k)),
        )

    def vertex_balance(self, value):
        """Combine only order-zero endpoint terms on continuous scalar tests."""
        value = self.density(value)
        if value.rank:
            raise TypeError("A vertex balance requires a scalar distribution.")
        endpoint = value.endpoints.get(0, 0)
        flux = 0 if _zero(endpoint) else self.basis.restriction.T @ endpoint
        return _sum(value.point, flux)

    def IVP(self, variables, *, namespace=None, tau_order, **kwargs):
        return _GraphProblem(self, problems.IVP, variables, namespace, tau_order, kwargs)

    def LBVP(self, variables, *, namespace=None, tau_order, **kwargs):
        return _GraphProblem(self, problems.LBVP, variables, namespace, tau_order, kwargs)

    def NLBVP(self, variables, *, namespace=None, tau_order, **kwargs):
        return _GraphProblem(self, problems.NLBVP, variables, namespace, tau_order, kwargs)


class _GraphProblem:
    """Second-order continuous-scalar lowering with an explicit tau recipe.

    Each graph equation supplies edge balance, continuity and vertex balance.
    Graph equations are associated with variables in declaration order, or by
    ``field=...``. Native generated equations are inspectable in ``lowered``
    and ``native_problem``. No equation or constraint is silently discarded.
    """

    def __init__(self, graph, problem_type, variables, namespace, tau_order, kwargs):
        if tau_order != 2:
            raise NotImplementedError("This builder currently supplies only tau_order=2.")
        self.graph = graph
        self.variables = tuple(variables)
        if not self.variables or len({id(v) for v in self.variables}) != len(self.variables):
            raise ValueError("Declare distinct continuous graph unknowns.")
        native_variables, interfaces = [], []
        self.tau_fields = {}
        self.lowered, self._assigned, self._dirichlet = [], set(), {}
        for value in self.variables:
            if (not isinstance(value, _ScalarField) or value.graph is not graph
                    or value.vertex is None):
                raise TypeError("This builder requires continuous graph ScalarFields.")
            taus = tuple(graph.dist.Field(name=f"{value.edge.name}_tau{j}", bases=graph.basis.edges)
                         for j in (1, 2))
            self.tau_fields[value] = taus
            native_variables.extend((value.edge, value.vertex, *taus))
            interfaces.append((value.edge, value.vertex))
        names = [v.name for v in native_variables]
        if len(set(names)) != len(names):
            raise ValueError("Graph unknown and generated field names must be distinct.")
        self.native_problem = problem_type(native_variables, network_interfaces=interfaces, **kwargs)
        functions = {name: getattr(graph, name) for name in (
            "scalar", "density", "point", "endpoint", "grad", "div", "dt",
            "H", "Hstar", "trace", "trace_adjoint",
        )}
        self.namespace = ChainMap(
            {v.edge.name: v for v in self.variables},
            namespace if namespace is not None else {}, functions,
        )
        self._lift_basis = graph.basis.interval.derivative_basis(2)

    def dirichlet(self, field, mask, value=0):
        """Replace selected vertex balances with prescribed common values.

        ``mask`` is a time-independent native vertex Field with entries 0 or 1;
        declare this before the graph equation. No unforced flux condition is
        also imposed at a prescribed vertex. Physical reaction flux is then
        computed from the solution, not assumed to vanish.
        """
        if field not in self.variables or field in self._assigned:
            raise ValueError("Prescribe values for a declared field before adding its equation.")
        if field in self._dirichlet:
            raise ValueError("Combine prescribed vertices into one mask per field.")
        self.graph._check_native(mask, "vertices")
        self.graph._check_native(value, "vertices")
        if not isinstance(mask, Field) or mask.domain.bases != (self.graph.basis.vertices,):
            raise TypeError("The Dirichlet mask must be a native vertex Field.")
        if self.graph.dist.comm.allreduce(int(not np.all(np.isin(mask["g"], (0, 1))))):
            raise ValueError("The Dirichlet mask must contain only zero or one.")
        if any(mask is v for v in self.native_problem.variables):
            raise ValueError("The Dirichlet mask cannot be a problem unknown.")
        self._dirichlet[field] = (mask, value)

    def add_equation(self, equation, *, field=None):
        if field is None:
            field = next((v for v in self.variables if v not in self._assigned), None)
        if field not in self.variables or field in self._assigned:
            raise ValueError("Each declared field needs exactly one graph balance equation.")
        if isinstance(equation, str):
            lhs, rhs = parsing.split_equation(equation)
            namespace = dict(self.namespace)
            lhs, rhs = eval(lhs, namespace), eval(rhs, namespace)
        else:
            lhs, rhs = equation
        lhs, rhs = self.graph.density(lhs), self.graph.density(rhs)
        if lhs.rank or rhs.rank:
            raise TypeError("Continuous scalar equations need scalar distributions.")
        if any(k != 0 for k in lhs.endpoints.keys() | rhs.endpoints.keys()):
            raise NotImplementedError("Higher Dirac derivatives need a higher-order constraint/tau builder.")
        edge_lhs = lhs.edge
        for j, tau in enumerate(self.tau_fields[field], 1):
            edge_lhs = _sum(edge_lhs, operators.Lift(tau, self._lift_basis, -j))
        vertex_lhs, vertex_rhs = map(self.graph.vertex_balance, (lhs, rhs))
        if field in self._dirichlet:
            mask, value = self._dirichlet[field]
            vertex_lhs = mask * field.vertex + _product(1-mask, vertex_lhs)
            vertex_rhs = mask * value + _product(1-mask, vertex_rhs)
        generated = [
            ("edge", edge_lhs, rhs.edge),
            ("continuity", self.graph.basis.dtrace(field.edge)
             - self.graph.basis.restriction @ field.vertex, 0),
            ("vertex", vertex_lhs, vertex_rhs),
        ]
        start = len(self.native_problem.eqs)
        try:
            for role, left, right in generated:
                self.native_problem.add_equation((left, right))
        except Exception:
            del self.native_problem.eqs[start:]
            raise
        self.lowered.extend((field.edge.name, *row) for row in generated)
        self._assigned.add(field)

    def build_solver(self, *args, **kwargs):
        if len(self._assigned) != len(self.variables):
            raise ValueError("Every graph unknown needs a balance equation before building the solver.")
        return self.native_problem.build_solver(*args, **kwargs)
