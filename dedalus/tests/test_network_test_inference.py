"""Inferred evolution associations preserve explicit matrices and reactions."""

import numpy as np
import pytest
import dedalus.public as d3


def setup(dtype=np.float64):
    network = d3.Network(endpoints=((0, 1), (2, 1)), lengths=(1., 1.7))
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=12)
    return dist, basis


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('backend', [None, 'dtn'])
def test_inferred_coupling_matches_explicit_matrices_and_evolution(dtype, backend):
    dist, basis = setup(dtype)
    solvers, fields, problems = [], [], []
    for explicit in (True, False):
        # Deliberately unrelated names and reversed variable/equation order.
        u = dist.Field(name='temperature', bases=basis, regularity='continuous')
        v = dist.Field(name='concentration', bases=basis, regularity='continuous')
        u['g'] = v['g'] = 0
        C = dist.Field(bases=basis.edges); C['g'] = np.array([[.7], [1.3]])
        p = d3.IVP([v, u], namespace=locals())
        p.add_equation('2*dt(u)-div(grad(u))+.3*(u-v)=1',
                       **({'test': u} if explicit else {}))
        p.add_equation('C*dt(v)-div(2*grad(v))+.3*(v-u)=.5',
                       **({'test': v} if explicit else {}))
        p.add_vertex_constraint('endpoint(v, on=[2])=0')
        p.add_vertex_constraint('endpoint(u, on=[0])=0')
        solvers.append(p.build_solver(d3.SBDF2, network_solver=backend))
        fields.append((u, v))
        problems.append(p)
    for a, b in zip(solvers[0].subproblems, solvers[1].subproblems, strict=True):
        for name in ('M_min', 'L_min'):
            np.testing.assert_array_equal(getattr(a, name).toarray(),
                                          getattr(b, name).toarray())
    for solver in solvers:
        for _ in range(3):
            solver.step(.001)
    for a, b in zip(*fields, strict=True):
        np.testing.assert_allclose(a['g'], b['g'], atol=1e-13, rtol=0)
        np.testing.assert_allclose(problems[0].vertex_field(a)['g'],
                                   problems[1].vertex_field(b)['g'], atol=1e-13, rtol=0)


@pytest.mark.parametrize('equation', [
    'dt(u)+dt(v)-div(grad(u))=0',
    'dt(u+v)-div(grad(u))=0',
    'dt(u)-div(grad(v))=0',
    'dt(u)-div(grad(u)+grad(v))=0',
    '-div(grad(u))+v=0',
])
def test_ambiguous_balance_rejected_without_mutation_and_explicit_override(equation):
    dist, basis = setup()
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    v = dist.Field(name='v', bases=basis, regularity='continuous')
    p = d3.IVP([u, v], namespace=locals())
    before = tuple(p.variables)
    with pytest.raises(ValueError, match='supply test=<field> explicitly'):
        p.add_equation(equation)
    assert tuple(p.variables) == before
    assert not p.equations
    assert not p.tau_fields
    assert not p.network_lowered_equations
    rows = p.add_equation(equation, test=u)
    assert rows and all(row['network_test'] is u for row in rows)


def test_vertex_capacity_uses_the_corresponding_continuous_unknown():
    dist, basis = setup()
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    v = dist.Field(name='v', bases=basis, regularity='continuous')
    capacity = dist.Field(bases=basis.dirac); capacity['g'] = 1
    p = d3.IVP([u, v], namespace=locals())
    rows = p.add_equation('(1+capacity)*dt(u)-div(grad(u))=v')
    assert all(row['network_test'] is u for row in rows)
    with pytest.raises(ValueError, match='different unknowns'):
        p.add_equation('dt(v)-div(grad(v))+capacity*u=0')


@pytest.mark.parametrize('equation', [
    'dt(u)+dt(q)-div(grad(u))=0',
    'dt(u)-div(grad(q))=0',
])
def test_piecewise_unknowns_are_not_ignored_when_inferring_a_balance(equation):
    dist, basis = setup()
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    v = dist.Field(name='v', bases=basis, regularity='continuous')
    q = dist.Field(name='q', bases=basis)
    p = d3.IVP([u, v, q], namespace=locals())
    before = tuple(p.variables)
    with pytest.raises(ValueError, match='supply test=<field> explicitly'):
        p.add_equation(equation)
    assert tuple(p.variables) == before
    assert not p.equations and not p.tau_fields
    assert p.add_equation(equation, test=u)
