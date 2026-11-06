"""Grouped traces and automatic beam reactions against independent solutions."""

import numpy as np
import pytest

import dedalus.public as d3


def setup(endpoints, lengths, dtype=np.float64):
    network = d3.Network(endpoints=endpoints, lengths=lengths)
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=12)
    return network, dist, basis


def values(expression):
    return expression.evaluate()['g'].copy().ravel()


@pytest.mark.parametrize('order', range(5))
def test_arbitrary_endpoint_trace_order_and_per_vertex_sums(order):
    network, dist, basis = setup(((0, 1), (2, 1), (1, 1)), (1., 2., .7))
    u = dist.Field(bases=basis)
    polynomial = np.polynomial.Polynomial([1., .3, -.2, .5, .7])
    u['g'] = polynomial(basis.local_physical_grid(dist))
    ends = np.array([0., 1., 0., 2., 0., .7])
    normals = np.array([-1., 1., -1., 1., -1., 1.])
    expected = normals**order * polynomial.deriv(order)(ends)
    np.testing.assert_allclose(values(d3.endpoint(u, deriv=order)), expected, atol=2e-7)
    selected = [1, 3, 4, 5]
    np.testing.assert_allclose(values(d3.endpoint(u, deriv=order, on=[1])),
                               expected[selected], atol=2e-7)
    np.testing.assert_allclose(values(d3.endpoint(u, deriv=order, on=[2, 1], sum=True)),
                               [expected[selected].sum(), expected[2]], atol=2e-7)


@pytest.mark.parametrize('order', [-1, .5, True, '1'])
def test_invalid_endpoint_trace_order(order):
    _, dist, basis = setup(((0, 1),), (1.,))
    u = dist.Field(bases=basis)
    with pytest.raises(ValueError, match='nonnegative integer'):
        d3.endpoint(u, deriv=order)


def test_network_parser_preserves_tensor_trace():
    _, dist, basis = setup(((0, 1),), (1.,))
    u = dist.Field(name='u', bases=basis, regularity='continuous'); u['g'] = 0
    A = dist.TensorField(basis.network.position_coord, bases=basis.edges)
    A['g'][0, 0] = 2
    problem = d3.LBVP([u], namespace=locals())
    assert problem.namespace['trace'] is d3.Trace
    problem.add_equation('-div(grad(u))+u=trace(A)')
    problem.build_solver().solve()
    np.testing.assert_allclose(u['g'], 2, atol=2e-11)


def beam_problem(dist, basis):
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    u['g'] = 0
    taus = [dist.Field(name=f'tau{j}', bases=basis.edges) for j in range(4)]
    tau = sum(d3.Lift(t, basis.interval.derivative_basis(4), -j-1)
              for j, t in enumerate(taus))
    return u, tau, d3.LBVP([u, *taus], namespace=locals())


def test_selected_traces_keep_loop_ends_and_sum_per_vertex():
    # At vertex 1, two incoming edges and BOTH ends of a self-loop meet.
    network, dist, basis = setup(((0, 1), (2, 1), (1, 1)), (1., 2., .7))
    u = dist.Field(bases=basis)
    x = basis.local_physical_grid(dist)
    a = np.array([.3, -.2, .5])[:, None]
    c = np.array([1., 2., 3.])[:, None]
    u['g'] = x*x + a*x + c
    n = np.array([-.3, 2.3, .2, 3.8, -.5, 1.9])
    d = np.array([1., 2.3, 2., 5.6, 3., 3.84])
    np.testing.assert_allclose(values(d3.NeumannTrace(u, on=[1])), n[[1, 3, 4, 5]], atol=1e-11)
    np.testing.assert_allclose(values(d3.DirichletTrace(u, on=[1])), d[[1, 3, 4, 5]], atol=1e-11)
    for trace, expected in ((d3.NeumannTrace, n), (d3.DirichletTrace, d)):
        # Selection order and duplicate indices do not change canonical order.
        actual = values(trace(u, on=[2, 1, 1], sum=True))
        np.testing.assert_allclose(actual, [expected[[1, 3, 4, 5]].sum(), expected[2]], atol=1e-11)
        np.testing.assert_allclose(values(trace(u, on=[False, True, True], sum=True)), actual)
    np.testing.assert_allclose(values(basis.ntrace(u, on=[1], sum=True)), [7.5], atol=1e-11)
    np.testing.assert_allclose(values(basis.dtrace(u, on=[1])), d[[1, 3, 4, 5]], atol=1e-11)


@pytest.mark.parametrize('on', [[], [3], [-1], [1.2], [[0]], [True]])
def test_invalid_trace_selection(on):
    _, dist, basis = setup(((0, 1),), (1.,))
    u = dist.Field(bases=basis)
    with pytest.raises(ValueError):
        d3.NeumannTrace(u, on=on)


def test_trace_sum_requires_boolean():
    _, dist, basis = setup(((0, 1),), (1.,))
    u = dist.Field(bases=basis)
    with pytest.raises(TypeError, match='Boolean'):
        d3.NeumannTrace(u, sum='yes')


def test_default_traces_remain_compatible_with_existing_port_maps():
    _, dist, basis = setup(((0, 1), (0, 2)), (1., 1.7))
    coefficient = dist.Field(bases=basis.edges)
    coefficient['g'] = np.array([[2.], [3.]])
    trace = d3.DirichletTrace(coefficient)
    assert trace.domain.bases == (basis.ports,)
    np.testing.assert_allclose(values(basis.restriction.T @ trace), [5., 2., 3.])
    np.testing.assert_allclose(values(d3.NeumannTrace(coefficient, on=[0], sum=True)), 0)


def test_legacy_traces_allow_isolated_vertices():
    network = d3.Network(vertex_ids=('left', 'right', 'isolated'),
                         endpoints=((0, 1),), lengths=(2.,))
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(bases=basis)
    x = basis.local_physical_grid(dist)
    u['g'] = 1 + 3*x + x*x
    for expression in (d3.dtrace(u), d3.endpoint(u)):
        np.testing.assert_allclose(values(expression), [1., 11.], atol=1e-12)
    for expression in (d3.ntrace(u), d3.endpoint(u, deriv=1)):
        np.testing.assert_allclose(values(expression), [-3., 7.], atol=1e-12)
    np.testing.assert_allclose(values(basis.R.T @ d3.ntrace(u)),
                               [-3., 7., 0.], atol=1e-12)


@pytest.mark.parametrize('backend', [None, 'dtn'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_branching_beam_with_nonzero_junction_slopes(backend, reverse, dtype):
    endpoints = ((0, 1), (2, 0), (0, 3)) if reverse else ((0, 1), (0, 2), (0, 3))
    _, dist, basis = setup(endpoints, (1., 1., 1.), dtype)
    u, tau, problem = beam_problem(dist, basis)
    load = dist.Field(bases=basis.edges)
    amplitude = 1+.3j if dtype == np.complex128 else 1.
    load['g'] = 24*amplitude*np.array([[1.], [2.], [3.]])
    problem.namespace.maps.insert(1, locals())
    # Constraints may precede their balance; association happens at build.
    problem.add_equation('ntrace(u, on=[0], sum=True)=0', constraint=True)
    problem.add_equation('Hstar(H(u))+tau=load')
    problem.add_equation('dtrace(u, on=[1,2,3])=0', constraint=True)
    problem.add_equation('ntrace(u, on=[1,2,3])=0', constraint=True)
    solver = problem.build_solver(network_solver=backend)
    solver.solve()
    x = basis.local_physical_grid(dist)
    if reverse:
        x = x.copy()
        x[1] = 1-x[1]
    exact = amplitude*(2*(1-x*x)**2 + np.array([[-1.], [0.], [1.]])
                       * (x**4-1.5*x**3+.5*x))
    np.testing.assert_allclose(u['g'], exact, atol=2e-10, rtol=0)
    np.testing.assert_allclose(values(d3.NeumannTrace(u, on=[0])),
                               amplitude*np.array([.5, 0., -.5]), atol=2e-10)
    np.testing.assert_allclose(values(d3.NeumannTrace(u, on=[0], sum=True)), 0, atol=2e-10)
    # The chosen slope relation transmits a common, nonzero bending moment.
    np.testing.assert_allclose(values(d3.DirichletTrace(d3.Hessian(u), on=[0])),
                               -8*amplitude, atol=2e-9)


@pytest.mark.parametrize('backend', [None, 'dtn'])
def test_internal_clamp_has_only_one_displacement_constraint(backend):
    network, dist, basis = setup(((0, 1), (0, 2), (0, 3)), (1., 1., 1.))
    u, tau, problem = beam_problem(dist, basis)
    problem.add_equation('Hstar(H(u))+tau=24')
    problem.add_equation('dtrace(u)=2', constraint=True)
    problem.add_equation('ntrace(u)=0', constraint=True)
    solver = problem.build_solver(network_solver=backend)
    solver.solve()
    x = basis.local_physical_grid(dist)
    np.testing.assert_allclose(u['g'], 2+x*x*(1-x)**2, atol=2e-10, rtol=0)
    force, moment = problem._hidden_network_constraint_fields
    assert force.domain.bases[0].size == network.num_vertices
    assert moment.domain.bases[0].size == network.num_ports
    shape = (len(problem.equations), len(problem.variables))
    problem.build_solver(network_solver=backend).solve()
    assert shape == (len(problem.equations), len(problem.variables))


@pytest.mark.parametrize('backend', [None, 'dtn'])
def test_cantilever_keeps_unselected_end_free(backend):
    _, dist, basis = setup(((0, 1),), (1.,))
    u, tau, problem = beam_problem(dist, basis)
    problem.add_equation('Hstar(H(u))+tau=24')
    problem.add_equation('dtrace(u, on=[0])=0', constraint=True)
    problem.add_equation('ntrace(u, on=[0])=0', constraint=True)
    problem.build_solver(network_solver=backend).solve()
    x = basis.local_physical_grid(dist)
    np.testing.assert_allclose(u['g'], x**4-4*x**3+6*x*x, atol=2e-10, rtol=0)
    np.testing.assert_allclose(values(d3.DirichletTrace(d3.Hessian(u), on=[1])), 0, atol=2e-9)
    np.testing.assert_allclose(values(d3.NeumannTrace(d3.Hessian(u), on=[1])), 0, atol=2e-8)


def test_overlapping_constraints_are_rejected():
    _, dist, basis = setup(((0, 1), (0, 2)), (1., 1.))
    u, tau, problem = beam_problem(dist, basis)
    problem.add_equation('ntrace(u, on=[0])=0', constraint=True)
    with pytest.raises(ValueError, match='Overlapping'):
        problem.add_equation('ntrace(u, on=[0], sum=True)=0', constraint=True)


def test_constraint_requires_matching_distributional_balance():
    _, dist, basis = setup(((0, 1),), (1.,))
    u, tau, problem = beam_problem(dist, basis)
    problem.add_equation('ntrace(u, on=[0])=0', constraint=True)
    original_variables = tuple(problem.variables)
    with pytest.raises(ValueError, match='distributional balance'):
        problem.build_solver()
    assert tuple(problem.variables) == original_variables
    assert not problem.equations


def test_mixed_balances_require_explicit_test_association():
    _, dist, basis = setup(((0, 1),), (1.,))
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    v = dist.Field(name='v', bases=basis, regularity='continuous')
    u['g'] = v['g'] = 0
    taus = [dist.Field(name=f'tau{j}', bases=basis.edges) for j in range(8)]
    lift_basis = basis.interval.derivative_basis(4)
    tau_u = sum(d3.Lift(t, lift_basis, -j-1) for j, t in enumerate(taus[:4]))
    tau_v = sum(d3.Lift(t, lift_basis, -j-1) for j, t in enumerate(taus[4:]))
    problem = d3.LBVP([u, v, *taus], namespace=locals())
    problem.add_equation('Hstar(H(u))+tau_u=24', test=u)
    problem.add_equation('Hstar(H(v))+tau_v=48', test=v)
    for name in ('u', 'v'):
        problem.add_equation(f'dtrace({name})=0', constraint=True)
        problem.add_equation(f'ntrace({name})=0', constraint=True)
    problem.build_solver().solve()
    x = basis.local_physical_grid(dist)
    np.testing.assert_allclose(u['g'], x*x*(1-x)**2, atol=2e-11)
    np.testing.assert_allclose(v['g'], 2*x*x*(1-x)**2, atol=2e-11)


def test_large_selection_uses_one_sparse_row_per_vertex():
    # Structural check only: no thousand-edge solve or global dense map.
    vertices = np.arange(1001)
    endpoints = np.column_stack((vertices[:-1], vertices[1:]))
    _, dist, basis = setup(endpoints, np.ones(1000))
    u = dist.Field(name='u', bases=basis)
    trace = d3.NeumannTrace(u, on=vertices[1:-1], sum=True)
    assert trace.mapping.matrix.shape == (999, 2000)
    assert trace.mapping.matrix.nnz == 1998
