"""Native graph interface checked against explicit equations and exact fields."""

import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3
from dedalus.core.network_operators import _NetworkLowerer


def context(dtype=np.float64, endpoints=((0, 1), (2, 1)), lengths=(1., 1.7), size=10):
    network = d3.Network(endpoints=endpoints, lengths=lengths)
    dist = d3.Distributor(network, dtype=dtype)
    return dist, d3.NetworkBasis(network, size=size)


def values(expression):
    if isinstance(expression, (int, float, complex)):
        return expression
    result = expression.evaluate()
    result.change_scales(1)
    return result['g'].copy()


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('backend', [None, 'dtn'])
def test_native_heat_matches_explicit_matrices_and_rhs(dtype, backend):
    d, b = context(dtype)
    u = d.Field(name='u', bases=b, regularity='continuous')
    u['g'] = 0
    tau1 = d.Field(name='tau1', bases=b.edges)
    tau2 = d.Field(name='tau2', bases=b.edges)
    K = d.Field(bases=b)
    x = b.local_physical_grid(d)
    K['g'] = np.array([[.7], [1.3]])*(1+.15*x+.2*x*x)
    ce = d.Field(bases=b.edges); ce['g'] = np.array([[1.1], [.6]])
    cv = d.Field(bases=b.dirac); cv['c'] = np.array([[1.4], [1.7], [2.1]])
    sv = d.Field(bases=b.dirac); sv['c'] = np.array([[.1], [-.3], [.2]])
    source = d.Field(bases=b); source['g'] = .4+x*x
    mass, heating = ce+cv, source+sv
    lb = b.interval.derivative_basis(2)
    native = d3.IVP([u, tau1, tau2], namespace=locals())
    native.add_equation('mass*dt(u)-div(K*grad(u))+lift(tau1,lb,-1)+lift(tau2,lb,-2)=heating')
    a = native.build_solver(d3.SBDF1, network_solver=backend)
    # Assemble the independent strong edge and vertex equations explicitly.
    vc = d.Field(bases=b.vertices); vc['g'] = cv['c']
    vs = d.Field(bases=b.vertices); vs['g'] = sv['c']
    explicit = d3.IVP([u, tau1, tau2], namespace=locals())
    U = explicit.vertex_field(u)
    dx, R, dtrace, ntrace = b.derivative, b.restriction, b.dtrace, b.ntrace
    explicit.namespace.maps.insert(1, locals())
    explicit.add_equation('ce*dt(u)-dx(K*dx(u))+lift(tau1,lb,-1)+lift(tau2,lb,-2)=source')
    explicit.add_equation('vc*dt(U)+R.T@(dtrace(K)*ntrace(u))=vs')
    c = explicit.build_solver(d3.SBDF1, network_solver=backend)
    rng = np.random.default_rng(172)
    for sa, sc in zip(a.subproblems, c.subproblems):
        for name in ('M_min', 'L_min'):
            ma, mc = getattr(sa, name), getattr(sc, name)
            np.testing.assert_allclose(ma.toarray(), mc.toarray(), rtol=0, atol=4e-12)
            z = rng.normal(size=ma.shape[1])
            if dtype == np.complex128:
                z = z+1j*rng.normal(size=z.size)
            np.testing.assert_allclose(ma@z, mc@z, rtol=0, atol=2e-11)
    for ea, ec in zip(native.equations, explicit.equations):
        np.testing.assert_allclose(values(ea['F']), values(ec['F']), rtol=0, atol=2e-13)


def test_native_nonlinear_point_reaction():
    d, b = context(size=8)
    u = d.Field(name='u', bases=b, regularity='continuous'); u['g'] = .96
    cv = d.Field(bases=b.dirac); cv['c'] = 2
    tau1 = d.Field(name='tau1', bases=b.edges)
    tau2 = d.Field(name='tau2', bases=b.edges)
    lb = b.interval.derivative_basis(2)
    p = d3.NLBVP([u, tau1, tau2], namespace=locals())
    p.add_equation('-div(grad(u))+(1+cv)*u**3+lift(tau1,lb,-1)+lift(tau2,lb,-2)=(1+cv)*1.2**3')
    solver = p.build_solver(network_solver='dtn')
    for _ in range(6):
        solver.newton_iteration()
    np.testing.assert_allclose(u['g'], 1.2, atol=2e-11)
    np.testing.assert_allclose(p.vertex_field(u)['g'], 1.2, atol=2e-11)


def test_initial_common_value_and_discontinuous_initial_data():
    d, b = context()
    u = d.Field(name='u', bases=b, regularity='continuous')
    u['g'] = np.array([[1.], [2.]])
    p = d3.IVP([u])
    with pytest.raises(ValueError, match='incompatible endpoint traces'):
        p._initialize_network_vertex(u)
    u['g'] = 3
    p._initialize_network_vertex(u)
    np.testing.assert_allclose(p.vertex_field(u)['g'], 3, atol=2e-14)
    assert u.copy().regularity == 'continuous'
    measure = d.Field(bases=b.dirac)
    assert measure.copy()._network_dirac is measure._network_dirac
    with pytest.raises(TypeError, match='common vertex value'):
        _NetworkLowerer(d, b).scalar(measure*d.Field(bases=b))
    with pytest.raises(TypeError, match='two distributions'):
        _NetworkLowerer(d, b).scalar(measure*measure)


def test_continuous_point_coefficient_uses_its_own_traces():
    d, b = context(size=8)
    u = d.Field(name='u', bases=b, regularity='continuous'); u['g'] = 0
    a = d.Field(name='a', bases=b, regularity='continuous'); a['g'] = 2
    # A previous binding must not leave a stale vertex coefficient in a new problem.
    previous = d3.LBVP([a]); previous.vertex_field(a)['g'] = 9
    point = d.Field(bases=b.dirac); point['c'] = 1
    tau1 = d.Field(name='tau1', bases=b.edges)
    tau2 = d.Field(name='tau2', bases=b.edges)
    lb = b.interval.derivative_basis(2)
    p = d3.LBVP([u, tau1, tau2], namespace=locals())
    p.add_equation('-div(grad(u))+u+point*a*u+lift(tau1,lb,-1)+lift(tau2,lb,-2)=3+6*point')
    solver = p.build_solver()
    solver.solve()
    np.testing.assert_allclose(u['g'], 3, atol=2e-11)
    a['g'] = np.array([[1.], [2.]])
    with pytest.raises(ValueError, match='incompatible endpoint traces'):
        p.build_solver()


@pytest.mark.parametrize('order', range(4))
def test_native_endpoint_dirac_product_pairing(order):
    d, b = context(endpoints=((0, 1),), lengths=(1.7,), size=12)
    f = d.Field(bases=b)
    fpoly = np.polynomial.Polynomial([1., 2., -.4, .6])
    phi = np.polynomial.Polynomial([.2, -.3, .4, .7])
    f['g'] = fpoly(b.local_physical_grid(d))
    a = d.Field(bases=b.endpoint_dirac(order)); a['c'] = np.array([[.7], [-1.1]])
    value = _NetworkLowerer(d, b).scalar(f*a)
    ends, normal = np.array([0., 1.7]), np.array([-1., 1.])
    actual = sum((-1)**k*np.sum(values(c).ravel()*normal**k*phi.deriv(k)(ends))
                 for k, c in value.endpoints.items())
    expected = (-1)**order*np.sum(np.array([.7, -1.1])*normal**order*(fpoly*phi).deriv(order)(ends))
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=3e-10)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_clamped_beam_exact_polynomial(dtype):
    d, b = context(dtype, endpoints=((0, 1),), lengths=(1.,), size=10)
    u = d.Field(name='u', bases=b, regularity='continuous'); u['g'] = 0
    # Prescribed endpoint displacement and slopes: reactions generated from C*.
    identity = d3.CardinalMap(b.ports, b.ports, sparse.eye(2))
    C0 = d3.TraceConstraint(b, {0: identity})
    C1 = d3.TraceConstraint(b, {1: identity})
    lam0 = d.Field(name='lam0', bases=C0.basis)
    lam1 = d.Field(name='lam1', bases=C1.basis)
    taus = [d.Field(name=f'tau{j}', bases=b.edges) for j in range(4)]
    lb = b.interval.derivative_basis(4)
    amplitude = 1+.3j if dtype == np.complex128 else 1.
    tau = sum(d3.Lift(t, lb, -j-1) for j, t in enumerate(taus))
    p = d3.LBVP([u, lam0, lam1, *taus], namespace=locals())
    p.add_equation('Hstar(H(u))+C0.adjoint(lam0)+C1.adjoint(lam1)+tau=24*amplitude')
    p.add_equation((C0(u), 0))
    p.add_equation((C1(u), 0))
    s = p.build_solver()
    s.solve()
    x = b.local_physical_grid(d)
    np.testing.assert_allclose(u['g'], amplitude*x*x*(1-x)**2, atol=2e-11)
    np.testing.assert_allclose(values(C1(u)), 0, atol=2e-11)
    roles = [e.get('network_role') for e in p.equations]
    assert roles.count('vertex') == roles.count('endpoint_1') == 1


def test_complex_coupled_trace_constraint_adjoint():
    d, b = context(np.complex128, endpoints=((0, 1), (0, 2), (0, 3)), lengths=(1., 1.2, .7))
    f = d.Field(bases=b)
    x = b.local_physical_grid(d)
    f['g'] = (1+.3j)*x*x + .2j*x + np.array([[.2], [.5], [-.1]])
    # Coupled slopes and values, using one native coefficient space for rows.
    rng = np.random.default_rng(342)
    maps = {k: d3.CardinalMap(b.ports, b.vertices,
                             rng.normal(size=(4, 6))+1j*rng.normal(size=(4, 6)))
            for k in (0, 1)}
    C = d3.TraceConstraint(b, maps)
    lam = d.Field(bases=C.basis); lam['g'] = rng.normal(size=(4, 1))+1j*rng.normal(size=(4, 1))
    reaction = _NetworkLowerer(d, b).scalar(C.adjoint(lam))
    actual = sum((-1)**k*np.vdot(values(d3.EndpointTrace(f, k)).ravel(), values(c).ravel())
                 for k, c in reaction.endpoints.items())
    np.testing.assert_allclose(actual, np.vdot(values(C(f)).ravel(), lam['g'].ravel()), atol=2e-12)


def test_direction_is_explicit_tangent_data():
    d, b = context()
    u = d.Field(bases=b)
    x = b.local_physical_grid(d)
    u['g'] = x*x
    E = d.VectorField(b.network.tangent, bases=b.edges)
    E['g'][0] = np.array([[1.], [-1.]])
    np.testing.assert_allclose(values(E @ d3.Gradient(u)), np.array([[1.], [-1.]])*2*x, atol=2e-12)
    assert d3.Gradient(u).tensorsig == (b.network.tangent,)
    assert d3.Gradient(u).evaluate()['g'].shape[0] == 1
    expression = d3.DotProduct(E, d3.Gradient(u))
    np.testing.assert_allclose(values(_NetworkLowerer(d, b).scalar(expression).edge),
                               np.array([[1.], [-1.]])*2*x, atol=2e-12)


@pytest.mark.parametrize('build_ivp_first', [False, True])
def test_native_heat_linearization_preserves_matching(build_ivp_first):
    d, b = context(np.complex128, endpoints=((0, 1),), lengths=(np.pi,), size=20)
    u = d.Field(name='u', bases=b, regularity='continuous'); u['g'] = 0
    tau1 = d.Field(name='tau1', bases=b.edges)
    tau2 = d.Field(name='tau2', bases=b.edges)
    lb = b.interval.derivative_basis(2)
    p = d3.IVP([u, tau1, tau2], namespace=locals())
    p.add_equation('dt(u)-div(grad(u))+lift(tau1,lb,-1)+lift(tau2,lb,-2)=0')
    if build_ivp_first:
        p.build_solver(d3.SBDF1)
    evp = p.build_EVP()
    s = evp.build_solver()
    s.solve_dense(s.subproblems[0])
    finite = s.eigenvalues[np.isfinite(s.eigenvalues)]
    np.testing.assert_allclose(np.sort(finite.real)[-3:], [-4., -1., 0.], atol=2e-7)
    assert len(evp.equations) == 3


def test_native_field_semantics_survive_output_and_restart(tmp_path):
    import h5py
    from dedalus.core.evaluator import Evaluator
    d, b = context()
    u = d.Field(name='u', bases=b, regularity='continuous'); u['g'] = 2
    cv = d.Field(name='cv', bases=(b.dirac,)); cv['c'] = np.array([[1.], [2.], [3.]])
    derivative = d.Field(name='derivative', bases=b.endpoint_dirac(2)); derivative['c'] = 4
    evaluator = Evaluator(d, {})
    handler = evaluator.add_file_handler(tmp_path/'native', parallel='gather', mode='overwrite')
    for f in (u, cv, derivative):
        handler.add_task(f, name=f.name, layout='c')
    evaluator.evaluate_handlers([handler], iteration=0)
    with h5py.File(handler.current_file, 'r') as file:
        assert file['tasks/cv'].attrs['network_field_kind'] == 'vertex_dirac'
        restored = d.Field(bases=b.dirac)
        restored.load_from_hdf5(file, 0, task='cv')
        np.testing.assert_array_equal(restored['c'], cv['c'])
        with pytest.raises(ValueError, match='semantics'):
            d.Field(bases=b.vertices).load_from_hdf5(file, 0, task='cv')
        with pytest.raises(ValueError, match='derivative order'):
            d.Field(bases=b.endpoint_dirac(1)).load_from_hdf5(file, 0, task='derivative')
        with pytest.raises(ValueError, match='regularity'):
            d.Field(bases=b).load_from_hdf5(file, 0, task='u')


def test_existing_tensor_trace_is_preserved():
    coords = d3.CartesianCoordinates('x', 'y')
    d = d3.Distributor(coords, dtype=np.float64)
    tensor = d.TensorField(coords)
    tensor['g'][0, 0] = 2
    tensor['g'][1, 1] = 3
    np.testing.assert_allclose(d3.Trace(tensor).evaluate()['g'], 5)


@pytest.mark.parametrize('backend', [None, 'dtn'])
def test_branching_beam_with_coupled_slopes(backend):
    lengths = np.array([1., 1.2, .7])
    weights = np.array([1., .5, 1.3])
    d, b = context(endpoints=((0, 1), (0, 2), (0, 3)), lengths=lengths, size=12)
    u = d.Field(name='u', bases=b, regularity='continuous'); u['g'] = 0
    R = b.network.restriction_matrix
    C0 = d3.TraceConstraint(b, {0: d3.CardinalMap(b.ports, b.vertices,
                                               sparse.diags([1/3, 1, 1, 1]) @ R.T)})
    slope_map = np.zeros((4, 6))
    slope_map[0, ::2] = weights
    slope_map[np.arange(1, 4), [1, 3, 5]] = 1
    C1 = d3.TraceConstraint(b, {1: d3.CardinalMap(b.ports, b.vertices, slope_map)})
    force = d.Field(name='force', bases=C0.basis)
    moment = d.Field(name='moment', bases=C1.basis)
    taus = [d.Field(name=f'tau{j}', bases=b.edges) for j in range(4)]
    tau = sum(d3.Lift(t, b.interval.derivative_basis(4), -j-1) for j, t in enumerate(taus))
    source = d.Field(bases=b.edges); source['g'] = (24*weights/lengths**2)[:, None]
    p = d3.LBVP([u, force, moment, *taus], namespace=locals())
    p.add_equation('Hstar(H(u))+C0.adjoint(force)+C1.adjoint(moment)+tau=source')
    p.add_equation('C0(u)=0')
    p.add_equation('C1(u)=0')
    solver = p.build_solver(network_solver=backend)
    solver.solve()
    x = b.local_physical_grid(d)
    s = x/lengths[:, None]
    exact = (weights*lengths**2)[:, None]*s*s*(1-s)**2
    np.testing.assert_allclose(u['g'], exact, atol=3e-11)
    np.testing.assert_allclose(values(d3.EndpointTrace(d3.Hessian(u)))[::2, 0],
                               2*weights, atol=3e-10)


def test_first_order_beam_ivp_exact_mode():
    d, b = context(endpoints=((0, 1),), lengths=(1.,), size=24)
    u = d.Field(name='u', bases=b, regularity='continuous')
    # Velocity is tied to displacement by the exact kinematic edge equation;
    # no duplicate displacement-matching constraint is imposed on that equation.
    v = d.Field(name='v', bases=b)
    C0 = d3.TraceConstraint(b, {0: d3.CardinalMap(b.ports, b.ports, sparse.eye(2))})
    force = d.Field(name='force', bases=C0.basis)
    taus = [d.Field(name=f'tau{j}', bases=b.edges) for j in range(4)]
    tau = sum(d3.Lift(t, b.interval.derivative_basis(4), -j-1) for j, t in enumerate(taus))
    p = d3.IVP([u, v, force, *taus], namespace=locals())
    p.add_equation('dt(u)-v=0')
    p.add_equation('dt(v)+Hstar(H(u))+C0.adjoint(force)+tau=0', test=u)
    p.add_equation('C0(u)=0')
    x = b.local_physical_grid(d)
    u['g'] = np.sin(np.pi*x); v['g'] = 0
    solver = p.build_solver(d3.SBDF1)
    dt = .001
    solver.step(dt)
    factor = 1/(1+dt*dt*np.pi**4)
    np.testing.assert_allclose(u['g'], factor*np.sin(np.pi*x), atol=2e-10)
    np.testing.assert_allclose(v['g'], -dt*np.pi**4*factor*np.sin(np.pi*x), atol=2e-9)
