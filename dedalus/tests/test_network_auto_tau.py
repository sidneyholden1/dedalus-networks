"""Automatic network taus against explicit discretizations and exact solutions."""

import numpy as np
import pytest
import dedalus.public as d3


def setup(dtype=np.float64, lengths=(1., 1.7), size=12):
    network = d3.Network(endpoints=((0, 1), (2, 1)), lengths=lengths)
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=size)
    return dist, basis


def problem(dist, basis, kind, order, manual=False, **kwargs):
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    u['g'] = 0
    if manual:
        U = dist.Field(name='U', bases=basis.vertices)
        taus = [dist.Field(name=f'tau_{j}', bases=basis.edges) for j in range(order)]
        # Match the automatic variable ordering, without relying on its assembly.
        p = kind([u, U, *taus], network_interfaces=[(u, U)], **kwargs)
        tau = sum(d3.Lift(t, basis.interval.derivative_basis(order), -j-1)
                  for j, t in enumerate(taus))
    else:
        p = kind([u], **kwargs)
        tau = 0
    p.namespace.maps.insert(1, dict(u=u, tau=tau))
    return u, p


def compare_matrices(a, b, names):
    for sa, sb in zip(a.subproblems, b.subproblems):
        for name in names:
            np.testing.assert_allclose(getattr(sa, name).toarray(),
                                       getattr(sb, name).toarray(), atol=3e-11, rtol=0)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('backend', [None, 'dtn'])
def test_heat_matrices_rhs_and_evolution_match_manual(dtype, backend):
    dist, basis = setup(dtype)
    x = basis.local_physical_grid(dist)
    K = dist.Field(bases=basis)
    K['g'] = np.array([[.7], [1.3]])*(1+.15*x+.2*x*x)
    cv = dist.Field(bases=basis.dirac); cv['c'] = np.array([[1.4], [1.7], [2.1]])
    source = dist.Field(bases=basis); source['g'] = .4+x*x
    solvers, problems, fields = [], [], []
    for manual in (False, True):
        u, p = problem(dist, basis, d3.IVP, 2, manual)
        p.namespace.maps.insert(1, dict(K=K, cv=cv, source=source))
        p.add_equation('(1+cv)*dt(u)-div(K*grad(u))+tau=source-.1*u*u')
        u['g'] = 1
        solvers.append(p.build_solver(d3.SBDF2, network_solver=backend))
        problems.append(p)
        fields.append(u)
    assert len(problems[0].tau_fields) == 2
    assert problems[1].tau_fields == ()
    compare_matrices(*solvers, ('M_min', 'L_min'))
    for s in solvers:
        for _ in range(3):
            s.step(.001)
    np.testing.assert_allclose(fields[0]['g'], fields[1]['g'], atol=3e-12, rtol=0)
    np.testing.assert_allclose(problems[0].vertex_field(fields[0])['g'],
                               problems[1].vertex_field(fields[1])['g'], atol=3e-12)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('backend', [None, 'dtn'])
@pytest.mark.parametrize('constraint_syntax', ['method', 'flag', 'legacy'])
def test_beam_matrices_reactions_and_exact_solution(dtype, backend, constraint_syntax):
    dist, basis = setup(dtype)
    amplitude = 1 + .4j if np.issubdtype(dtype, np.complexfloating) else 1.
    solvers, fields, problems = [], [], []
    for manual in (False, True):
        u, p = problem(dist, basis, d3.LBVP, 4, manual)
        p.namespace.maps.insert(1, dict(amplitude=amplitude))
        equation = 'Hstar(H(u))+tau=24*amplitude' if manual else 'H.T(H(u))=24*amplitude'
        p.add_equation(equation)
        for order in (0, 1):
            equation = f'endpoint(u, deriv={order})=0'
            if manual or constraint_syntax == 'legacy':
                p.add_equation(equation, constraint=True)
            elif constraint_syntax == 'method':
                p.add_vertex_constraint(equation)
            else:
                p.add_equation(equation, vertex_constraint=True)
        solvers.append(p.build_solver(network_solver=backend))
        fields.append(u); problems.append(p)
    x = basis.local_physical_grid(dist)
    length = np.array([[1.], [1.7]])
    for s, u in zip(solvers, fields):
        s.solve()
        np.testing.assert_allclose(u['g'], amplitude*x*x*(length-x)**2, atol=3e-11, rtol=0)
    compare_matrices(*solvers, ('L_min',))
    for a, b in zip(problems[0]._hidden_network_constraint_fields,
                    problems[1]._hidden_network_constraint_fields):
        np.testing.assert_allclose(a['c'], b['c'], atol=3e-11, rtol=0)
    p = problems[0]
    taus = p.tau_fields
    assert len(taus) == 4 and all(isinstance(t, d3.Field) for t in taus)
    assert p.network_lowered_equations[0][0]['network_tau_fields'] == taus
    p.build_solver(network_solver=backend).solve()
    assert p.tau_fields == taus


def test_nonlinear_problem_extends_newton_variables():
    dist, basis = setup(size=8)
    u, p = problem(dist, basis, d3.NLBVP, 2)
    cv = dist.Field(bases=basis.dirac); cv['c'] = 2
    p.namespace.maps.insert(1, dict(cv=cv))
    u['g'] = .96
    p.add_equation('-div(grad(u))+(1+cv)*u**3=(1+cv)*1.2**3')
    assert len(p.variables) == len(p.perturbations) == 4
    solver = p.build_solver(network_solver='dtn')
    for _ in range(6):
        solver.newton_iteration()
    np.testing.assert_allclose(u['g'], 1.2, atol=2e-11)


@pytest.mark.parametrize('build_first', [False, True])
@pytest.mark.parametrize('custom_perturbation', [False, True])
def test_linearized_evp_keeps_taus_hidden(build_first, custom_perturbation):
    dist, basis = setup(np.complex128, lengths=(np.pi/2, np.pi/2), size=16)
    u, p = problem(dist, basis, d3.IVP, 2)
    p.add_equation('dt(u)-div(grad(u))=0')
    if build_first:
        p.build_solver(d3.SBDF1)
    options = {}
    if custom_perturbation:
        options['perturbations'] = [dist.Field(name='du', bases=basis,
                                             regularity='continuous')]
    evp = p.build_EVP(**options)
    assert len(evp.tau_fields) == 2
    s = evp.build_solver()
    s.solve_dense(s.subproblems[0])
    finite = s.eigenvalues[np.isfinite(s.eigenvalues)]
    np.testing.assert_allclose(np.sort(finite.real)[-3:], [-4, -1, 0], atol=2e-7)


def test_custom_tau_opt_out_and_existing_lifts():
    dist, basis = setup()
    u, p = problem(dist, basis, d3.LBVP, 2)
    p.add_equation('-div(grad(u))+u=0', tau=False)
    assert p.tau_fields == ()
    u, explicit = problem(dist, basis, d3.LBVP, 2, manual=True)
    explicit.add_equation('-div(grad(u))+u+tau=1')
    explicit.build_solver().solve()
    assert explicit.tau_fields == ()
    np.testing.assert_allclose(u['g'], 1, atol=2e-12)


def test_failed_lowering_rolls_back_automatic_fields():
    dist, basis = setup()
    u, p = problem(dist, basis, d3.LBVP, 2)
    original = tuple(p.variables)
    with pytest.raises(Exception, match='linear'):
        p.add_equation('-div(grad(u))+u*u=1')
    assert tuple(p.variables) == original
    assert p.tau_fields == () and p.equations == []
    p.add_equation('-div(grad(u))+u=1')
    p.build_solver().solve()
    np.testing.assert_allclose(u['g'], 1, atol=2e-12)


def test_differentiated_coefficient_does_not_increase_tau_order():
    dist, basis = setup()
    u, p = problem(dist, basis, d3.LBVP, 2)
    K = dist.Field(bases=basis)
    x = basis.local_physical_grid(dist)
    K['g'] = 1+x**4
    coefficient = d3.Hessian(d3.Hessian(K))
    p.namespace.maps.insert(1, dict(coefficient=coefficient))
    p.add_equation('-div(grad(u))+coefficient*u=24')
    assert len(p.tau_fields) == 2
    p.build_solver().solve()
    np.testing.assert_allclose(u['g'], 1, atol=3e-11)


def test_mixed_fields_get_separate_tau_groups():
    dist, basis = setup()
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    v = dist.Field(name='v', bases=basis, regularity='continuous')
    u['g'] = v['g'] = 0
    p = d3.LBVP([u, v], namespace=locals())
    p.add_equation('-div(grad(u))+2*u-v=0', test=u)
    p.add_equation('-div(grad(v))+2*v-u=3', test=v)
    assert len(p.tau_fields) == 4
    assert len({t.name for t in p.tau_fields}) == 4
    p.build_solver(network_solver='dtn').solve()
    np.testing.assert_allclose(u['g'], 1, atol=3e-12)
    np.testing.assert_allclose(v['g'], 2, atol=3e-12)


def test_generated_taus_survive_checkpoint_and_restart(tmp_path):
    dist, basis = setup(size=8)

    def build():
        u, p = problem(dist, basis, d3.IVP, 2)
        p.add_equation('dt(u)-div(grad(u))=-.1*u*u')
        u['g'] = 1
        return u, p, p.build_solver(d3.SBDF1)

    u, p, solver = build()
    solver.step(.001)
    checkpoint = solver.evaluator.add_file_handler(tmp_path/'checkpoint', parallel='gather')
    checkpoint.add_tasks(solver.state, layout='c')
    solver.evaluator.evaluate_handlers([checkpoint], iteration=solver.iteration,
        sim_time=solver.sim_time, timestep=.001, wall_time=0)
    v, restarted, other = build()
    assert [t.name for t in p.tau_fields] == [t.name for t in restarted.tau_fields]
    other.load_state(checkpoint.current_file)
    for a, b in zip(solver.state, other.state):
        np.testing.assert_allclose(a['c'], b['c'], atol=1e-14, rtol=0)
    solver.step(.001)
    other.step(.001)
    np.testing.assert_allclose(u['g'], v['g'], atol=2e-13, rtol=0)
