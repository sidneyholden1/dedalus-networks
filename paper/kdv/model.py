"""Geometry, initialization and diagnostics for the native ordinary-KdV example.

The physical equation and crossing laws are declared in evolve.py. The legacy
assembly is retained in explicit_reference.py solely for the translation audit.
The native physical-L2 assembly currently supports serial execution.
"""
from functools import lru_cache
import numpy as np
from mpi4py import MPI
from numpy.polynomial.legendre import legvander, leggauss, legder
from numpy.polynomial import Polynomial, Legendre
import dedalus.public as d3
from linear_solvers import BlockEquilibratedLU, constrain_stage_rhs
from conditions import constraints, boundary_budget, periodic_geometry


@lru_cache(maxsize=16)
def legendre_projection(n, nq):
    q, weights = leggauss(nq)
    V = legvander(q, n-1)
    projection = weights[:, None]*V*(2*np.arange(n)+1)[None, :]/2
    return q, weights, V, projection


def prepare(n=24, periodic_size=3, comm=None):
    """Build the metric graph and initialize the paper's compact polynomial."""
    if n < 18:
        raise ValueError('The initial polynomial requires at least 18 modes')
    comm = MPI.COMM_WORLD if comm is None else comm
    if comm.size != 1:
        raise NotImplementedError('The native physical-L2 example is currently serial')
    xy, endpoints, lengths, directions = periodic_geometry(periodic_size)
    network = d3.Network(endpoints=endpoints, lengths=lengths)
    dist = d3.Distributor(network, dtype=np.float64, comm=comm)
    basis = d3.NetworkBasis(network, size=n, interval_basis=d3.Legendre, dealias=1.5)
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    t = dist.Field(name='t')
    E = dist.VectorField(network.tangent, name='E', bases=basis.edges)
    E['g'][0] = 1
    horizontal = dist.Field(name='horizontal', bases=basis.edges)
    vertical = dist.Field(name='vertical', bases=basis.edges)
    edge, s = basis.local_grids(dist, scales=(1, 1))
    indices = edge.astype(int).ravel()
    horizontal['g'] = directions[indices, 0, None]
    vertical['g'] = directions[indices, 1, None]
    source_edge = (periodic_size//2)*periodic_size
    initial = (Polynomial([1., 0., -1.])**8).convert(kind=Legendre).coef
    u['c'] = 0
    u['c'][indices == source_edge, :len(initial)] = initial/np.sqrt((2*np.arange(len(initial))+1)/2)
    matrices, rows = constraints(xy, endpoints, lengths, directions)
    return dict(dist=dist, basis=basis, u=u, t=t, E=E, horizontal=horizontal,
        vertical=vertical, beta=1., xy=xy, endpoints=endpoints, lengths=lengths,
        edge_directions=directions, periodic_size=periodic_size, source_edge=source_edge,
        matrices=matrices, rows=rows, local_edges=indices, s=s.ravel(), exact=[],
        manufactured=False)


def finish(model, problem):
    """Build the unchanged RKGFY integrator and independent constrained rate."""
    from dedalus.core.timesteppers import RKGFY
    dist, basis, u, E, horizontal, vertical = (
        model[key] for key in ('dist', 'basis', 'u', 'E', 'horizontal', 'vertical'))
    options = dict(matsolver=BlockEquilibratedLU, network_solver='dtn',
                   network_separator_method='replicated_direct', ncc_cutoff=0, entry_cutoff=0)
    solver = problem.build_solver(RKGFY, enforce_real_cadence=0, **options)
    active = [sp for sp in solver.subproblems if sp.size]
    if len(active) != 1:
        raise RuntimeError('Expected one coupled subproblem')
    sp = active[0]
    indicators = []
    for j, f in enumerate(solver.F):
        marker = f.copy(); marker.change_scales(1); marker['c'] = float(j == 0)
        indicators.append(marker)
    zero_rows = ~np.any(sp.gather_outputs(indicators) != 0, axis=1)
    if sp.M_min[zero_rows].nnz:
        raise RuntimeError('Constraint mask touched a differential row')
    constrain_stage_rhs(solver, zero_rows)
    # The instantaneous rate belongs to the same constrained space as u.
    ut = dist.Field(name='ut', bases=basis, regularity='continuous')
    w = E @ d3.grad(E @ d3.grad(u))
    rate_lhs = d3.Convert(ut, basis.interval.derivative_basis(3))
    rp = d3.LBVP([ut], namespace=locals())
    rp.add_equation('rate_lhs = -6*u*(E@grad(u)) - E@grad(w)', tau='adjoint')
    rp.add_vertex_constraint('endpoint(horizontal*ut, deriv=1, sum=True) = 0')
    rp.add_vertex_constraint('endpoint(vertical*ut, deriv=1, sum=True) = 0')
    rp.add_vertex_constraint('endpoint(E@grad(ut), deriv=1, sum=True) = 0')
    rate_solver = rp.build_solver(**options)
    model.update(problem=problem, solver=solver, ut=ut, rate_solver=rate_solver)


def capture(model):
    """Independent quadrature and unscaled trace diagnostics, including c_e=0."""
    b = model['basis']; u = model['u']; n = b.size
    u.change_scales(1)
    raw_u = u.allgather_data('c').copy()
    try:
        model['rate_solver'].solve()
    finally:
        # Evaluating the nonlinear rate changes the shared field's layout.
        # Diagnostics must leave the time-integrator's coefficients intact.
        u.change_scales(1); u.change_layout('c'); u.set_global_data(raw_u)
    model['ut'].change_scales(1)
    # Dedalus uses orthonormal Legendre modes on [-1,1]. Read coefficients
    # directly: an additional grid/quadrature round trip contaminates high
    # derivatives with avoidable tiny high-mode errors.
    normalization = np.sqrt((2*np.arange(n)+1)/2)
    cu = raw_u*normalization
    ct = model['ut'].allgather_data('c')*normalization
    polynomials = [cu]
    for order in range(1, 4):
        polynomials.append(legder(cu, m=order, axis=-1)*(2/model['lengths'][:, None])**order)
    polynomials.append(ct)
    q, qw, V, _ = legendre_projection(n, 2*n+1)
    Ve = legvander(np.array([-1., 1.]), n-1)
    arrays = [c@V[:, :c.shape[-1]].T for c in polynomials]
    traces = [(c@Ve[:, :c.shape[-1]].T).ravel() for c in polynomials]
    uq, uxq, wq, uxxxq, utq = arrays
    weights = model['lengths'][:, None]*qw[None, :]/2
    integrate = lambda x: float(np.sum(x*weights))
    budget = boundary_budget(traces[:3], model['endpoints'], model['beta'])
    residual = utq+uxxxq+6*model['beta']*uq*uxq
    source = np.zeros_like(uq)
    if model['manufactured']:
        from numpy.polynomial import Polynomial
        h = Polynomial([1., 0., -1.])**4
        e = model['source_edge']; a = 2/model['lengths'][e]
        now = float(model['solver'].sim_time)
        source[e] = (-h(q)*np.sin(now)+a**3*h.deriv(3)(q)*np.cos(now)
                     +6*model['beta']*h(q)*a*h.deriv(1)(q)*np.cos(now)**2)
        residual -= source
    source_power = integrate(uq*source)
    scale = max(integrate(utq**2)+integrate(uxxxq**2)+
                integrate((6*model['beta']*uq*uxq)**2), 1e-30)
    constraint = sum(C@tr for C, tr in zip(model['matrices'], traces[:3]))
    rate = integrate(uq*utq)
    record = dict(t=float(model['solver'].sim_time), iteration=model['solver'].iteration,
        u_min=float(uq.min()), u_max=float(uq.max()), mass=integrate(uq),
        squared_u_l2=integrate(uq**2), squared_w_l2=integrate(wq**2),
        squared_u_l2_half_rate=rate,
        mass_rate=integrate(utq),
        mass_balance_residual=integrate(utq)-budget['mass_boundary_rate']-integrate(source),
        source_power=source_power,
        l2_balance_residual=rate-budget['l2_boundary_power']-source_power,
        l2_balance_relative=abs(rate-budget['l2_boundary_power']-source_power)/max(
            1.,abs(rate),abs(budget['l2_boundary_power']),abs(source_power)),
        constraint_max=float(np.max(abs(constraint))),
        edge_equation_relative_l2=float(np.sqrt(integrate(residual**2)/scale)),
        edge_constants=(np.sum(residual*weights, axis=-1)/model['lengths']).tolist(),
        **budget)
    if model['exact']:
        reference = np.zeros_like(uq)
        reference[model['source_edge']] = np.cos(record['t'])*(1-q*q)**4
        record['u_exact_error'] = float(np.max(abs(uq-reference)))
    # Output is sampled on Chebyshev roots for compatibility with the audited
    # comparison/3D tools. This is exact evaluation of the Legendre polynomial.
    sample_s = .5*(1-np.cos(np.pi*(np.arange(n)+.5)/n))
    sample_V = legvander(2*sample_s-1, n-1)
    saved = [c@sample_V[:, :c.shape[-1]].T for c in (cu, polynomials[2], ct)]
    if not all(np.isfinite(a).all() for a in saved):
        raise FloatingPointError('Nonfinite fields')
    return record, saved, sample_s
