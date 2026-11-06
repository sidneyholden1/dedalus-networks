"""Physical-L2 ordinary KdV assembly used for the manuscript application.

The six constraints at each periodic crossing and their adjoints are explicit.
The native third-order graph compiler is not substituted for this discretization.
"""
from functools import lru_cache
import numpy as np
from mpi4py import MPI
from scipy import sparse
from numpy.polynomial.legendre import legval, legvander, leggauss, legder
from numpy.polynomial import Polynomial, Legendre
import dedalus.public as d3
from dedalus.core.basis import CardinalBasis
from linear_solvers import EquilibratedLU, geometry
from conditions import constraints, boundary_budget, periodic_geometry


@lru_cache(maxsize=16)
def legendre_projection(n, nq):
    q, weights = leggauss(nq)
    V = legvander(q, n-1)
    projection = weights[:, None]*V*(2*np.arange(n)+1)[None, :]/2
    return q, weights, V, projection



def build(n=48, beta=1., lead=32., lead_panels=4, manufactured=False,
          mesh=None, source_edge=2, periodic_size=0, *, vertex_interface=False,
          comm=None, network_partition=None, network_solver='auto'):
    comm = MPI.COMM_WORLD if comm is None else comm
    if comm.size != 1 and not vertex_interface:
        raise ValueError('This small-grid implementation is serial; pack independent checks')
    if vertex_interface and not periodic_size:
        raise ValueError('The grouped vertex interface currently requires a periodic grid')
    if network_partition is not None and not vertex_interface:
        raise ValueError('Distributed storage requires the grouped vertex interface')
    if n < 12 or (not manufactured and n < 18):
        raise ValueError('Insufficient basis for the specified initial polynomial')
    if periodic_size:
        if mesh is not None:
            raise ValueError('Do not combine a custom mesh with a periodic grid')
        xy, endpoints, lengths, directions = periodic_geometry(periodic_size)
        source_edge = (periodic_size//2)*periodic_size
    else:
        xy, endpoints, lengths = geometry(lead=lead, lead_panels=lead_panels) if mesh is None else mesh
        directions = (xy[endpoints[:, 1]]-xy[endpoints[:, 0]]) / lengths[:, None]
    network = d3.Network(endpoints=endpoints, lengths=lengths)
    dist = d3.Distributor(network, dtype=np.float64, comm=comm,
                          network_partition=network_partition, network_vertex_storage='auto')
    basis = d3.NetworkBasis(network, size=n, interval_basis=d3.Legendre, dealias=1.5)
    matrices, rows = constraints(xy, endpoints, lengths, directions)
    if vertex_interface:
        # Reorder the SAME constraints into six ordinary vertex fields. This
        # exposes the interface to Dedalus without altering any row or adjoint.
        by_vertex = [[] for _ in range(network.num_vertices)]
        for j, row in enumerate(rows):
            by_vertex[row['vertex']].append(j)
        if any(len(indices) != 6 for indices in by_vertex):
            raise ValueError('Expected six constraints at every periodic crossing')
        row_groups = np.asarray(by_vertex, dtype=int).T
        cb = basis.vertices
    else:
        row_groups = np.arange(len(rows))[None, :]
        cb = CardinalBasis(network.edge_coord, size=len(rows))
    # Scale each constraint and its adjoint by the same factor.
    scales = np.array([(float(np.min(lengths))/n**2)**r['order'] for r in rows])
    map_groups = [{j: d3.CardinalMap(basis.ports, cb,
                    sparse.diags(scales[group])@C[group], name=f'C{i}_{j}')
                   for j, C in enumerate(matrices) if C[group].nnz}
                  for i, group in enumerate(row_groups)]
    u = dist.Field(name='u', bases=basis)
    def multipliers(prefix):
        return [dist.Field(name=f'{prefix}_{i}' if vertex_interface else prefix, bases=cb)
                for i in range(len(row_groups))]
    multiplier = multipliers('multiplier')
    dx = basis.derivative
    uxxx = dx(dx(dx(u)))
    edge, s = basis.local_grids(dist, scales=(1, 1))
    indices = edge.astype(int).ravel()
    z = 2*s-1
    k = np.arange(n, dtype=float)
    derivatives_at_one = (np.ones(n), k*(k+1)/2,
                          (k-1)*k*(k+1)*(k+2)/8)
    left, right = basis._left_trace.T, basis._right_trace.T
    kernels = []
    for derivative in range(3):
        side_kernels = []
        for side in (-1, 1):
            coefficients = (2*k+1)*derivatives_at_one[derivative]
            if side == -1:
                coefficients *= (-1.)**(k+derivative)
            f = dist.Field(name=f'r{derivative}_{side}', bases=basis)
            f['g'] = legval(z, coefficients)*(2/lengths[indices, None])**derivative / lengths[indices, None]
            side_kernels.append(f)
        kernels.append(side_kernels)

    def adjoint(lam):
        terms = []
        for maps, coefficient in zip(map_groups, lam):
            for order, C in maps.items():
                kl, kr = kernels[order]
                port = C.T @ coefficient
                terms.append(kl*(left@port) + kr*(right@port))
        return sum(terms)

    def condition(field):
        return [sum(C @ d3.endpoint(field if j == 0 else
                    dx(field) if j == 1 else dx(dx(field))) for j, C in maps.items())
                for maps in map_groups]

    penalty = adjoint(multiplier)
    boundary = condition(u)
    t = dist.Field(name='t')
    exact = []
    h = None
    if manufactured:
        p = Polynomial([0., 1.]); h = 256*p**4*(1-p)**4
        for poly in (h, h.deriv(1)/lengths[source_edge],
                     h.deriv(3)/lengths[source_edge]**3):
            f = dist.Field(bases=basis)
            f['g'] = poly(s)*(indices[:, None] == source_edge)
            exact.append(f)
    # Initialize the known polynomial directly. Avoid creating high-mode
    # transform noise that is amplified by endpoint second derivatives.
    initial = (Polynomial([1., 0., -1.])**(4 if manufactured else 8)).convert(kind=Legendre).coef
    u['c'] = 0
    local_source = np.flatnonzero(indices == source_edge)
    u['c'][local_source, :len(initial)] = initial/np.sqrt((2*np.arange(len(initial))+1)/2)
    def projected_rhs(field, time_field):
        values = field.data
        q, _, V, project = legendre_projection(n, values.shape[-1])
        coefficients = values@project
        derivative = legder(coefficients, axis=-1)*(2/lengths[indices, None])
        physical_rhs = -6*beta*values*(derivative@V[:, :n-1].T)
        if manufactured:
            now = float(time_field.data.ravel()[0]); coordinate = (q+1)/2
            hv = h(coordinate); hx = h.deriv(1)(coordinate)/lengths[source_edge]
            hxxx = h.deriv(3)(coordinate)/lengths[source_edge]**3
            physical_rhs[local_source] += (-hv*np.sin(now)+hxxx*np.cos(now)
                                          +6*beta*hv*hx*np.cos(now)**2)
        # Dedalus differentiates into higher Jacobi parameters. Project in
        # physical L2 first, then convert that polynomial upward for assembly.
        # Otherwise projection of the RHS in Jacobi(3,3) breaks the L2 identity.
        return (physical_rhs@project)@V.T

    rhs = d3.GeneralFunction(dist, u.domain, tensorsig=(), dtype=np.float64,
                             layout='g', func=projected_rhs, args=[u, t])
    problem = d3.IVP([u, *multiplier], time=t, namespace=locals())
    problem.add_equation('dt(u) + uxxx + penalty = rhs')
    # Share one explicitly named zero leaf. Six implicitly generated unnamed
    # vertex RHS fields are ambiguous to the owned/ghost input synchronizer.
    zero_boundary = dist.Field(name='zero_boundary', bases=cb) if vertex_interface else 0
    for bc in boundary:
        problem.add_equation((bc, zero_boundary))
    from dedalus.core.timesteppers import RKGFY
    if vertex_interface:
        from linear_solvers import BlockEquilibratedLU, constrain_stage_rhs
        matrix_solver = BlockEquilibratedLU
    else:
        matrix_solver = EquilibratedLU
    # Preserve direct, refined solves for this small interface. The automatic
    # large-interface GMRES default has not been qualified for this KdV DAE.
    separator_options = dict(network_separator_method='replicated_direct') if vertex_interface else {}
    solver = problem.build_solver(RKGFY, matsolver=matrix_solver, network_solver=network_solver,
                                  ncc_cutoff=0, entry_cutoff=0,
                                  enforce_real_cadence=0, **separator_options)
    # All bases and coefficients are real: there are no Fourier Hermitian
    # pairs to repair. The generic c->g->c symmetry pass adds endpoint noise.
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
    if vertex_interface:
        constrain_stage_rhs(solver, zero_rows)
    else:
        solver._kdv_zero_rhs_rows = zero_rows
    # Independent instantaneous constrained rate, used at diagnostic frames only.
    ut = dist.Field(name='ut', bases=basis)
    rate_multiplier = multipliers('rate_multiplier')
    rate_penalty, rate_boundary = adjoint(rate_multiplier), condition(ut)
    rate_lhs = d3.Convert(ut+rate_penalty, basis.interval.derivative_basis(3))
    rp = d3.LBVP([ut, *rate_multiplier], namespace=locals())
    rp.add_equation('rate_lhs = rhs - uxxx')
    for bc in rate_boundary:
        rp.add_equation((bc, zero_boundary))
    rate_solver = rp.build_solver(matsolver=matrix_solver, network_solver=network_solver,
                                  ncc_cutoff=0, entry_cutoff=0, **separator_options)
    return dict(solver=solver, u=u, ut=ut, basis=basis, dist=dist, beta=beta,
        rate_solver=rate_solver, matrices=matrices, rows=rows, condition=boundary,
        xy=np.asarray(xy), endpoints=np.asarray(endpoints), lengths=np.asarray(lengths),
        edge_directions=directions, periodic_size=periodic_size,
        vertex_interface=vertex_interface, constraint_row_order=row_groups.ravel(),
        local_edges=indices,
        s=s.ravel(), exact=exact, source_edge=source_edge, manufactured=manufactured)



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

