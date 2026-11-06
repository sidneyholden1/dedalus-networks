"""Exact fixed-carrier representation of the passive optical Network EVP.

For a fixed complex carrier c and eigenvalue delta=k-c, write
  E(x) = exp(i*c*n*x) A(x) + exp(i*c*n*(L-x)) B(x),
  F(x) = n [exp(i*c*n*x) A(x) - exp(i*c*n*(L-x)) B(x)].
The exact equations are A'=i*delta*n*A and B'=-i*delta*n*B. There
is no slowly-varying-envelope approximation. Outgoing and vertex laws are
unchanged; their fixed endpoint phase factors enter the linear pencil.
"""
from dataclasses import dataclass

import dedalus.public as d3
from mpi4py import MPI
import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.legendre import leggauss

from photonic_dedalus import solve, pencil_residuals
from photonic_reference import BondOperator, edge_moments, field_values


@dataclass
class CarrierProblem:
    graph: object
    dist: object
    basis: object
    problem: object
    A: object
    B: object
    U: object
    carrier: complex
    resolution: int


def build_carrier_problem(graph, resolution, carrier, *, comm=MPI.COMM_WORLD, owners=None):
    carrier = complex(carrier)
    if resolution < 8 or not np.isfinite(carrier):
        raise ValueError('Require at least eight coefficients and a finite fixed carrier')
    network = d3.Network(endpoints=graph.endpoints, lengths=graph.lengths)
    if owners is None:
        owners = d3.NetworkPartition.balanced(network, comm.size).edge_owners
    dist = d3.Distributor(network, dtype=np.complex128, comm=comm,
        network_partition=owners, network_vertex_storage='owned_ghost')
    basis = d3.NetworkBasis(network, size=resolution)
    A = dist.Field(name='A', bases=basis)
    B = dist.Field(name='B', bases=basis)
    U = dist.Field(name='U', bases=basis.vertices)
    E = dist.VectorField(network.tangent, name='E', bases=basis.edges)
    E['g'][0] = 1
    n = dist.Field(name='n', bases=basis.edges)
    phaseA = dist.Field(name='phaseA', bases=basis.ports)
    phaseB = dist.Field(name='phaseB', bases=basis.ports)
    signed_index = dist.Field(name='signed_index', bases=basis.ports)
    nout = dist.Field(name='nout', bases=basis.vertices)
    endpoint_phase = np.exp(1j*carrier*graph.index*graph.lengths)
    if not np.all(np.isfinite(endpoint_phase)) or np.any(abs(endpoint_phase) == 0):
        raise ValueError('Carrier phases overflow/underflow; no clipping is permitted')
    n.set_global_data(graph.index[:, None])
    phaseA.set_global_data(np.column_stack((np.ones(graph.edges), endpoint_phase)).reshape(-1, 1))
    phaseB.set_global_data(np.column_stack((endpoint_phase, np.ones(graph.edges))).reshape(-1, 1))
    signed_index.set_global_data((graph.index[:, None]*np.array([-1., 1.])).reshape(-1, 1))
    nout.set_global_data(graph.exterior_index[:, None])
    delta = dist.Field(name='delta')
    problem = d3.EVP([A, B, U], delta, namespace=locals())
    problem.add_equation('E@grad(A) - 1j*delta*n*A = 0')
    problem.add_equation('E@grad(B) + 1j*delta*n*B = 0')
    problem.add_equation('phaseA*endpoint(A) + phaseB*endpoint(B) - endpoint(U) = 0')
    problem.add_equation('endpoint(signed_index*(phaseA*endpoint(A) - phaseB*endpoint(B)), sum=True) - nout*U = 0')
    return CarrierProblem(graph, dist, basis, problem, A, B, U, carrier, int(resolution))


def solve_carrier(app, *, target=None, modes=6, ncv=64, dense=False, tolerance=1e-10,
                  separator_direct_max_size=256,timings=None):
    target = app.carrier if target is None else complex(target)
    return solve(app, modes=modes, target=target-app.carrier, ncv=ncv,
                 dense=dense, tolerance=tolerance,
                 separator_direct_max_size=separator_direct_max_size,timings=timings)


def physical_eigenvalues(app, solver):
    return app.carrier+solver.eigenvalues


def composite_quadrature(phase, *, order=16, multiplier=1):
    """Resolve |E|^4: at most pi optical phase per 16-point Gauss panel.

    The multiplier supports an independent twice-refined physical quadrature.
    Constant-size panel rules avoid a large dense Legendre eigensolve.
    """
    if not np.isfinite(phase) or phase < 0 or order < 8 or multiplier not in (1, 2):
        raise ValueError('Invalid physical quadrature request')
    panels = max(1, int(np.ceil(phase/np.pi)))*multiplier
    if panels*order > 200000:
        raise ValueError('Small audit quadrature limit; use a bounded production path')
    q, w = leggauss(order)
    fractions = (np.arange(panels)[:, None]+(q+1)/2)/panels
    weights = np.broadcast_to(w/(2*panels), fractions.shape)
    return fractions.ravel(), weights.ravel()


def evaluate_coefficients(graph, carrier, ac, bc, fractions):
    """Reconstruct physical fields and derivatives from numerical envelopes."""
    fractions = np.asarray(fractions)
    if ac.shape != bc.shape or ac.shape[0] != graph.edges or fractions.ndim != 1:
        raise ValueError('Inconsistent envelope coefficient dimensions')
    n = ac.shape[1]
    values = cheb.chebvander(2*fractions-1, n-1)
    derivatives = cheb.chebvander(2*fractions-1, n-2)
    a, b = ac@values.T, bc@values.T
    ax = cheb.chebder(ac, axis=1)@derivatives.T*2/graph.lengths[:, None]
    bx = cheb.chebder(bc, axis=1)@derivatives.T*2/graph.lengths[:, None]
    q = carrier*graph.index[:, None]
    phase_a = np.exp(1j*q*graph.lengths[:, None]*fractions)
    phase_b = np.exp(1j*q*graph.lengths[:, None]*(1-fractions))
    pa, pb = phase_a*a, phase_b*b
    E = pa+pb
    F = graph.index[:, None]*(pa-pb)
    Ex = phase_a*ax+phase_b*bx+1j*q*(pa-pb)
    Fx = graph.index[:, None]*(phase_a*ax-phase_b*bx+1j*q*(pa+pb))
    return E, F, Ex, Fx


def small_carrier_audit(app, solver, mode, *, quadrature_multiplier=1):
    """Independent physical checks for small graphs only; no large field gather."""
    graph = app.graph
    if graph.edges > 1000:
        raise ValueError('The small carrier audit is not a production field collector')
    solver.set_state(int(mode))
    for value in (app.A, app.B, app.U):
        value.change_scales(1)
    a = app.A.allgather_data('g')
    b = app.B.allgather_data('g')
    u = app.U.allgather_data('g').ravel()
    x = app.dist.local_grid(app.basis.interval).ravel()
    ac = np.linalg.solve(cheb.chebvander(2*x-1, app.resolution-1), a.T).T
    bc = np.linalg.solve(cheb.chebvander(2*x-1, app.resolution-1), b.T).T
    k = complex(physical_eigenvalues(app, solver)[mode])
    endpoints = cheb.chebvander(np.array([-1., 1.]), app.resolution-1)
    ap, bp = ac@endpoints.T, bc@endpoints.T
    waves = np.column_stack((ap[:, 0], bp[:, 1]))
    ep, fp, _, _ = evaluate_coefficients(graph, app.carrier, ac, bc, np.array([0., 1.]))
    scale = max(np.max(abs(ep)), np.max(abs(fp)), np.finfo(float).tiny)
    flux = np.zeros(graph.vertices, complex)
    np.add.at(flux, graph.endpoints.ravel(), (fp*np.array([-1., 1.])).ravel())
    continuity = np.max(abs(ep-u[graph.endpoints]))/scale
    vertex = np.max(abs(flux-graph.exterior_index*u))/scale
    operator = BondOperator(graph).matrix(k)
    bond = np.linalg.norm(operator@waves.ravel())/max(
        np.linalg.norm(operator.data)*np.linalg.norm(waves), np.finfo(float).tiny)
    largest_phase = float(max(abs((k*graph.index).real)*graph.lengths))
    fractions, weights = composite_quadrature(largest_phase, multiplier=quadrature_multiplier)
    if graph.edges*len(fractions) > 2_000_000:
        raise ValueError('Physical audit memory bound; use local chunks in production')
    eq, fq, ex, fx = evaluate_coefficients(graph, app.carrier, ac, bc, fractions)
    weights = graph.lengths[:, None]*weights
    r1, r2 = ex-1j*k*fq, fx-1j*k*graph.index[:, None]**2*eq
    physical_scale = np.sum(weights*(abs(ex)**2+abs(k*fq)**2+abs(fx)**2+
                                     abs(k*graph.index[:, None]**2*eq)**2))
    differential = np.sqrt(np.sum(weights*(abs(r1)**2+abs(r2)**2))/max(physical_scale, 1e-300))
    i2, i4 = np.sum(weights*abs(eq)**2, axis=1), np.sum(weights*abs(eq)**4, axis=1)
    expected2, expected4 = edge_moments(graph, k, waves)
    exact, _ = field_values(graph, k, waves, fractions)
    interpolation = np.sqrt(np.sum(weights*abs(eq-exact)**2)/max(np.sum(weights*abs(exact)**2), 1e-300))
    integrals = max(np.linalg.norm(i2-expected2)/max(np.linalg.norm(expected2), 1e-300),
                    np.linalg.norm(i4-expected4)/max(np.linalg.norm(expected4), 1e-300))
    I2, I4 = sum(i2[graph.internal]), sum(i4[graph.internal])
    if min(I2, I4) <= 0:
        raise ValueError('Mode has no resolved internal intensity')
    audit = dict(k_real=k.real, k_imag=k.imag, delta_real=float(solver.eigenvalues[mode].real),
        delta_imag=float(solver.eigenvalues[mode].imag), carrier_real=app.carrier.real,
        carrier_imag=app.carrier.imag, pencil=float(pencil_residuals(app, solver)[mode]),
        continuity=float(continuity), vertex=float(vertex), bond=float(bond),
        differential=float(differential), physical_interpolation=float(interpolation),
        integral_relative=float(integrals), I2=float(I2), I4=float(I4), Lpart=float(I2**2/I4),
        physical_quadrature_points=len(fractions), envelope_coefficients=app.resolution)
    return audit, dict(A=a, B=b, U=u, x=x, amplitudes=waves,
                       coefficients_A=ac, coefficients_B=bc)
