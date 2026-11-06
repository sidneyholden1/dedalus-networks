"""First-order passive optical Network EVP and bounded qualification audits.

This initial audit path intentionally permits only small graphs. Production
field output requires distributed reductions/checkpointing before admission.
"""
from dataclasses import dataclass

import dedalus.public as d3
from mpi4py import MPI
import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.legendre import leggauss

from photonic_reference import BondOperator, edge_moments


@dataclass
class OpticalProblem:
    graph: object
    dist: object
    basis: object
    problem: object
    E: object
    F: object
    U: object


def build_problem(graph, resolution, *, comm=MPI.COMM_WORLD, owners=None):
    if resolution < 8:
        raise ValueError('Use at least eight coefficients per numerical interval')
    network = d3.Network(endpoints=graph.endpoints, lengths=graph.lengths)
    if owners is None:
        owners = d3.NetworkPartition.balanced(network, comm.size).edge_owners
    dist = d3.Distributor(network, dtype=np.complex128, comm=comm,
        network_partition=owners, network_vertex_storage='owned_ghost')
    basis = d3.NetworkBasis(network, size=resolution)
    E = dist.Field(name='E', bases=basis, regularity='continuous')
    F = dist.Field(name='F', bases=basis)
    tangent = dist.VectorField(network.tangent, name='tangent', bases=basis.edges)
    tangent['g'][0] = 1
    n2 = dist.Field(name='n2', bases=basis.edges)
    nout = dist.Field(name='nout', bases=basis.dirac)
    n2.set_global_data((graph.index**2)[:, None])
    nout.set_global_data(graph.exterior_index[:, None])
    k = dist.Field(name='k')
    problem = d3.EVP([E, F], k, namespace=locals())
    problem.add_equation('tangent@grad(E) - 1j*k*F = 0')
    problem.add_equation('div(tangent*F) - 1j*k*n2*E + nout*E = 0', test=E)
    U = problem.vertex_field(E)
    return OpticalProblem(graph, dist, basis, problem, E, F, U)


def solve(application, *, modes=6, target=1., ncv=64, dense=False, tolerance=1e-10,
          separator_direct_max_size=256, timings=None):
    import time
    comm = application.dist.comm
    if type(separator_direct_max_size) is not int or not 0<=separator_direct_max_size<=8192:
        raise ValueError('Explicit separator direct threshold must be bounded by 8192')
    began=time.monotonic()
    solver = application.problem.build_solver(
        network_solver=None if comm.size == 1 else 'dtn', entry_cutoff=0, ncc_cutoff=0,
        network_separator_direct_max_size=separator_direct_max_size)
    if timings is not None:
        timings['solver_initialization_seconds']=time.monotonic()-began
        def instrument(name,key):
            original=getattr(solver,name)
            def measured(*args,**kwargs):
                started=time.monotonic()
                try:return original(*args,**kwargs)
                finally:timings[key]=timings.get(key,0.)+time.monotonic()-started
            setattr(solver,name,measured)
        instrument('build_matrices','assembly_seconds')
        instrument('build_subproblem_solver','factorization_seconds')
    started=time.monotonic()
    if dense:
        if comm.size != 1 or application.graph.edges > 12:
            raise ValueError('Dense qualification is restricted to serial tiny graphs')
        solver.solve_dense(solver.subproblems[0])
    else:
        options = dict(ncv=ncv, tol=tolerance)
        if comm.size == 1:
            np.random.seed(0)
            options['maxiter'] = 500
        solver.solve_sparse(solver.subproblems[0], N=modes, target=target, **options)
    if timings is not None:
        timings['solve_seconds']=time.monotonic()-started
        timings['krylov_and_verification_seconds']=max(0.,timings['solve_seconds']-
            timings.get('assembly_seconds',0.)-timings.get('factorization_seconds',0.))
        timings['arnoldi_dimension']=int(getattr(solver,'arnoldi_dimension',0))
    return solver


def pencil_residuals(application, solver):
    """||Ax-kBx||_2 / (||Ax||_2+|k| ||Bx||_2), counting owned rows once."""
    if application.dist.comm.size > 1:
        return np.asarray(solver.eigenvalue_residuals)
    subproblem = solver.subproblems[0]
    vectors = subproblem.pre_right_pinv@solver.eigenvectors
    av = subproblem.L_min@vectors
    bv = -subproblem.M_min@vectors
    return np.linalg.norm(av-bv*solver.eigenvalues, axis=0)/np.maximum(
        np.linalg.norm(av, axis=0)+abs(solver.eigenvalues)*np.linalg.norm(bv, axis=0),
        np.finfo(float).tiny)


def small_mode_audit(application, solver, mode):
    graph = application.graph
    if graph.edges > 1000:
        raise ValueError('Small-field audit cannot gather a production graph')
    solver.set_state(int(mode))
    for field in (application.E, application.F, application.U):
        field.change_scales(1)
    E = application.E.allgather_data('g')
    F = application.F.allgather_data('g')
    U = application.U.allgather_data('g').ravel()
    x = application.dist.local_grid(application.basis.interval).ravel()
    k = complex(solver.eigenvalues[mode])
    n = E.shape[-1]
    vandermonde = cheb.chebvander(2*x-1, n-1)
    ec = np.linalg.solve(vandermonde, E.T).T
    fc = np.linalg.solve(vandermonde, F.T).T
    nodes, weights = leggauss(2*n+4)
    value = cheb.chebvander(nodes, n-1)
    derivative = cheb.chebvander(nodes, n-2)
    eq, fq = ec@value.T, fc@value.T
    ex = cheb.chebder(ec, axis=1)@derivative.T*2/graph.lengths[:, None]
    fx = cheb.chebder(fc, axis=1)@derivative.T*2/graph.lengths[:, None]
    w = weights[None, :]*graph.lengths[:, None]/2
    r1 = ex-1j*k*fq
    r2 = fx-1j*k*graph.index[:, None]**2*eq
    differential = np.sqrt(np.sum(w*(abs(r1)**2+abs(r2)**2))/max(
        np.sum(w*(abs(ex)**2+abs(k*fq)**2+abs(fx)**2+abs(k*graph.index[:, None]**2*eq)**2)),
        np.finfo(float).tiny))
    endpoint = cheb.chebvander(np.array([-1., 1.]), n-1)
    ep, fp = ec@endpoint.T, fc@endpoint.T
    norm = max(np.max(abs(ep)), np.max(abs(fp)), np.finfo(float).tiny)
    flux = np.zeros(graph.vertices, complex)
    np.add.at(flux, graph.endpoints.ravel(), (fp*np.array([-1., 1.])).ravel())
    continuity = np.max(abs(ep-U[graph.endpoints]))/norm
    vertex = np.max(abs(flux-graph.exterior_index*U))/norm
    waves = np.column_stack(((ep[:, 0]+fp[:, 0]/graph.index)/2,
                             (ep[:, 1]-fp[:, 1]/graph.index)/2))
    operator = BondOperator(graph).matrix(k)
    bond = np.linalg.norm(operator@waves.ravel())/max(
        np.linalg.norm(waves)*np.linalg.norm(operator.data), np.finfo(float).tiny)
    i2 = np.sum(w*abs(eq)**2, axis=1)
    i4 = np.sum(w*abs(eq)**4, axis=1)
    a2, a4 = edge_moments(graph, k, waves)
    norm2, norm4 = sum(i2[graph.internal]), sum(i4[graph.internal])
    integrals = max(np.linalg.norm(a2-i2)/max(np.linalg.norm(a2), 1e-300),
                    np.linalg.norm(a4-i4)/max(np.linalg.norm(a4), 1e-300))
    result = dict(k_real=k.real, k_imag=k.imag, differential=float(differential),
        continuity=float(continuity), vertex=float(vertex), bond=float(bond),
        integral_relative=float(integrals), I2=float(norm2), I4=float(norm4),
        Lpart=float(norm2**2/norm4))
    return result, dict(E=E, F=F, U=U, x=x, amplitudes=waves)
