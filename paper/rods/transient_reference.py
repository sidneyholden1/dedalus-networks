"""Figure 1 heterogeneous rods: C u_t - K u_xx + a u = 0.

The heated corner follows h(t)=1-exp(-4t^2). Write u=v+h(t)w, where w
is the exact steady profile, to impose homogeneous conditions on v.
Different mode counts share one vertex field across the basis buckets.
Defaults take three short steps; the paper case uses --modes 32 --steps 128.
"""
import argparse
import json
import numpy as np
from numpy.polynomial import chebyshev as cheb
from mpi4py import MPI
import dedalus.public as d3
from inputs import geometry, mode_counts, steady, profile

# Geometry, coefficients and resolution
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--modes', type=int, default=8, help='mean coefficients per edge')
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--dt', type=float, default=1/128)
args = parser.parse_args()
if args.modes < 4 or args.steps < 1 or args.dt <= 0:
    raise ValueError('Require modes >= 4, steps >= 1 and dt > 0')
comm = MPI.COMM_WORLD
xy, edges, lengths, conductivity, capacity, reaction = geometry()
W, q = steady(edges, lengths, conductivity, reaction)
network = d3.Network(endpoints=edges, lengths=lengths)
partition = d3.NetworkPartition.spatial(network, comm.size, xy, weights=lengths)
dist = d3.Distributor(network, dtype=np.float64, network_partition=partition,
                      network_vertex_storage='owned_ghost')
sizes = mode_counts(lengths, args.modes)
bases = d3.NetworkBasisSet(network, {
    str(n): dict(size=int(n), edge_indices=np.flatnonzero(sizes == n))
    for n in np.unique(sizes)
})

# Shared vertex values and boundary masks
V = dist.Field(name='V', bases=bases.vertices)
t = dist.Field(name='t')
alpha = dist.Field(name='alpha', bases=bases.vertices)
beta = dist.Field(name='beta', bases=bases.vertices)
hot = np.zeros((16, 1)); hot[0] = 1
alpha.set_global_data(hot)
beta.set_global_data(1-hot)
variables, equations, interfaces, buckets, fluxes = [V], [], [], [], []

# Edge balances and endpoint matching for each resolution bucket
for name, basis in bases.items():
    edge, s = basis.local_grids(dist, scales=(1, 1))
    edge, s = np.asarray(edge).ravel(), np.asarray(s).ravel()
    v = dist.Field(name='v'+name, bases=basis)
    tau1 = dist.Field(name='tau1_'+name, bases=basis.edges)
    tau2 = dist.Field(name='tau2_'+name, bases=basis.edges)
    K = dist.Field(name='K'+name, bases=basis)
    C = dist.Field(name='C'+name, bases=basis)
    a = dist.Field(name='a'+name, bases=basis)
    w = dist.Field(name='w'+name, bases=basis)
    K['g'], C['g'], a['g'] = conductivity[edge, None], capacity[edge, None], reaction[edge, None]
    w['g'] = profile(s, edges, lengths, W, q)[edge]
    dx, R = basis.derivative, basis.restriction
    # A resolution bucket contains only a subset of the network's vertices.
    incident_vertices = np.unique(network.endpoints[basis.edge_indices])
    lift_basis = basis.interval.derivative_basis(2)
    balance = (C*d3.TimeDerivative(v) - K*dx(dx(v)) + a*v
               + d3.Lift(tau1, lift_basis, -1) + d3.Lift(tau2, lift_basis, -2))
    forcing = -C*(8*t*np.exp(-4*t*t))*w
    equations.extend([(balance, forcing), (d3.endpoint(v, on=incident_vertices) - R@V, 0)])
    fluxes.append(R.T @ d3.endpoint(K*v, deriv=1, on=incident_vertices))
    variables.extend([v, tau1, tau2])
    interfaces.append((v, V))
    buckets.append((basis, v, edge, s))

problem = d3.IVP(variables, time=t, network_interfaces=interfaces)
for equation in equations:
    problem.add_equation(equation)
problem.add_equation((alpha*V + beta*sum(fluxes), 0))
solver = problem.build_solver(d3.RK222, network_solver='dtn',
    network_interior_elimination='coupled', network_separator_method='gmres',
    network_separator_preconditioner='jacobi',
    network_separator_rtol=1e-13, network_separator_atol=1e-14)

# Advance and check the actual endpoint and flux conditions
for step in range(args.steps):
    solver.step(args.dt)
coefficients = [None]*len(edges)
local = []
for basis, v, edge, s in buckets:
    v.change_scales(1)
    local.extend((int(e), cheb.chebfit(2*s-1, row, int(sizes[e])-1))
                 for e, row in zip(edge, v['g']))
for payload in comm.allgather(local):
    for e, coefficient in payload:
        coefficients[e] = coefficient
values = V.allgather_data('g').ravel()
flux, mismatches = np.zeros(16), []
for e, (tail, head) in enumerate(edges):
    coefficient = coefficients[e]
    slope = cheb.chebder(coefficient)*2/lengths[e]
    flux[tail] -= conductivity[e]*cheb.chebval(-1, slope)
    flux[head] += conductivity[e]*cheb.chebval(1, slope)
    mismatches.extend([cheb.chebval(-1, coefficient)-values[tail],
                       cheb.chebval(1, coefficient)-values[head]])
constraint = float(max(np.max(np.abs(mismatches)), np.max(np.abs(flux[1:])), abs(values[0])))
if not np.isfinite(constraint) or constraint > 1e-8:
    raise AssertionError(f'Rod vertex residual {constraint:g}')
if not all(np.isfinite(c).all() for c in coefficients):
    raise FloatingPointError('Nonfinite rod field')
if comm.rank == 0:
    print('RESULT ' + json.dumps(dict(example='transient rods', steps=args.steps,
        time=float(solver.sim_time), vertex_residual=constraint,
        heated_vertex=float(-np.expm1(-4*solver.sim_time**2)))))
