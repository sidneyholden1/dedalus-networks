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

# Shared physical vertex values across the resolution groups
V = dist.Field(name='V', bases=bases.vertices)
t = dist.Field(name='t')
variables, interfaces, buckets, coefficients = [V], [], [], []

# Physical fields and coefficients; each group keeps its original resolution
for name, basis in bases.items():
    edge, s = basis.local_grids(dist, scales=(1, 1))
    edge, s = np.asarray(edge).ravel(), np.asarray(s).ravel()
    v = dist.Field(name='v'+name, bases=basis, regularity='continuous')
    K = dist.Field(name='K'+name, bases=basis)
    C = dist.Field(name='C'+name, bases=basis)
    a = dist.Field(name='a'+name, bases=basis)
    w = dist.Field(name='w'+name, bases=basis)
    K['g'], C['g'], a['g'] = conductivity[edge, None], capacity[edge, None], reaction[edge, None]
    w['g'] = profile(s, edges, lengths, W, q)[edge]
    variables.append(v)
    interfaces.append((v, V))
    buckets.append((basis, v, edge, s))
    coefficients.append((v, K, C, a, w))

# Native balances: Dedalus combines the vertex fluxes across all groups.
problem = d3.IVP(variables, time=t, network_interfaces=interfaces, namespace=locals())
for v, K, C, a, w in coefficients:
    problem.add_equation('C*dt(v) - div(K*grad(v)) + a*v = -C*(8*t*exp(-4*t*t))*w')
hot_field = next(v for basis, v, edge, s in buckets
                 if 0 in network.endpoints[basis.edge_indices])
problem.add_vertex_constraint('endpoint(hot_field, on=[0]) = 0')
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
