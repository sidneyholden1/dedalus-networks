"""Two coupled reaction-diffusion fields used by the scaling study.

Defaults exercise heterogeneous edge resolutions on a small graph. This is
an execution/accuracy check, not a replacement measurement of Figure 2.
Saved --input and --owners arrays can preserve a study's graph and partition.
"""
import argparse
import json
import time
import numpy as np
from mpi4py import MPI
import dedalus.public as d3
from inputs import fixture

# Graph, partition and timestep settings
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--family', choices=['spatial', 'irregular', 'hub', 'mesh',
                                        'community', 'triangular', 'junctions'], default='spatial')
parser.add_argument('--size', type=int, default=4)
parser.add_argument('--input', help='NPZ containing endpoints, lengths and modes')
parser.add_argument('--owners', help='NPY containing one owner rank per edge')
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--warmup', type=int, default=1)
parser.add_argument('--dt', type=float, default=0.001)
args = parser.parse_args()
if args.steps < 1 or args.warmup < 0 or args.dt <= 0:
    raise ValueError('Require steps >= 1, warmup >= 0 and dt > 0')
comm = MPI.COMM_WORLD
if args.input:
    with np.load(args.input, allow_pickle=False) as data:
        network = d3.Network(endpoints=data['endpoints'], lengths=data['lengths'])
        sizes = data['modes'].copy()
else:
    network, sizes, xy = fixture(args.family, args.size, seed=20261010)
owners = np.load(args.owners, allow_pickle=False) if args.owners else None
partition = (d3.NetworkPartition(network, owners, num_parts=comm.size) if owners is not None
             else d3.NetworkPartition.balanced(network, comm.size))
dist = d3.Distributor(network, dtype=np.float64, comm=comm,
                      network_partition=partition, network_vertex_storage='owned_ghost')
bases = d3.NetworkBasisSet(network, {
    str(n): dict(size=int(n), edge_indices=np.flatnonzero(sizes == n))
    for n in np.unique(sizes)
})

# Shared time and boundary masks
t = dist.Field(name='t')
alpha = dist.Field(name='alpha', bases=bases.vertices)
beta = dist.Field(name='beta', bases=bases.vertices)
leaf = (network.degree == 1).astype(float)
alpha.set_global_data(leaf[:, None])
beta.set_global_data((1-leaf)[:, None])
variables, equations, interfaces, terms, buckets = [], [], [], {}, []
coupling, nonlinear = 0.2, 0.1

# Two fields with different coefficients, coupled on each edge
for k in range(2):
    U = dist.Field(name=f'U{k}', bases=bases.vertices)
    H = (1+0.2*np.cos(0.41*np.arange(network.num_vertices)+k))*(1-leaf)
    U.set_global_data(H[:, None])
    variables.append(U)
    fluxes = []
    for name, basis in bases.items():
        edge, s = basis.local_grids(dist, scales=(1, 1))
        edge, s = np.asarray(edge).ravel(), np.asarray(s).ravel()[None, :]
        u = dist.Field(name=f'u{k}_{name}', bases=basis)
        tau1 = dist.Field(name=f'tau1_{k}_{name}', bases=basis.edges)
        tau2 = dist.Field(name=f'tau2_{k}_{name}', bases=basis.edges)
        tail = H[network.endpoints[edge, 0], None]
        jump = H[network.endpoints[edge, 1], None]-tail
        h = tail+jump*(3*s*s-2*s**3)+0.1*s*s*(1-s)**2
        hxx = (jump*(6-12*s)+0.1*(2-12*s+12*s*s))/network.lengths[edge, None]**2
        u['g'] = h
        K = dist.Field(name=f'K{k}_{name}', bases=basis.edges)
        C = dist.Field(name=f'C{k}_{name}', bases=basis.edges)
        a = dist.Field(name=f'a{k}_{name}', bases=basis.edges)
        kv = 0.8+0.3*np.sin(0.19*edge+k)**2
        cv = 0.7+0.4*np.cos(0.23*edge+k)**2
        av = 0.6+0.2*np.sin(0.31*edge+k)**2
        K['g'], C['g'], a['g'] = kv[:, None], cv[:, None], av[:, None]
        f = dist.Field(name=f'f{k}_{name}', bases=basis)
        exact = dist.Field(name=f'h{k}_{name}', bases=basis)
        exact3 = dist.Field(name=f'h3_{k}_{name}', bases=basis)
        f['g'] = (av-cv)[:, None]*h-kv[:, None]*hxx
        exact['g'], exact3['g'] = h, h**3
        dx, R = basis.derivative, basis.restriction
        incident_vertices = np.unique(network.endpoints[basis.edge_indices])
        lift_basis = basis.interval.derivative_basis(2)
        lhs = (C*d3.TimeDerivative(u) - K*dx(dx(u)) + a*u
               + d3.Lift(tau1, lift_basis, -1) + d3.Lift(tau2, lift_basis, -2))
        rhs = np.exp(-t)*f + nonlinear*(u**3-np.exp(-3*t)*exact3)
        terms[k, name] = lhs, rhs, u, exact
        equations.append((d3.endpoint(u, on=incident_vertices)-R@U, 0))
        fluxes.append(R.T @ d3.endpoint(K*u, deriv=1, on=incident_vertices))
        variables.extend([u, tau1, tau2])
        interfaces.append((u, U))
        buckets.append((u, h))
    equations.append((alpha*U+beta*sum(fluxes), 0))

for (k, name), (lhs, rhs, u, exact) in terms.items():
    other_lhs, other_rhs, other, other_exact = terms[1-k, name]
    equations.append((lhs+coupling*(u-other),
                      rhs+coupling*np.exp(-t)*(exact-other_exact)))
problem = d3.IVP(variables, time=t, network_interfaces=interfaces)
for equation in equations:
    problem.add_equation(equation)
solver = problem.build_solver(d3.RK222, network_solver='dtn',
    network_interior_elimination='coupled', network_separator_method='gmres',
    network_separator_rtol=1e-12, network_separator_atol=1e-13)

# Warm up, advance, and compare with the manufactured solution exp(-t) h
for step in range(args.warmup):
    solver.step(args.dt)
comm.Barrier()
started = time.perf_counter()
for step in range(args.steps):
    solver.step(args.dt)
comm.Barrier()
elapsed = time.perf_counter()-started
error = 0.
for u, h in buckets:
    u.change_scales(1)
    error = max(error, float(np.max(np.abs(u['g']-np.exp(-solver.sim_time)*h), initial=0.)))
error = comm.allreduce(error, op=MPI.MAX)
if not np.isfinite(error) or error > 1e-5:
    raise AssertionError(f'Scaling fixture error {error:g}')
if comm.rank == 0:
    print('RESULT ' + json.dumps(dict(example='scaling equations', edges=network.num_edges,
        coefficients=int(2*sizes.sum()), ranks=comm.size, steps=args.steps,
        warmup=args.warmup, maximum_error=error, smoke_seconds=elapsed)))
