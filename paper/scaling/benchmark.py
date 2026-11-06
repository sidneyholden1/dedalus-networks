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

# Two physical fields, sharing vertex values across their resolution groups
t = dist.Field(name='t')
U = dist.Field(name='U', bases=bases.vertices)
V = dist.Field(name='V', bases=bases.vertices)
u_fields, v_fields = {}, {}
for name, basis in bases.items():
    u_fields[name] = dist.Field(name='u'+name, bases=basis, regularity='continuous')
    v_fields[name] = dist.Field(name='v'+name, bases=basis, regularity='continuous')

variables = [U, *u_fields.values(), V, *v_fields.values()]
interfaces = [(u, U) for u in u_fields.values()] + [(v, V) for v in v_fields.values()]
problem = d3.IVP(variables, time=t, network_interfaces=interfaces, namespace=locals())

# Known vertex profiles; both fields vanish at the outer ends
leaf = network.degree == 1
vertex = np.arange(network.num_vertices)
H_u = (1+0.2*np.cos(0.41*vertex))*(1-leaf)
H_v = (1+0.2*np.cos(0.41*vertex+1))*(1-leaf)
U.set_global_data(H_u[:, None])
V.set_global_data(H_v[:, None])
coupling, nonlinear = 0.2, 0.1
profiles = []

# Add the same pair of equations on each resolution group
for name, basis in bases.items():
    u, v = u_fields[name], v_fields[name]
    edge, s = basis.local_grids(dist, scales=(1, 1))
    edge, s = np.asarray(edge).ravel(), np.asarray(s).ravel()[None, :]

    # Conductivity, capacity and reaction coefficients for u and v
    K_u = dist.Field(name='K_u'+name, bases=basis.edges)
    C_u = dist.Field(name='C_u'+name, bases=basis.edges)
    a_u = dist.Field(name='a_u'+name, bases=basis.edges)
    K_v = dist.Field(name='K_v'+name, bases=basis.edges)
    C_v = dist.Field(name='C_v'+name, bases=basis.edges)
    a_v = dist.Field(name='a_v'+name, bases=basis.edges)
    K_u['g'] = (0.8+0.3*np.sin(0.19*edge)**2)[:, None]
    C_u['g'] = (0.7+0.4*np.cos(0.23*edge)**2)[:, None]
    a_u['g'] = (0.6+0.2*np.sin(0.31*edge)**2)[:, None]
    K_v['g'] = (0.8+0.3*np.sin(0.19*edge+1)**2)[:, None]
    C_v['g'] = (0.7+0.4*np.cos(0.23*edge+1)**2)[:, None]
    a_v['g'] = (0.6+0.2*np.sin(0.31*edge+1)**2)[:, None]

    # Smooth edge profiles with the prescribed vertex values and zero end slopes
    tail, head = network.endpoints[edge].T
    jump_u = (H_u[head]-H_u[tail])[:, None]
    jump_v = (H_v[head]-H_v[tail])[:, None]
    profile_u = H_u[tail, None]+jump_u*(3*s*s-2*s**3)+0.1*s*s*(1-s)**2
    profile_v = H_v[tail, None]+jump_v*(3*s*s-2*s**3)+0.1*s*s*(1-s)**2
    curvature_u = (jump_u*(6-12*s)+0.1*(2-12*s+12*s*s))/network.lengths[edge, None]**2
    curvature_v = (jump_v*(6-12*s)+0.1*(2-12*s+12*s*s))/network.lengths[edge, None]**2
    u['g'], v['g'] = profile_u, profile_v

    # Known forcing makes exp(-t)*h_u and exp(-t)*h_v exact solutions.
    # Store the sampled cubes before spectral conversion, as in the benchmark.
    h_u = dist.Field(name='h_u'+name, bases=basis)
    h_v = dist.Field(name='h_v'+name, bases=basis)
    h_u3 = dist.Field(name='h_u3'+name, bases=basis)
    h_v3 = dist.Field(name='h_v3'+name, bases=basis)
    f_u = dist.Field(name='f_u'+name, bases=basis)
    f_v = dist.Field(name='f_v'+name, bases=basis)
    h_u['g'], h_v['g'] = profile_u, profile_v
    h_u3['g'], h_v3['g'] = profile_u**3, profile_v**3
    f_u['g'] = (a_u['g']-C_u['g'])*profile_u-K_u['g']*curvature_u
    f_v['g'] = (a_v['g']-C_v['g'])*profile_v-K_v['g']*curvature_v

    problem.add_equation(
        'C_u*dt(u) - div(K_u*grad(u)) + a_u*u + coupling*(u-v) = '
        'exp(-t)*f_u + nonlinear*(u**3-exp(-3*t)*h_u3) '
        '+ coupling*exp(-t)*(h_u-h_v)')
    problem.add_equation(
        'C_v*dt(v) - div(K_v*grad(v)) + a_v*v + coupling*(v-u) = '
        'exp(-t)*f_v + nonlinear*(v**3-exp(-3*t)*h_v3) '
        '+ coupling*exp(-t)*(h_v-h_u)')

    boundary_vertices = np.intersect1d(np.flatnonzero(leaf), network.endpoints[basis.edge_indices])
    if boundary_vertices.size:
        problem.add_vertex_constraint('endpoint(u, on=boundary_vertices) = 0')
        problem.add_vertex_constraint('endpoint(v, on=boundary_vertices) = 0')
    profiles.extend([(u, profile_u), (v, profile_v)])
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
for field, profile in profiles:
    field.change_scales(1)
    error = max(error, float(np.max(np.abs(field['g']-np.exp(-solver.sim_time)*profile), initial=0.)))
error = comm.allreduce(error, op=MPI.MAX)
if not np.isfinite(error) or error > 1e-5:
    raise AssertionError(f'Scaling fixture error {error:g}')
if comm.rank == 0:
    print('RESULT ' + json.dumps(dict(example='scaling equations', edges=network.num_edges,
        coefficients=int(2*sizes.sum()), ranks=comm.size, steps=args.steps,
        warmup=args.warmup, maximum_error=error, smoke_seconds=elapsed)))
