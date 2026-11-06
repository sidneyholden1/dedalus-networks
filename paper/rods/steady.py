"""Figure 1 steady rod problem, expressed with the native graph calculus.

One corner is held at u=1. The other vertices have Kirchhoff flux balance.
The hyperbolic-function solution provides an independent accuracy check.
"""
import argparse
import json
import numpy as np
import dedalus.public as d3
from inputs import steady, profile

# Network and resolution
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--modes', type=int, default=16)
args = parser.parse_args()
edges = np.array([(4*y+x, 4*y+x+1) for y in range(4) for x in range(3)]
                 + [(4*y+x, 4*(y+1)+x) for y in range(3) for x in range(4)])
lengths = np.ones(len(edges))
network = d3.Network(endpoints=edges, lengths=lengths,
                     vertex_groups={'hot': (0,)})
dist = d3.Distributor(network, dtype=np.float64)
basis = d3.NetworkBasis(network, size=args.modes)

# Physical field and balance
u = dist.Field(name='u', bases=basis, regularity='continuous')
hot = network.vertex_groups['hot']
problem = d3.LBVP([u], namespace=locals())
problem.add_equation('-div(grad(u)) + u = 0')
problem.add_vertex_constraint('endpoint(u, on=hot) = 1')

# Solve and compare with the exact edge profiles
solver = problem.build_solver()
solver.solve()
vertex_values, q = steady(edges, lengths, np.ones(len(edges)), np.ones(len(edges)))
edge, s = basis.local_grids(dist, scales=(1, 1))
exact = profile(np.asarray(s).ravel(), edges, lengths, vertex_values, q)
error = float(np.max(np.abs(u['g'] - exact[np.asarray(edge).ravel()])))
if not np.isfinite(error) or error > 1e-8:
    raise AssertionError(f'Steady reference error {error:g}')
print('RESULT ' + json.dumps(dict(example='steady rods', modes=args.modes,
                                maximum_error=error)))
