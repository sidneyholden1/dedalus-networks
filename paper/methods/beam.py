"""Static Euler--Bernoulli beams on a three-edge star.

All coefficients are nondimensional. The outer ends are clamped. At the
central junction displacement is continuous and outward slopes sum to zero;
this is an explicit chosen joint law. Individual junction slopes may differ.
The unequal edge loads give an exact polynomial solution for verification.
Run this small example with one process.
"""

import json
import numpy as np
import dedalus.public as d3

# Network and basis
network = d3.Network(
    vertex_ids=('junction', 'end_1', 'end_2', 'end_3'),
    endpoints=((0, 1), (0, 2), (0, 3)),
    lengths=(1., 1., 1.),
    vertex_groups={'clamped': (1, 2, 3), 'coupled': (0,)},
)
dist = d3.Distributor(network, dtype=np.float64)
basis = d3.NetworkBasis(network, size=12)

# Displacement and coefficients
u = dist.Field(name='u', bases=basis, regularity='continuous')
u['g'] = 0

clamped = network.vertex_groups['clamped']
selected_junctions = network.vertex_groups['coupled']
EI = 1.
load = dist.Field(name='load', bases=basis.edges)
edge = dist.local_grid(basis.edges)
load['g'] = 24 * (edge + 1)

# Problem
problem = d3.LBVP([u], namespace=locals())
problem.add_equation('H.T(EI*H(u)) = load')
# Vertex constraints add their unknown adjoint reactions to the beam balance.
problem.add_vertex_constraint('endpoint(u, deriv=0, on=clamped) = 0')
problem.add_vertex_constraint('endpoint(u, deriv=1, on=clamped) = 0')
problem.add_vertex_constraint(
    'endpoint(u, deriv=1, on=selected_junctions, sum=True) = 0'
)

# Solver
solver = problem.build_solver()
solver.solve()

# Check against the exact solution
x = basis.local_physical_grid(dist)
exact = 2*(1-x*x)**2 + (edge-1)*(x**4 - 1.5*x**3 + 0.5*x)
error = np.max(np.abs(u['g'] - exact))
if not np.isfinite(error) or error > 1e-8:
    raise AssertionError(f'Beam displacement error {error:g}')
print('RESULT ' + json.dumps(dict(example='beam methods listing', maximum_error=float(error))))
