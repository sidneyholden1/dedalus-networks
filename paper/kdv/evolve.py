"""Ordinary nonlinear KdV on a periodic square network.

The native calculus generates the physical-L2 reactions to the declared
junction laws. Defaults advance a 3x3 graph by three steps. The paper trajectory
uses a 16x16 graph, 64 coefficients per edge and extends to T=200.
"""
import argparse
import json
import numpy as np
import dedalus.public as d3
from model import prepare, finish, capture

# Graph, resolution and initial compact pulse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--size', type=int, default=3)
parser.add_argument('--modes', type=int, default=24)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--dt', type=float, default=0.000125)
args = parser.parse_args()
if args.steps < 1 or args.dt <= 0:
    raise ValueError('Require steps >= 1 and dt > 0')
model = prepare(n=args.modes, periodic_size=args.size)
dist, basis, u, t, E, horizontal, vertical = (
    model[key] for key in ('dist', 'basis', 'u', 't', 'E', 'horizontal', 'vertical'))

# Ordinary KdV: continuous displacement, piecewise curvature, physical flux.
# u was declared with regularity='continuous'; curvature need not be continuous.
w = E @ d3.grad(E @ d3.grad(u))
problem = d3.IVP([u], time=t, namespace=locals())
# Construct tau corrections from the adjoints of the junction conditions.
problem.add_equation('dt(u) + div(E*w) = -6*u*(E@grad(u))', tau='adjoint')
problem.add_vertex_constraint('endpoint(horizontal*u, deriv=1, sum=True) = 0')
problem.add_vertex_constraint('endpoint(vertical*u, deriv=1, sum=True) = 0')
# On this balanced oriented grid, the vertex contributions of div(3*E*u**2)
# cancel because u is continuous. The regular product above preserves the
# paper's physical-L2 projection before conversion to derivative bases.
# div(E*w) supplies signed curvature balance; the constraints pair slopes.
finish(model, problem)
solver = model['solver']
initial, _, _ = capture(model)

# Time integration: u_t + u_xxx + 6 u u_x = 0
for step in range(args.steps):
    solver.step(args.dt)
final, fields, position = capture(model)

# Unscaled endpoint constraints and independent quadrature balances
checks = ('constraint_max', 'mass_balance_residual', 'l2_balance_residual')
if any(not np.isfinite(final[key]) or abs(final[key]) > 1e-8 for key in checks):
    raise AssertionError({key: final[key] for key in checks})
mass_drift = abs(final['mass']-initial['mass'])/max(abs(initial['mass']), 1.)
if not np.isfinite(mass_drift) or mass_drift > 1e-8:
    raise AssertionError(f'KdV mass drift {mass_drift:g}')
if model['dist'].comm.rank == 0:
    print('RESULT ' + json.dumps(dict(example='KdV', edges=len(model['endpoints']),
        modes=args.modes, steps=args.steps, time=final['t'], mass_drift=mass_drift,
        **{key: final[key] for key in checks})))
