"""Passive optical resonances in an exact fixed-carrier representation.

The default is a five-edge open loop. --graph accepts an OpticalGraph JSON.
The carrier removes rapid phase oscillations without an envelope approximation.
"""
import argparse
import json
import numpy as np
from photonic_reference import OpticalGraph, branched_loop, dielectric_interval, interval_resonance
from photonic_carrier import build_carrier_problem, solve_carrier, physical_eigenvalues, small_carrier_audit

# Graph and spectral window
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--graph', help='OpticalGraph JSON input')
parser.add_argument('--interval-check', action='store_true', help='use the analytic dielectric interval')
parser.add_argument('--resolution', type=int, default=16)
parser.add_argument('--carrier', type=complex, default=10.68-0.05j)
args = parser.parse_args()
if args.graph and args.interval_check:
    raise ValueError('Choose a graph file or the analytic interval')
graph = (OpticalGraph.load(args.graph) if args.graph else
         dielectric_interval(length=20., index=1.5) if args.interval_check else branched_loop())

# Two first-order envelope equations, physical-field continuity and outgoing flux
# Full Dedalus equations are in photonic_carrier.py:build_carrier_problem.
application = build_carrier_problem(graph, args.resolution, args.carrier)
solver = solve_carrier(application, modes=4, ncv=28)
k = physical_eigenvalues(application, solver)
if not np.isfinite(k).all():
    raise FloatingPointError('Nonfinite optical eigenvalue')
mode = int(np.argmin(np.abs(k-args.carrier)))

# Check the physical field, endpoint laws and independent bond operator
audit, fields = small_carrier_audit(application, solver, mode)
checks = ('pencil', 'continuity', 'vertex', 'bond', 'differential',
          'physical_interpolation', 'integral_relative')
if any(not np.isfinite(audit[key]) or audit[key] > 1e-7 for key in checks):
    raise AssertionError({key: audit[key] for key in checks})
if args.interval_check:
    reference = interval_resonance(102, length=20., index=1.5)
    audit['analytic_eigenvalue_error'] = float(np.min(np.abs(k-reference)))
    if audit['analytic_eigenvalue_error'] > 1e-10:
        raise AssertionError('Dielectric interval resonance mismatch')
if application.dist.comm.rank == 0:
    print('RESULT ' + json.dumps(dict(example='photonic resonances', edges=graph.edges, **audit)))
