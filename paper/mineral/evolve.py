"""Dedalus pore-network flow, two-species transport and finite-grain evolution.

This short Gamma=0.3 example checks the complete update path on a small graph.
The paper's stored 29,996-pore history remains the source of its figure.
Transport uses the exact integrating factor for each uniform pore.
"""
import argparse
import json
import numpy as np
from mpi4py import MPI
from mineral_reference import Chemistry
from mineral_geometry import periodic_graph, initial_inventory
from mineral_evolution import State, choose_step, advance
from mineral_dedalus import hydraulic, transport

# Geometry, chemistry and initial inventories
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--size', type=int, default=4)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--max-dt', type=float, default=1e-4)
args = parser.parse_args()
if args.steps < 1 or args.max_dt <= 0:
    raise ValueError('Require steps >= 1 and max-dt > 0')
geometry = periodic_graph(args.size, args.size, seed=20261029)
chemistry = Chemistry(gamma=0.3, kappa=1., da=0.5)
primary, secondary = initial_inventory(geometry, chemistry.d0, method='uniform-equation21')
initial_primary = float(primary.sum())
diameter = np.full(len(geometry.endpoints), chemistry.d0)
state = State(diameter, primary, secondary, np.ones(len(diameter), dtype=bool))
total_flow = chemistry.mean_initial_flow*len(geometry.inlet)
dmin = 0.01*chemistry.d0
maximum_water_residual = maximum_solute_defect = maximum_algebraic_residual = 0.

# Rebuild pressure and transport for each geometry, then advance finite inventories
for step in range(args.steps):
    lengths = geometry.lengths(state.primary, state.secondary)
    water = hydraulic(geometry.endpoints, geometry.node_lengths, lengths,
        state.diameter, geometry.inlet, geometry.outlet, total_flow, active=state.active)
    if water['clogged']:
        raise RuntimeError('The small check unexpectedly lost a conducting path')
    access = np.asarray(geometry.incidence.T @ (state.primary > 0)).ravel() > 0
    solute = transport(geometry.endpoints, geometry.node_lengths, lengths,
        state.diameter, water['flow'], geometry.inlet, chemistry,
        dissolving=access, modes=2)
    dissolved, precipitated = solute['dissolved_rate'], solute['precipitated_rate']
    update = choose_step(geometry, state, lengths, dissolved, precipitated,
        gamma=chemistry.gamma, dmin=dmin, max_dt=args.max_dt)
    state, event = advance(geometry, state, update, dissolved=dissolved,
        precipitated=precipitated, gamma=chemistry.gamma, dmin=dmin)
    maximum_water_residual = max(maximum_water_residual, water['conservation_residual'])
    maximum_solute_defect = max(maximum_solute_defect, solute['integrated_balance_defect'])
    maximum_algebraic_residual = max(maximum_algebraic_residual,
        water['algebraic_residual'], solute['algebraic_residual'])
    if solute['minimum_concentration'] < -1e-10:
        raise AssertionError('Negative solute concentration')

# Global inventory and conservation checks
inventory_defect = max(abs(state.primary.sum()+state.dissolved-initial_primary),
                       abs(state.secondary.sum()-state.precipitated))
checks = [maximum_water_residual, maximum_solute_defect,
          maximum_algebraic_residual, float(inventory_defect)]
if not np.isfinite(checks).all() or max(checks) > 1e-8:
    raise AssertionError(f'Mineral balance defects: {checks}')
if not all(np.isfinite(value).all() for value in (state.diameter, state.primary, state.secondary)):
    raise FloatingPointError('Nonfinite mineral state')
if MPI.COMM_WORLD.rank == 0:
    print('RESULT ' + json.dumps(dict(example='mineral replacement', pores=len(diameter),
        steps=state.step, scaled_time=chemistry.scaled_time(state.time),
        water_residual=maximum_water_residual, solute_defect=maximum_solute_defect,
        algebraic_residual=maximum_algebraic_residual, inventory_defect=float(inventory_defect))))
