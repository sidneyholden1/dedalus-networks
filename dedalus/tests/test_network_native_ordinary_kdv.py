"""Small translation audit against the ordinary-KdV paper discretization.

This exercises the actual paper entry point and its retained explicit reference.
"""

import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from mpi4py import MPI
from scipy import sparse


ROOT = Path(os.environ.get('DEDALUS_PAPER_DIR', Path(__file__).resolve().parents[2] / 'paper'))


@pytest.mark.skipif('DEDALUS_PAPER_DIR' not in os.environ and not ROOT.is_dir(),
                    reason='Paper integration checks require the companion source checkout')
@pytest.mark.skipif(MPI.COMM_WORLD.size != 1, reason='Serial adjoint translation audit')
def test_native_ordinary_kdv_matches_paper_discretization(monkeypatch):
    directory = ROOT / 'kdv'
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, 'argv', [str(directory / 'evolve.py')])
    # Execute the public example, including its three steps and diagnostics.
    result = runpy.run_path(str(directory / 'evolve.py'))
    new = result['model']
    spec = importlib.util.spec_from_file_location('ordinary_kdv_explicit_reference',
                                                 directory / 'explicit_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    old = reference.build(n=24, periodic_size=3, vertex_interface=True,
                          network_solver='dtn')
    a, b = old['solver'], new['solver']
    sa, sb = a.subproblems[0], b.subproblems[0]
    assert len(a.subproblems) == len(b.subproblems) == 1
    # Distributional div contributes minus the outward flux. The old code
    # wrote the opposite constraint. Reverse that row AND its multiplier;
    # continuity and straight-through slope rows have identical ordering.
    signs = np.ones(sa.L_min.shape[0])
    signs[-new['basis'].network.num_vertices:] = -1
    sign = sparse.diags(signs)
    errors = {}
    for name in ('M_min', 'L_min'):
        raw_a = sa.pre_left.T @ getattr(sa, name) @ sa.pre_right.T
        raw_b = sb.pre_left.T @ getattr(sb, name) @ sb.pre_right.T
        difference = (sign @ raw_a @ sign - raw_b).tocsr()
        errors[name] = float(np.max(np.abs(difference.data), initial=0))
        assert errors[name] < 2e-10
    # Probe arbitrary retained states, including the highest modes: smooth
    # initial data alone can conceal an incorrect weighted RHS projection.
    saved_old = old['u']['c'].copy()
    saved_new = new['u']['c'].copy()
    rng = np.random.default_rng(20261105)
    coefficients = rng.normal(size=saved_old.shape)*.01
    old['u']['c'] = coefficients
    new['u']['c'] = coefficients
    fa = a.problem.equations[0]['F'].evaluate()['c'].copy()
    fb = b.problem.equations[0]['F'].evaluate()['c'].copy()
    errors['nonlinear_forcing'] = float(np.max(np.abs(fa-fb)))
    np.testing.assert_allclose(fa, fb, rtol=0, atol=3e-11)
    old['u']['c'] = saved_old
    new['u']['c'] = saved_new
    for _ in range(3):
        a.step(.000125)
    errors['three_step_coefficients'] = float(np.max(np.abs(old['u']['c']-new['u']['c'])))
    np.testing.assert_allclose(old['u']['c'], new['u']['c'], rtol=0, atol=3e-11)
    old_record, _, _ = reference.capture(old)
    errors['instantaneous_rate'] = float(np.max(np.abs(old['ut']['c']-new['ut']['c'])))
    np.testing.assert_allclose(old['ut']['c'], new['ut']['c'], rtol=0, atol=3e-10)
    for key in ('mass', 'squared_u_l2', 'squared_w_l2'):
        assert abs(old_record[key] - result['final'][key]) < 3e-10
    assert result['final']['constraint_max'] < 1e-10
    assert len(b.problem._hidden_network_constraint_fields) == 6
    assert b.problem.tau_fields == ()
    assert new['u'].regularity == 'continuous'
    print('TRANSLATION ' + json.dumps(errors, sort_keys=True))
