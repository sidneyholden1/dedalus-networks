"""Small physical-solution comparisons with the retained explicit paper code."""

import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from mpi4py import MPI


ROOT = Path(os.environ.get('DEDALUS_PAPER_DIR', Path(__file__).resolve().parents[2] / 'paper'))
pytestmark = [
    pytest.mark.skipif(MPI.COMM_WORLD.size != 1, reason='Bounded serial paper checks'),
    pytest.mark.skipif('DEDALUS_PAPER_DIR' not in os.environ and not ROOT.is_dir(),
                       reason='Paper integration checks require the companion source checkout'),
]


def module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, result)
    spec.loader.exec_module(result)
    return result


def run(monkeypatch, path, arguments=()):
    monkeypatch.setattr(sys, 'argv', [str(path), *arguments])
    return runpy.run_path(str(path))


@pytest.mark.parametrize('application,entry,reference,arguments', [
    ('rods', 'transient.py', 'transient_reference.py', ()),
    ('scaling', 'benchmark.py', 'explicit_reference.py', ()),
    ('scaling', 'benchmark.py', 'explicit_reference.py', ('--family', 'hub', '--size', '1')),
])
def test_native_heterogeneous_evolution(monkeypatch, application, entry, reference, arguments):
    directory = ROOT / application
    monkeypatch.syspath_prepend(str(directory))
    module(monkeypatch, 'inputs', directory / 'inputs.py')
    old = run(monkeypatch, directory / reference, arguments)
    new = run(monkeypatch, directory / entry, arguments)
    if application == 'rods':
        old_fields = [b[1] for b in old['buckets']] + [old['V']]
        new_fields = [b[1] for b in new['buckets']] + [new['V']]
        edge_fields = new_fields[:-1]
    else:
        old_fields = [b[0] for b in old['buckets']]
        new_fields = [*new['u_fields'].values(), *new['v_fields'].values()]
        edge_fields = new_fields
    errors = []
    for a, b in zip(old_fields, new_fields, strict=True):
        a.change_scales(1); b.change_scales(1)
        error = float(np.max(np.abs(a['c'] - b['c'])))
        errors.append(error)
        np.testing.assert_allclose(a['c'], b['c'], rtol=0, atol=3e-11)
    assert len(new['problem'].tau_fields) == 2*len(edge_fields)
    assert all(field.regularity == 'continuous' for field in edge_fields)
    print('TRANSLATION ' + json.dumps(dict(application=application,
          arguments=list(arguments), maximum_coefficient_difference=max(errors), steps=new['args'].steps)))


def test_native_optical_pencil_and_modes(monkeypatch):
    directory = ROOT / 'photonic'
    monkeypatch.syspath_prepend(str(directory))
    module(monkeypatch, 'photonic_carrier', directory / 'carrier_reference.py')
    old = run(monkeypatch, directory / 'eigenmodes.py')
    module(monkeypatch, 'photonic_carrier', directory / 'photonic_carrier.py')
    new = run(monkeypatch, directory / 'eigenmodes.py')
    a, b = old['solver'].subproblems[0], new['solver'].subproblems[0]
    errors = {}
    for name in ('L_min', 'M_min'):
        ma = a.pre_left.T @ getattr(a, name) @ a.pre_right.T
        mb = b.pre_left.T @ getattr(b, name) @ b.pre_right.T
        errors[name] = float(np.max(np.abs((ma-mb).data), initial=0))
        assert errors[name] < 3e-11
    errors['eigenvalues'] = float(np.max(np.abs(np.sort_complex(old['k'])-np.sort_complex(new['k']))))
    assert errors['eigenvalues'] < 1e-10
    assert len(new['application'].problem.tau_fields) == 2
    print('TRANSLATION ' + json.dumps(dict(application='optics', **errors)))


def test_native_mineral_pressure_transport_and_evolution(monkeypatch):
    directory = ROOT / 'mineral'
    monkeypatch.syspath_prepend(str(directory))
    reference = module(monkeypatch, 'mineral_dedalus', directory / 'explicit_reference.py')
    previous_transport = reference.transport
    # The old entry explicitly selected its exact uniform-pore branch.
    monkeypatch.setattr(reference, 'transport',
                        lambda *args, **kw: previous_transport(*args, fitted=True, **kw))
    old = run(monkeypatch, directory / 'evolve.py')
    module(monkeypatch, 'mineral_dedalus', directory / 'mineral_dedalus.py')
    new = run(monkeypatch, directory / 'evolve.py')
    errors = {}
    pairs = {key: (old['water'][key], new['water'][key]) for key in ('pressure', 'flow')}
    pairs.update({key: (old['solute'][key], new['solute'][key]) for key in (
        'b', 'c', 'node_b', 'node_c', 'dissolved_rate', 'precipitated_rate')})
    pairs.update({key: (getattr(old['state'], key), getattr(new['state'], key))
                  for key in ('diameter', 'primary', 'secondary')})
    for key, (a, b) in pairs.items():
        errors[key] = float(np.max(np.abs(a-b)))
        np.testing.assert_allclose(a, b, rtol=3e-11, atol=3e-12)
    print('TRANSLATION ' + json.dumps(dict(application='mineral', **errors)))


def test_native_direct_optical_field(monkeypatch):
    directory = ROOT / 'photonic'
    monkeypatch.syspath_prepend(str(directory))
    reference = module(monkeypatch, 'physical_reference', directory / 'physical_reference.py')
    native = module(monkeypatch, 'photonic_dedalus', directory / 'photonic_dedalus.py')
    from photonic_reference import dielectric_interval, interval_resonance
    graph = dielectric_interval(length=1., index=1.5)
    exact = interval_resonance(1, length=1., index=1.5)
    eigenvalues = []
    for implementation in (reference, native):
        application = implementation.build_problem(graph, 24)
        solver = implementation.solve(application, modes=2, target=exact+.1, ncv=20)
        mode = int(np.argmin(abs(solver.eigenvalues-exact)))
        audit, _ = implementation.small_mode_audit(application, solver, mode)
        assert max(audit[key] for key in ('differential', 'continuity', 'vertex', 'bond')) < 1e-8
        eigenvalues.append(solver.eigenvalues[mode])
    assert len(application.problem.tau_fields) == 2
    np.testing.assert_allclose(eigenvalues, exact, rtol=0, atol=1e-10)
    print('TRANSLATION ' + json.dumps(dict(application='optics_direct',
          eigenvalue_difference=float(abs(eigenvalues[0]-eigenvalues[1])),
          analytic_eigenvalue_error=float(abs(eigenvalues[1]-exact)))))


def test_mineral_pressure_defect_and_cached_reset(monkeypatch):
    directory = ROOT / 'mineral'
    monkeypatch.syspath_prepend(str(directory))
    native = module(monkeypatch, 'mineral_dedalus', directory / 'mineral_dedalus.py')
    from mineral_lifecycle import clear_pressure
    endpoints = np.array([[0, 1], [1, 2], [2, 3]])
    ones = np.ones(3)
    cache = {}
    try:
        first = native.hydraulic(endpoints, ones, ones, ones, [0], [3], 1., cache=cache)
        network, owned, P, k, boundary, defect, solver = cache['kernel']
        native.put(boundary, np.zeros(4))
        native.put(defect, [0., 1., 0., 0.])
        native.solve_pressure_system(solver)
        np.testing.assert_allclose(native.get(P).ravel(), [0., 2/3, 1/3, 0.], atol=2e-12)
        second = native.hydraulic(endpoints, ones, ones, ones, [0], [3], 1., cache=cache)
        assert second['immutable_pressure_graph_reused']
        np.testing.assert_allclose(first['pressure'], second['pressure'], rtol=2e-12)
        np.testing.assert_allclose(first['flow'], second['flow'], rtol=2e-12)
    finally:
        clear_pressure(cache)


def test_native_direct_mineral_transport_with_reversed_pore(monkeypatch):
    directory = ROOT / 'mineral'
    monkeypatch.syspath_prepend(str(directory))
    old = module(monkeypatch, 'mineral_explicit_reference', directory / 'explicit_reference.py')
    module(monkeypatch, 'mineral_dedalus', directory / 'mineral_dedalus.py')
    new = module(monkeypatch, 'mineral_transport_verification',
                 Path(__file__).with_name('mineral_transport_verification.py'))
    from mineral_reference import Chemistry
    chemistry = Chemistry(gamma=.3)
    endpoints = np.array([[0, 2], [2, 1], [2, 3]])
    flow = chemistry.mean_initial_flow*np.array([.6, 1., -.4])
    options = dict(modes=16, fitted=False)
    args = (endpoints, np.ones(3), np.array([.9, 1.3, .7]),
            np.full(3, chemistry.d0), flow, [0, 3], chemistry)
    reference = old.transport(*args, **options)
    native = new.transport(*args, **options)
    errors = {}
    for key in ('b', 'c', 'node_b', 'node_c', 'dissolved_rate', 'precipitated_rate'):
        errors[key] = float(np.max(abs(reference[key]-native[key])))
        np.testing.assert_allclose(reference[key], native[key], rtol=3e-11, atol=3e-12)
    assert native['integrated_balance_defect'] < 1e-10
    print('TRANSLATION ' + json.dumps(dict(application='mineral_direct', **errors)))
