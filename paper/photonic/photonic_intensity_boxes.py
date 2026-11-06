"""Intensity-defined square crops of a saved travelling-wave mode; no EVP solve."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


from photonic_reference import integrated_exponential


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_moment_exponential(rate, length):
    """Integral of x exp(rate*x), with a series at the removable singularity."""
    rate, length = np.broadcast_arrays(rate, length)
    z = rate * length
    value = np.empty_like(z, dtype=complex)
    small = abs(z) < .01
    term = np.ones_like(z[small])
    series = term / 2
    for n in range(1, 12):
        term = term * z[small] / n
        series += term / (n + 2)
    value[small] = series
    w = z[~small]
    value[~small] = (np.exp(w) * (w - 1) + 1) / w**2
    return length**2 * value


def clipped_parameters(segments, low, high):
    """Slab intersection: exact straight-edge parameter interval inside a box."""
    start = segments[:, 0]
    direction = segments[:, 1] - start
    enter = np.zeros(len(start))
    leave = np.ones(len(start))
    for axis in range(2):
        moving = direction[:, axis] != 0
        a = np.full(len(start), -np.inf)
        b = np.full(len(start), np.inf)
        a[moving] = (low[axis] - start[moving, axis]) / direction[moving, axis]
        b[moving] = (high[axis] - start[moving, axis]) / direction[moving, axis]
        enter = np.maximum(enter, np.minimum(a, b))
        leave = np.minimum(leave, np.maximum(a, b))
        outside = ~moving & ((start[:, axis] < low[axis]) | (start[:, axis] > high[axis]))
        leave[outside] = -1
    return enter, np.maximum(enter, leave)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[.90, .95, .99],
                        help='Fractions of internal integrated intensity to enclose.')
    args = parser.parse_args()
    if not all(0 < fraction < 1 for fraction in args.fractions):
        parser.error('Fractions must lie strictly between zero and one.')
    selection = json.loads((args.selection / 'selection.json').read_text())
    source = args.selection / 'selected-fields.npz'
    assert selection['passed'] and sha(source) == selection['selected_fields_sha256']
    with np.load(source, allow_pickle=False) as data:
        mask = data['internal']
        segments = data['positions_um'][data['endpoints']][mask]
        lengths = data['lengths_um'][mask]
        amplitudes = data['amplitudes'][0][mask]
        stored = data['edge_I2'][0][mask]
        q = complex(data['k'][0]) * data['index'][mask]
        domain = [data['positions_um'].min(axis=0), data['positions_um'].max(axis=0)]
    assert np.allclose(np.linalg.norm(np.diff(segments, axis=1)[:, 0], axis=1), lengths)
    a = amplitudes[:, 0]
    b = amplitudes[:, 1] * np.exp(1j * q * lengths)
    coefficients = np.array([abs(a)**2, abs(b)**2, a*b.conj(), a.conj()*b])
    rates = np.array([-2*q.imag, 2*q.imag, 2j*q.real, -2j*q.real])
    full = np.sum(coefficients * integrated_exponential(rates, lengths), axis=0).real
    error = float(np.max(abs(full - stored) / np.maximum(stored, 1e-30)))
    assert error < 1e-9 and np.all(full >= 0)
    total = float(full.sum())
    moment = np.sum(coefficients * first_moment_exponential(rates, lengths), axis=0).real
    assert np.all(moment >= 0) and np.all(moment <= lengths * full)
    center = np.sum(segments[:, 0] * full[:, None] +
                    (segments[:, 1] - segments[:, 0]) * (moment / lengths)[:, None], axis=0) / total

    def intervals_integral(ids, x0, x1):
        return np.sum(coefficients[:, ids] * np.exp(rates[:, ids] * x0) *
                      integrated_exponential(rates[:, ids], x1 - x0), axis=0).real

    def mass(low, high):
        t0, t1 = clipped_parameters(segments, low, high)
        inside = (t0 == 0) & (t1 == 1)
        partial = np.flatnonzero((t1 > t0) & ~inside)
        integrals = intervals_integral(partial, t0[partial]*lengths[partial], t1[partial]*lengths[partial])
        assert np.all(integrals >= -1e-15)
        return float((full[inside].sum() + integrals.sum()) / total)

    boxes = []
    for target in sorted(set(args.fractions)):
        left, right = 0., float(np.max(abs(segments - center)))
        assert mass(center-right, center+right) > 1-1e-12
        for _ in range(42):
            half = (left + right) / 2
            if mass(center-half, center+half) < target:
                left = half
            else:
                right = half
        low, high = center-right, center+right
        fraction = mass(low, high)
        assert abs(fraction-target) < 1e-8
        boxes.append(dict(target_fraction=target, actual_fraction=fraction,
            width_um=2*right, bounds_um=[low.tolist(), high.tolist()],
            inside_graph_bounds=bool(np.all(low>domain[0]) and np.all(high<domain[1]))))

    # Independent quadrature of the longest partial edge intervals in each box.
    # Subintervals <= .25 um resolve the oscillatory density with 16-point Gauss.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    errors = []
    centroid_errors = []
    for box in boxes:
        t0, t1 = clipped_parameters(segments, *np.array(box['bounds_um']))
        partial = np.flatnonzero((t1>t0) & ((t0>0) | (t1<1)))
        chosen = partial[np.argsort((t1-t0)[partial]*lengths[partial])[-12:]]
        for edge in chosen:
            x0, x1 = t0[edge]*lengths[edge], t1[edge]*lengths[edge]
            partitions = np.linspace(x0, x1, int(np.ceil((x1-x0)/.25))+1)
            half = np.diff(partitions)/2
            x = (partitions[:-1]+half)[:, None] + half[:, None]*nodes
            u = amplitudes[edge, 0]*np.exp(1j*q[edge]*x) + amplitudes[edge, 1]*np.exp(1j*q[edge]*(lengths[edge]-x))
            quadrature = np.sum(half[:, None]*weights*abs(u)**2)
            analytic = intervals_integral(np.array([edge]), x0, x1)[0]
            errors.append(float(abs(quadrature-analytic)/max(abs(analytic),1e-30)))
    for edge in np.argsort(full)[-12:]:
        partitions = np.linspace(0, lengths[edge], int(np.ceil(lengths[edge]/.25))+1)
        half = np.diff(partitions)/2
        x = (partitions[:-1]+half)[:, None] + half[:, None]*nodes
        u = amplitudes[edge, 0]*np.exp(1j*q[edge]*x) + amplitudes[edge, 1]*np.exp(1j*q[edge]*(lengths[edge]-x))
        numerical = np.sum(half[:, None]*weights*x*abs(u)**2)
        centroid_errors.append(float(abs(numerical-moment[edge])/max(abs(moment[edge]),1e-30)))
    assert max(errors) < 1e-9 and max(centroid_errors) < 1e-9
    peak = np.asarray(selection['selected']['peak_position_um'])
    report = dict(passed=True, preliminary=selection['preliminary'], fields=str(source),
        fields_sha256=sha(source), script_sha256=sha(__file__),
        reference_script_sha256=sha(Path(__file__).with_name('photonic_reference.py')),
        mode_index=selection['selected']['mode_index'], center_um=center.tolist(),
        definition='Smallest axis-aligned square grown about the exact internal intensity centroid; analytic edge integrals clipped at the box boundary.',
        density='abs(u_e(x))**2 integrated over internal edges only; no colour-floor truncation',
        total_internal_intensity=total, boxes=boxes,
        old_peak_centered_900um_box_fraction=mass(peak-450, peak+450),
        checks=dict(full_edge_integrals_max_relative_error=error,
            clipped_quadrature_count=len(errors), clipped_quadrature_max_relative_error=max(errors),
            centroid_moment_quadrature_count=len(centroid_errors), centroid_moment_max_relative_error=max(centroid_errors)))
    args.output.parent.mkdir(mode=0o700, exist_ok=False)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
