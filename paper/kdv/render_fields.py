"""Static snapshots and an orientation diagram from qualified KdV data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', dest='source', type=Path, default=Path(__file__).resolve().parents[1]/'data/kdv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-halfwidth-cells', type=float, default=.95)
    parser.add_argument('--grid-width', type=float, default=.64)
    parser.add_argument('--grid-color', default='#4a6376')
    parser.add_argument('--times', type=float, nargs='+', default=[0., 6., 20., 200.])
    parser.add_argument('--style-reference', type=Path,
                        help='Previous figure provenance supplying the initial amplitude scale')
    args = parser.parse_args()
    began = time.monotonic()
    args.validation = args.source/'selected-frames-validation.json'
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.tri import Triangulation
    from plotting import sample
    qualified = json.loads(args.validation.read_text())
    if not qualified['passed']:
        raise ValueError('The supplied validation does not cover this trajectory')
    result = json.loads((args.source / 'result.json').read_text())
    if not result.get('complete', result.get('render_only', False) and qualified['passed']):
        raise ValueError('Cannot render an incomplete stage as a qualified result')
    config = result['config']
    requested_times = np.array(args.times)
    if not 1 <= len(requested_times) <= 4 or not np.isfinite(requested_times).all():
        raise ValueError('Request between one and four finite snapshot times')
    if 'validation' in qualified and not qualified['validation']['passed']:
        raise ValueError('Stage validation failed')
    trajectory = args.source / 'trajectory.npz'
    source_sha_before = digest(trajectory)
    with np.load(trajectory) as saved:
        times = saved['t']
        indices = np.array([np.argmin(abs(times - t)) for t in requested_times])
        if np.max(abs(times[indices] - requested_times)) >= 1e-8:
            raise ValueError('An exact requested saved frame is missing')
        fields = saved['u'][indices]
        xy, endpoints = saved['xy'], saved['endpoints']
        lengths, directions = saved['lengths'], saved['edge_directions']
    if (config['periodic_size'] != 16 or config['n'] != 64 or
            len(xy) != 256 or len(endpoints) != 512):
        raise ValueError('Unexpected source geometry or resolution')
    horizontal = np.all(directions == [1, 0], axis=1)
    vertical = np.all(directions == [0, 1], axis=1)
    if horizontal.sum() != 256 or vertical.sum() != 256:
        raise ValueError('A right/up key would not describe all saved edges')

    period = config['period']
    center = xy[endpoints[config['source_edge'], 0]] + [8., 0.]
    lower = center - period / 2
    displayed = (xy - lower) % period + lower
    origins = displayed[endpoints[:, 0]]
    deltas = lengths[:, None] * directions
    ends = origins + deltas
    wrapped_ends = (ends - lower) % period + lower
    if not np.allclose(wrapped_ends, displayed[endpoints[:, 1]], atol=1e-12):
        raise ValueError('Outgoing edges do not match the stored periodic graph')
    padding = .15 * config['edge_length']
    bounds = (lower[0] - padding, lower[0] + period + padding,
              lower[1] - padding, lower[1] + period + padding)
    segments = np.stack((origins, ends), axis=1)
    s = np.linspace(0, 1, 513)
    points = origins[:, None, :] + s[None, :, None] * deltas[:, None, :]
    base = np.arange(len(endpoints) * len(s) * 2).reshape(len(endpoints), len(s), 2)
    triangles = np.stack((np.stack((base[:, :-1, 0], base[:, 1:, 0], base[:, :-1, 1]), axis=-1),
                          np.stack((base[:, 1:, 0], base[:, 1:, 1], base[:, :-1, 1]), axis=-1)),
                         axis=2).reshape(-1, 3)
    limit = 1.1
    if args.style_reference:
        previous_style = json.loads(args.style_reference.read_text())
        if (not previous_style['passed'] or previous_style['period'] != period or
                previous_style['edge_length'] != config['edge_length'] or
                previous_style['initial_halfwidth_cells'] != args.initial_halfwidth_cells):
            raise ValueError('The reference style does not match this graph and width scale')
        initial_peak = previous_style['initial_peak']
    else:
        zero_indices = np.flatnonzero(abs(requested_times) < 1e-12)
        if not len(zero_indices):
            raise ValueError('A noninitial snapshot needs an existing style reference')
        initial_peak = float(np.max(abs(sample(fields[zero_indices[0]], s))))
    initial_halfwidth = args.initial_halfwidth_cells * config['edge_length']
    width_coefficient = 2 * initial_halfwidth / initial_peak
    if not np.isfinite(width_coefficient) or width_coefficient <= 0 or args.grid_width <= 0:
        raise ValueError('Invalid ribbon or grid display scale')
    # Colour now carries sign only. All visible pulses have the same opaque
    # blue/gold colour; magnitude is encoded by width. The grey topology is a
    # separate line, so the coloured ribbon has zero width at zero amplitude.
    sign_zero_tolerance = 1e-12
    norm = Normalize(vmin=-1, vmax=1)
    palette = ['#65dfff', args.grid_color, '#ffe080']
    cmap = ListedColormap(palette, name='kdv_sign_only')
    background, foreground, neutral = '#05080d', '#f1f5f9', args.grid_color
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    args.output.mkdir(parents=True,exist_ok=False)

    def setup(ax):
        ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal')
        ax.set_axis_off()
        ax.set_facecolor(background)

    def orientation_key(fig, column_center):
        # Both keys occupy a separate narrow column beside the four snapshots.
        arrow_inches = .44
        dx, dy = arrow_inches / fig.get_figwidth(), arrow_inches / fig.get_figheight()
        origin = (column_center - dx / 2, .325 - dy / 2)
        right = origin[0] + dx
        fig.text(column_center, .44, 'Edge\norientation', ha='center', va='center',
                 color=foreground, fontsize=14)
        style = dict(arrowstyle='-|>', color=foreground, linewidth=1.45,
                     mutation_scale=12, shrinkA=0, shrinkB=0)
        for end in ((right, origin[1]), (origin[0], origin[1] + dy)):
            fig.add_artist(FancyArrowPatch(origin, end, transform=fig.transFigure, **style))
        return dict(right_edge_fraction=float(right), origin=list(map(float, origin)),
                    arrow_inches=arrow_inches)

    # Preserve the physical size of each snapshot while appending the key
    # column, rather than shrinking the networks to make room for the legends.
    panels_inches = 6 * len(requested_times)
    figure_size = (panels_inches + 1.6, 6.55)
    fig = plt.figure(figsize=figure_size, facecolor=background)
    panel_scale = 24 / fig.get_figwidth()
    panel_width, gap, left = np.array([.239, .009, .008]) * panel_scale
    column_center = (panels_inches + .7) / fig.get_figwidth()
    values_metadata = []
    for panel, (u, target) in enumerate(zip(fields, requested_times)):
        values = sample(u, s)
        if not np.isfinite(values).all() or np.max(abs(values)) > limit:
            raise ValueError('Field outside the shared fixed colour scale')
        normal = np.stack((-directions[:, 1], directions[:, 0]), axis=-1)
        half = width_coefficient * abs(values) / 2
        offset = half[..., None] * normal[:, None, :]
        vertices = np.stack((points-offset, points+offset), axis=2).reshape(-1, 2)
        colors = np.repeat(values[..., None], 2, axis=-1).ravel()
        colors = np.where(abs(colors) <= sign_zero_tolerance, 0., np.sign(colors))
        triangulation = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
        ax = fig.add_axes([left + panel * (panel_width + gap), .018, panel_width, .876])
        setup(ax)
        # A brighter, fixed-width topology underlay keeps the unexcited grid legible
        # after shrinking four full networks to a single paper-width strip.
        ax.add_collection(LineCollection(segments, colors=neutral, linewidths=args.grid_width, zorder=0))
        ax.tripcolor(triangulation, colors, shading='gouraud', cmap=cmap, norm=norm,
                     alpha=1., rasterized=True, zorder=1)
        fig.text(left + panel * (panel_width + gap) + .005 * panel_scale, .942,
                 r'$\mathbf{t=' + f'{target:g}' + '}$', color=foreground,
                 fontsize=25, ha='left', va='center')
        values_metadata.append(dict(time=float(target), minimum=float(values.min()),
                                    maximum=float(values.max())))
    orientation_metadata = orientation_key(fig, column_center)
    for y, colour, label in ((.68, palette[0], r'$u<0$'),
                              (.60, palette[2], r'$u>0$')):
        x = (panels_inches + .08) / fig.get_figwidth()
        fig.add_artist(Rectangle((x, y - .014), .24 / fig.get_figwidth(), .028,
                                 transform=fig.transFigure,
                                 facecolor=colour, edgecolor='none'))
        fig.text(x + .36 / fig.get_figwidth(), y, label, ha='left', va='center',
                 color=foreground, fontsize=18)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headers = [text.get_window_extent(renderer) for text in fig.texts]
    if any(a.overlaps(b) for i, a in enumerate(headers) for b in headers[i+1:]):
        raise ValueError('Overlapping panel titles or orientation label')
    fig.savefig(args.output / 'kdv-time-snapshots.png', dpi=300, facecolor=background)
    fig.savefig(args.output / 'kdv-time-snapshots.pdf', dpi=450, facecolor=background)
    plt.close(fig)
    print(f'{len(requested_times)}-panel PNG/PDF exported', flush=True)

    # A separate optional panel contains actual edge arrows, one per physical
    # edge. The diagram uses exactly the same outgoing-period representation.
    diagram = plt.figure(figsize=(7.2, 7.65), facecolor=background)
    ax = diagram.add_axes([.025, .018, .95, .894])
    setup(ax)
    ax.add_collection(LineCollection(segments, colors='#587185', linewidths=.8))
    midpoints = origins + .45 * deltas
    arrows = .24 * deltas
    ax.quiver(midpoints[:, 0], midpoints[:, 1], arrows[:, 0], arrows[:, 1],
              angles='xy', scale_units='xy', scale=1, color='#dbe9f3',
              width=.0021, headwidth=3.5, headlength=4.5, headaxislength=4.2,
              pivot='middle', zorder=3)
    ax.scatter(displayed[:, 0], displayed[:, 1], s=5, c='#9cb1c2', zorder=4)
    diagram.text(.5, .963, 'Edge orientations', ha='center', va='center',
                 fontsize=23, color=foreground)
    diagram.savefig(args.output / 'kdv-edge-orientations.png', dpi=220, facecolor=background)
    diagram.savefig(args.output / 'kdv-edge-orientations.pdf', facecolor=background)
    plt.close(diagram)

    source_sha = digest(trajectory)
    if source_sha != qualified.get('trajectory_sha256', source_sha_before):
        raise ValueError('Saved trajectory checksum does not match validation')
    caption = (
        'Ordinary KdV on a periodic 16 by 16 square metric graph at t='
        + ','.join(f'{t:g}' for t in requested_times) + '. '
        'All panels use fixed opaque gold for positive values and blue for negative values; '
        'colour brightness and opacity do not depend on magnitude. Magnitude is represented '
        f'by ribbon width {width_coefficient:g}|u| in graph coordinates. '
        'The coloured ribbon has zero width at zero amplitude; the neutral grid is drawn separately. '
        f'The initial pulse reaches {args.initial_halfwidth_cells:g} grid spacings '
        'on either side of its source edge; the same width encoding applies to every panel. '
        'Values with absolute magnitude at most1e-12 are coloured neutral to avoid encoding '
        'floating-point roundoff as a sign. A brighter fixed-width underlay shows the graph. '
        'The common orientation key gives the chosen positive edge coordinates: '
        'horizontal edges point right,vertical edges point up. These are fixed '
        'coordinate orientations,not instantaneous wave-propagation directions. '
        'Each of512physical edges appears once at its full length8; outgoing '
        'right/top ends identify with corresponding left/bottom vertices. '
        'Snapshots are exact saved frames of the qualified N64,dt0.000125 '
        'Dedalus solution; no temporal interpolation or numerical rerun.\n')
    (args.output / 'caption.txt').write_text(caption)
    outputs = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
               for p in (args.output / name for name in
                         ('kdv-time-snapshots.png', 'kdv-time-snapshots.pdf',
                          'kdv-edge-orientations.png', 'kdv-edge-orientations.pdf', 'caption.txt'))}
    provenance = dict(passed=True, source=str(args.source), source_sha256=source_sha,
        validation=str(args.validation), validation_sha256=digest(args.validation),
        requested_times=requested_times.tolist(), actual_times=times[indices].tolist(),
        original_frame_indices=indices.tolist(), field_extrema=values_metadata,
        edges=512, vertices=256, rightward_edges=int(horizontal.sum()),
        upward_edges=int(vertical.sum()), period=period, edge_length=8.,
        full_length_outgoing_periodic_edges=True, bounds=list(map(float, bounds)),
        numerical_rerun=False, temporal_interpolation=False, spatial_samples_per_edge=513,
        color_scale='sign only; fixed opaque blue/gold', palette=palette,
        sign_zero_tolerance=sign_zero_tolerance, ribbon_alpha=1.,
        width_coefficient=width_coefficient, width_encoding=f'{width_coefficient:g}*abs(u)',
        minimum_colored_width=0.,
        initial_halfwidth_cells=args.initial_halfwidth_cells,
        initial_halfwidth_graph_units=initial_halfwidth, initial_peak=initial_peak,
        underlay_width_points=args.grid_width, underlay_color=neutral,
        figure_size_inches=list(figure_size),
        legend_column_inches=[float(panels_inches), panels_inches + 1.4],
        style_reference=str(args.style_reference) if args.style_reference else None,
        style_reference_sha256=digest(args.style_reference) if args.style_reference else None,
        orientation_key_location='separate narrow column to the right of all snapshots',
        orientation_key_geometry=orientation_metadata,
        color_legend=dict(blue='u<0',yellow='u>0',location='right column, above orientation key',
                          magnitude_encoding='ribbon width'),
        script_sha256=digest(Path(__file__)), job_id=os.environ.get('SLURM_JOB_ID'),
        elapsed_seconds=time.monotonic()-began,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, outputs=outputs)
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k:provenance[k] for k in ('passed','elapsed_seconds','peak_rss_kib','actual_times')}), flush=True)


if __name__ == '__main__':
    main()
