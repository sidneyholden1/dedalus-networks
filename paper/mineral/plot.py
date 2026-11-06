"""Redraw Figure 4 from the saved mineral-replacement fields and histories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.text import Text


INPUT_SHA = '0f6781648ad4f72aebed2714bdce124d5b15decc1369433b37f4a1d7550e0a38'
CURVE_COLOR = '#15566e'
COLORMAP = 'viridis_r'
COLOR_LIMITS = (0., 12.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def periodic_segments(z, geometry):
    points, ends = z['geometry_points'], z['geometry_endpoints']
    width = geometry['nx'] / geometry['original_mean_node_distance']
    segments = points[ends].copy()
    segments[:, 1, 0] += z['geometry_edge_shift'] * width
    lo, hi = segments[:, :, 0].min(axis=1), segments[:, :, 0].max(axis=1)
    first = np.ceil(-hi / width).astype(int)
    last = np.floor((width - lo) / width).astype(int)
    pieces, indices = [], []
    for shift in range(int(first.min()), int(last.max()) + 1):
        selected = np.flatnonzero((first <= shift) & (last >= shift))
        pieces.append(segments[selected] + [shift * width, 0.])
        indices.append(selected)
    ids = np.concatenate(indices)
    assert np.array_equal(np.unique(ids), np.arange(len(ends)))
    return np.concatenate(pieces), ids, width, (points[:, 1].min(), points[:, 1].max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', dest='input', type=Path, default=Path(__file__).resolve().parents[1]/'data/mineral/plot-data.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--line-weight', type=float, default=1.8)
    args = p.parse_args()
    reference = json.loads((Path(__file__).resolve().parents[1]/'figure-style.json').read_text())
    geometry_scale = (6.5/reference['inclusion_scale'])/7.65
    font = reference['source_font_tex_points']
    title_font = reference['source_heading_font_tex_points']
    tick_font = font
    axis_font = 12.
    if not 1. <= args.line_weight <= 3.:
        p.error('--line-weight must be between 1 and 3')
    assert sha(args.input) == INPUT_SHA
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('mineral-channeling.png', 'mineral-channeling.pdf', 'provenance.json'):
        if (args.output / name).exists():
            raise FileExistsError('Preserve prior exports: ' + name)
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'], 'font.size': font,
        'axes.labelsize': axis_font, 'xtick.labelsize': tick_font, 'ytick.labelsize': tick_font,
        'axes.linewidth': .65*geometry_scale, 'savefig.facecolor': 'white',
        'text.latex.preamble': (r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}'
                               if reference else r'\usepackage{amsmath}'),
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    with np.load(args.input, allow_pickle=False) as z:
        meta = json.loads(z['metadata_json'].item())
        audit = json.loads(args.input.with_name('replay-audit.json').read_text())
        assert audit['passed'] and audit['full_replay_completed']
        assert audit['plot_archive']['sha256'] == INPUT_SHA
        assert len(meta['checks']) == 4
        segments, ids, width, (ylo, yhi) = periodic_segments(z, meta['geometry_metadata'])
        times, permeability = z['history__paper_time'], z['history__permeability_ratio']
        assert len(times) == len(permeability) == 12497
        assert np.all(np.diff(times) >= 0) and np.all(permeability > 0)
        cmap = matplotlib.colormaps[COLORMAP]
        norm = Normalize(*COLOR_LIMITS, clip=False)
        assert np.isclose(norm(1 + 1e-10) - norm(1 - 1e-10), 2e-10 / 12)
        fig_width, fig_height = 7.65, 3.63
        fig = plt.figure(figsize=(fig_width*geometry_scale, fig_height*geometry_scale))
        headings = []
        square = 1.55
        records = []
        network_axes = []
        for frame, left in enumerate((.48, 2.15, 3.82, 5.49), 1):
            ax = fig.add_axes([left / fig_width, 1.73 / fig_height, square / fig_width, square / fig_height])
            network_axes.append(ax)
            d = z[f'frame_{frame:02d}__diameter']
            assert len(d) == 29996 and np.all(np.isfinite(d)) and np.all(d > 0)
            ratio = d / meta['d0']
            assert ratio.min() >= COLOR_LIMITS[0] and ratio.max() <= COLOR_LIMITS[1]
            # No diameter bins or threshold jumps in either colour or line width.
            order = np.argsort(ratio[ids], kind='stable')
            weights = np.clip(.10 + .13 * (ratio - 1.), .07, .82)
            weights *= (square / 3.03) * args.line_weight * geometry_scale
            ax.add_collection(LineCollection(
                segments[order], colors=cmap(norm(ratio[ids[order]])),
                linewidths=weights[ids[order]], rasterized=True,
                capstyle='round', joinstyle='round'))
            ax.set(xlim=(0, width), ylim=(yhi, ylo), aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#bdc7ca')
                spine.set_linewidth(.4*geometry_scale)
            check = meta['checks'][frame - 1]
            ix = np.flatnonzero(z['history__step'] == check['step']).item()
            assert times[ix] == check['paper_time']
            assert permeability[ix] == check['permeability_ratio']
            headings.append(ax.text(0, 1.04,
                    rf'({chr(96 + frame)})\quad $t^\prime={times[ix]:.1f}$',
                    transform=ax.transAxes, ha='left', va='bottom', fontsize=title_font))
            records.append(dict(panel=chr(96 + frame), **check,
                                diameter_ratio_min=float(ratio.min()),
                                diameter_ratio_max=float(ratio.max())))

        # Align the lower axes with the full upper row, through the colour bar.
        for network_ax in network_axes:
            network_ax.apply_aspect()
        panel_bounds = network_axes[0].get_position()
        panel_gap = network_axes[1].get_position().x0 - panel_bounds.x1
        colourbar_left = network_axes[-1].get_position().x1 + panel_gap
        colourbar_width = .10 / fig_width
        curve_width = colourbar_left + colourbar_width - panel_bounds.x0
        ax = fig.add_axes([panel_bounds.x0, .48 / fig_height,
                           curve_width, 1.05 / fig_height])
        history_line, = ax.plot(times, np.log(permeability), color=CURVE_COLOR, lw=1.5*geometry_scale)
        assert np.array_equal(history_line.get_xdata(), times)
        assert np.array_equal(history_line.get_ydata(), np.log(permeability))
        ax.set(xlim=(0, 300), ylim=(-.04, 3.22), xlabel=r'$t^\prime$',
               ylabel=r'$\log(K/K_0)$')
        ax.set_xticks(np.arange(0, 301, 50))
        ax.set_yticks([0, 1, 2, 3])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', width=.65*geometry_scale, length=3*geometry_scale,
                       pad=3.5*geometry_scale)
        ax.xaxis.labelpad = 4*geometry_scale
        label_x = -.352 / (curve_width * fig_width)
        ax.yaxis.set_label_coords(label_x, .5)
        ax.yaxis.label.set_verticalalignment('center')
        panel_e = ax.text(label_x, 3., '(e)', transform=ax.get_yaxis_transform(),
                          ha='center', va='center', fontsize=title_font)
        headings.append(panel_e)
        for record in records:
            x, y = record['paper_time'], np.log(record['permeability_ratio'])
            ax.plot(x, y, 'o', ms=4.2*geometry_scale, color=CURVE_COLOR,
                    markeredgecolor='white', markeredgewidth=.7*geometry_scale, zorder=4)
            offset = np.array((-5, 7) if record['panel'] == 'd' else (0, 7))*geometry_scale
            ax.annotate(f"({record['panel']})", (x, y), xytext=offset,
                        textcoords='offset points',
                        ha='right' if record['panel'] == 'd' else 'center',
                        va='bottom', fontsize=tick_font)

        # Use the same gap as between squares and exactly the same height.
        cax = fig.add_axes([colourbar_left, panel_bounds.y0,
                            colourbar_width, panel_bounds.height])
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                            orientation='vertical', ticks=[0, 1, 2, 4, 8, 12])
        cbar.outline.set_linewidth(.4*geometry_scale)
        cbar.ax.tick_params(direction='out', width=.5*geometry_scale, length=2*geometry_scale,
                           pad=2*geometry_scale, labelsize=tick_font)
        colourbar_title = cax.set_title(r'$d/d_0$', fontsize=font, pad=7*geometry_scale)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert np.allclose([cax.get_position().y0, cax.get_position().y1],
                           [panel_bounds.y0, panel_bounds.y1], rtol=0, atol=1e-12)
        assert np.isclose(cax.get_position().x0 - network_axes[-1].get_position().x1,
                          panel_gap, rtol=0, atol=1e-12)
        assert np.isclose(ax.get_position().x1, cax.get_position().x1, rtol=0, atol=1e-12)
        for text_artist in (ax.xaxis.label, ax.yaxis.label, panel_e,
                            colourbar_title, *cax.get_yticklabels()):
            bounds = text_artist.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.y0 >= 0
            assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
        if reference:
            for text in fig.findobj(Text):
                if text.get_visible() and text.get_text():
                    assert text.get_fontsize() == (title_font if text in headings else axis_font if text in (ax.xaxis.label,ax.yaxis.label) else font)
                    bounds = text.get_window_extent(renderer)
                    assert bounds.x0 >= 0 and bounds.y0 >= 0, (text.get_text(), bounds.extents)
                    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, (text.get_text(), bounds.extents)
            for target in (ax,cax):
                for labels in (target.get_xticklabels(), target.get_yticklabels()):
                    boxes = [t.get_window_extent(renderer) for t in labels if t.get_visible()]
                    assert not any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
        print('Rendering compact layout; panel and colour-bar gaps match, lower plot spans upper row, labels fit.', flush=True)
        for ext in ('png', 'pdf'):
            fig.savefig(args.output / f'mineral-channeling.{ext}', dpi=350/geometry_scale)
        plt.close(fig)

    caption = (
        r'Channel formation during mineral replacement on a random network of '
        r'$29\,996$ pores, with $\Gamma=0.3$, $\kappa=1$ and $\mathrm{Da}=0.5$. '
        r'(a--d) Pore diameters at the indicated dimensionless times, with flow '
        r'from top to bottom and periodic transverse boundaries. Colours show '
        r'diameter relative to the uniform initial value $d_0$ on one shared continuous, '
        r'linear colour scale from $0$ to $12$, using the standard reversed viridis map '
        r'with no clipping of the saved values. '
        r'Each model pore has one uniform diameter; widened pores '
        r'are emphasised by thicker lines. (e) Evolution of the natural logarithm '
        r'of normalised permeability, with the four displayed states marked. '
        r'The dimensionless time is $t^\prime=t/t_0$, where '
        r'$t_0=d_0/(2k_1\chi_A\nu_A c_B^{\mathrm{in}})$ is the dissolution timescale '
        r'[Budek and Szymczak (2025), Eq.~(16)].' + '\n'
    )
    (args.output / 'caption.tex').write_text(caption)
    (args.output / 'plot-source.py').write_bytes(Path(__file__).read_bytes())
    outputs = {name: sha(args.output / name) for name in (
        'mineral-channeling.png', 'mineral-channeling.pdf', 'caption.tex', 'plot-source.py')}
    provenance = dict(
        input=str(args.input), input_sha256=INPUT_SHA,
        plot_script=str(Path(__file__).resolve()), plot_script_sha256=sha(__file__),
        replay_audit=str(args.input.with_name('replay-audit.json')), replay_audit_sha256=sha(args.input.with_name('replay-audit.json')),
        frames=records, physical_pores=29996, all_physical_pores_displayed=True,
        periodic_edge_pieces=int(len(ids)), history_samples=len(times),
        continuous_colour=True, colormap=COLORMAP, colormap_modified=False,
        matplotlib_version=matplotlib.__version__,
        colour_limits=COLOR_LIMITS, colour_normalization='linear in d/d0',
        colour_clipping=False, diameter_bins=None,
        spatial_interpolation=False, diameter_uniform_on_each_pore=True,
        tolerance=None, line_weight_multiplier=args.line_weight,
        width_mapping=f'{args.line_weight}*(1.55/3.03)*clip(0.10+0.13*(d/d0-1),0.07,0.82)pt',
        parent_renderer='docs/research/plot_mineral_four_stage_viridis_compact_01.py',
        font=('Latin Modern, matched to Figure 1 at full manuscript width' if reference else
              'LaTeX Computer Modern, labels 10pt, ticks/legend 9pt'),
        font_reference=reference, geometry_scale=geometry_scale,
        source_axis_label_font_tex_points=axis_font,
        all_text_font_sizes_and_bounds_checked=bool(reference),
        exact_history_values_checked=True,
        source_figure_inches=[fig_width*geometry_scale, fig_height*geometry_scale],
        colourbar_orientation='vertical', colourbar_title='d/d0',
        colourbar_matches_panel_height=True,
        colourbar_bounds=list(cax.get_position().bounds),
        network_panel_bounds=[list(a.get_position().bounds) for a in network_axes],
        lower_plot_bounds=list(ax.get_position().bounds),
        row_gap_inches=(panel_bounds.y0 - ax.get_position().y1) * fig_height,
        panel_gap_inches=panel_gap * fig_width,
        colourbar_gap_matches_panels=True, lower_plot_spans_top_row=True,
        panel_e_aligned_with_y_tick=3,
        label_bounds_checked=True, figure_inches=[fig_width, fig_height], permeability_ordinate='ln(K/K0)',
        new_simulations=False, field_interpolation=False, scientific_data_modified=False,
        time_rescaling=False, displayed_time_convention='tprime=t/t0: existing tprime=20T; numbers unchanged',
        physical_time_scale='d0/(2*k1*chi_A*nu_A*c_B_in), reference Eq16',
        scope='Preview only; no Overleaf edits. One accepted qualitative Gamma=0.3 example.',
        limitations='Original timing discrepancies and model/geometry limitations remain '
                    'documented; this is not exact reproduction of the authors\' realization.',
        outputs=outputs,
    )
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), times=[r['paper_time'] for r in records])))


if __name__ == '__main__':
    main()
