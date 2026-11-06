"""Pointwise mode maps with integrated-intensity boxes and a resolved hotspot."""
import argparse
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
import matplotlib.patheffects as effects
from matplotlib.ticker import NullLocator
import numpy as np

import plot_photonic_full_pointwise as renderer
from plot_photonic_full_pointwise import sha
from plot_photonic_pointwise_panel import field, pointwise_peak
from plot_photonic_interior_figure import scale
from photonic_intensity_boxes import clipped_parameters


def check_display_floor(linear, floor, stroke_width):
    """Check actual raster colours, clipping counts and a solid stroke interior."""
    lut = np.array([[.2, .5, .7], [.9, .4, .3]])
    errors = []
    densities = (0., floor/100, floor, np.sqrt(floor), .25, .5, 1., 2.)
    for density in densities:
        waves = np.array([[np.sqrt(density)+0j, 0j]])
        raster, under, over = renderer.rasterize(
            np.array([[[1., 5.], [9., 5.]]]), waves, np.zeros(1, complex),
            np.array([8.]), np.array([128], np.int64), np.array([4], np.int64),
            np.array([0], np.int64), np.zeros(2), 10/64, 64, floor, 1., lut,
            linear=linear, stroke_width=stroke_width)
        actual_density = float(abs(waves[0, 0])**2)
        fraction = np.clip((actual_density-floor)/(1-floor) if linear else
                           np.log(max(actual_density, floor)/floor)/np.log(1/floor), 0, 1)
        expected = (1-fraction)*lut[0]+fraction*lut[1]
        error = float(abs(raster[31:33, 12:52]-expected).max())
        assert error < 2e-7
        assert under == 512*(actual_density < floor)
        assert over == 512*(actual_density > 1)
        errors.append(error)
    return dict(constant_density_cases=len(densities), maximum_rgb_error=max(errors),
                color_scale='linear' if linear else 'logarithmic',
                floor=floor, stroke_width_pixels=stroke_width,
                lower_and_upper_clipping_counts_verified=True,
                continuous_stroke_interior_verified=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--boxes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', choices=('mixed', 'linear', 'log'), default='mixed')
    parser.add_argument('--color-floor', type=float, default=1e-6)
    parser.add_argument('--palette', choices=('original', 'vivid'), default='original')
    parser.add_argument('--overview-stroke-width', type=float, default=1.7)
    parser.add_argument('--hotspot-width', type=float, default=40.)
    parser.add_argument('--hotspot-stroke-width', type=float, default=6.8)
    parser.add_argument('--zoom-fraction', type=float, choices=(.5, .9, .95, .99),
                        help='Use an integrated-intensity square for panel (b).')
    parser.add_argument('--edge-profile', action='store_true',
                        help='Add a resolved, linear intensity trace as panel (c).')
    args = parser.parse_args()
    assert 0 < args.color_floor < 1
    assert 1 <= args.hotspot_width <= 100
    assert 1.7 <= args.hotspot_stroke_width <= 32
    assert 1.7 <= args.overview_stroke_width <= 32
    shared_linear = args.scales == 'linear'
    shared_scale = args.scales != 'mixed'
    assert not args.edge_profile or shared_scale
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    started = time.monotonic()
    renderer.BACKEND = renderer.backend()
    display_checks = (check_display_floor(shared_linear, args.color_floor, args.hotspot_stroke_width)
                      if shared_scale else None)
    source = args.selection / 'selected-fields.npz'
    selection = json.loads((args.selection / 'selection.json').read_text())
    boxes = json.loads(args.boxes.read_text())
    assert selection['passed'] and selection['refinement_passed'] and not selection['preliminary']
    assert boxes['passed'] and not boxes['preliminary']
    assert sha(source) == selection['selected_fields_sha256'] == boxes['fields_sha256']
    chosen_boxes = [next(b for b in boxes['boxes'] if b['target_fraction'] == f)
                    for f in (.5, .9, .99)]
    with np.load(source, allow_pickle=False) as data:
        positions = data['positions_um']
        segments = positions[data['endpoints']]
        lengths = data['lengths_um']
        waves = data['amplitudes'][0]
        q = complex(data['k'][0]) * data['index']
        internal = data['internal']
        means = data['edge_I2'][0] / lengths
    peak, peak_info = pointwise_peak(waves, q, lengths, internal)
    center = segments[peak_info['edge']].mean(axis=0)
    hotspot_width = args.hotspot_width
    zoom_box = None
    if args.zoom_fraction is not None:
        zoom_box = next(b for b in boxes['boxes'] if b['target_fraction'] == args.zoom_fraction)
        assert abs(zoom_box['actual_fraction']-args.zoom_fraction) < 1e-8
        center = np.asarray(boxes['center_um'])
        hotspot_width = zoom_box['width_um']
    views = [(positions.min(axis=0), positions.max(axis=0)),
             (center-hotspot_width/2, center+hotspot_width/2)]
    if zoom_box is not None:
        np.testing.assert_allclose(views[1], zoom_box['bounds_um'], rtol=0, atol=1e-10)
    colours = (['#327eae', '#465c9b', '#593574', '#aa3970', '#e77851'] if shared_scale else
               ['#e5f0f8', '#a8cce5', '#528db7', '#365986', '#553574', '#aa3970', '#e77851'])
    if args.palette == 'vivid':
        assert shared_scale
        colours = ['#1671b5', '#3f3cad', '#7b239e', '#d32965', '#f57b25']
    cmap = LinearSegmentedColormap.from_list('photonic_density', colours, N=4096)
    lut = cmap(np.linspace(0, 1, 4096))[:, :3]
    if shared_scale:
        norm_class = Normalize if shared_linear else LogNorm
        norms = [norm_class(args.color_floor, 1, clip=True)]*2
    else:
        norms = [LogNorm(1e-20, 1, clip=False), Normalize(0, 1, clip=False)]
    rasters, receipts = [], []
    for panel, (low, high) in enumerate(views):
        assert np.isclose(*(high-low))
        t0, t1 = clipped_parameters(segments, low, high)
        ids = np.flatnonzero(t1 > t0)
        start = t0[ids] * lengths[ids]
        stop = t1[ids] * lengths[ids]
        cropped_lengths = stop-start
        cropped_segments = (segments[ids, 0, None, :] +
            np.stack([t0[ids], t1[ids]], axis=1)[:, :, None] *
            (segments[ids, 1]-segments[ids, 0])[:, None, :])
        cropped_waves = np.column_stack([
            waves[ids, 0] * np.exp(1j*q[ids]*start),
            waves[ids, 1] * np.exp(1j*q[ids]*(lengths[ids]-stop))])
        check = np.unique(np.r_[np.arange(0, len(ids), max(1, len(ids)//15)),
                                np.argmax(means[ids])])
        x = cropped_lengths[check, None] * np.array([0., .137, .5, .831, 1.])
        direct = (waves[ids[check], 0, None] * np.exp(1j*q[ids[check], None]*(start[check, None]+x)) +
                  waves[ids[check], 1, None] * np.exp(1j*q[ids[check], None]*(lengths[ids[check], None]-start[check, None]-x)))
        shifted = (cropped_waves[check, 0, None] * np.exp(1j*q[ids[check], None]*x) +
                   cropped_waves[check, 1, None] * np.exp(1j*q[ids[check], None]*(cropped_lengths[check, None]-x)))
        defect = float(np.max(abs(shifted-direct) /
            np.maximum(np.sum(abs(cropped_waves[check]), axis=1)[:, None], 1e-300)))
        assert defect < 5e-12
        pixels = 2400
        pixel_um = float((high-low)[0]/pixels)
        bins = np.maximum(1, np.ceil(cropped_lengths/pixel_um*2).astype(np.int64))
        per_bin = np.maximum(4, np.ceil(cropped_lengths/bins*abs(q[ids].real)/np.pi*16).astype(np.int64))
        count = int(np.sum(bins*per_bin))
        assert count < 220000000
        raster, under, over = renderer.rasterize(
            cropped_segments, cropped_waves, q[ids], cropped_lengths, bins, per_bin,
            np.argsort(means[ids], kind='stable'), low, pixel_um, pixels,
            norms[panel].vmin*peak, peak, lut,
            linear=shared_linear or (not shared_scale and bool(panel)),
            stroke_width=args.overview_stroke_width if panel == 0 else args.hotspot_stroke_width)
        assert over == 0 and np.isfinite(raster).all()
        assert shared_scale or under == 0
        rasters.append(raster)
        receipts.append(dict(visible_edges=len(ids), optical_samples=count,
            spatial_bins=int(bins.sum()), pixels=pixels, pixel_um=pixel_um,
            minimum_samples_per_period=16, maximum_bin_pixels=.5,
            shifted_field_max_relative_defect=defect,
            under_range_samples=int(under), over_range_samples=int(over),
            intensity_period_pixels=float(np.pi/abs(q[peak_info['edge']].real)/pixel_um),
            bounds_um=[low.tolist(), high.tolist()],
            color_scale='logarithmic' if isinstance(norms[panel], LogNorm) else 'linear',
            color_limits=[norms[panel].vmin, norms[panel].vmax],
            stroke_width_pixels=args.overview_stroke_width if panel == 0 else args.hotspot_stroke_width))
        print(json.dumps(dict(panel=panel, **receipts[-1], cpu_seconds=time.process_time())), flush=True)

    font_size = 12
    callout_font_size = 14
    plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'font.size': font_size,
        'axes.labelsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'axes.linewidth': .8, 'savefig.facecolor': 'white', 'pdf.compression': 3})
    figure_size = ((14.35, 4.85) if args.edge_profile else
                   ((14.35, 6.55) if shared_scale else (14.35, 6.2)))
    height = figure_size[1]
    panel_size = 4. if args.edge_profile else (5.9 if shared_scale else 5.5)
    panel_bottom = .6 if args.edge_profile else .35
    fig = plt.figure(figsize=figure_size, facecolor='white')
    axes, bars = [], []
    layout = ((.4, None), (6.75, 12.92)) if shared_scale else ((.4, 6.05), (7.2, 12.85))
    if args.edge_profile:
        layout = ((.35, None), (4.65, 8.82))
    for panel, (x0, bar_x) in enumerate(layout):
        ax = fig.add_axes([x0/14.35, panel_bottom/height, panel_size/14.35, panel_size/height])
        axes.append(ax)
        low, high = views[panel]
        ax.imshow(rasters[panel], origin='lower', extent=(low[0], high[0], low[1], high[1]),
                  interpolation='lanczos', aspect='equal')
        ax.set_axis_off()
        fig.text((x0-.28)/14.35, (panel_bottom+panel_size)/height, rf'({chr(97+panel)})',
                 ha='left', va='top', fontsize=font_size)
        bar_length = 100 if hotspot_width >= 300 else (5 if hotspot_width >= 30 else 2)
        scale(ax, 1000 if panel == 0 else bar_length,
              r'$1\,\mathrm{mm}$' if panel == 0 else rf'${bar_length}\,\mu\mathrm{{m}}$',
              fontsize=font_size, color='black', linewidth=1.)
        if bar_x is None:
            continue
        cax = fig.add_axes([bar_x/14.35, panel_bottom/height, .15/14.35, panel_size/height])
        bars.append(cax)
        ticks = [1e-20, 1e-15, 1e-10, 1e-5, 1.] if panel == 0 else [norms[panel].vmin, .25, .5, .75, 1.]
        labels = ([r'$10^{-20}$', r'$10^{-15}$', r'$10^{-10}$', r'$10^{-5}$', '$1$'] if panel == 0
                  else ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'])
        if shared_linear:
            labels[0] = rf'$10^{{{np.log10(args.color_floor):g}}}$'
        elif shared_scale:
            exponents = np.linspace(np.log10(args.color_floor), 0, 4)
            ticks = 10.**exponents
            labels = [rf'$10^{{{e:g}}}$' if e else '$1$' for e in exponents]
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norms[panel], cmap=cmap), cax=cax, ticks=ticks)
        cb.ax.set_yticklabels(labels)
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.ax.tick_params(length=3, width=.8)
        cb.set_label(r'$|u_e(x)|^2/\rho_{\max}$', labelpad=8)
    halo = [effects.Stroke(linewidth=2., foreground='white'), effects.Normal()]
    for box in chosen_boxes:
        low, high = np.array(box['bounds_um'])
        rect = Rectangle(low, box['width_um'], box['width_um'], fill=False,
                         linewidth=.75, edgecolor='black', zorder=5)
        rect.set_path_effects(halo)
        axes[0].add_patch(rect)
        axes[0].annotate(rf"${100*box['target_fraction']:.0f}\%$", xy=low,
            xytext=(3, 3), textcoords='offset points', fontsize=font_size,
            ha='left', va='bottom', zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    if zoom_box is None or args.zoom_fraction not in (.5, .9, .99):
        hotspot = Rectangle(views[1][0], hotspot_width, hotspot_width,
                            fill=False, linewidth=.75, edgecolor='black', zorder=7)
        hotspot.set_path_effects(halo)
        axes[0].add_patch(hotspot)
    profile = None
    if args.edge_profile:
        edge = peak_info['edge']
        edge_xy = segments[edge].mean(axis=0)
        assert np.all(edge_xy > views[1][0]) and np.all(edge_xy < views[1][1])
        callout_halo = [effects.Stroke(linewidth=4.4, foreground='white'), effects.Normal()]
        marker = axes[1].scatter(*edge_xy, s=500, facecolors='none',
                                 edgecolors='black', linewidths=1.6, zorder=7)
        marker.set_path_effects(callout_halo)
        callout = axes[1].annotate(r'$e_*$', xy=edge_xy, xytext=(30, -30),
                         textcoords='offset points', ha='left', va='top',
                         fontsize=callout_font_size,
                         arrowprops=dict(arrowstyle='-', color='black', linewidth=1.2,
                                         shrinkA=4, shrinkB=13),
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='black', linewidth=1.2), zorder=8)
        callout.arrow_patch.set_path_effects(callout_halo)
        trace = fig.add_axes([10.55/14.35, panel_bottom/height, 3.55/14.35, panel_size/height])
        axes.append(trace)
        x = np.linspace(0, min(3., lengths[edge]), 2049)
        u = field(x, waves[edge], q[edge], lengths[edge])
        density = abs(u)**2 / peak
        period = float(np.pi/abs(q[edge].real))
        assert period/(x[1]-x[0]) > 100
        assert np.isfinite(density).all() and density.min() >= 0 and density.max() <= 1+1e-12
        # Check the reconstructed trace independently in the sine/cosine basis.
        left, right = waves[edge]
        endpoint_right = right*np.exp(1j*q[edge]*lengths[edge])
        alternate = ((left+endpoint_right)*np.cos(q[edge]*x) +
                     1j*(left-endpoint_right)*np.sin(q[edge]*x))
        trace_defect = float(abs(u-alternate).max()/np.sqrt(peak))
        assert trace_defect < 1e-12
        trace.plot(x, density, color='#7b239e' if args.palette == 'vivid' else '#623b76', linewidth=1.3)
        trace.set(xlim=(0, x[-1]), ylim=(0, 1.06), xticks=[0, 1, 2, 3],
                  yticks=[0, .25, .5, .75, 1])
        trace.set_xlabel(r'$x\;(\mu\mathrm{m})$', labelpad=5)
        trace.set_ylabel(r'$|u_{e_*}(x)|^2/\rho_{\max}$', labelpad=6)
        trace.tick_params(direction='out', length=3, width=.8, pad=3.5)
        trace.spines[['top', 'right']].set_visible(False)
        fig.text((10.55-.28)/14.35, (panel_bottom+panel_size)/height, '(c)',
                 ha='left', va='top', fontsize=font_size)
        profile = dict(edge=int(edge), edge_length_um=float(lengths[edge]),
                       interval_um=[float(x[0]), float(x[-1])], samples=len(x),
                       intensity_period_um=period, samples_per_period=float(period/(x[1]-x[0])),
                       normalization='same internal pointwise maximum as maps',
                       vertical_scale='linear', display_floor=None,
                       sine_cosine_reconstruction_defect=trace_defect,
                       marker_position_um=edge_xy.tolist(),
                       minimum=float(density.min()), maximum=float(density.max()))
    fig.canvas.draw()
    extents = np.array([a.get_window_extent().bounds for a in axes+bars])
    assert np.allclose(extents[:, [1, 3]], extents[0, [1, 3]])
    args.output.mkdir(mode=0o700)
    if args.edge_profile:
        np.savetxt(args.output/'resolved-trace.csv', np.column_stack([x, u.real, u.imag, density]),
                   delimiter=',', header='x_um,u_real,u_imag,density_over_pointwise_peak', comments='')
    basename = 'interior-mode' if args.edge_profile else 'quantile-pointwise-pair'
    for ext in ('png', 'pdf'):
        options = {'pil_kwargs': {'compress_level': 3}} if ext == 'png' else {}
        fig.savefig(args.output/f'{basename}.{ext}', dpi=300, **options)
    plt.close(fig)
    caption = (r'Pointwise intensity of the refined $100{,}485$-edge mode. '
        r'(a) The full network on a logarithmic scale; no evaluated intensity lies '
        r'outside the colour range. Nested squares enclose $50\%$, $90\%$ and $99\%$ '
        r'of $\sum_e\int |u_e|^2\,\mathrm{d}x$ over internal edges, with exact '
        r'integrals clipped at their boundaries. All squares share the intensity '
        r'centroid; their widths are $0.599$, $1.084$ and $2.076\,\mathrm{mm}$. '
        rf'The small marked square locates (b), a ${hotspot_width:g}\,\mu\mathrm{{m}}$ hotspot view '
        r'on a linear scale, resolving the standing-wave intensity. Both panels '
        r'use the same internal pointwise maximum $\rho_{\max}$ for normalization. '
        r'At network scale, unresolved fringes are represented by spatially averaged '
        r'pointwise colours; edges are not assigned a single mean intensity. '+'\n')
    if shared_scale:
        zoom_description = (
            rf'and the ${100*args.zoom_fraction:g}\%$ integrated-intensity square (b), '
            rf'of width ${hotspot_width:.3f}\,\mu\mathrm{{m}}$. '
            if zoom_box is not None else
            rf'and a ${hotspot_width:g}\,\mu\mathrm{{m}}$ hotspot square (b). ')
        resolution_description = (
            r'Panel (b) magnifies the corresponding square in (a) with thick continuous strokes. '
            r'At these scales optical fringes are unresolved; pointwise colours '
            r'are spatially averaged within half-pixel bins. '
            if zoom_box is not None else
            r'The small square marks (b), where thick continuous strokes retain the resolved '
            r'standing-wave variation in colour. At full-network scale, pointwise colours '
            r'are spatially averaged within half-pixel bins. ')
        caption = (r'Pointwise intensity of the refined $100{,}485$-edge mode on the full network (a) '
            + zoom_description + r'Both panels use the same '
            rf'{"linear" if shared_linear else "logarithmic"} colour scale from '
            rf'$10^{{{np.log10(args.color_floor):g}}}$ to $1$, normalized '
            r'by the internal pointwise maximum $\rho_{\max}$. Values below the display floor '
            r'are clamped to the blue endpoint; no edges are removed. Nested squares in (a) '
            r'enclose $50\%$, $90\%$ and $99\%$ of the integrated internal intensity, '
            r'computed from the unchanged field with exact boundary-clipped edge integrals. '
            + resolution_description + '\n')
    if args.edge_profile:
        caption = caption.rstrip() + (
            r' (c) Resolved $|u_{e_*}(x)|^2/\rho_{\max}$ over the first '
            r'$3\,\mu\mathrm{m}$ of the circled edge, which contains the internal pointwise maximum. '
            r'The trace uses a linear vertical axis and no display floor; its '
            r'intensity oscillation period is $0.1961\,\mu\mathrm{m}$.' + '\n')
    (args.output/'caption.tex').write_text(caption)
    names = [f'{basename}.png', f'{basename}.pdf', 'caption.tex']
    if args.edge_profile:
        names.append('resolved-trace.csv')
    report = dict(fields=str(source), fields_sha256=sha(source),
        selection_sha256=sha(args.selection/'selection.json'), boxes_sha256=sha(args.boxes),
        displayed_boxes=chosen_boxes, box_definition=boxes['definition'],
        zoom_box=zoom_box,
        physical_edges=len(lengths), pointwise_peak=peak, peak_search=peak_info,
        quantity='pointwise abs(u_e(x))**2 / maximum internal pointwise density',
        shared_peak_normalization=True, shared_colorbar=shared_scale,
        color_norm_clip=shared_scale, colormap_anchors=colours,
        palette=args.palette,
        display_mapping_checks=display_checks,
        panels=receipts, aligned_panel_and_colorbar_heights=True,
        resolved_profile=profile,
        figure_size_inches=list(figure_size), text_renderer='LaTeX Computer Modern',
        typography=dict(base_font_points=font_size, panel_font_points=font_size,
                        callout_font_points=callout_font_size,
                        reference='Figure 2 scaling figure uses 12-point Computer Modern'),
        edge_callout=dict(marker_area_points_squared=500, marker_linewidth_points=1.6,
                          white_halo_linewidth_points=4.4, label_offset_points=[30, -30],
                          label_box='opaque white, rounded, black 1.2-point outline',
                          connector_linewidth_points=1.2) if args.edge_profile else None,
        edge_means_used_only_for_overlap_order=True,
        range_definition=(f'{"Linear" if shared_linear else "Logarithmic"} scale with a lower display floor; subfloor samples use the same blue endpoint.'
            if shared_scale else 'Log lower bound contains every evaluated positive sample; not a certified continuum minimum.'),
        script_sha256=sha(__file__), dependencies={n: sha(Path(__file__).with_name(n)) for n in
            ['plot_photonic_full_pointwise.py', 'photonic_pointwise_raster.cpp',
             'plot_photonic_pointwise_panel.py', 'plot_photonic_interior_figure.py', 'photonic_intensity_boxes.py']},
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time(),
        artifacts={n: sha(args.output/n) for n in names})
    (args.output/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(args.output/f'{basename}.png', flush=True)


if __name__ == '__main__':
    main()
