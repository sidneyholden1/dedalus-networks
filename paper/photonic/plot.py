"""Redraw Figure 3 from the saved optical intensity maps and resolved trace.

Checks the physical field and endpoint laws before composing the figure.
Requires the plotting dependencies listed in INSTALL.md."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
import numpy as np
from PIL import Image


FONT = 10.95

def figure1_typography():
    return json.loads((Path(__file__).resolve().parents[1]/'figure-style.json').read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def optical(work, revision=4):
    reference = figure1_typography() if revision >= 3 else None
    left_padding = .28 if revision == 4 else 0.
    width, height = 6.5+left_padding, 4.88
    # Keep the approved 0.7-text-width footprint, but use the same source
    # fonts and physical inclusion scale as Figure 1, including optical design.
    geometry_scale = (reference['manuscript_width_fractions']['photonic-interior-mode.pdf'] * 6.5 /
                      (reference['inclusion_scale']*width)) if reference else 1.
    font = reference['source_font_tex_points'] if reference else FONT
    title_font = reference['source_heading_font_tex_points'] if reference else FONT
    axis_font = 12 if revision == 4 else font
    source = OPTICAL/'interior-mode.pdf'
    provenance = json.loads((OPTICAL/'provenance.json').read_text())
    assert sha(source) == provenance['artifacts']['interior-mode.pdf']
    assert sha(OPTICAL/'resolved-trace.csv') == provenance['artifacts']['resolved-trace.csv']
    subprocess.run(['pdfimages', '-png', str(source), str(work/'optical-map')], check=True)
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}',
        'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'font.size': font, 'axes.labelsize': font, 'axes.titlesize': font,
        'xtick.labelsize': font, 'ytick.labelsize': font, 'legend.fontsize': font,
        'axes.linewidth': .7*geometry_scale, 'pdf.compression': 6,
    })
    fig = plt.figure(figsize=(width*geometry_scale, height*geometry_scale), facecolor='white', dpi=220)
    headings = []
    def axes(x, y, w, h):
        return fig.add_axes([(x+left_padding)/width, y/height, w/width, h/height])
    maps = [axes(.10, 2.03, 2.55, 2.55), axes(2.94, 2.03, 2.55, 2.55)]
    image_records = []
    for i, ax in enumerate(maps):
        path = work/f'optical-map-{i:03d}.png'
        pixels = np.asarray(Image.open(path).convert('RGB'))
        assert pixels.shape == (1200, 1200, 3)
        low, high = np.asarray(provenance['panels'][i]['bounds_um'])
        # PDF image objects are in top-to-bottom row order. No colour remapping.
        ax.imshow(pixels, origin='upper', extent=(low[0], high[0], low[1], high[1]),
                  interpolation='none', aspect='equal')
        ax.set_axis_off()
        if revision == 4:
            box=ax.get_position()
            headings.append(fig.text(box.x0-.055/width,box.y1,f'({chr(97+i)})',
                                     ha='right',va='top',fontsize=title_font))
        else:
            headings.append(fig.text((.10 if i == 0 else 2.94)/width, 4.69/height,
                     f'({chr(97+i)})', ha='left', va='baseline', fontsize=title_font))
        length = 1000 if i == 0 else 100
        label = r'$1\,\mathrm{mm}$' if i == 0 else r'$100\,\mu\mathrm{m}$'
        start = low+.065*(high-low)
        line, = ax.plot([start[0], start[0]+length], [start[1], start[1]],
                        lw=1.2*geometry_scale, color='black', zorder=9)
        line.set_path_effects([effects.Stroke(linewidth=3.5*geometry_scale, foreground='white'), effects.Normal()])
        ax.text(start[0]+length/2, start[1]+.04*(high[1]-low[1]), label,
                ha='center', va='bottom', zorder=10,
                bbox=dict(facecolor='white', edgecolor='none', pad=.8, alpha=.95))
        image_records.append({'source_pdf_object': [13, 14][i], 'extracted_sha256': sha(path),
                              'rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                              'pixels': [1200, 1200], 'bounds_um': [low.tolist(), high.tolist()]})

    halo = [effects.Stroke(linewidth=2.5*geometry_scale, foreground='white'), effects.Normal()]
    label_positions = {0.5: (.98, .53), 0.9: (.98, .70), 0.99: (.98, .88)}
    for box in provenance['displayed_boxes']:
        low, high = np.asarray(box['bounds_um'])
        rect = Rectangle(low, box['width_um'], box['width_um'], fill=False,
                         linewidth=.8*geometry_scale, edgecolor='black', zorder=5)
        rect.set_path_effects(halo)
        maps[0].add_patch(rect)
        label = maps[0].annotate(rf"${100*box['target_fraction']:.0f}\%$",
            xy=high, xycoords='data', xytext=label_positions[box['target_fraction']],
            textcoords='axes fraction', ha='right', va='center', zorder=7,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.1),
            arrowprops=dict(arrowstyle='-', color='black', linewidth=.7*geometry_scale, shrinkA=2*geometry_scale, shrinkB=0))
        label.arrow_patch.set_path_effects(halo)

    edge_xy = provenance['resolved_profile']['marker_position_um']
    marker = maps[1].scatter(*edge_xy, s=180*geometry_scale**2, facecolors='none', edgecolors='black',
                            linewidths=geometry_scale, zorder=7)
    marker.set_path_effects([effects.Stroke(linewidth=3.2*geometry_scale, foreground='white'), effects.Normal()])
    callout = maps[1].annotate(r'$e_*$', xy=edge_xy, xytext=(20*geometry_scale, -24*geometry_scale),
        textcoords='offset points', ha='left', va='top', zorder=8,
        arrowprops=dict(arrowstyle='-', color='black', linewidth=.9*geometry_scale, shrinkA=3*geometry_scale, shrinkB=8*geometry_scale),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=.8*geometry_scale))
    callout.arrow_patch.set_path_effects(halo)
    cmap = LinearSegmentedColormap.from_list('published_vivid', provenance['colormap_anchors'], N=4096)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=LogNorm(1e-6, 1), cmap=cmap),
                      cax=axes(5.65, 2.03, .105, 2.55), ticks=[1e-6, 1e-4, 1e-2, 1])
    bar.ax.set_yticklabels([r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', '$1$'])
    bar.ax.yaxis.set_minor_locator(NullLocator())
    bar.ax.tick_params(length=3*geometry_scale, width=.7*geometry_scale, pad=3*geometry_scale)
    bar.set_label(r'$|u_e(x)|^2/\rho_{\max}$', labelpad=5*geometry_scale)

    trace_data = np.loadtxt(OPTICAL/'resolved-trace.csv', delimiter=',', skiprows=1)
    assert trace_data.shape == (2049, 4)
    # Verify the saved profile against the retained physical travelling waves.
    with np.load(OPTICAL/'selected-fields.npz', allow_pickle=False) as z:
        edge = provenance['resolved_profile']['edge']
        q = z['k'][0]*z['index'][edge]
        a, b = z['amplitudes'][0,edge]
        x = trace_data[:,0]
        u = a*np.exp(1j*q*x)+b*np.exp(1j*q*(z['lengths_um'][edge]-x))
        np.testing.assert_allclose(abs(u)**2/provenance['pointwise_peak'],
                                   trace_data[:,3], rtol=2e-12, atol=2e-13)
    # Revision 2 aligns the data axes with the right edge of the colour bar.
    trace_right = (maps[1].get_position().x1 * width if revision >= 3 else
                   bar.ax.get_position().x1 * width if revision == 2 else 6.08)
    trace = axes(.61, .48, trace_right-.61-left_padding, 1.13)
    if revision == 2:
        assert abs(trace.get_position().x1-bar.ax.get_position().x1) < 1e-12
    if revision >= 3:
        assert abs(trace.get_position().x1-maps[1].get_position().x1) < 1e-12
    line, = trace.plot(trace_data[:, 0], trace_data[:, 3], color='#7b239e', linewidth=geometry_scale)
    assert np.array_equal(line.get_xdata(),trace_data[:,0]) and np.array_equal(line.get_ydata(),trace_data[:,3])
    trace.set(xlim=(0, 3), ylim=(0, 1.06), xticks=[0, 1, 2, 3], yticks=[0, .5, 1])
    trace.set_xlabel(r'$x\;(\mu\mathrm{m})$', labelpad=4*geometry_scale, fontsize=axis_font)
    trace.set_ylabel(r'$|u_{e_*}(x)|^2/\rho_{\max}$', labelpad=5*geometry_scale, fontsize=axis_font)
    trace.tick_params(direction='out', length=3*geometry_scale, width=.7*geometry_scale, pad=3*geometry_scale)
    trace.spines[['top', 'right']].set_visible(False)
    if revision == 4:
        headings.append(fig.text((.10+left_padding-.055)/width, trace.get_position().y1,
                                 '(c)',ha='right',va='top',fontsize=title_font))
    else:
        headings.append(fig.text(.10/width, 1.68/height, '(c)', ha='left', va='baseline', fontsize=title_font))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if revision == 4:
        for heading,target in zip(headings,[*maps,trace]):
            box=heading.get_window_extent(renderer)
            assert box.x1 < target.bbox.x0 and abs(box.y1-target.bbox.y1)<1e-7
        assert headings[1].get_window_extent(renderer).x0 > maps[0].bbox.x1
        assert not headings[2].get_window_extent(renderer).overlaps(trace.yaxis.label.get_window_extent(renderer))
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            assert t.get_fontsize() == (title_font if t in headings else axis_font if t in (trace.xaxis.label,trace.yaxis.label) else font), (t.get_text(), t.get_fontsize())
            box = t.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= fig.bbox.width, (t.get_text(), box.extents)
            assert box.y0 >= 0 and box.y1 <= fig.bbox.height, (t.get_text(), box.extents)
    output = work/'interior-mode'
    for ext in ('pdf', 'png'):
        fig.savefig(output.with_suffix('.'+ext), dpi=260/geometry_scale)
    plt.close(fig)
    receipt = dict(purpose='Typography/layout preview from published plotting assets',
        source_pdf=str(source), source_pdf_sha256=sha(source),
        provenance_sha256=sha(OPTICAL/'provenance.json'),
        source_trace=str(OPTICAL/'resolved-trace.csv'), source_trace_sha256=sha(OPTICAL/'resolved-trace.csv'),
        maps=image_records, map_rgb_preserved=True, trace_values_preserved=True,
        box_geometry_preserved=True, normalization_and_color_scale_preserved=True,
        scientific_computation=False, font_tex_points=font, heading_font_tex_points=title_font,
        source_axis_label_font_tex_points=axis_font, panel_labels_outside_upper_left=revision==4,
        font_reference=reference, figure_size_inches=[width*geometry_scale, height*geometry_scale],
        layout=('A and B above; shallow C aligned with B data-plot right edge' if revision >= 3 else
                'A and B above; shallow C aligned with colour-bar right edge' if revision == 2 else 'A and B above; full-width shallow C'),
        trace_right_inches=trace_right*geometry_scale,
        colour_bar_right_inches=bar.ax.get_position().x1*width*geometry_scale,
        panel_b_right_inches=maps[1].get_position().x1*width*geometry_scale, geometry_scale=geometry_scale,
        all_text_bounds_checked=True,
        outputs={ext:sha(output.with_suffix('.'+ext)) for ext in ('pdf', 'png')})
    write_json(output.with_suffix('.json'), receipt)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Redraw the optical figure from retained maps and physical fields.')
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/photonic')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    OPTICAL=args.data
    args.output.mkdir(parents=True,exist_ok=False)
    print(optical(args.output,revision=4))
