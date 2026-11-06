"""Redraw Figure 2 from the 180 saved scaling measurements.

Requires the plotting dependencies listed in INSTALL.md, including TeX Live."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/scaling')
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
ROOT=args.data
SOURCE=ROOT/'results-original.json'
PUBLISHED=ROOT/'published.pdf'
OUTPUT=args.output/'parallel-scaling'
FONT, TITLE_FONT, AXIS_FONT, INCLUSION_SCALE = 10.95, 12., 12., .6
compact = inset = True
font_reference=json.loads((Path(__file__).resolve().parents[1]/'figure-style.json').read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
provenance=json.loads((ROOT/'plot-provenance.json').read_text())
assert sha(SOURCE)==provenance['source_sha256']
results = json.loads(SOURCE.read_text())
samples = results['samples']
assert results['passed'] and len(samples) == 180
families = [('spatial', 'Square grid', '#0072B2', 'o'),
            ('junctions', 'Branching', '#D55E00', 's'),
            ('triangular', 'Triangular mesh', '#009E73', '^')]
targets = [24576, 65536, 163840, 393216]
styles = [':', '-.', '--', '-']
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}',
    'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    'font.size': FONT, 'axes.labelsize': AXIS_FONT, 'axes.titlesize': FONT,
    'xtick.labelsize': FONT, 'ytick.labelsize': FONT, 'legend.fontsize': FONT,
    'legend.title_fontsize': FONT, 'axes.linewidth': .65/INCLUSION_SCALE,
    'axes.spines.top': False, 'axes.spines.right': False,
})
width, height = 6.5, (1.99 if inset else (2.35 if compact else 3.64))
plot_bottom = .29 if inset else (.65 if compact else 1.16)
heading_y = 1.66 if inset else (2.02 if compact else 3.29)
fig = plt.figure(figsize=(width/INCLUSION_SCALE, height/INCLUSION_SCALE), facecolor='white', dpi=220)
geometry = [(.34, 1.55), (2.30, 1.55), (4.31, 2.10)] if compact else [(.46, 1.27), (2.16, 1.27), (4.04, 2.38)]
axes = [fig.add_axes([left/width, plot_bottom/height,
                     span/width, (1.20 if compact else 1.66)/height])
        for left, span in geometry]
line_width, marker_size = (.9, 2.7) if compact else (1., 3.2)
line_width /= INCLUSION_SCALE
marker_size /= INCLUSION_SCALE
heading_text = []
curves, used = [], []
for ax, q, base, ranks, title in zip(
        axes[:2], [4000000, 512000000], [1, 96],
        [[1, 2, 4, 8, 16, 32, 64], [96, 192, 384, 768, 1536]],
        ['(a) Strong scaling\nwithin one node', '(b) Strong scaling\nacross nodes']):
    for family, _, colour, marker in families:
        points = sorted([s for s in samples if s['study'] == 'strong'
                         and s['target'] == q and s['family'] == family],
                        key=lambda s: s['ranks'])
        assert [s['ranks'] for s in points] == ranks
        used.extend(points)
        x = [s['ranks']/base for s in points]
        y = [points[0]['seconds']/s['seconds'] for s in points]
        line, = ax.plot(x, y, color=colour, marker=marker, lw=line_width, ms=marker_size,
                        markeredgewidth=.5/INCLUSION_SCALE)
        assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y)
        curves.append(dict(study='strong', family=family, target=q, ranks=ranks,
                           plotted_x=x, plotted_y=y))
    ticks = [p/base for p in ranks]
    ax.plot(ticks, ticks, ':', color='.55', lw=.8/INCLUSION_SCALE, zorder=0)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlim(ticks[0]/1.13, ticks[-1]*1.13)
    ax.set_ylim(.85, ticks[-1]*1.2)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel('Processes' if base == 1 else ('Nodes (96 processes per node)' if compact else 'Nodes\n(96 processes per node)'), labelpad=(3 if compact else 5)/INCLUSION_SCALE)
    ax.set_ylabel('Speedup', labelpad=3/INCLUSION_SCALE)
    ax.text(.04, .95, ('4M coefficients\n'+r'$\approx$50k edges') if base == 1
            else ('512M coefficients\n'+r'$\approx$6.4M edges'),
            transform=ax.transAxes, va='top', linespacing=1.1)
    centre = (ax.get_position().x0+ax.get_position().x1)/2
    heading_text.append(fig.text(centre, heading_y/height,
                                title.replace('\n', ': ') if compact else title,
                                ha='center', va='baseline' if compact else 'top',
                                fontsize=TITLE_FONT, linespacing=1.15))

ax = axes[2]
weak_ranks = [1, 2, 4, 8, 16, 32, 64, 96, 192, 384, 768, 1536]
for family, _, colour, marker in families:
    for q, style in zip(targets, styles):
        points = sorted([s for s in samples if s['study'] == 'weak'
                         and s['target'] == q and s['family'] == family],
                        key=lambda s: s['ranks'])
        assert [s['ranks'] for s in points] == weak_ranks
        used.extend(points)
        y = [s['seconds'] for s in points]
        line, = ax.plot(weak_ranks, y, color=colour, marker=marker, ls=style,
                        lw=line_width, ms=marker_size, markeredgewidth=.5/INCLUSION_SCALE)
        assert np.array_equal(line.get_xdata(), weak_ranks) and np.array_equal(line.get_ydata(), y)
        curves.append(dict(study='weak', family=family, target=q, ranks=weak_ranks,
                           plotted_x=weak_ranks, plotted_y=y))
ax.set_xscale('log', base=2)
ax.set_yscale('linear' if inset else 'log')
ax.set_xlim(.8, 1900)
ax.set_ylim((0, .4) if inset else (.009, .55))
ax.xaxis.set_major_locator(FixedLocator(weak_ranks))
ax.yaxis.set_major_locator(FixedLocator([0, .1, .2, .3, .4] if inset else [.01, .02, .05, .1, .2, .5]))
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_locator(NullLocator())
# Retain all twelve requested process-count labels at a readable point size.
ax.tick_params(axis='x', labelrotation=0 if compact else 90)
if compact:
    for value, label in zip(weak_ranks, ax.get_xticklabels()):
        if value == 64:
            label.set_ha('right')
ax.set_xlabel('Processes', labelpad=(3 if compact else 5)/INCLUSION_SCALE)
ax.set_ylabel('Time per timestep (s)', labelpad=3/INCLUSION_SCALE)
ax.axvline(96, color='.75', lw=.6/INCLUSION_SCALE, zorder=0)
if inset:
    node_text = ax.annotate('1 full node', (96, .388), xytext=(4/INCLUSION_SCALE, 0),
                            textcoords='offset points', ha='left', va='top', color='.4')
else:
    node_text = ax.text(96, .48, '1 full node', ha='center', va='top', color='.4')
heading_text.append(fig.text((ax.get_position().x0+ax.get_position().x1)/2,
                            heading_y/height,
                            '(c) Weak scaling', ha='center',
                            va='baseline' if compact else 'top', fontsize=TITLE_FONT))
for ax in axes:
    ax.tick_params(length=(2.5 if compact else 3)/INCLUSION_SCALE,
                   width=.65/INCLUSION_SCALE, pad=(2 if compact else 3)/INCLUSION_SCALE)

fig.legend(handles=[Line2D([], [], color=c, marker=m, label=n, lw=line_width, ms=(3. if compact else 4.)/INCLUSION_SCALE)
                    for _, n, c, m in families],
           loc='upper center', bbox_to_anchor=(.5, 1), ncol=3,
           frameon=False, handlelength=1.8, columnspacing=1.5)
workload_handles = [Line2D([], [], color='.3', ls=style, label=label, lw=1./INCLUSION_SCALE)
                    for label, style in zip(['25K', '66K', '164K', '393K'], styles)]
if inset:
    workload_legend = axes[2].legend(
        handles=workload_handles, loc='upper left', bbox_to_anchor=(.02, .99),
        ncol=2, title='Approximate coefficients\nper process', frameon=False,
        borderaxespad=0, borderpad=.2, handlelength=1.6, columnspacing=.8,
        labelspacing=.35, handletextpad=.5)
else:
    workload_legend = fig.legend(
        handles=workload_handles, loc='lower center', bbox_to_anchor=(.5, .02/height), ncol=4,
        title='Weak scaling: approximate coefficients per process',
        frameon=False, handlelength=2.1, columnspacing=1.5, labelspacing=.5)

assert len(used) == 180
assert len({(s['study'], s['family'], s['target'], s['ranks']) for s in used}) == 180
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
if inset:
    legend_box = workload_legend.get_window_extent(renderer)
    assert axes[2].bbox.contains(legend_box.x0, legend_box.y0)
    assert axes[2].bbox.contains(legend_box.x1, legend_box.y1)
    # Check whole connecting segments, not only the measurement markers.
    # A small safety margin accounts for marker radii and line thickness.
    clearance = (marker_size/2 + 1)*fig.dpi/72
    protected_box = legend_box.padded(clearance)
    for line in axes[2].lines:
        path = line.get_transform().transform_path(line.get_path())
        assert not path.intersects_bbox(protected_box, filled=False), 'Workload legend overlaps a curve'
    assert not legend_box.overlaps(node_text.get_window_extent(renderer))
axis_text = [label for ax in axes for label in (ax.xaxis.label, ax.yaxis.label)]
for text in fig.findobj(Text):
    if text.get_visible() and text.get_text():
        assert text.get_fontsize() == (TITLE_FONT if text in heading_text else AXIS_FONT if text in axis_text else FONT), (text.get_text(), text.get_fontsize())
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.x1 <= fig.bbox.width, (text.get_text(), box.extents)
        assert box.y0 >= 0 and box.y1 <= fig.bbox.height, (text.get_text(), box.extents)
for ax in axes:
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        visible = [t for t in labels if t.get_visible()]
        boxes = [t.get_window_extent(renderer) for t in visible]
        overlaps = [(visible[i].get_text(), visible[j].get_text(), a.extents, b.extents)
                    for i, a in enumerate(boxes) for j, b in enumerate(boxes[i+1:], i+1)
                    if a.overlaps(b)]
        assert not overlaps, overlaps
if compact:
    boxes = [t.get_window_extent(renderer) for t in axes[2].get_xticklabels()]
    gaps = [(right.x0-left.x1)/fig.dpi*72*INCLUSION_SCALE
            for left,right in zip(boxes,boxes[1:])]
    assert min(gaps) > 1.5, gaps
for ext in ('pdf', 'png'):
    fig.savefig(OUTPUT.with_suffix('.'+ext), dpi=260*INCLUSION_SCALE)
plt.close(fig)
receipt = dict(
    purpose='Three-panel scaling typography preview; unchanged published measurements',
    source=str(SOURCE), source_sha256=sha(SOURCE),
    published_pdf=str(PUBLISHED), published_pdf_sha256=sha(PUBLISHED),
    points=180, curves=curves, plotted_arrays_checked=True,
    strong_normalization=provenance['strong_normalization'],
    weak_timing=provenance['weak_timing'],
    font_tex_points=FONT*INCLUSION_SCALE, heading_font_tex_points=TITLE_FONT*INCLUSION_SCALE,
    source_font_tex_points=FONT, source_heading_font_tex_points=TITLE_FONT,
    source_axis_label_font_tex_points=AXIS_FONT, axis_label_font_tex_points=AXIS_FONT*INCLUSION_SCALE,
    inclusion_scale=INCLUSION_SCALE, font_reference=font_reference,
    figure_size_inches=[width, height], source_figure_size_inches=[width/INCLUSION_SCALE,height/INCLUSION_SCALE],
    layout=('Three panels in one row; shared topology legend above, workload legend inside panel C'
            if inset else 'Three panels in one row; shared topology and workload legends outside data axes'),
    weak_y_scale='linear' if inset else 'log', weak_y_limits=[0, .4] if inset else [.009, .55],
    workload_legend_location='inside panel C, upper left' if inset else 'below all panels',
    workload_legend_bounds_and_curve_clearance_checked=inset,
    all_process_tick_labels_preserved=True, text_bounds_and_tick_overlap_checked=True,
    horizontal_process_tick_labels=compact,
    scientific_computation=False, overleaf_updated=False,
    outputs={ext: sha(OUTPUT.with_suffix('.'+ext)) for ext in ('pdf', 'png')})
OUTPUT.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(OUTPUT, flush=True)
