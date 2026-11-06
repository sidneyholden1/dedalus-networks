"""Redraw Figure 5 from saved fields and compose its two-by-two layout."""
import sys
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


def kdv(work, revision=2):
    reference = figure1_typography() if revision >= 3 else None
    two_rows = revision >= 2
    source = KDV/'kdv-time-snapshots.pdf'
    provenance = json.loads((KDV/'provenance.json').read_text())
    assert provenance['passed']
    assert sha(source) == provenance['outputs']['kdv-time-snapshots.pdf']['sha256']
    output = work/'kdv-time-snapshots'
    # Remove the eight old text blocks losslessly before cropping. Otherwise
    # PDF text extraction can return the clipped old labels as duplicates.
    # qpdf preserves all image samples and non-text drawing operations.
    qdf = work/'kdv-source.qdf.pdf'
    subprocess.run(['qpdf','--qdf','--object-streams=disable',str(source),str(qdf)],check=True)
    raw = qdf.read_bytes()
    start = raw.index(b'\nstream\n',raw.index(b'%% Contents for page 1'))+8
    end = raw.index(b'\nendstream',start)
    original_commands = raw[start:end]
    commands, count = re.subn(rb'(?m)^BT\n.*?^ET\n',b'',original_commands,flags=re.S)
    assert count == 8
    assert b'\nBT\n' not in commands
    qdf.write_bytes(raw[:start]+commands+raw[end:])
    fixed = work/'kdv-graphics-fixed.pdf'
    with fixed.open('wb') as stream:
        subprocess.run(['fix-qdf',str(qdf)],stdout=stream,check=True)
    graphics = output.with_name(output.name+'-graphics.pdf')
    subprocess.run(['qpdf','--stream-data=compress',str(fixed),str(graphics)],check=True)
    assert not subprocess.check_output(['pdftotext',str(graphics),'-']).strip()
    # Equal-aspect axes in the published 25.6 x 6.55 inch PDF. These are data
    # panels only; original labels and the old key lie outside the crop boxes.
    crop_side, crop_bottom = 5.736, .1188
    crops = []
    height = 6.24 if two_rows else 1.82
    side = 2.70 if two_rows else 1.37
    right_margin = 6.5-(.025+2.835+side)
    left_padding = right_margin-.025 if revision == 4 else 0.
    width = 6.5+left_padding
    inclusion_scale = reference['inclusion_scale'] if reference else 1.
    if revision == 4:
        inclusion_scale = .7
    if revision == 4:
        assert abs((.025+left_padding)-right_margin)<1e-12
    lines = [r'\documentclass[11pt,border=0pt]{standalone}',
             r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern,amsmath,graphicx,tikz}',
             r'\usetikzlibrary{arrows.meta}', r'\definecolor{background}{HTML}{05080D}',
             r'\definecolor{foreground}{HTML}{F1F5F9}', r'\definecolor{negative}{HTML}{65DFFF}',
             r'\definecolor{positive}{HTML}{FFE080}', r'\begin{document}',
             r'\fontsize{10.95}{13.6}\selectfont',
             r'\begin{tikzpicture}[x=1in,y=1in,every node/.style={text=foreground,inner sep=0pt}]',
             rf'\path[use as bounding box] (0,0) rectangle ({width:.7f},{height});',
             rf'\fill[background] (0,0) rectangle ({width:.7f},{height});',
             rf'\begin{{scope}}[xshift={left_padding:.7f}in]']
    for i, target in enumerate(provenance['requested_times']):
        left = .192+i*5.952
        right, top = 25.6-left-crop_side, 6.55-crop_bottom-crop_side
        x = .025+(i % 2)*2.835 if two_rows else .025+i*1.42
        y = (3.18 if i < 2 else .10) if two_rows else .08
        title_y = y+side+.18 if two_rows else 1.64
        lines.append(rf'\node[anchor=south west] at ({x:.5f},{y:.5f}) {{\includegraphics[trim={left:.7f}in {crop_bottom:.7f}in {right:.7f}in {top:.7f}in,clip,width={side:.5f}in]{{{graphics}}}}};')
        if revision >= 3:
            lines.append(rf'\node[anchor=west,font=\fontsize{{12}}{{14.4}}\selectfont] at ({x+.015:.5f},{title_y:.5f}) {{({chr(97+i)})}};')
            lines.append(rf'\node[font=\fontsize{{12}}{{14.4}}\selectfont] at ({x+side/2:.5f},{title_y:.5f}) {{$t={target:g}$}};')
        else:
            lines.append(rf'\node[anchor=west] at ({x+.015:.5f},{title_y:.5f}) {{$\mathbf{{t={target:g}}}$}};')
        crops.append(dict(time=target, source_bounds_inches=[left,crop_bottom,left+crop_side,crop_bottom+crop_side],
                          placed_bounds_inches=[x+left_padding,y,x+side+left_padding,y+side]))
    if two_rows:
        lines.append(r'\begin{scope}[yshift=2.45in]')
    lines += [r'\fill[negative] (5.77,1.24) rectangle (5.86,1.33);',
              r'\node[anchor=west] at (5.93,1.285) {$u<0$};',
              r'\fill[positive] (5.77,1.01) rectangle (5.86,1.10);',
              r'\node[anchor=west] at (5.93,1.055) {$u>0$};',
              r'\node[align=center] at (6.08,.70) {Edge\\orientation};',
              r'\draw[foreground,line width=.8pt,-{Stealth[length=3pt]}] (5.96,.22) -- (6.23,.22);',
              r'\draw[foreground,line width=.8pt,-{Stealth[length=3pt]}] (5.96,.22) -- (5.96,.49);']
    if two_rows:
        lines.append(r'\end{scope}')
    lines += [r'\end{scope}', r'\end{tikzpicture}', r'\end{document}']
    tex = work/'kdv-11pt.tex'
    tex.write_text('\n'.join(lines)+'\n')
    with (work/'kdv-build.log').open('w') as log:
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',tex.name],
                       cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert not any(s in (work/'kdv-build.log').read_text() for s in ['Overfull', 'Underfull', 'LaTeX Warning'])
    shutil.copyfile(work/'kdv-11pt.pdf', output.with_suffix('.pdf'))
    shutil.copyfile(tex, output.with_suffix('.tex'))
    subprocess.run(['pdftoppm','-png','-singlefile','-r','260',str(output.with_suffix('.pdf')),str(output)],check=True)
    receipt = dict(purpose='Typography-only composition from the exact published PDF',
        source_pdf=str(source),source_pdf_sha256=sha(source),
        source_provenance_sha256=sha(KDV/'provenance.json'),source_panels=crops,
        graphics_only_pdf=str(graphics),graphics_only_pdf_sha256=sha(graphics),
        removed_old_text_blocks=count,nontext_drawing_commands_sha256=hashlib.sha256(commands).hexdigest(),
        data_panels_reused_without_rerendering=True, temporal_interpolation=False,
        scientific_computation=False,font_tex_points=FONT,heading_font_tex_points=12 if revision >= 3 else FONT,
        font_reference=reference,figure_size_inches=[width,height],
        added_left_padding_inches=left_padding, left_panel_margin_inches=.025+left_padding,
        right_panel_margin_inches=right_margin, equal_panel_side_margins=revision==4,
        manuscript_inclusion_scale=inclusion_scale,
        manuscript_width_fraction=inclusion_scale*width/6.5,
        layout='Two by two snapshots with shared sign/orientation key on the right' if two_rows else 'Four snapshots in one row with sign/orientation key on the right',
        caption_compatible=True,outputs={ext:sha(output.with_suffix('.'+ext)) for ext in ('pdf','png','tex')})
    write_json(output.with_suffix('.json'),receipt)
    return output



if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Redraw the four saved KdV fields in the paper layout.')
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/kdv')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    KDV=args.output/'field-panels'
    subprocess.run([sys.executable,str(Path(__file__).with_name('render_fields.py')),
                    '--data',str(args.data.resolve()),'--output',str(KDV)],check=True)
    print(kdv(args.output,revision=4))
