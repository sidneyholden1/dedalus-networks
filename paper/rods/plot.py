"""Redraw Figure 1 from the saved steady and transient rod solutions.

Requires the plotting dependencies listed in INSTALL.md, including TeX Live."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from numpy.polynomial import chebyshev as cheb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator,LogLocator
from matplotlib.text import Text
from matplotlib.offsetbox import DrawingArea,VPacker,HPacker,AnchoredOffsetbox
from plotting import evaluate

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/rods')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--layout',choices=('two-rows','one-row'),default='two-rows')
p.add_argument('--font-size',type=float,default=10.95)
p.add_argument('--title-size',type=float,default=12.)
p.add_argument('--axis-label-size',type=float,default=12.)
p.add_argument('--panel-d-title-drop',type=float,default=.12)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
args.output_stem=args.output/'rod-grid'
args.two_rank_data=args.data
args.transient_file=args.data/'transient-n32-s128-r1.json'
one_row=args.layout=='one-row'
steady_data=args.data
reports=[json.loads((steady_data/f'rods-{r}r-{m}.json').read_text()) for r,m in [(1,'direct'),(2,'dtn'),(4,'dtn')]]
checks=json.loads((args.data/'transient-verification.json').read_text());assert checks['passed']
transient=json.loads(args.transient_file.read_text());assert transient['correctness']['passed']
steady=next(r for r in reports[2]['results'] if r['N']==transient['N'])
xy_a=np.array([(x,y) for y in range(4) for x in range(4)])
edges_a=np.array([(4*y+x,4*y+x+1) for y in range(4) for x in range(3)]+[(4*y+x,4*(y+1)+x) for y in range(3) for x in range(4)])
xy_b=np.array(transient['xy']);edges_b=np.array(transient['edges'])
# Standard LaTeX article[11pt] uses a 10.95 TeX-point normal font. With usetex
# this is the same font and physical size as the manuscript, including maths.
label_size=args.font_size
title_size=args.title_size if args.title_size is not None else label_size
axis_size=args.axis_label_size if args.axis_label_size is not None else label_size
plt.rcParams.update({'font.size':label_size,'axes.labelsize':label_size,
    'axes.titlesize':label_size,'xtick.labelsize':label_size,
    'ytick.labelsize':label_size,'legend.fontsize':label_size,
    'text.usetex':True,'font.family':'serif','font.serif':['Computer Modern Roman'],
    'text.latex.preamble':r'\usepackage[T1]{fontenc}\usepackage{lmodern}',
    'axes.formatter.use_mathtext':True})

two_checks=json.loads((args.two_rank_data/'two-rank-verification.json').read_text());assert two_checks['passed']
transient_reports={r:json.loads(((args.two_rank_data if r==2 else args.data)/f'transient-n24-s128-r{r}.json').read_text()) for r in (1,2,4)}
width,height=6.5,(2.72 if one_row else 6.65)
fig=plt.figure(figsize=(width,height),dpi=220)
def axes(x,y,w,h):return fig.add_axes([x/width,y/height,w/width,h/height])
if one_row:
    a=axes(.04,1.18,.94,.94);ac=axes(1.05,1.18,.06,.94)
    b=axes(2.12,1.18,1.07,.94)
    c=axes(3.29,1.18,.94*np.ptp(xy_b[:,0])/np.ptp(xy_b[:,1]),.94);cc=axes(4.30,1.18,.06,.94)
    d=axes(5.37,1.18,1.07,.94)
else:
    a=axes(.15,4.00,2.25,2.25);ac=axes(2.60,4.00,.10,2.03)
    b=axes(4.05,4.00,2.25,2.25)
    c=axes(.15,.70,2.25*np.ptp(xy_b[:,0])/np.ptp(xy_b[:,1]),2.25);cc=axes(2.60,.70,.10,2.03)
    d=axes(4.05,.70,2.25,2.25)
z=np.linspace(-1,1,401);s=(z+1)/2
steady_values=np.array([cheb.chebval(z,q) for q in steady['coefficients']])
transient_values=evaluate(transient,transient['snapshots'][-1],z)
assert steady_values.min()>0
print('Field ranges:',(steady_values.min(),steady_values.max()),(transient_values.min(),transient_values.max()))
def draw_network(target,xy,edges,values):
    minimum=values.min()
    # Keep the small coarse-timestep undershoot visible; never clip or mask it.
    norm=(LogNorm(10.**np.floor(np.log10(minimum)),1) if minimum>0 else
          SymLogNorm(linthresh=1e-6,vmin=-2e-6,vmax=1,base=10))
    segments=[];temperatures=[]
    for (v,w),value in zip(edges,values):
        points=(1-s[:,None])*xy[v]+s[:,None]*xy[w]
        segments.extend(np.stack([points[:-1],points[1:]],axis=1));temperatures.extend((value[:-1]+value[1:])/2)
    assert min(temperatures)>=norm.vmin and max(temperatures)<=norm.vmax
    lc=LineCollection(segments,array=np.array(temperatures),norm=norm,cmap='inferno',linewidths=2.2 if one_row else 4,capstyle='round',joinstyle='round',clip_on=False)
    target.add_collection(lc);target.set(xlim=(xy[:,0].min(),xy[:,0].max()),ylim=(xy[:,1].min(),xy[:,1].max()),aspect='equal');target.set_axis_off()
    return lc
la=draw_network(a,xy_a,edges_a,steady_values);lc=draw_network(c,xy_b,edges_b,transient_values)
# Shared baselines within each row; all prose uses the manuscript's size.
top_text=[];bottom_text=[];panel_letters=[]
if one_row:
    headings=[(a,'(a) Steady\n'+rf'$N={steady["N"]}$',''),
              (b,'(b) Convergence\nExact reference','Modes per unit\nlength, $N$'),
              (c,'(c) Transient\n'+rf'$t=1$, $N={transient["N"]}$',rf'$\Delta t=1/{round(1/transient["dt"])}$'),
              (d,'(d) Convergence\n'+r'$N=96$ reference','Modes per unit\nlength, $N$')]
    for target,heading,footer in headings:
        box=target.get_position();centre=box.x0+box.width/2
        top_text.append(fig.text(centre,(2.38-(args.panel_d_title_drop if target is d else 0))/height,heading,ha='center',va='baseline',fontsize=title_size,linespacing=1.1))
        if footer:
            bottom_text.append(fig.text(centre,(.77 if target in (b,d) else .97)/height,footer,ha='center',va='baseline',fontsize=label_size,linespacing=1.1))
else:
    for label,x,y in (('(a)',.03,6.47),('(b)',3.63,6.47),('(c)',.03,3.17),('(d)',3.63,3.17)):
        panel_letters.append(fig.text(x/width,y/height,label,ha='left',va='baseline',fontsize=title_size))
    for target,heading,footer,top,bottom in (
    (a,'Steady',rf'$N={steady["N"]}$',6.47,3.56),
    (b,'Reference: exact solution','Modes per unit length, $N$',6.47,3.56),
    (c,r'Transient, $t=1$',rf'$N={transient["N"]}$, $\Delta t=1/{round(1/transient["dt"])}$',3.17,.26),
    (d,r'$t=1$, $\Delta t=1/128$'+'\n'+r'Reference: $N=96$','Modes per unit length, $N$',3.17,.26)):
        box=target.get_position();centre=box.x0+box.width/2
        top_text.append(fig.text(centre,(top-(args.panel_d_title_drop if target is d else 0))/height,heading,ha='center',va='baseline',fontsize=title_size,linespacing=1.15))
        bottom_text.append(fig.text(centre,bottom/height,footer,ha='center',va='baseline',fontsize=label_size))
bar_labels=[]
for field_ax,bar_ax,collection in ((a,ac,la),(c,cc,lc)):
    ticks=(10.**np.arange(int(np.log10(collection.norm.vmin)),1) if isinstance(collection.norm,LogNorm)
           else np.array([-1e-6,0,1e-6,1e-5,1e-4,.001,.01,.1,1]))
    if one_row and isinstance(collection.norm,SymLogNorm):
        ticks=np.array([-1e-6,1e-4,.01,1])
    bar=fig.colorbar(collection,cax=bar_ax,ticks=ticks)
    bar.ax.set_yticklabels([('$'+('-' if x<0 else '')+f'10^{{{int(round(np.log10(abs(x))))}}}$') if 0<abs(x)<.001 else f'${x:g}$' for x in ticks]);bar.ax.yaxis.set_minor_locator(NullLocator())
    if one_row:
        bar.ax.tick_params(length=2,width=.6,pad=2)
    label=fig.text(bar_ax.get_position().x0+bar_ax.get_position().width/2,field_ax.get_position().y1+(.15/height if one_row else 0),'$u$',ha='center',va='top',fontsize=label_size)
    bar_labels.append((field_ax,bar_ax,label))
styles={1:dict(color='black',marker='o',ms=4.5,ls='-',label='Direct'),2:dict(color='#2379b5',marker='s',ms=4,mfc='none',ls='--',label='2 processes'),4:dict(color='#c44030',marker='x',ms=5,ls=':',label='4 processes')}
if one_row:
    for style in styles.values():
        style['ms']*=.65
        style['lw']=.85
for report in reports:
    b.semilogy([r['N'] for r in report['results']],[r['metrics']['relative_l2'] for r in report['results']],**styles[report['ranks']])
all_rows=checks['spatial']+two_checks['spatial']
for ranks in (1,2,4):
    rows=sorted((r for r in all_rows if r['ranks']==ranks),key=lambda r:r['N'])
    assert [r['N'] for r in rows]==[4,6,8,10,12,16,24,32,48,64]
    d.semilogy([r['N'] for r in rows],[r['relative_l2'] for r in rows],**styles[ranks])
for target in (b,d):
    # Fewer labelled ticks avoid collisions at actual text size; every original
    # data point remains plotted and the x/y limits are unchanged.
    target.set(ylabel='Relative $L^2$ error',xticks=[4,16,32,48,64],xlim=(2.6,66),ylim=(1e-16,.1))
    target.set_yticks(10.**np.arange(-4,-17,-4) if one_row else 10.**np.arange(-2,-17,-2))
    target.yaxis.set_minor_locator(NullLocator())
    target.tick_params(axis='both',labelsize=label_size,pad=3)
    if one_row:
        target.tick_params(length=2.5,width=.6,pad=2)
        target.set_ylabel('Relative $L^2$ error',labelpad=2)
        for spine in target.spines.values():
            spine.set_linewidth(.6)
    target.grid(True,which='major',alpha=.2)
palette=['#0072B2','#E69F00','#009E73','#CC79A7']
def partition_legend(target,xy,edges,saved,anchor,location):
    rows=[]
    for ranks in (1,2,4):
        report=saved[ranks];owners=np.asarray(report['owners']);assert set(owners)==set(range(ranks))
        for diagnostic in report['diagnostics']:
            assert np.count_nonzero(owners==diagnostic['rank'])==diagnostic['local_edges']
        style=styles[ranks];row=DrawingArea(65,40,0,0) if one_row else DrawingArea(126,24,0,0)
        sample_style={k:v for k,v in style.items() if k!='label'}
        sample_style['lw']=1.5
        row.add_artist(Line2D([3,11,19] if one_row else [0,10,20],[12,12,12],markevery=[1],**sample_style))
        row.add_artist(Text(32.5 if one_row else 28,33 if one_row else 12,style['label'],ha='center' if one_row else 'left',va='center',fontsize=label_size))
        scaled=(xy-xy.min(axis=0))*22/np.ptp(xy,axis=0).max()
        for (v,w),owner in zip(edges,owners):
            points=scaled[[v,w]]
            row.add_artist(Line2D((34 if one_row else 101)+points[:,0],1+points[:,1],color=palette[int(owner)],lw=1.1 if one_row else 1.5,solid_capstyle='round'))
        rows.append(row)
    if one_row:
        legend=AnchoredOffsetbox(loc='lower left',child=HPacker(children=rows,align='bottom',pad=0,sep=9),frameon=False,bbox_to_anchor=((.05 if target is b else 3.30)/width,.04/height),bbox_transform=fig.transFigure,pad=0,borderpad=0)
    else:
        legend=AnchoredOffsetbox(loc=location,child=VPacker(children=rows,align='left',pad=0,sep=3),frameon=False,bbox_to_anchor=anchor,bbox_transform=target.transAxes,pad=0,borderpad=0)
    target.add_artist(legend)
partition_legend(b,xy_a,edges_a,{r['ranks']:next(v for v in r['results'] if v['N']==24) for r in reports},(.98,.97),'upper right')
partition_legend(d,xy_b,edges_b,transient_reports,(.98,.97),'upper right')
axis_text=[b.yaxis.label,d.yaxis.label]+[t for t in bottom_text if t.get_text().startswith('Modes per unit')]
for t in axis_text:
    t.set_fontsize(axis_size)
fig.canvas.draw()
if not one_row:
    for target in (b,d):
        box=target.get_window_extent();assert abs(box.width-box.height)<1e-8
for group,expected in ((top_text,title_size),(bottom_text,label_size),(panel_letters,title_size)):
    assert all(text.get_fontsize()==(axis_size if text in axis_text else expected) for text in group)
heading_text=set(top_text+panel_letters)
visible_text=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
for t in visible_text:
    assert t.get_fontsize()==(title_size if t in heading_text else axis_size if t in axis_text else label_size),(t.get_text(),t.get_fontsize())
    box=t.get_window_extent(fig.canvas.get_renderer())
    assert box.x0>=0 and box.x1<=fig.bbox.width,(t.get_text(),list(box.extents))
    assert box.y0>=0 and box.y1<=fig.bbox.height,(t.get_text(),list(box.extents))
for target in (b,d,ac,cc):
    for labels in (target.get_xticklabels(),target.get_yticklabels()):
        boxes=[t.get_window_extent(fig.canvas.get_renderer()) for t in labels if t.get_visible()]
        assert not any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
for ext in ('png','pdf'):
    fig.savefig(args.output_stem.with_suffix('.'+ext),dpi=220)

input_paths=[args.data/f'rods-{r}r-{m}.json' for r,m in [(1,'direct'),(2,'dtn'),(4,'dtn')]]
input_paths += [args.data/'transient-verification.json',args.two_rank_data/'two-rank-verification.json',args.transient_file]
input_paths += [(args.two_rank_data if r==2 else args.data)/f'transient-n24-s128-r{r}.json' for r in (1,2,4)]
hash_file=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
receipt=dict(purpose='Typography-only Figure 1 preview; no numerical evolution',
    manuscript_documentclass='article[11pt,letterpaper]',font_size_tex_pt=label_size,
    title_and_panel_letter_size_tex_pt=title_size,panel_d_heading_drop_inches=args.panel_d_title_drop,
    axis_label_size_tex_pt=axis_size,
    final_width_inches=width,height_inches=height,layout='1 row of 4 panels' if one_row else '2 rows by 2 columns',
    all_visible_text_has_requested_size=True,text_within_canvas=True,tick_overlap_checked=True,
    input_sha256={str(path):hash_file(path) for path in input_paths},
    renderer_sha256=hash_file(Path(__file__)),
    sampled_steady_sha256=hashlib.sha256(steady_values.tobytes()).hexdigest(),
    sampled_transient_sha256=hashlib.sha256(transient_values.tobytes()).hexdigest(),
    steady_range=[float(steady_values.min()),float(steady_values.max())],
    transient_range=[float(transient_values.min()),float(transient_values.max())],
    outputs={ext:hash_file(args.output_stem.with_suffix('.'+ext)) for ext in ('png','pdf')})
args.output_stem.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(args.output_stem.with_suffix('.png'))
