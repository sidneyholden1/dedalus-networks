"""Compact interior-mode preview, with edge means and a resolved field trace."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap,LogNorm,Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale(ax,length,label,fontsize=7,color='#28394b',linewidth=1.1):
    lo=np.array([ax.get_xlim()[0],ax.get_ylim()[0]])
    span=np.array([np.diff(ax.get_xlim())[0],np.diff(ax.get_ylim())[0]])
    start=lo+np.array([.065,.065])*span
    ax.plot([start[0],start[0]+length],[start[1],start[1]],lw=linewidth+2.2,color='white',zorder=8)
    ax.plot([start[0],start[0]+length],[start[1],start[1]],lw=linewidth,color=color,zorder=9)
    ax.text(start[0]+length/2,start[1]+.03*span[1],label,ha='center',va='bottom',fontsize=fontsize,
            bbox=dict(facecolor='white',edgecolor='none',pad=.5,alpha=.95),zorder=10)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preliminary',action='store_true')
    parser.add_argument('--color-floor',type=float,choices=(1e-8,1e-6,1e-4),default=1e-4)
    parser.add_argument('--color-scale',choices=('log','linear'),default='log')
    parser.add_argument('--zoom-color-scale',choices=('match','log','linear'),default='match')
    parser.add_argument('--figure-one-style',action='store_true')
    parser.add_argument('--panel-a-only',action='store_true')
    parser.add_argument('--intensity-boxes',type=Path)
    parser.add_argument('--box-fraction',type=float,choices=(.90,.95,.99),default=.90)
    args=parser.parse_args()
    if args.figure_one_style:
        assert not args.panel_a_only and args.zoom_color_scale=='match'
    folder=args.selection
    selection=json.loads((folder/'selection.json').read_text())
    assert selection['passed'] and selection['preliminary']==args.preliminary
    assert selection['refinement_passed']==(not args.preliminary)
    chosen=selection['selected'];assert chosen['clearly_interior']
    fp=folder/'selected-fields.npz';assert sha(fp)==selection['selected_fields_sha256']
    with np.load(fp,allow_pickle=False) as d:
        positions=d['positions_um'];endpoints=d['endpoints'];lengths=d['lengths_um']
        internal=d['internal'];eta=d['index'];a=d['amplitudes'][0];i2=d['edge_I2'][0];k=complex(d['k'][0])
    assert len(endpoints)==100485 and abs(i2[internal].sum()-1)<1e-10
    assert [k.real,k.imag]==chosen['k']
    segments=positions[endpoints];mean=i2/lengths;maximum=mean[internal].max();relative=mean/maximum
    peak=chosen['peak_edge'];assert mean[peak]==maximum
    low,high=positions.min(axis=0),positions.max(axis=0)
    peak_xy=segments[peak].mean(axis=0)
    width=900.0
    zoom_low=peak_xy-width/2;zoom_high=peak_xy+width/2
    box_provenance=None
    box_caption=r'The box identifies a $900\,\mu\mathrm{m}$ square. '
    if args.intensity_boxes:
        boxes=json.loads(args.intensity_boxes.read_text())
        assert boxes['passed'] and boxes['fields_sha256']==sha(fp)
        assert boxes['mode_index']==chosen['mode_index']
        box=next(b for b in boxes['boxes'] if b['target_fraction']==args.box_fraction)
        assert abs(box['actual_fraction']-args.box_fraction)<1e-8
        zoom_low,zoom_high=np.array(box['bounds_um']);width=box['width_um']
        assert np.allclose(zoom_high-zoom_low,width)
        box_provenance=dict(source=str(args.intensity_boxes),source_sha256=sha(args.intensity_boxes),
            definition=boxes['definition'],center_um=boxes['center_um'],**box)
        box_caption=(rf'The ${width/1000:.3f}\,\mathrm{{mm}}$ square is centred at the internal intensity centroid '
            rf'and contains ${100*args.box_fraction:g}\%$ of $\sum_{{e\,\mathrm{{internal}}}}\int_e |u_e|^2\,\mathrm{{d}}x$, '
            r'including partial-edge contributions at its boundary. ')
    assert np.all(zoom_low>low) and np.all(zoom_high<high)
    floor_exponent=int(round(np.log10(args.color_floor)))
    norm=LogNorm(args.color_floor,1,clip=True) if args.color_scale=='log' else Normalize(0,1,clip=True)
    ticks=[args.color_floor,np.sqrt(args.color_floor),1] if args.color_scale=='log' else [0,.5,1]
    tick_labels=[rf'$10^{{{floor_exponent}}}$',rf'$10^{{{floor_exponent//2}}}$',r'$1$'] if args.color_scale=='log' else [r'$0$',r'$0.5$',r'$1$']
    scale_caption=(rf'Every physical edge is drawn with one continuous logarithmic colour scale; values below $10^{{{floor_exponent}}}I_{{\max}}$ use its lower blue endpoint. '
        if args.color_scale=='log' else r'Every physical edge is drawn with a continuous linear colour scale from zero to $I_{\max}$. ')
    cmap=LinearSegmentedColormap.from_list('interior_intensity',
        ['#e5f0f8','#a8cce5','#528db7','#365986','#553574','#aa3970','#e77851'])
    plt.rcParams.update({'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath}',
        'font.family':'sans-serif','font.sans-serif':['Helvetica'],'font.size':7.5,
        'axes.labelsize':8,'axes.linewidth':.6,'axes.edgecolor':'#28394b','text.color':'#28394b',
        'xtick.color':'#28394b','ytick.color':'#28394b','xtick.labelsize':7,'ytick.labelsize':7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    if args.figure_one_style:
        # Figure 1 is 14.35 inches wide with 10-point CM labels; identical
        # canvas width preserves their size when both are included at textwidth.
        plt.rcParams.update({'font.family':'serif','font.serif':['Computer Modern Roman'],
            'font.size':10,'axes.labelsize':10,'axes.linewidth':.8,
            'xtick.labelsize':10,'ytick.labelsize':10,'axes.edgecolor':'black',
            'text.color':'black','xtick.color':'black','ytick.color':'black'})
    if args.panel_a_only:
        fig=plt.figure(figsize=(2.3,2.65),facecolor='white')
        whole=fig.add_axes([.02,.20,.96,.75])
        ids=np.argsort(relative,kind='stable')
        whole.add_collection(LineCollection(segments[ids],colors=cmap(norm(relative[ids])),linewidths=.28,rasterized=True))
        lo,hi=low-.012*(high-low),high+.012*(high-low)
        whole.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');whole.set_axis_off()
        whole.add_patch(Rectangle(zoom_low,width,width,fill=False,lw=.65,edgecolor='#28394b',zorder=7))
        scale(whole,1000,r'$1\,\mathrm{mm}$')
        fig.text(.035,.97,r'\textbf{a}',fontsize=9,ha='left',va='center')
        cax=fig.add_axes([.27,.105,.65,.024])
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=ticks)
        cb.ax.xaxis.set_minor_locator(NullLocator());cb.ax.set_xticklabels(tick_labels)
        cb.ax.tick_params(length=2,width=.5,pad=2,labelsize=6.5);cb.outline.set_visible(False)
        fig.text(.23,.117,r'$I_e/I_{\max}$',ha='right',va='center',fontsize=7.5)
        if args.preliminary:
            fig.text(.98,.015,'Preliminary: refinement pending',ha='right',fontsize=5.5,color='#7b5361')
        out=args.output;out.mkdir(mode=0o700)
        for suffix in ('png','pdf','svg'):
            fig.savefig(out/('panel-a.'+suffix),dpi=400)
        plt.close(fig)
        caption=(r'Interior passive resonance of the $100{,}485$-edge network L3. '
            r'Edge-mean intensity $I_e=\ell_e^{-1}\int_0^{\ell_e}|u_e(x)|^2\,\mathrm{d}x$, '
            r'normalized by $I_{\max}=\max_e I_e$. '+scale_caption+
            box_caption+
            f'Mode {chosen["mode_index"]}. '+
            ('Physical checks pass; coarse/fine refinement remains pending.\n' if args.preliminary else 'Physical and refinement checks pass.\n'))
        (out/'caption.tex').write_text(caption)
        provenance=dict(preliminary=args.preliminary,panels=['a'],selection=str(folder/'selection.json'),
            selection_sha256=sha(folder/'selection.json'),fields_sha256=sha(fp),script_sha256=sha(__file__),selected=chosen,
            text_renderer='LaTeX via Matplotlib usetex',color_scale=args.color_scale,color_floor=float(norm.vmin),color_ceiling=1,
            separate_gray_category=False,plotted_quantity_maps='edge mean intensity / maximum internal-edge mean',
            zoom_bounds_um=[zoom_low.tolist(),zoom_high.tolist()],intensity_box=box_provenance,physical_edges=len(endpoints),
            artifacts={name:sha(out/name) for name in ('panel-a.png','panel-a.pdf','panel-a.svg','caption.tex')})
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(out/'panel-a.png')
        return
    zoom_scale=args.color_scale if args.zoom_color_scale=='match' else args.zoom_color_scale
    zoom_norm=LogNorm(args.color_floor,1,clip=True) if zoom_scale=='log' else Normalize(0,1,clip=True)
    if args.figure_one_style:
        figure_size=(14.35,4.55);save_dpi=220;stroke_scale=14.35/7
        fig=plt.figure(figsize=figure_size,facecolor='white',dpi=save_dpi)
        def axes_inches(x,y,w,h):
            return fig.add_axes([x/figure_size[0],y/figure_size[1],w/figure_size[0],h/figure_size[1]])
        whole=axes_inches(.40,.55,3.65,3.65)
        close=axes_inches(4.55,.55,3.65,3.65)
        cax=axes_inches(8.42,.55,.14,3.65)
        trace=axes_inches(10.10,.55,3.85,3.65)
    else:
        figure_size=(7.0,2.60);save_dpi=400;stroke_scale=1.
        fig=plt.figure(figsize=figure_size,facecolor='white')
        whole=fig.add_axes([.02,.20,.30,.74])
        close=fig.add_axes([.345,.20,.30,.74])
        trace=fig.add_axes([.745,.27,.238,.62])
    for ax,local in [(whole,False),(close,True)]:
        lo,hi=(zoom_low,zoom_high) if local else (low-.012*(high-low),high+.012*(high-low))
        visible=np.all(segments.max(axis=1)>=lo,axis=1)&np.all(segments.min(axis=1)<=hi,axis=1)
        ids=np.flatnonzero(visible);ids=ids[np.argsort(relative[ids],kind='stable')]
        panel_norm=zoom_norm if local else norm
        ax.add_collection(LineCollection(segments[ids],colors=cmap(panel_norm(np.maximum(relative[ids],1e-300))),
            linewidths=(.60 if local else .28)*stroke_scale,rasterized=True))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');ax.set_axis_off()
    whole.add_patch(Rectangle(zoom_low,width,width,fill=False,lw=.65*stroke_scale,edgecolor='#28394b',zorder=7))
    close.add_patch(Rectangle(zoom_low,width,width,fill=False,lw=.4*stroke_scale,edgecolor='#adbecb',clip_on=False))
    # Circle marks the single physical edge used in the third panel.
    close.scatter([peak_xy[0]],[peak_xy[1]],s=70*stroke_scale**2,facecolors='none',edgecolors='#28394b',linewidths=.7*stroke_scale,zorder=6)
    close.annotate(r'$e_*$',xy=peak_xy,xytext=(15*stroke_scale,13*stroke_scale),textcoords='offset points',
        fontsize=10 if args.figure_one_style else 7.5,ha='left',va='bottom',zorder=7,
        arrowprops=dict(arrowstyle='-',lw=.5*stroke_scale,color='#28394b',shrinkB=6*stroke_scale),
        bbox=dict(facecolor='white',edgecolor='none',pad=.5,alpha=.9))
    scale_options=dict(fontsize=10,color='black',linewidth=1.1*stroke_scale) if args.figure_one_style else {}
    scale(whole,1000,r'$1\,\mathrm{mm}$',**scale_options);scale(close,200,r'$200\,\mu\mathrm{m}$',**scale_options)
    x=np.linspace(0,min(3.,lengths[peak]),2049)
    q=k*eta[peak]
    u=a[peak,0]*np.exp(1j*q*x)+a[peak,1]*np.exp(1j*q*(lengths[peak]-x))
    intensity=abs(u)**2/maximum
    trace.plot(x,intensity,color='#623b76',lw=.8*stroke_scale)
    trace.set(xlim=(0,x[-1]),ylim=(0,1.06*intensity.max()))
    trace.set_xlabel(r'$x\;(\mu\mathrm{m})$',labelpad=4)
    trace.set_ylabel(r'$|u_{e_*}(x)|^2/I_{\max}$',labelpad=5)
    trace.tick_params(direction='out',length=3.5 if args.figure_one_style else 2.4,
        width=.8 if args.figure_one_style else .6,pad=3.5 if args.figure_one_style else 3)
    trace.spines[['top','right']].set_visible(False)
    if args.figure_one_style:
        for target,letter,offset in ((whole,'a',.045*3.65),(close,'b',.045*3.65),(trace,'c',.17*2.55)):
            fig.text(target.get_position().x0-offset/figure_size[0],4.10/figure_size[1],
                f'({letter})',fontsize=10,ha='right',va='baseline')
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
            orientation='vertical',ticks=[1e-4,1e-3,1e-2,1e-1,1])
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.ax.set_yticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$1$'])
        cb.ax.tick_params(length=3.5,width=.8,pad=3.5,labelsize=10)
        cb.outline.set_linewidth(.8)
        bar_label=fig.text(8.49/figure_size[0],4.30/figure_size[1],r'$I_e/I_{\max}$',
            ha='center',va='bottom',fontsize=14.3)
        bar_specs=[]
    else:
        for xpos,letter in [(.02,'a'),(.345,'b'),(.745,'c')]:
            fig.text(xpos,.96,rf'\textbf{{{letter}}}',fontsize=9,ha='left',va='center')
        bar_specs=([(.20,.40,norm,args.color_scale)] if zoom_scale==args.color_scale else
                   [(.115,.19,norm,args.color_scale),(.440,.19,zoom_norm,zoom_scale)])
    for left,bar_width,bar_norm,bar_scale in bar_specs:
        bar_ticks=([args.color_floor,np.sqrt(args.color_floor),1] if bar_scale=='log' else [0,.5,1])
        bar_labels=([rf'$10^{{{floor_exponent}}}$',rf'$10^{{{floor_exponent//2}}}$',r'$1$']
                    if bar_scale=='log' else [r'$0$',r'$0.5$',r'$1$'])
        cax=fig.add_axes([left,.11,bar_width,.027])
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=bar_norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=bar_ticks)
        cb.ax.xaxis.set_minor_locator(NullLocator());cb.ax.set_xticklabels(bar_labels)
        cb.ax.tick_params(length=2,width=.5,pad=2,labelsize=6.5);cb.outline.set_visible(False)
        fig.text(left-.015,.1235,r'$I_e/I_{\max}$',ha='right',va='center',fontsize=7.5)
    if args.preliminary:
        fig.text(.98,.015,'Preliminary: refinement pending',ha='right',fontsize=6,color='#7b5361')
    layout_receipt=None
    if args.figure_one_style:
        fig.canvas.draw()
        boxes={name:ax.get_window_extent().bounds for name,ax in [('a',whole),('b',close),('c',trace),('colorbar',cax)]}
        assert max(b[1] for b in boxes.values())-min(b[1] for b in boxes.values())<1e-7
        assert max(b[3] for b in boxes.values())-min(b[3] for b in boxes.values())<1e-7
        renderer=fig.canvas.get_renderer()
        for text in fig.texts+[trace.xaxis.label,trace.yaxis.label]+trace.get_xticklabels()+trace.get_yticklabels()+cb.ax.get_yticklabels():
            box=text.get_window_extent(renderer)
            assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
        layout_receipt=dict(axes_pixels={name:list(b) for name,b in boxes.items()},equal_bottoms_and_heights=True,
            all_checked_labels_within_canvas=True)
    out=args.output;out.mkdir(mode=0o700)
    for suffix in ('png','pdf','svg'):
        fig.savefig(out/('interior-mode.'+suffix),dpi=save_dpi)
    plt.close(fig)
    map_caption=(rf'Panel (a) uses a logarithmic scale clipped below $10^{{{floor_exponent}}}$; '
        r'panel (b) uses a linear scale with the same normalization. '
        if args.color_scale=='log' and zoom_scale=='linear' else scale_caption)
    caption=(r'Interior passive resonance of a $100{,}485$-edge network. '
        r'(a) Edge-mean intensity $I_e=\ell_e^{-1}\int_0^{\ell_e}|u_e(x)|^2\,\mathrm{d}x$, '
        r'normalized by $I_{\max}=\max_e I_e$. '
        r'(b) The boxed region. '+box_caption
        +map_caption+
        r'(c) Resolved intensity over the first '+f'{x[-1]:g}'+r'$\,\mu\mathrm{m}$ of the circled edge $e_*$. '
        rf'The resonance has $k=({k.real:.8f}{k.imag:+.9f}i)\,\mu\mathrm{{m}}^{{-1}}$, '
        rf'$Q\simeq {chosen["Q"]:.0f}$ and $L_{{\mathrm{{part}}}}={chosen["Lpart_um"]/1000:.3f}\,\mathrm{{mm}}$. '+
        ('Refinement remains pending.\n' if args.preliminary else
         'The mode is unchanged to numerical precision under refinement from $24$ to $32$ envelope coefficients per edge.\n'))
    (out/'caption.tex').write_text(caption)
    np.savetxt(out/'resolved-trace.csv',np.column_stack([x,u.real,u.imag,intensity]),delimiter=',',
        header='x_um,u_real,u_imag,intensity_over_max_edge_mean',comments='')
    provenance=dict(preliminary=args.preliminary,selection=str(folder/'selection.json'),selection_sha256=sha(folder/'selection.json'),
        fields_sha256=sha(fp),script_sha256=sha(__file__),selected=chosen,
        text_renderer='LaTeX via Matplotlib usetex',color_scale=args.color_scale,color_floor=float(norm.vmin),color_ceiling=1,separate_gray_category=False,
        zoom_color_scale=zoom_scale,zoom_color_floor=float(zoom_norm.vmin),zoom_color_ceiling=1,
        figure_size_inches=list(figure_size),network_raster_dpi=save_dpi,trace_and_labels_vector=True,
        figure_one_style=args.figure_one_style,colorbar_orientation='vertical' if args.figure_one_style else 'horizontal',
        layout_checks=layout_receipt,
        plotted_quantity_maps='edge mean intensity / maximum internal-edge mean',
        plotted_quantity_trace='resolved absolute squared complex field / maximum internal-edge mean',
        zoom_bounds_um=[zoom_low.tolist(),zoom_high.tolist()],intensity_box=box_provenance,trace_edge=peak,trace_interval_um=[0,float(x[-1])],
        artifacts={name:sha(out/name) for name in ('interior-mode.png','interior-mode.pdf','interior-mode.svg','caption.tex','resolved-trace.csv')})
    if args.figure_one_style:
        reference=Path('/mnt/home/sholden/projects/dedalus_network/docs/research/overleaf/figures/rod-grid.pdf')
        reference_source=Path('/mnt/home/sholden/projects/dedalus_network/runs/rod-grid-20261009/preview_aligned_dt128.py')
        provenance['typography_reference']=dict(pdf=str(reference),pdf_sha256=sha(reference),
            source=str(reference_source),source_sha256=sha(reference_source),family='Computer Modern',
            font_size_pt=10,panel_label_size_pt=10,panel_label_weight='normal',panel_label_format='(a), (b), (c)',
            colorbar_label_size_pt=14.3,reference_width_inches=14.35,matched_width_inches=figure_size[0],
            typesetting='Actual LaTeX with Computer Modern, visually matching Figure 1 CM mathtext fonts')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(out/'interior-mode.png')


if __name__=='__main__':main()
