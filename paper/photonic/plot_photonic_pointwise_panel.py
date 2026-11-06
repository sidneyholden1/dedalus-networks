"""Pointwise optical intensity in the accepted 95% crop; no EVP solve."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.optimize import brentq

from photonic_intensity_boxes import clipped_parameters
from plot_photonic_interior_figure import scale


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field(x, waves, q, length):
    return waves[0]*np.exp(1j*q*x)+waves[1]*np.exp(1j*q*(length-x))


def pointwise_peak(waves, q, lengths, internal):
    """Bound all edges, then find every extremum on competitive edges."""
    bounds=(np.sum(abs(waves), axis=1)*np.exp(np.maximum(0, -q.imag*lengths)))**2
    ids=np.flatnonzero(internal)
    ids=ids[np.argsort(bounds[ids])[::-1]]
    best=0.; chosen=None; checked=0
    for edge in ids:
        if bounds[edge] <= best:
            break
        qe=q[edge]; le=lengths[edge]; w=waves[edge]
        def derivative(x):
            left=w[0]*np.exp(1j*qe*x)
            right=w[1]*np.exp(1j*qe*(le-x))
            return float(2*np.real(np.conj(left+right)*1j*qe*(left-right)))
        count=max(32, int(np.ceil(le*abs(qe.real)/np.pi*16)))
        xs=np.linspace(0, le, count+1)
        u=field(xs, w, qe, le)
        du=1j*qe*(w[0]*np.exp(1j*qe*xs)-w[1]*np.exp(1j*qe*(le-xs)))
        derivatives=2*np.real(u.conj()*du)
        changes=np.flatnonzero(derivatives[:-1]*derivatives[1:]<0)
        roots=[brentq(derivative, xs[j], xs[j+1], xtol=1e-13) for j in changes]
        candidates=np.array([0., le]+roots)
        values=abs(field(candidates, w, qe, le))**2
        j=int(np.argmax(values)); checked+=1
        if values[j]>best:
            best=float(values[j]); chosen=dict(edge=int(edge), x_um=float(candidates[j]))
    assert chosen is not None and best>0
    return best, dict(**chosen, checked_edges=checked, bound_unchecked=True)


def render(segments, lengths, q, waves, means, low, high, peak, cmap, norm, pixels):
    """Integrate sampled pointwise RGB along each edge into bilinear pixels.

    Strong edges are composited last, as in the edge-mean figure. Sampling
    resolves the intensity period before pixel integration; no sparse samples
    are displayed as if they resolved the optical waves at the crop scale.
    """
    t0,t1=clipped_parameters(segments, low, high)
    ids=np.flatnonzero(t1>t0)
    ids=ids[np.argsort(means[ids], kind='stable')]
    counts=np.maximum(1, np.ceil((t1[ids]-t0[ids])*lengths[ids]*abs(q[ids].real)/np.pi*16).astype(int))
    assert counts.sum()<15000000
    image=np.ones((pixels*pixels, 3), dtype=np.float32)
    pixel_um=float((high-low)[0]/pixels)
    offsets=np.array([(dx,dy) for dy in range(-2,3) for dx in range(-2,3)])
    kernel=np.exp(-np.sum(offsets**2,axis=1)/2);kernel/=kernel.sum()
    maximum=0.; samples=0
    for edge, count in zip(ids, counts):
        delta=(t1[edge]-t0[edge])/count
        t=t0[edge]+(np.arange(count)+.5)*delta
        x=t*lengths[edge]
        intensity=abs(field(x, waves[edge], q[edge], lengths[edge]))**2/peak
        maximum=max(maximum,float(intensity.max()))
        rgb=cmap(norm(intensity))[:,:3]
        xy=(segments[edge,0]+t[:,None]*(segments[edge,1]-segments[edge,0])-low)/pixel_um-.5
        ix=np.floor(xy).astype(np.int64); frac=xy-ix
        keys=[]; weights=[]; colors=[]
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            xp=ix[:,0]+dx; yp=ix[:,1]+dy
            valid=(xp>=0)&(xp<pixels)&(yp>=0)&(yp<pixels)
            weight=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
            keys.append((yp[valid]*pixels+xp[valid]))
            weights.append(weight[valid])
            colors.append(rgb[valid])
        keys=np.concatenate(keys); weights=np.concatenate(weights); colors=np.concatenate(colors)
        unique,inverse=np.unique(keys,return_inverse=True)
        total=np.bincount(inverse,weights=weights)
        valid=total>0
        color=np.column_stack([np.bincount(inverse,weights=weights*colors[:,c]) for c in range(3)])
        color=color[valid]/total[valid,None]
        target=unique[valid]
        # Spread the already integrated colour across the displayed stroke.
        # This changes line weight only; it does not average over whole edges.
        coverage=total[valid]*delta*lengths[edge]/pixel_um
        xp=target%pixels;yp=target//pixels
        wide_keys=[];wide_weights=[];wide_colors=[]
        for (dx,dy),weight in zip(offsets,kernel):
            xx=xp+dx;yy=yp+dy
            keep=(xx>=0)&(xx<pixels)&(yy>=0)&(yy<pixels)
            wide_keys.append(yy[keep]*pixels+xx[keep])
            wide_weights.append(coverage[keep]*weight*5.)
            wide_colors.append(color[keep])
        keys=np.concatenate(wide_keys);weights=np.concatenate(wide_weights);colors=np.concatenate(wide_colors)
        target,inverse=np.unique(keys,return_inverse=True)
        alpha=np.bincount(inverse,weights=weights)
        color=np.column_stack([np.bincount(inverse,weights=weights*colors[:,c]) for c in range(3)])/alpha[:,None]
        alpha=np.minimum(1.,alpha)
        image[target]=image[target]*(1-alpha[:,None])+color*alpha[:,None]
        samples+=int(count)
    assert maximum<=1+1e-10
    return image.reshape(pixels,pixels,3),dict(visible_edges=len(ids),samples=samples,
        samples_per_intensity_period=16,pixel_um=pixel_um,sampled_peak_fraction=maximum,
        stroke_integrated_width_pixels=5.,stroke_kernel_sigma_pixels=1.,
        antialiasing='Bilinear integration of pointwise colour samples, followed by edge compositing; no edge-mean colouring.')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--boxes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(60,65));started=time.monotonic()
    source=args.selection/'selected-fields.npz'
    selection=json.loads((args.selection/'selection.json').read_text())
    boxes=json.loads(args.boxes.read_text())
    assert selection['passed'] and selection['refinement_passed'] and not selection['preliminary']
    assert sha(source)==selection['selected_fields_sha256']==boxes['fields_sha256']
    box=next(b for b in boxes['boxes'] if b['target_fraction']==.95)
    low,high=np.array(box['bounds_um']);assert np.allclose(high-low,box['width_um'])
    with np.load(source,allow_pickle=False) as d:
        segments=d['positions_um'][d['endpoints']];lengths=d['lengths_um']
        waves=d['amplitudes'][0];q=complex(d['k'][0])*d['index']
        means=d['edge_I2'][0]/lengths;internal=d['internal']
    peak,peak_info=pointwise_peak(waves,q,lengths,internal)
    cmap=LinearSegmentedColormap.from_list('pointwise_intensity',
        ['#e5f0f8','#a8cce5','#528db7','#365986','#553574','#aa3970','#e77851'],N=1024)
    norm=LogNorm(1e-4,1,clip=True)
    raster,receipt=render(segments,lengths,q,waves,means,low,high,peak,cmap,norm,1800)
    plt.rcParams.update({'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath}',
        'font.family':'sans-serif','font.sans-serif':['Helvetica'],'font.size':8,
        'text.color':'#28394b','xtick.color':'#28394b','ytick.color':'#28394b',
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(3.4,3.8),facecolor='white')
    ax=fig.add_axes([.03,.20,.94,.76])
    ax.imshow(raster,origin='lower',extent=(low[0],high[0],low[1],high[1]),
        interpolation='lanczos',aspect='equal',zorder=1)
    ax.set_axis_off();scale(ax,200,r'$200\,\mu\mathrm{m}$')
    fig.text(.035,.98,r'\textbf{b}',fontsize=10,va='top')
    cax=fig.add_axes([.15,.085,.73,.026])
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
        orientation='horizontal',ticks=[1e-4,1e-2,1])
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.ax.set_xticklabels([r'$10^{-4}$',r'$10^{-2}$',r'$1$'])
    cb.ax.tick_params(length=2.5,width=.5,pad=3);cb.outline.set_visible(False)
    fig.text(.515,.15,r'$|u_e(x)|^2/\max_{e,x}|u_e(x)|^2$',ha='center',va='center',fontsize=9)
    args.output.mkdir(mode=0o700)
    for suffix in ('png','pdf'):
        fig.savefig(args.output/('panel-b.'+suffix),dpi=500)
    plt.close(fig)
    caption=(r'Pointwise intensity $|u_e(x)|^2$ in the same $95\%$ internal-intensity square, '
        r'computed from the accepted complex amplitudes '
        r'$u_e(x)=a_e e^{ik\eta_e x}+b_e e^{ik\eta_e(\ell_e-x)}$. '
        r'Colours are normalized to the maximum pointwise intensity over internal edges, '
        r'with a logarithmic floor of $10^{-4}$. '
        r'The crop is $1.306\,\mathrm{mm}$ across; the approximately $0.196\,\mu\mathrm{m}$ '
        r'intensity period is unresolved at this display scale. Pointwise colours are integrated '
        r'within pixels to suppress aliasing; no edge-mean intensities determine the colours.'+'\n')
    (args.output/'caption.tex').write_text(caption)
    provenance=dict(selection=str(args.selection/'selection.json'),selection_sha256=sha(args.selection/'selection.json'),
        fields=str(source),fields_sha256=sha(source),boxes=str(args.boxes),boxes_sha256=sha(args.boxes),
        box=box,quantity='abs(u_e(x))**2 / maximum internal pointwise intensity',
        pointwise_peak=peak,peak_search=peak_info,peak_over_maximum_edge_mean=peak/means[internal].max(),
        color_floor=1e-4,color_ceiling=1,text_renderer='LaTeX via Matplotlib usetex',
        intensity_period_um=float(np.pi/abs(q[selection['selected']['peak_edge']].real)),
        rendering=receipt,script_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time(),artifacts={n:sha(args.output/n) for n in ('panel-b.png','panel-b.pdf','caption.tex')})
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()
