"""Full-network pointwise intensity, using the full positive sampled range."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
import numpy as np
import ctypes
import os
import subprocess
import tempfile

from plot_photonic_pointwise_panel import field, pointwise_peak
from plot_photonic_interior_figure import scale


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend():
    source=Path(__file__).with_name('photonic_pointwise_raster.cpp')
    folder=Path(tempfile.mkdtemp(prefix='pointwise-raster-',dir=os.environ['TMPDIR']))
    library=folder/'raster.so'
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],
                   check=True,timeout=45,capture_output=True)
    lib=ctypes.CDLL(str(library))
    pointer=ctypes.c_void_p; index=ctypes.c_int64; real=ctypes.c_double
    lib.scan_field.argtypes=[index]+[pointer]*6
    lib.scan_field.restype=None
    lib.raster_field.argtypes=[index]+[pointer]*8+[real,index,real,real,pointer,index,pointer,pointer]
    lib.raster_field.restype=None
    lib.raster_field_linear.argtypes=lib.raster_field.argtypes
    lib.raster_field_linear.restype=None
    lib.raster_field_linear_width.argtypes=lib.raster_field.argtypes+[real]
    lib.raster_field_linear_width.restype=None
    lib.raster_field_log_width.argtypes=lib.raster_field.argtypes+[real]
    lib.raster_field_log_width.restype=None
    return lib


BACKEND=None


def pointers(*arrays):
    return [ctypes.c_void_p(a.ctypes.data) for a in arrays]


def scan(waves,q,lengths,bins,per_bin):
    arrays=[np.ascontiguousarray(a) for a in (waves,q,lengths,bins,per_bin)]
    result=np.empty(6)
    BACKEND.scan_field(len(lengths),*pointers(*arrays,result))
    return tuple(result)


def rasterize(segments,waves,q,lengths,bins,per_bin,order,low,pixel_um,pixels,minimum,peak,lut,linear=False,stroke_width=1.7):
    arrays=[np.ascontiguousarray(a) for a in (segments,waves,q,lengths,bins,per_bin,order,low)]
    lut=np.ascontiguousarray(lut)
    canvas=np.empty((pixels,pixels,3),np.float32)
    result=np.empty(2)
    function=BACKEND.raster_field_linear if linear else BACKEND.raster_field
    extra=[]
    if stroke_width!=1.7:
        assert 1.7<=stroke_width<=32
        function=BACKEND.raster_field_linear_width if linear else BACKEND.raster_field_log_width
        extra=[stroke_width]
    function(len(lengths),*pointers(*arrays),pixel_um,pixels,minimum,peak,
             *pointers(lut),len(lut),*pointers(canvas,result),*extra)
    return canvas,*result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--boxes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(60,65))
    started=time.monotonic()
    global BACKEND
    BACKEND=backend()
    source=args.selection/'selected-fields.npz'
    selection=json.loads((args.selection/'selection.json').read_text())
    boxes=json.loads(args.boxes.read_text())
    assert selection['passed'] and selection['refinement_passed'] and not selection['preliminary']
    assert sha(source)==selection['selected_fields_sha256']==boxes['fields_sha256']
    with np.load(source,allow_pickle=False) as d:
        positions=d['positions_um'];segments=positions[d['endpoints']]
        waves=d['amplitudes'][0];lengths=d['lengths_um']
        q=complex(d['k'][0])*d['index'];internal=d['internal']
        means=d['edge_I2'][0]/lengths
    peak,peak_info=pointwise_peak(waves,q,lengths,internal)
    low=positions.min(axis=0);high=positions.max(axis=0)
    assert np.isclose(*(high-low))
    pixels=2400;pixel_um=float((high-low)[0]/pixels)
    bins=np.maximum(1,np.ceil(lengths/pixel_um*2).astype(np.int64))
    per_bin=np.maximum(4,np.ceil(lengths/bins*abs(q.real)/np.pi*16).astype(np.int64))
    planned=int(np.sum(bins*per_bin))
    assert planned<225000000
    # Check recurrence and range discovery against direct complex exponentials
    # on representative weak/strong/long/terminal edges before the full pass.
    chosen=np.unique(np.r_[np.argsort(means)[::10000],np.argmax(lengths),
                           np.argmax(means),np.flatnonzero(~internal)[:3]])
    expected=[]
    for e in chosen:
        count=int(bins[e]*per_bin[e])
        x=(np.arange(count)+.5)*lengths[e]/count
        expected.append(abs(field(x,waves[e],q[e],lengths[e]))**2)
    direct=np.concatenate(expected)
    trial=scan(waves[chosen],q[chosen],lengths[chosen],bins[chosen],per_bin[chosen])
    np.testing.assert_allclose([trial[0],trial[1]],[direct.min(),direct.max()],rtol=2e-8,atol=0)
    minimum,maximum,zeros,samples,min_edge,min_x=scan(waves,q,lengths,bins,per_bin)
    assert zeros==0 and minimum>0 and maximum<=peak*(1+1e-10) and samples==planned
    print(json.dumps(dict(stage='range',samples=samples,minimum_relative=minimum/peak,
                          peak=peak,cpu_seconds=time.process_time())),flush=True)
    cmap=LinearSegmentedColormap.from_list('full_pointwise_intensity',
        ['#e5f0f8','#a8cce5','#528db7','#365986','#553574','#aa3970','#e77851'],N=4096)
    raster,under,over=rasterize(segments,waves,q,lengths,bins,per_bin,
        np.argsort(means,kind='stable'),low,pixel_um,pixels,minimum,peak,
        cmap(np.linspace(0,1,4096))[:,:3])
    assert under==0 and over==0
    print(json.dumps(dict(stage='raster',cpu_seconds=time.process_time())),flush=True)
    norm=LogNorm(minimum/peak,1,clip=False)
    plt.rcParams.update({'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath}',
        'font.family':'serif','font.serif':['Computer Modern Roman'],'font.size':10,
        'axes.linewidth':.8,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(8.2,7.25),facecolor='white')
    ax=fig.add_axes([.035,.035,6.65/8.2,6.65/7.25])
    ax.imshow(raster,origin='lower',extent=(low[0],high[0],low[1],high[1]),
              interpolation='lanczos',aspect='equal')
    ax.set_axis_off()
    box=next(b for b in boxes['boxes'] if b['target_fraction']==.95)
    ax.add_patch(Rectangle(box['bounds_um'][0],box['width_um'],box['width_um'],
                          fill=False,linewidth=.75,edgecolor='black'))
    scale(ax,1000,r'$1\,\mathrm{mm}$',fontsize=10,color='black',linewidth=1.)
    cax=fig.add_axes([.87,.035,.021,6.65/7.25])
    lowest=int(np.ceil(np.log10(minimum/peak)/5)*5)
    exponents=list(range(lowest,1,5))
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
                    ticks=[10.**e for e in exponents])
    cb.ax.set_yticklabels([rf'$10^{{{e}}}$' if e else '$1$' for e in exponents])
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.tick_params(length=3,width=.8)
    cax.set_title(r'$|u_e(x)|^2/\rho_{\max}$',fontsize=11,pad=10)
    args.output.mkdir(mode=0o700)
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'full-pointwise.{ext}',dpi=350)
    plt.close(fig)
    caption=(r'Pointwise intensity on all $100{,}485$ physical edges, reconstructed '
        r'from the saved refined complex mode and normalized by the largest '
        r'internal pointwise intensity $\rho_{\max}$. The logarithmic colour '
        r'scale extends from the smallest positive evaluated intensity to '
        r'$\rho_{\max}$, without a prescribed cutoff or clipped samples. '
        r'The square retains the $95\%$ internal-intensity region. '
        r'Optical fringes are unresolved at full-network scale; pointwise colours '
        r'are averaged within half-pixel spatial bins before antialiasing. '
        r'Edges are not assigned a single mean intensity. '+'\n')
    (args.output/'caption.tex').write_text(caption)
    provenance=dict(selection=str(args.selection/'selection.json'),
        selection_sha256=sha(args.selection/'selection.json'),fields=str(source),
        fields_sha256=sha(source),script_sha256=sha(__file__),
        dependencies={n:sha(Path(__file__).with_name(n)) for n in
            ['plot_photonic_pointwise_panel.py','plot_photonic_interior_figure.py',
             'photonic_pointwise_raster.cpp']},
        physical_edges=len(lengths),box=box,boxes_sha256=sha(args.boxes),
        quantity='pointwise abs(u_e(x))**2 / internal pointwise maximum',
        pointwise_peak=peak,peak_search=peak_info,minimum_sampled=minimum,
        minimum_sampled_relative=minimum/peak,minimum_edge=int(min_edge),
        minimum_x_um=float(min_x),maximum_sampled=maximum,
        color_limits=[minimum/peak,1.],color_norm_clip=False,
        under_range_samples=int(under),over_range_samples=int(over),zero_samples=int(zeros),
        optical_samples=int(samples),minimum_samples_per_period=16,
        spatial_bins=int(bins.sum()),maximum_bin_pixels=.5,pixels=pixels,pixel_um=pixel_um,
        intensity_period_um=float(np.pi/abs(q[selection['selected']['peak_edge']].real)),
        sampling_check='Recurrence extrema match direct complex exponentials on representative edges.',
        range_definition='Full positive sampled range; not a certified continuum minimum.',
        antialiasing='Mean of pointwise mapped RGB in half-pixel bins, then bilinear edge compositing.',
        edge_means_used_only_for_overlap_order=True,text_renderer='LaTeX Computer Modern',
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time(),
        artifacts={n:sha(args.output/n) for n in ['full-pointwise.png','full-pointwise.pdf','caption.tex']})
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(args.output/'full-pointwise.png',flush=True)


if __name__=='__main__':
    main()
