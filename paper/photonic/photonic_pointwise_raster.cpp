// Bounded postprocessing of saved travelling-wave amplitudes; no eigenproblem.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

using Complex=std::complex<double>;
using Index=std::int64_t;
static Complex read_complex(const double* p,Index i) { return {p[2*i],p[2*i+1]}; }

extern "C" void scan_field(Index edges,const double* waves,const double* q,
    const double* lengths,const Index* bins,const Index* per_bin,double* result) {
    double smallest=std::numeric_limits<double>::infinity(),largest=0;
    Index zeros=0,samples=0,min_edge=-1;
    double min_x=0;
    for(Index e=0;e<edges;++e) {
        const Complex qe=read_complex(q,e),a=read_complex(waves,2*e),b=read_complex(waves,2*e+1);
        const double step=lengths[e]/bins[e]/per_bin[e];
        const Complex forward=std::exp(Complex(0,1)*qe*step),backward=std::exp(-Complex(0,1)*qe*step);
        for(Index bin=0;bin<bins[e];++bin) {
            const double x=(bin*per_bin[e]+.5)*step;
            Complex left=a*std::exp(Complex(0,1)*qe*x);
            Complex right=b*std::exp(Complex(0,1)*qe*(lengths[e]-x));
            for(Index j=0;j<per_bin[e];++j) {
                const double value=std::norm(left+right);
                if(value>0 && value<smallest) {smallest=value;min_edge=e;min_x=x+j*step;}
                largest=std::max(largest,value);zeros+=(value==0);++samples;
                left*=forward;right*=backward;
            }
        }
    }
    result[0]=smallest;result[1]=largest;result[2]=zeros;result[3]=samples;
    result[4]=min_edge;result[5]=min_x;
}

template<bool logarithmic>
static void raster_impl(Index edges,const double* segments,const double* waves,
    const double* q,const double* lengths,const Index* bins,const Index* per_bin,
    const Index* order,const double* low,double pixel_um,Index pixels,
    double minimum,double peak,const double* lut,Index lut_size,float* canvas,double* result,
    double stroke_width=1.7) {
    const Index total=pixels*pixels;
    std::fill(canvas,canvas+3*total,1.f);
    std::vector<float> accum(4*total,0);
    const Index across=stroke_width==1.7?1:Index(std::ceil(stroke_width));
    const double transverse_weight=stroke_width/(1.7*across);
    std::vector<Index> touched(4*across*(*std::max_element(bins,bins+edges)));
    const double loglo=logarithmic?std::log(minimum):0;
    const double logspan=logarithmic?std::log(peak)-loglo:1;
    Index under=0,over=0;
    for(Index ordinal=0;ordinal<edges;++ordinal) {
        const Index e=order[ordinal];Index touched_count=0;
        const double normal_x=-(segments[4*e+3]-segments[4*e+1])/lengths[e];
        const double normal_y=(segments[4*e+2]-segments[4*e])/lengths[e];
        const Complex qe=read_complex(q,e),a=read_complex(waves,2*e),b=read_complex(waves,2*e+1);
        const double step=lengths[e]/bins[e]/per_bin[e],coverage=lengths[e]/bins[e]/pixel_um;
        const Complex forward=std::exp(Complex(0,1)*qe*step),backward=std::exp(-Complex(0,1)*qe*step);
        for(Index bin=0;bin<bins[e];++bin) {
            const double x=(bin*per_bin[e]+.5)*step;
            Complex left=a*std::exp(Complex(0,1)*qe*x);
            Complex right=b*std::exp(Complex(0,1)*qe*(lengths[e]-x));
            double rgb[3]={0,0,0};
            for(Index j=0;j<per_bin[e];++j) {
                const double value=std::norm(left+right);
                under+=(value<minimum);over+=(value>peak);
                // Clamp the display, not the saved field. The same finite
                // interval defines the Python colour bar and this LUT mapping.
                const double mapped=std::clamp(logarithmic?(std::log(value)-loglo)/logspan:
                    (value-minimum)/(peak-minimum),0.,1.);
                const double z=mapped*(lut_size-1);
                const Index idx=std::min(lut_size-2,std::max(Index(0),Index(z)));
                const double fraction=z-idx;
                for(int c=0;c<3;++c) rgb[c]+=lut[3*idx+c]*(1-fraction)+lut[3*(idx+1)+c]*fraction;
                left*=forward;right*=backward;
            }
            for(int c=0;c<3;++c) rgb[c]/=per_bin[e];
            const double t=(bin+.5)/bins[e];
            const double px=(segments[4*e]+t*(segments[4*e+2]-segments[4*e])-low[0])/pixel_um-.5;
            const double py=(segments[4*e+1]+t*(segments[4*e+3]-segments[4*e+1])-low[1])/pixel_um-.5;
            // Widen perpendicular to the edge, retaining each bin's field colour.
            for(Index strip=0;strip<across;++strip) {
            const double offset=stroke_width*((strip+.5)/across-.5);
            const double sx=px+offset*normal_x,sy=py+offset*normal_y;
            const Index ix=std::floor(sx),iy=std::floor(sy);
            const double fx=sx-ix,fy=sy-iy;
            for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) {
                const Index xx=ix+dx,yy=iy+dy;
                if(xx<0 || xx>=pixels || yy<0 || yy>=pixels) continue;
                const double weight=(dx?fx:1-fx)*(dy?fy:1-fy)*coverage*transverse_weight;
                if(weight<=0) continue;
                const Index key=yy*pixels+xx;
                if(accum[4*key]==0) touched[touched_count++]=key;
                accum[4*key]+=weight;
                for(int c=0;c<3;++c) accum[4*key+c+1]+=weight*rgb[c];
            }
            }
        }
        for(Index j=0;j<touched_count;++j) {
            const Index key=touched[j];
            const double weight=accum[4*key],alpha=std::min(1.,1.7*weight);
            for(int c=0;c<3;++c) {
                canvas[3*key+c]=canvas[3*key+c]*(1-alpha)+alpha*accum[4*key+c+1]/weight;
                accum[4*key+c+1]=0;
            }
            accum[4*key]=0;
        }
    }
    result[0]=under;result[1]=over;
}

#define RASTER_ARGUMENTS Index edges,const double* segments,const double* waves, \
    const double* q,const double* lengths,const Index* bins,const Index* per_bin, \
    const Index* order,const double* low,double pixel_um,Index pixels, \
    double minimum,double peak,const double* lut,Index lut_size,float* canvas,double* result
#define RASTER_VALUES edges,segments,waves,q,lengths,bins,per_bin,order,low,pixel_um, \
    pixels,minimum,peak,lut,lut_size,canvas,result
extern "C" void raster_field(RASTER_ARGUMENTS) { raster_impl<true>(RASTER_VALUES); }
extern "C" void raster_field_linear(RASTER_ARGUMENTS) { raster_impl<false>(RASTER_VALUES); }
extern "C" void raster_field_linear_width(RASTER_ARGUMENTS,double stroke_width) {
    raster_impl<false>(RASTER_VALUES,stroke_width);
}
extern "C" void raster_field_log_width(RASTER_ARGUMENTS,double stroke_width) {
    raster_impl<true>(RASTER_VALUES,stroke_width);
}
