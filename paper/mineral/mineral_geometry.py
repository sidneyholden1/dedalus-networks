"""Pore/grain identities and explicit reconstruction of periodic planar inputs.

The paper does not deposit graph inputs or define all boundary/initial overlap
conventions. These builders declare their reconstruction; source equivalence
is a separate qualification gate, never implied by successful construction.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay


@dataclass
class Geometry:
    points: np.ndarray
    endpoints: np.ndarray
    edge_shift: np.ndarray
    node_lengths: np.ndarray
    triangles: np.ndarray
    triangle_points: np.ndarray
    triangle_edges: np.ndarray
    maximum_grain_volume: np.ndarray
    inlet: np.ndarray
    outlet: np.ndarray
    width: float
    metadata: dict

    def __post_init__(self):
        n,e,g=len(self.points),len(self.endpoints),len(self.triangles)
        if self.endpoints.shape!=(e,2) or self.triangle_edges.shape!=(g,3):raise ValueError('Geometry shape')
        if np.any(self.node_lengths<=0) or np.any(self.maximum_grain_volume<=0):raise ValueError('Degenerate geometry')
        if self.endpoints.max()>=n or self.triangle_edges.max()>=e:raise ValueError('Invalid geometry identity')
        if np.intersect1d(self.inlet,self.outlet).size:raise ValueError('Overlapping boundary sets')
        rows=np.repeat(np.arange(g),3);cols=self.triangle_edges.ravel()
        self.incidence=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(g,e))
        self.edge_grain_count=np.asarray(self.incidence.sum(axis=0)).ravel()
        if np.any((self.edge_grain_count<1)|(self.edge_grain_count>2)):raise ValueError('Nonmanifold pore/grain adjacency')

    def lengths(self,primary,secondary):
        """Equation 20: average adjacent homothetic grain-size ratios."""
        total=np.asarray(primary)+np.asarray(secondary)
        if np.any(total<0):raise ValueError('Negative grain inventory')
        ratio=np.cbrt(total/self.maximum_grain_volume)
        return self.node_lengths*np.asarray(self.incidence.T@ratio).ravel()/self.edge_grain_count

    def fingerprint(self):
        h=hashlib.sha256(json.dumps(self.metadata,sort_keys=True).encode())
        for name,array in self.arrays().items():
            h.update(name.encode());h.update(array.dtype.str.encode());h.update(str(array.shape).encode());h.update(array.tobytes())
        return h.hexdigest()

    def arrays(self):
        return {key:np.asarray(getattr(self,key)) for key in ['points','endpoints','edge_shift','node_lengths','triangles',
            'triangle_points','triangle_edges','maximum_grain_volume','inlet','outlet']}


def periodic_graph(nx,ny,*,seed=20261029,kind='random'):
    if min(nx,ny)<4:raise ValueError('At least four nodes in each direction')
    if kind=='random':
        points=np.random.default_rng(seed).uniform(size=(nx*ny,2))*np.array([nx,ny])
        width=float(nx)
    elif kind=='regular':
        j,i=np.meshgrid(np.arange(ny),np.arange(nx),indexing='ij')
        points=np.column_stack(((i+.5*(j%2)).ravel(),(j*np.sqrt(3)/2).ravel()))
        width=float(nx)
    else:raise ValueError('Unknown graph process')
    n=len(points)
    images=np.concatenate([points+np.array([s*width,0]) for s in [-1,0,1]])
    shifts=np.repeat([-1,0,1],n)
    simplices=Delaunay(images).simplices
    coordinates=images[simplices]
    center=coordinates[:,:,0].mean(axis=1)
    keep=(center>=0)&(center<width)
    simplices=simplices[keep];coordinates=coordinates[keep]
    # Stable IDs independent of Qhull output order.
    ids=simplices%n;image_shift=shifts[simplices]
    # With iid y coordinates, the outermost periodic convex-hull cap can
    # contain two images of the same reservoir point. It spans the full width
    # outside the local porous domain; remove these declared reservoir caps.
    cap=(ids[:,0]==ids[:,1])|(ids[:,1]==ids[:,2])|(ids[:,2]==ids[:,0])
    removed_caps=int(np.count_nonzero(cap))
    ids,coordinates,image_shift=ids[~cap],coordinates[~cap],image_shift[~cap]
    face_keys=[]
    for verts,offsets in zip(ids,image_shift):
        order=np.argsort(verts);v=verts[order];s=offsets[order]
        face_keys.append(tuple(zip(v.tolist(),(s-s[0]).tolist())))
    order=sorted(range(len(face_keys)),key=lambda k:face_keys[k])
    if len(set(face_keys))!=len(face_keys):raise ValueError('Periodic triangle deduplication failed')
    ids,coordinates,image_shift=ids[order],coordinates[order],image_shift[order]
    keys=set();face_edges=[]
    for verts,offsets in zip(ids,image_shift):
        row=[]
        for a,b in [(0,1),(1,2),(2,0)]:
            u,v=int(verts[a]),int(verts[b]);delta=int(offsets[b]-offsets[a])
            if u==v:raise ValueError('Periodic self-pore needs a wider domain')
            if u>v:u,v,delta=v,u,-delta
            key=(u,v,delta);keys.add(key);row.append(key)
        face_edges.append(row)
    edges=sorted(keys);lookup={edge:k for k,edge in enumerate(edges)}
    endpoints=np.array([x[:2] for x in edges],np.int64);edge_shift=np.array([x[2] for x in edges],np.int64)
    vector=points[endpoints[:,1]]-points[endpoints[:,0]]
    vector[:,0]+=edge_shift*width
    lengths=np.linalg.norm(vector,axis=1);l0=float(lengths.mean())
    points/=l0;coordinates/=l0;lengths/=l0;width/=l0
    a=coordinates[:,1]-coordinates[:,0];b=coordinates[:,2]-coordinates[:,0]
    area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
    if kind=='regular':
        inlet=np.flatnonzero(points[:,1]==points[:,1].min());outlet=np.flatnonzero(points[:,1]==points[:,1].max())
        boundary_rule='first and last regular rows'
    else:
        # Explicit reconstruction: one mean-node-distance-thick reservoir band.
        # No seed selection and no movement of random coordinates.
        inlet=np.flatnonzero(points[:,1]<points[:,1].min()+1.)
        outlet=np.flatnonzero(points[:,1]>points[:,1].max()-1.)
        boundary_rule='one normalized mean-node-distance inlet/outlet bands'
    return Geometry(points,endpoints,edge_shift,lengths,ids,coordinates,
        np.array([[lookup[e] for e in row] for row in face_edges],np.int64),area*.5,inlet,outlet,width,
        dict(kind=kind,nx=nx,ny=ny,seed=seed,point_generator='NumPy default_rng PCG64',
             random_process='independent uniform points' if kind=='random' else 'equilateral triangular node lattice',
             boundary_rule=boundary_rule,periodicity='x only; triplicated Delaunay, centroid strip selection',
             removed_periodic_reservoir_cap_faces=removed_caps,
             original_mean_node_distance=l0,thickness=.5,
             source_equivalence='reconstruction, not author graph inputs'))


def initial_inventory(geometry,d0,*,method='cylindrical-subtraction'):
    """Declared direct pore-volume subtraction; invalid overlap fails closed.

    This deliberately exposes thin-grain conflicts rather than clipping a
    negative inventory or silently changing the iid point process. Other
    initializations require their own source/geometry qualification.
    """
    if method in ('equation21-regular','uniform-equation21'):
        if method=='equation21-regular' and (geometry.metadata['kind']!='regular' or not np.allclose(geometry.node_lengths,1,rtol=0,atol=1e-13)):
            raise ValueError('Equation 21 is established only for the regular uniform geometry')
        # Equation 20 gives l/l0=(1-phi)^(1/3). Substitute it and s=pi*d*l
        # in Equation 21 to recover the initial solid fraction without a fit.
        scale2=1-4*np.pi*d0*d0/(3*np.sqrt(3))
        if scale2<=0:raise ValueError('Regular grain exhausted at the initial pore diameter')
        primary=geometry.maximum_grain_volume*scale2**1.5
        return primary,np.zeros_like(primary)
    if method=='local-area-equation21':
        # Sensitivity reconstruction, not a reported author rule. Extend the
        # regular expression using each triangle's area S instead of sqrt(3)/4.
        # A triangle whose modeled pore area exceeds S is initially empty;
        # exact zero-length contraction handles pairs of empty grains.
        area=geometry.maximum_grain_volume/geometry.metadata['thickness']
        scale2=1-np.pi*d0*d0/(3*area)
        primary=geometry.maximum_grain_volume*np.maximum(scale2,0)**1.5
        return primary,np.zeros_like(primary)
    if method!='cylindrical-subtraction':raise ValueError('Unqualified initialization rule')
    pore_volume=np.pi*d0*d0*geometry.node_lengths/4
    occupied=geometry.incidence@(pore_volume/geometry.edge_grain_count)
    primary=geometry.maximum_grain_volume-occupied
    if np.any(primary<=0):
        bad=np.flatnonzero(primary<=0)
        raise ValueError(f'Initial cylindrical pores exhaust/overlap {len(bad)} triangular grains; '
                         f'minimum reconstructed A={primary.min():.9g}; source geometry audit required')
    return primary,np.zeros_like(primary)
