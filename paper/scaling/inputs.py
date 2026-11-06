"""The manuscript graph generators and length-dependent mode counts."""
import os
import numpy as np
import dedalus.public as d3


def base_fixture(family,scale,homogeneous=False,seed=20261010,resolution=0,hubs=8):
    rng=np.random.default_rng(seed);xy=None
    if family=='spatial':
        xy=np.array([(x,y) for y in range(scale) for x in range(scale)],float)
        edges=[(i,i+1) for i in range(scale**2) if i%scale<scale-1]+[(i,i+scale) for i in range(scale*(scale-1))]
    elif family=='irregular':
        n=scale*scale;edges=[(i,int(rng.integers(i))) for i in range(1,n)];used={tuple(sorted(e)) for e in edges}
        while len(edges)<2*n:
            e=tuple(sorted(map(int,rng.integers(n,size=2))))
            if e[0]!=e[1] and e not in used:edges.append(e);used.add(e)
    elif family=='hub':
        edges=[(i,(i+1)%hubs) for i in range(hubs)];v=hubs
        for hub in range(hubs):
            for _ in range(scale):edges.extend([(hub,v),(v,v+1)]);v+=2
    else:raise ValueError(family)
    edges=np.asarray(edges,dtype=np.int64)
    sizes=rng.choice([16,32,64,128],len(edges),p=[.4,.3,.2,.1])
    if homogeneous:sizes[:]=32
    if family=='hub' and not homogeneous:sizes[hubs:hubs+2*scale]=128
    if resolution:sizes[:]=resolution
    lengths=.7+.6*rng.random(len(edges))
    if homogeneous:lengths[:]=1
    return d3.Network(endpoints=edges,lengths=lengths),sizes,xy



def two_cycles(vertices,rng):
    vertices=np.asarray(vertices,dtype=np.int64)
    first=list(zip(vertices,np.roll(vertices,-1)))
    used={tuple(sorted(map(int,e))) for e in first}
    for _ in range(1000):
        perm=rng.permutation(vertices)
        second=list(zip(perm,np.roll(perm,-1)))
        if not any(tuple(sorted(map(int,e))) in used for e in second):
            return first+second
    raise RuntimeError('Could not generate disjoint Hamiltonian cycles')



def general_fixture(family,scale,homogeneous=False,seed=20261010,resolution=0,hubs=8):
    if family in ['spatial','hub']:
        return base_fixture(family,scale,homogeneous,seed,resolution,hubs)
    if scale<4: raise ValueError('General fixtures require scale>=4')
    rng=np.random.default_rng(seed)
    n=scale*scale
    xy=None
    if family=='mesh':
        xy=np.array([(x,y) for y in range(scale) for x in range(scale)],float)
        edges=[(y*scale+x,y*scale+(x+1)%scale) for y in range(scale) for x in range(scale)]
        edges += [(y*scale+x,((y+1)%scale)*scale+x) for y in range(scale) for x in range(scale)]
    elif family=='irregular':
        edges=two_cycles(np.arange(n),rng)
    elif family=='community':
        groups=np.array_split(np.arange(n),max(2,int(np.ceil(n/128))))
        pieces=[two_cycles(g,rng) for g in groups]
        # Swap one edge across each adjacent pair of communities. Every degree
        # stays four, the graph is connected, and each community retains a cycle.
        for k in range(len(groups)-1):
            u,v=pieces[k][1]
            x,y=pieces[k+1][0]
            pieces[k][1]=(u,x)
            pieces[k+1][0]=(v,y)
        edges=[e for group in pieces for e in group]
        permutation=rng.permutation(n)
        edges=[(permutation[u],permutation[v]) for u,v in edges]
    else: raise ValueError(family)
    edges=np.asarray(edges,dtype=np.int64)
    sizes=rng.choice([16,32,64,128],len(edges),p=[.4,.3,.2,.1])
    lengths=.7+.6*rng.random(len(edges))
    if homogeneous: sizes[:]=32;lengths[:]=1
    if resolution: sizes[:]=resolution
    net=d3.Network(endpoints=edges,lengths=lengths)
    if not np.all(net.degree==4): raise AssertionError('Fixture must have no degree-two vertices')
    return net,sizes,xy



def cubic(vertices,rng):
    vertices=np.asarray(vertices,dtype=np.int64)
    edges=list(zip(vertices,np.roll(vertices,-1)))
    used={tuple(sorted(map(int,e))) for e in edges}
    for _ in range(1000):
        pairs=rng.permutation(vertices).reshape(-1,2)
        if not any(tuple(sorted(map(int,e))) in used for e in pairs):
            return edges+[tuple(e) for e in pairs]
    raise RuntimeError('Failed to generate cubic component')



def topology(family,scale,homogeneous=False,seed=20261010,resolution=0,hubs=8):
    if family not in ['triangular','junctions']:
        return general_fixture(family,scale,homogeneous,seed,resolution,hubs)
    rng=np.random.default_rng(seed);n=scale*scale;xy=None
    if family=='junctions':n += n%2
    if family=='triangular':
        xy=np.array([(x,y) for y in range(scale) for x in range(scale)],float)
        edges=[(y*scale+x,y*scale+(x+1)%scale) for y in range(scale) for x in range(scale)]
        edges += [(y*scale+x,((y+1)%scale)*scale+x) for y in range(scale) for x in range(scale)]
        for y in range(scale):
            for x in range(scale):
                if rng.integers(2):edges.append((y*scale+x,((y+1)%scale)*scale+(x+1)%scale))
                else:edges.append((y*scale+(x+1)%scale,((y+1)%scale)*scale+x))
    else:
        if n%2 or n<8:raise ValueError('Junction fixture requires even scale>=4')
        groups=[np.arange(i,min(i+128,n)) for i in range(0,n,128)]
        if len(groups)>1 and len(groups[-1])<8:
            groups[-2]=np.r_[groups[-2],groups[-1]];groups.pop()
        pieces=[]
        for vertices in groups:
            piece=cubic(vertices,rng);hub=int(vertices[0])
            neighbors={v if u==hub else u for u,v in piece if u==hub or v==hub}
            available=[int(v) for v in vertices if v!=hub and v not in neighbors]
            chosen=rng.choice(available,size=len(vertices)//2,replace=False)
            piece.extend((hub,int(v)) for v in chosen);pieces.append(piece)
        for k in range(len(groups)-1):
            u,v=pieces[k][1];x,y=pieces[k+1][0]
            pieces[k][1]=(u,x);pieces[k+1][0]=(v,y)
        permutation=rng.permutation(n)
        edges=[(permutation[u],permutation[v]) for piece in pieces for u,v in piece]
    edges=np.asarray(edges,dtype=np.int64)
    sizes=rng.choice([16,32,64,128],len(edges),p=[.4,.3,.2,.1]);lengths=.7+.6*rng.random(len(edges))
    if homogeneous:sizes[:]=32;lengths[:]=1
    if resolution:sizes[:]=resolution
    net=d3.Network(endpoints=edges,lengths=lengths)
    if np.any(net.degree<3):raise AssertionError('Every vertex must branch')
    if len({tuple(sorted(e)) for e in edges})!=len(edges):raise AssertionError('Unexpected duplicate edge')
    return net,sizes,xy



def fixture(family,scale,homogeneous=False,seed=20261026,resolution=0,hubs=8):
 net,old,xy=topology(family,scale,homogeneous,seed,0,hubs)
 if homogeneous:lengths=np.ones(net.num_edges);modes=np.full(net.num_edges,32,dtype=np.int64)
 else:
  lengths=np.random.default_rng(seed+1).uniform(.5,2.,net.num_edges)
  modes=(8*np.floor(4*lengths+.5)).astype(np.int64)
 modes*=int(os.environ.get('PAPER_MODE_FACTOR','1'))
 if resolution:modes[:]=resolution
 return d3.Network(endpoints=net.endpoints,lengths=lengths),modes,xy

