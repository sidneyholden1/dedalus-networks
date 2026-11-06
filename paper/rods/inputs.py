"""Graph, resolution apportionment and analytic steady reference for Figure 1."""
import numpy as np


def geometry():
    from scipy.spatial import Delaunay
    rng=np.random.default_rng(7304)
    xy=np.array([(x,y) for y in range(4) for x in range(4)],dtype=float)
    xy+=rng.uniform(-.28,.28,xy.shape)
    xy[0]=[0,0]
    tris=Delaunay(xy).simplices
    chosen={tuple(sorted((int(t[i]),int(t[(i+1)%3])))) for t in tris for i in range(3)}
    # Remove triangulation edges while preserving connectivity and degree >=2.
    candidates=sorted(chosen)
    score={e:np.linalg.norm(xy[e[1]]-xy[e[0]])*(.8+.4*rng.random()) for e in candidates}
    candidates.sort(key=lambda e:score[e],reverse=True)
    for edge in candidates:
        if len(chosen)==24:break
        trial=chosen-{edge};degree=np.bincount(np.array(sorted(trial)).ravel(),minlength=16)
        if degree.min()<2:continue
        reached={0}
        for _ in range(16):
            reached|={w for v,w in trial if v in reached}|{v for v,w in trial if w in reached}
        if len(reached)==16:chosen=trial
    assert len(chosen)==24
    edges=np.array(sorted(chosen));lengths=np.linalg.norm(xy[edges[:,1]]-xy[edges[:,0]],axis=1)
    xy*=24/lengths.sum();xy-=xy.min(axis=0)
    lengths=np.linalg.norm(xy[edges[:,1]]-xy[edges[:,0]],axis=1)
    assert abs(lengths.sum()-24)<1e-12
    K=rng.permutation(np.linspace(.7,1.7,24));C=rng.permutation(np.linspace(.6,1.4,24));a=rng.permutation(np.linspace(.15,.45,24))
    return xy,edges,lengths,K,C,a



def mode_counts(lengths,N):
    # Minimum four modes; optimal integer apportionment for squared distance
    # to N*length, subject to exactly 24*N edge coefficients.
    target=N*lengths
    counts=np.maximum(4,np.floor(target).astype(int))
    while counts.sum()<24*N:
        counts[np.argmax(target-counts)]+=1
    while counts.sum()>24*N:
        penalty=counts-target;penalty[counts<=4]=-np.inf
        counts[np.argmax(penalty)]-=1
    assert counts.sum()==24*N and counts.min()>=4
    return counts



def steady(edges,L,K,a):
    q=np.sqrt(a/K); A=np.zeros((16,16))
    for e,(v,w) in enumerate(edges):
        diag=K[e]*q[e]/np.tanh(q[e]*L[e]); off=-K[e]*q[e]/np.sinh(q[e]*L[e])
        A[v,v]+=diag;A[w,w]+=diag;A[v,w]+=off;A[w,v]+=off
    U=np.ones(16);U[1:]=np.linalg.solve(A[1:,1:],-A[1:,0])
    assert np.max(np.abs((A@U)[1:]))<1e-13
    return U,q



def profile(s,edges,L,U,q):
    z=q[:,None]*L[:,None]
    return (U[edges[:,0],None]*np.sinh(z*(1-s))+
            U[edges[:,1],None]*np.sinh(z*s))/np.sinh(z)

