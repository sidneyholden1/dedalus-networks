"""Exact zero-length contraction, preserving every original pore/grain ID.

The paper does not specify the fully exhausted, zero-length limit. This is an
explicit reconstruction: contract exactly zero lengths in pressure, then route
flow inside each contracted component using the limit l_e=epsilon*Lref_e.
No positive hydraulic length or conductance floor is inserted in the model.
"""
import heapq

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def two_sum(a,b):
    """Error-free sum of two finite binary64 arrays (without overflow)."""
    value=a+b;part=value-a
    error=(a-(value-part))+(b-part)
    return value,error


def reservoir_constraints(endpoints,n,inlet,outlet):
    """Exact prescribed pressure on components touching one reservoir value.

    Remove reservoir vertices before finding interior components. If every
    boundary of one such component has the same prescribed pressure, the
    unique positive-conductance Kirchhoff solution is constant there. Enforce
    that identity before pressure differencing and defect correction; tiny
    genuine flows in components touching both reservoir values are untouched.
    ``endpoints`` must contain only the active positive-length pressure graph
    (after any exact zero-length contraction). No flow magnitude is consulted.
    """
    endpoints=np.asarray(endpoints,np.int64)
    inlet,outlet=np.asarray(inlet,np.int64),np.asarray(outlet,np.int64)
    prescribed=np.zeros(n,np.uint8);prescribed[inlet]=1;prescribed[outlet]=2
    fixed=prescribed!=0
    interior=endpoints[~fixed[endpoints].any(axis=1)]
    u,v=interior.T
    graph=sparse.csr_matrix((np.ones(2*len(u)),(np.r_[u,v],np.r_[v,u])),shape=(n,n))
    _,components=connected_components(graph)
    touches=np.zeros(n,np.uint8)
    for source,target in ((0,1),(1,0)):
        edge=endpoints[fixed[endpoints[:,source]]&~fixed[endpoints[:,target]]]
        np.bitwise_or.at(touches,components[edge[:,target]],prescribed[edge[:,source]])
    incident=touches[components]
    constant=(~fixed)&((incident==1)|(incident==2))
    values=(prescribed==1).astype(float)
    values[constant]=(incident[constant]==1)
    return fixed|constant,values


def refine_pressure(endpoints,conductance,pressure,free,inlet,solve_correction,*,iterations=12):
    """Compensated pressure defect correction, using an unchanged factorization.

    Near closure, small pressure differences sit on a large common pressure.
    Store that pressure as a binary64 high/low pair, form physical edge fluxes
    from compensated differences, and solve the actual Kirchhoff defect. The
    callback is either a direct reference factor or the ordinary Network LBVP.
    No conductance, connectivity, or conservation target is modified.
    """
    u,v=np.asarray(endpoints).T;n=len(pressure)
    high=np.asarray(pressure,float).copy();low=np.zeros(n)
    weight=np.asarray(conductance,np.longdouble)
    free=np.asarray(free,np.int64);initial=None
    for iteration in range(iterations+1):
        difference,error=two_sum(high[u],-high[v])
        # Retain the low pressure part when forming the physical residual.
        # Rounding every edge product to binary64 first can erase a tiny cross
        # flow while the global water tolerance already passes.
        flow=weight*(np.asarray(difference,np.longdouble)+np.asarray(error,np.longdouble)+
            np.asarray(low[u],np.longdouble)-np.asarray(low[v],np.longdouble))
        net=np.zeros(n,np.longdouble)
        np.add.at(net,u,flow);np.add.at(net,v,-flow)
        q0=float(np.sum(net[inlet],dtype=np.longdouble))
        if not np.isfinite(q0):raise ArithmeticError('Connected pressure solve has nonfinite throughput')
        # The first binary64 solve can round an inlet pressure drop to zero.
        # An unresolved interior defect still supplies a scale for correction.
        scale=max(abs(q0),float(np.max(np.abs(net),initial=0)),np.finfo(float).tiny)
        defect=float(np.max(np.abs(net[free]),initial=0))/scale
        if initial is None:initial=defect
        # Two corrections also resolve near-balanced pores whose absolute
        # flux is small but whose incoming concentration is not. This changes
        # neither a flow cutoff nor the stated conservation tolerance.
        if (defect<=1e-12 and iteration>=2) or iteration==iterations:break
        correction=np.asarray(solve_correction(-np.asarray(net,float)),float)
        if correction.shape!=high.shape or not np.all(np.isfinite(correction)):
            raise ArithmeticError('Invalid pressure defect correction')
        update,error=two_sum(high,correction)
        high,low=two_sum(update,low+error)
    if q0<=0 or defect>1e-10:
        raise ArithmeticError(f'Pressure refinement failed water conservation: {defect:.6g} after {iteration} corrections')
    return dict(high=high,low=low,flow=np.asarray(flow,float),net=np.asarray(net,float),throughput=q0,
                iterations=iteration,before=initial,after=defect)


def flow_order(endpoints,flow,n):
    """Deterministic directed topological order, including equal-pressure pores."""
    live=np.flatnonzero(np.asarray(flow)!=0)
    pair=np.asarray(endpoints)[live];q=np.asarray(flow)[live]
    upstream=np.where(q>0,pair[:,0],pair[:,1]);downstream=np.where(q>0,pair[:,1],pair[:,0])
    sort=np.argsort(upstream,kind='stable')
    ptr=np.r_[0,np.cumsum(np.bincount(upstream,minlength=n))]
    indegree=np.bincount(downstream,minlength=n)
    ready=np.flatnonzero(indegree==0).tolist();heapq.heapify(ready);order=[]
    while ready:
        node=heapq.heappop(ready);order.append(node)
        for edge in sort[ptr[node]:ptr[node+1]]:
            target=downstream[edge];indegree[target]-=1
            if indegree[target]==0:heapq.heappush(ready,int(target))
    if len(order)!=n:raise ArithmeticError('Nonzero hydraulic flows contain a directed cycle')
    return np.asarray(order,np.int64)


def source_free_branches(endpoints,n,terminals):
    """Identify exact equal potentials in pendant source-free subgraphs.

    Terminals carry nonzero external flux or a prescribed routing potential.
    A component attached through one articulation and containing no terminal
    has constant potential, independent of its positive conductances. Return
    the attachment vertex for every vertex in such a component. An iterative
    low-link traversal includes pendant cycles and parallel edges, without a
    flow threshold, recursion-depth limit, or dense graph representation.
    """
    endpoints=np.asarray(endpoints,np.int64);terminals=np.asarray(terminals,bool)
    if endpoints.ndim!=2 or endpoints.shape[1]!=2 or terminals.shape!=(n,):
        raise ValueError('Invalid routing graph or terminal mask')
    if np.any(endpoints<0) or np.any(endpoints>=n):raise ValueError('Invalid routing vertex')
    u,v=endpoints.T;edge=np.arange(len(u))
    source=np.r_[u,v];target=np.r_[v,u];ids=np.r_[edge,edge]
    sort=np.argsort(source,kind='stable');target=target[sort];ids=ids[sort]
    ptr=np.r_[0,np.cumsum(np.bincount(source,minlength=n))]
    entered=np.full(n,-1,np.int64);low=np.empty(n,np.int64)
    parent=np.full(n,-1,np.int64);parent_edge=np.full(n,-1,np.int64)
    end=np.empty(n,np.int64);count=terminals.astype(np.int64);cursor=ptr[:-1].copy()
    order=[]
    # Root each connected component at a terminal, so every source-free
    # pendant component is a descendant subtree, even after relabeling IDs.
    for root in np.r_[np.flatnonzero(terminals),np.flatnonzero(~terminals)]:
        if entered[root]>=0:continue
        entered[root]=low[root]=len(order);order.append(int(root));stack=[int(root)]
        while stack:
            node=stack[-1]
            if cursor[node]<ptr[node+1]:
                index=cursor[node];cursor[node]+=1
                other=int(target[index]);eid=ids[index]
                if eid==parent_edge[node]:continue
                if entered[other]<0:
                    parent[other]=node;parent_edge[other]=eid
                    entered[other]=low[other]=len(order);order.append(other);stack.append(other)
                else:low[node]=min(low[node],entered[other])
            else:
                stack.pop();end[node]=len(order);previous=parent[node]
                if previous>=0:
                    low[previous]=min(low[previous],low[node]);count[previous]+=count[node]
    representative=np.arange(n);order=np.asarray(order,np.int64);index=0
    while index<n:
        node=order[index];previous=parent[node]
        if previous>=0 and count[node]==0 and low[node]>=entered[previous]:
            representative[order[index:end[node]]]=previous;index=int(end[node])
        else:index+=1
    return representative


def contract_zero_lengths(endpoints,reference_lengths,lengths,diameters,inlet,outlet,total_flow,
                          *,active,positive_solver):
    endpoints=np.asarray(endpoints,np.int64);lengths=np.asarray(lengths,float)
    reference_lengths=np.asarray(reference_lengths,float);diameters=np.asarray(diameters,float)
    n=int(endpoints.max())+1;active=np.asarray(active,bool)
    zero=active&(lengths==0)
    if not np.any(zero):raise ValueError('Contraction called without a zero-length pore')
    if np.any(lengths[active]<0) or np.any(reference_lengths<=0):raise ValueError('Invalid hydraulic metric')
    u,v=endpoints[zero].T
    graph=sparse.csr_matrix((np.ones(2*len(u)),(np.r_[u,v],np.r_[v,u])),shape=(n,n))
    count,labels=connected_components(graph)
    mapped=labels[endpoints];mapped_in=np.unique(labels[inlet]);mapped_out=np.unique(labels[outlet])
    if np.intersect1d(mapped_in,mapped_out).size:
        raise ArithmeticError('Zero-resistance reservoir short requires an earlier breakthrough terminal state')
    finite=active&(lengths>0)&(mapped[:,0]!=mapped[:,1])
    result=positive_solver(mapped,lengths,diameters,mapped_in,mapped_out,total_flow,active=finite)
    result['pressure']=result['pressure'][labels]
    for key in ['unit_pressure_hi','unit_pressure_lo']:
        if key in result:result[key]=result[key][labels]
    result['contraction_labels']=labels
    result['zero_length_routing_potential']=np.zeros(n)
    result['zero_length_pores']=np.flatnonzero(zero)
    if result['clogged']:return result
    flow=result['flow'].copy()
    external=np.bincount(endpoints[:,0],weights=flow,minlength=n)-np.bincount(endpoints[:,1],weights=flow,minlength=n)
    # The common pi/(128 mu epsilon) cancels in the internal routing problem.
    weight=diameters[zero]**4/reference_lengths[zero]
    weight/=weight.max()
    reservoirs=np.zeros(n,bool);reservoirs[np.r_[inlet,outlet]]=True
    fixed=reservoirs.copy()
    for label in np.unique(labels[np.r_[u,v]]):
        nodes=np.flatnonzero(labels==label)
        if not np.any(reservoirs[nodes]):fixed[nodes[0]]=True
    # A numerical gauge is not a physical source: a gauge vertex can itself
    # belong to a dead end. Move its constraint to the exact representative,
    # retaining the original minimum-ID zero gauge in the reported potential.
    representative=source_free_branches(endpoints[zero],n,reservoirs|(external!=0))
    gauge_vertices=representative[np.flatnonzero(fixed)]
    fixed=np.zeros(n,bool);fixed[gauge_vertices]=True
    ru,rv=representative[u],representative[v];live=ru!=rv
    ru,rv,w=ru[live],rv[live],weight[live]
    lap=sparse.csr_matrix((np.r_[w,w,-w,-w],
                          (np.r_[ru,rv,ru,rv],np.r_[ru,rv,rv,ru])),shape=(n,n))
    free=np.flatnonzero((np.asarray(lap.diagonal())>0)&~fixed)
    high=np.zeros(n);low=np.zeros(n)
    factor=splu(lap[free][:,free].tocsc()) if len(free) else None
    if len(free):high[free]=factor.solve(-external[free])
    for iteration in range(13):
        high=high[representative];low=low[representative]
        difference,error=two_sum(high[u],-high[v])
        routed=np.asarray(weight,np.longdouble)*(np.asarray(difference,np.longdouble)+
            np.asarray(error,np.longdouble)+np.asarray(low[u],np.longdouble)-np.asarray(low[v],np.longdouble))
        defect=np.asarray(external,np.longdouble).copy()
        np.add.at(defect,u,routed);np.add.at(defect,v,-routed)
        relative=float(np.max(np.abs(defect[free]),initial=0))/total_flow
        if (relative<=1e-12 and iteration>=2) or iteration==12:break
        correction=np.zeros(n)
        if len(free):correction[free]=factor.solve(-np.asarray(defect[free],float))
        update,error=two_sum(high,correction)
        high,low=two_sum(update,low+error)
    potential=high+low
    flow[zero]=np.asarray(routed,float)
    balance=np.bincount(endpoints[:,0],weights=flow,minlength=n)-np.bincount(endpoints[:,1],weights=flow,minlength=n)
    residual=float(np.max(np.abs(balance[~reservoirs]),initial=0))/total_flow
    if not np.isfinite(residual) or residual>1e-10:raise ArithmeticError('Contracted component flux reconstruction failed')
    flow_order(endpoints,flow,n)
    result.update(flow=flow,conservation_residual=residual,zero_length_routing_potential=potential,
                  contraction_routing_residual=residual)
    return result
