"""Retained native mineral transport alternatives for verification only.

Extracted before simplifying the paper example to exact uniform-pore transport.
The direct concentration and optional custom mixing formulations stay available
here for comparisons. Paper entry scripts do not import this module.
"""
import time
import numpy as np
from mpi4py import MPI
from scipy import sparse
import dedalus.public as d3
from mineral_dedalus import (layout, put, get, fundamental_matrix, build,
                             algebraic_residual)


def transport(endpoints, reference_lengths, lengths, diameters, flow, inlet,
              chemistry, *, dissolving=None, modes=16, comm=MPI.COMM_WORLD,
              owners=None, network_solver='dtn',fitted=False,concentration_map=None):
    """First-order two-species LBVP with incoming-port and vertex-flux laws.

    Flow direction changes the incoming port map. Concentration continuity is
    imposed only at upstream ports; each downstream port keeps its edge value.
    A vertex concentration is the incoming solute flux divided by water flux.
    An optional conservative edge-to-edge concentration map instead prescribes
    distinct outgoing ports; the vertex fields then remain mean diagnostics.
    """
    start=time.perf_counter()
    endpoints=np.asarray(endpoints,np.int64)
    reference_lengths,lengths,diameters,flow=map(np.asarray,(reference_lengths,lengths,diameters,flow))
    if concentration_map is not None:
        from mineral_mixing import validate_network_map
        concentration_map=validate_network_map(concentration_map,endpoints,flow,inlet)
    live=np.flatnonzero(flow!=0)
    if not len(live):
        raise ValueError('Transport requires a transmitting network')
    vertices=np.unique(endpoints[live]);n=int(endpoints.max())+1
    mapping=np.full(n,-1,dtype=np.int64);mapping[vertices]=np.arange(len(vertices))
    network=d3.Network(vertex_ids=tuple(map(int,vertices)),edge_ids=tuple(map(int,live)),
                       endpoints=mapping[endpoints[live]],lengths=reference_lengths[live])
    dist=layout(network,comm=comm,owners=None if owners is None else np.asarray(owners)[live])
    basis=d3.NetworkBasis(network,size=modes)
    owned=dist.network_partition.edges[comm.rank]
    B=dist.Field(name='B',bases=basis);C=dist.Field(name='C',bases=basis)
    VB=dist.Field(name='VB',bases=basis.vertices);VC=dist.Field(name='VC',bases=basis.vertices)
    E=dist.VectorField(network.tangent,name='E',bases=basis.edges)
    E['g'][0]=1
    Q=dist.Field(name='Q',bases=basis.edges)
    A1=dist.Field(name='A1',bases=basis.edges);A2=dist.Field(name='A2',bases=basis.edges)
    alpha=dist.Field(name='alpha',bases=basis.vertices)
    beta=dist.Field(name='beta',bases=basis.vertices)
    incoming=dist.Field(name='incoming',bases=basis.ports)
    prescribed=dist.Field(name='prescribed',bases=basis.vertices)
    k1,k2=chemistry.rates(diameters[live])
    if dissolving is not None:
        access=np.asarray(dissolving,float)[live]
        if np.any(~np.isfinite(access)) or np.any((access<0)|(access>1)):
            raise ValueError('Primary reactive fraction must lie in [0,1]')
        k1=k1*access
    r1=np.pi*diameters[live]*lengths[live]/reference_lengths[live]*k1
    r2=np.pi*diameters[live]*lengths[live]/reference_lengths[live]*k2
    da1=r1*reference_lengths[live]/np.abs(flow[live])
    da2=r2*reference_lengths[live]/np.abs(flow[live])
    if not np.all(np.isfinite(da1+da2)):
        raise ArithmeticError('Unrepresentable local Damkohler number; no rate truncation is allowed')
    norm=np.maximum(np.abs(flow[live])/reference_lengths[live],np.maximum(r1,r2))
    Q['g']=(flow[live]/norm)[owned,None]
    A1['g']=(r1/norm)[owned,None];A2['g']=(r2/norm)[owned,None]
    positive=flow[live]>0
    source=np.where(positive,network.endpoints[:,0],network.endpoints[:,1])
    target=np.where(positive,network.endpoints[:,1],network.endpoints[:,0])
    # One incoming trace per physical pore. Target rows are edge rows, not
    # vertex continuity equations for the two distinct physical endpoints.
    e=np.arange(len(live))
    upstream=(~positive).astype(int)
    inc=np.zeros(2*len(live));inc[2*e+positive]=np.abs(flow[live])
    # Scale the vertex balance to a weighted average for numerical conditioning.
    water=np.bincount(target,weights=np.abs(flow[live]),minlength=len(vertices))
    port_vertex=network.endpoints.ravel()
    inc=np.divide(inc,water[port_vertex],out=np.zeros_like(inc),where=water[port_vertex]>0)
    put(incoming,inc)
    if fitted:
        eb,ec,h,_=fundamental_matrix(da1,da2)
        outlet_port=2*e+positive
        port_b=np.ones(2*len(live));port_b[outlet_port]=eb
        port_c=np.ones(2*len(live));port_c[outlet_port]=ec
        port_h=np.zeros(2*len(live));port_h[outlet_port]=h
        TB=dist.Field(name='TB',bases=basis.ports);TC=dist.Field(name='TC',bases=basis.ports)
        TH=dist.Field(name='TH',bases=basis.ports)
        put(TB,port_b);put(TC,port_c);put(TH,port_h)
    boundary=np.isin(vertices,inlet)|(water==0)
    put(alpha,boundary);put(beta,~boundary);put(prescribed,np.isin(vertices,inlet))
    mixing_unknowns=[];mixing_equations=[]
    if concentration_map is not None:
        port_map=concentration_map[live][:,live].tocoo()
        outlet_ports=2*port_map.col+positive[port_map.col]
        # Separate outgoing concentrations are vertex unknowns, one slot per
        # outgoing physical port. Port -> vertex -> edge maps use the existing
        # endpoint halo and Schur coupling across MPI owners. A direct implicit
        # port -> edge map cannot cross owners. Slots do not identify distinct
        # outgoing concentrations or impose full mixing at a junction.
        order=np.argsort(source,kind='stable')
        counts=np.bincount(source,minlength=len(vertices))
        slots=np.empty(len(live),dtype=np.int64)
        slots[order]=np.arange(len(live))-np.repeat(np.cumsum(counts)-counts,counts)
        btrace=TB*d3.endpoint(B) if fitted else d3.endpoint(B)
        ctrace=TC*d3.endpoint(C)+TH*d3.endpoint(B) if fitted else d3.endpoint(C)
        outgoing_b=[];outgoing_c=[]
        for slot in range(int(counts.max())):
            edges=np.flatnonzero(slots==slot)
            entries=slots[port_map.row]==slot
            Mix=d3.CardinalMap(basis.ports,basis.vertices,
                sparse.csr_matrix((port_map.data[entries],(source[port_map.row[entries]],outlet_ports[entries])),
                                  shape=(len(vertices),2*len(live))),name=f'MixSlot{slot}')
            UpSlot=d3.CardinalMap(basis.vertices,basis.edges,
                sparse.csr_matrix((np.ones(len(edges)),(edges,source[edges])),
                                  shape=(len(live),len(vertices))),name=f'UpSlot{slot}')
            SB=dist.Field(name=f'portB{slot}',bases=basis.vertices)
            SC=dist.Field(name=f'portC{slot}',bases=basis.vertices)
            injected=dist.Field(name=f'injected{slot}',bases=basis.vertices)
            injected_c=dist.Field(name=f'injectedC{slot}',bases=basis.vertices)
            injection=np.zeros(len(vertices))
            injection[source[edges]]=np.isin(vertices[source[edges]],inlet)
            put(injected,injection)
            put(injected_c,np.zeros(len(vertices)))
            mixing_unknowns.extend([SB,SC])
            # Named zero RHS fields keep distributed evaluator identities
            # unique; literal zero creates anonymous fields per equation.
            mixing_equations.extend([(SB-Mix@btrace,injected),(SC-Mix@ctrace,injected_c)])
            outgoing_b.append(UpSlot@SB);outgoing_c.append(UpSlot@SC)
        mixed_b=sum(outgoing_b);mixed_c=sum(outgoing_c)
    problem=d3.LBVP([B,C,VB,VC]+mixing_unknowns,namespace=locals())
    if fitted:
        # c=F(s)w makes dw/dx=0. Two modes suffice for the constant
        # amplitudes; sharp physical concentration layers are reconstructed
        # from F, with exact integrated reaction. This is not axial resolution
        # or extra physical pores, and applies only to uniform coefficients.
        problem.add_equation('E@grad(B)=0')
        problem.add_equation('E@grad(C)=0')
    else:
        problem.add_equation('Q*(E@grad(B))+A1*B=0')
        problem.add_equation('Q*(E@grad(C))-A1*B+A2*C=0')
    if concentration_map is None:
        problem.add_equation('endpoint(B,side=upstream)-endpoint(VB,side=upstream)=0')
        problem.add_equation('endpoint(C,side=upstream)-endpoint(VC,side=upstream)=0')
    else:
        problem.add_equation('endpoint(B,side=upstream)-mixed_b=0')
        problem.add_equation('endpoint(C,side=upstream)-mixed_c=0')
    if fitted:
        problem.add_equation('alpha*VB+beta*(VB-endpoint(incoming*TB*endpoint(B),sum=True))=prescribed')
        problem.add_equation('alpha*VC+beta*(VC-endpoint(incoming*(TC*endpoint(C)+TH*endpoint(B)),sum=True))=0')
    else:
        problem.add_equation('alpha*VB+beta*(VB-endpoint(incoming*endpoint(B),sum=True))=prescribed')
        problem.add_equation('alpha*VC+beta*(VC-endpoint(incoming*endpoint(C),sum=True))=0')
    for equation in mixing_equations:problem.add_equation(equation)
    setup=time.perf_counter()
    solver=build(problem,network_solver)
    solver.solve()
    residual=algebraic_residual(solver)
    bg,cg=get(B),get(C)
    bports=get(d3.endpoint(B).evaluate()).ravel().reshape(-1,2)
    cports=get(d3.endpoint(C).evaluate()).ravel().reshape(-1,2)
    bi=bports[e,(~positive).astype(int)];ci=cports[e,(~positive).astype(int)]
    if fitted:
        eb,ec,h,pr= fundamental_matrix(da1,da2)
        bo=eb*bports[e,positive.astype(int)]
        co=ec*cports[e,positive.astype(int)]+h*bports[e,positive.astype(int)]
        dissolved=np.abs(flow[live])*bi*(-np.expm1(-da1))
        precipitated=np.abs(flow[live])*(ci*(-np.expm1(-da2))+bi*pr)
        s=dist.local_grid(basis.interval).ravel()[None,:]
        x=np.where(positive[:,None],s,1-s)
        eb,ec,h,_=fundamental_matrix(da1[:,None]*x,da2[:,None]*x)
        bg,cg=eb*bg,ec*cg+h*bg
    else:
        bo=bports[e,positive.astype(int)];co=cports[e,positive.astype(int)]
        integrate=lambda f:get(d3.Integrate(f,network.position_coord).evaluate()).ravel()
        dissolved=np.pi*diameters[live]*lengths[live]*k1*integrate(B)
        precipitated=np.pi*diameters[live]*lengths[live]*k2*integrate(C)
    bdefect=np.abs(np.abs(flow[live])*(bi-bo)-dissolved)
    cdefect=np.abs(np.abs(flow[live])*(ci-co)+dissolved-precipitated)
    size=len(flow)
    arrays={name:np.zeros(size) for name in ['inlet_b','inlet_c','outlet_b','outlet_c','dissolved_rate','precipitated_rate']}
    for name,values in zip(arrays,[bi,ci,bo,co,dissolved,precipitated]):arrays[name][live]=values
    # Keep original vertex indexing even after closure creates isolated nodes.
    # Interior zero-flow components have the explicit inert concentration
    # gauge zero; reservoir concentrations retain their prescribed values.
    node_b=np.zeros(n);node_c=np.zeros(n);node_b[inlet]=1.
    node_b[vertices]=get(VB).ravel();node_c[vertices]=get(VC).ravel()
    result=dict(**arrays,live=live,b=bg,c=cg,s=dist.local_grid(basis.interval).ravel().copy(),
                node_ids=vertices,node_b=node_b,node_c=node_c,
                integrated_balance_defect=float(max(np.sum(bdefect),np.sum(cdefect))),
                algebraic_residual=residual,
                representation='exact uniform-pore integrating factor' if fitted else 'physical concentration Chebyshev expansion',
                vertex_concentration_role='fully mixed junction concentration' if concentration_map is None else 'incoming-flux mean diagnostic',
                minimum_concentration=float(min(np.min(bg),np.min(cg),np.min(node_b),np.min(node_c),np.min(bi),np.min(ci))),
                seconds=time.perf_counter()-start,setup_seconds=setup-start)
    from mineral_lifecycle import retire
    retire(dist)
    return result

