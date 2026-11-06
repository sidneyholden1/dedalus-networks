"""Quasi-static mineral-replacement equations expressed in Dedalus Network.

Immutable active-network snapshots may be retained while topology is unchanged;
all pressure matrices/factors update after geometry changes. Retired private
snapshots explicitly release their constructor-cache roots. Uniform-pore
analytical solutions live in mineral_reference and are not used by these solves.
"""
import time
import hashlib

import numpy as np
from mpi4py import MPI
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import dedalus.public as d3


def put(field, values):
    """Set canonical input for both replicated and owned/ghost vertex storage."""
    field.change_layout('g')
    field.set_global_data(np.asarray(values, float).reshape(-1, 1))


def get(field):
    field.change_layout('g')
    return np.asarray(field.allgather_data('g')).copy()


def layout(network, *, comm, owners=None, storage='owned_ghost'):
    if owners is None:
        owners = np.arange(network.num_edges, dtype=np.int64)*comm.size//network.num_edges
    return d3.Distributor(network, dtype=np.float64, comm=comm,
                          network_partition=owners, network_vertex_storage=storage)


def coefficient_vectors(solver):
    for field in solver.state+solver.F:field.change_layout('c')
    return ([sp.gather_inputs(solver.state).copy() for sp in solver.subproblems],
            [sp.gather_outputs(solver.F).copy() for sp in solver.subproblems])


def algebraic_residual(solver,state=None,rhs=None,low=None):
    """Residual of the specified assembled solution against its original RHS.

    High/low coefficient vectors allow pressure defect corrections to be
    audited as one solution. Residuals of individually rescaled auxiliary
    solves are retained as diagnostics, not mistaken for this final residual.
    """
    if state is None:state,rhs=coefficient_vectors(solver)
    error=scale=0.
    for index,subproblem in enumerate(solver.subproblems):
        lhs=solver.apply_subproblem_matrix(subproblem.L_min,state[index],subproblem=subproblem)
        if low is not None:
            lhs=lhs+solver.apply_subproblem_matrix(subproblem.L_min,low[index],subproblem=subproblem)
        error=max(error,float(np.max(np.abs(lhs-rhs[index]),initial=0.)))
        scale=max(scale,float(np.max(np.abs(lhs),initial=0.)),float(np.max(np.abs(rhs[index]),initial=0.)))
    error=solver.dist.comm.allreduce(error,op=MPI.MAX)
    scale=solver.dist.comm.allreduce(scale,op=MPI.MAX)
    return error/max(scale,np.finfo(float).tiny)


def build(problem,network_solver):
    # Retain arbitrarily small active conductances. A sparse-entry cutoff must
    # not become an unreported pore-closure rule near d_min.
    # Small separators can stagnate in restarted GMRES during pressure defect
    # correction. Keep the existing direct method through 1024 unknowns (8 MiB
    # per real dense matrix/rank); larger systems retain the collective method.
    # This changes the linear algebra choice, not the equations or tolerances.
    return problem.build_solver(network_solver=network_solver,entry_cutoff=0.,ncc_cutoff=0.,
                                network_separator_rtol=1e-12,network_separator_atol=1e-14,
                                network_separator_direct_max_size=1024)


def solve_pressure_system(solver,*,rebuild_matrices=False):
    """Correct reconstruction error against the unchanged assembled equations.

    Condensation can leave a small error in a reconstructed edge coefficient
    which a large local matrix entry magnifies. Correct the full matrix defect
    using its existing factorization before extracting physical pressures.
    This is distinct from the later compensated Kirchhoff flux correction.
    """
    solver.solve(rebuild_matrices=rebuild_matrices)
    state,rhs=coefficient_vectors(solver)
    comm=solver.dist.comm
    if not comm.allreduce(all(np.all(np.isfinite(x)) for x in state+rhs),op=MPI.LAND):
        raise ArithmeticError('Nonfinite assembled pressure solution or RHS')
    before=residual=algebraic_residual(solver,state,rhs)
    iterations=0
    while residual>1e-12 and iterations<3:
        for index,sp in enumerate(solver.subproblems):
            lhs=solver.apply_subproblem_matrix(sp.L_min,state[index],subproblem=sp)
            delta=solver.subproblem_matsolvers[sp].solve(rhs[index]-lhs)
            state[index]=state[index]+delta
        if not comm.allreduce(all(np.all(np.isfinite(x)) for x in state),op=MPI.LAND):
            raise ArithmeticError('Nonfinite assembled pressure defect correction')
        for field in solver.state:field.change_layout('c')
        for sp,value in zip(solver.subproblems,state):sp.scatter_inputs(value,solver.state)
        iterations+=1
        residual=algebraic_residual(solver,state,rhs)
    return state,rhs,residual,dict(before=before,after=residual,iterations=iterations)


def fundamental_matrix(a,b):
    """Exact triangular propagator for a *uniform* reactive pore.

    Used as a change of unknown c=F(s)w in the fitted formulation below.
    Network unknown amplitudes w are still solved by the ordinary Dedalus
    LBVP and its port/vertex equations. No independent-reference solve enters.
    """
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    if np.any(a<0) or np.any(b<0) or not np.all(np.isfinite(a+b)):
        raise ValueError('Propagator requires finite nonnegative exponents')
    delta=np.abs(a-b);smaller=np.minimum(a,b)
    exponential_average=np.ones_like(delta)
    np.divide(-np.expm1(-delta),delta,out=exponential_average,where=delta!=0)
    cross=a*np.exp(-smaller)*exponential_average
    precip=-np.expm1(-smaller)-smaller*np.exp(-smaller)*exponential_average
    small=np.maximum(a,b)<.1
    if np.any(small):
        # Independent integral expansion of the two-reaction completion
        # probability. All 18 terms are below roundoff after this threshold.
        aa,bb=a[small],b[small];series=np.zeros_like(aa);factorial=2.
        for degree in range(18):
            if degree:factorial*=degree+2
            series+=(-1.)**degree*sum(aa**j*bb**(degree-j) for j in range(degree+1))/factorial
        precip=np.array(precip,copy=True);precip[small]=aa*bb*series
    return np.exp(-a),np.exp(-b),cross,precip


def hydraulic(endpoints, reference_lengths, lengths, diameters, inlet, outlet,
              total_flow, *, mu=1., active=None, modes=4, comm=MPI.COMM_WORLD,
              owners=None, network_solver='dtn',cache=None):
    active=np.ones(len(endpoints),bool) if active is None else np.asarray(active,bool)
    options=dict(mu=mu,modes=modes,comm=comm,owners=owners,network_solver=network_solver,cache=cache)
    if np.any((np.asarray(lengths)==0)&active):
        from mineral_topology import contract_zero_lengths
        def positive(mapped,ll,dd,ii,oo,qq,**kw):
            return _hydraulic_positive(mapped,reference_lengths,ll,dd,ii,oo,qq,**options,**kw)
        return contract_zero_lengths(endpoints,reference_lengths,lengths,diameters,inlet,outlet,total_flow,
                                     active=active,positive_solver=positive)
    return _hydraulic_positive(endpoints,reference_lengths,lengths,diameters,inlet,outlet,total_flow,active=active,**options)


def _hydraulic_positive(endpoints, reference_lengths, lengths, diameters, inlet, outlet,
              total_flow, *, mu=1., active=None, modes=4, comm=MPI.COMM_WORLD,
              owners=None, network_solver='dtn',cache=None):
    """Solve pressure PDE + flux conservation with explicit evolving metric.

    k_ref = pi*d**4*L_ref/(128*mu*L_current) gives the physical conductance
    after differentiation on the reference network. Inactive pores are absent.
    """
    start=time.perf_counter()
    endpoints=np.asarray(endpoints,np.int64)
    reference_lengths,lengths,diameters=map(np.asarray,(reference_lengths,lengths,diameters))
    n=int(endpoints.max())+1
    active=np.ones(len(endpoints),bool) if active is None else np.asarray(active,bool)
    inlet,outlet=map(lambda a:np.asarray(a,np.int64),(inlet,outlet))
    u,v=endpoints[active].T
    adjacency=sparse.coo_matrix((np.ones(2*len(u)),(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    _,labels=connected_components(adjacency)
    transmitting=np.intersect1d(labels[inlet],labels[outlet])
    pressure=np.zeros(n);flow=np.zeros(len(endpoints))
    if not transmitting.size:
        if cache is not None:
            from mineral_lifecycle import clear_pressure
            clear_pressure(cache)
        return dict(clogged=True,pressure=pressure,flow=flow,inlet_pressure=None,
                    seconds=time.perf_counter()-start)
    live=np.flatnonzero(active & np.isin(labels[endpoints[:,0]],transmitting))
    vertices=np.unique(endpoints[live]);mapping=np.full(n,-1,dtype=np.int64)
    mapping[vertices]=np.arange(len(vertices))
    conductance=np.pi*diameters[live]**4/(128*mu*lengths[live])
    # A common scalar row scale preserves pressure and avoids tiny PDE entries.
    row_scale=float(np.max(conductance*reference_lengths[live]))
    from mineral_topology import reservoir_constraints
    constrained,prescribed_values=reservoir_constraints(endpoints[live],n,inlet,outlet)
    boundary_mask=constrained[vertices]
    boundary_values=prescribed_values[vertices]
    physical_free=~np.isin(vertices,np.r_[inlet,outlet])
    key=None;reused=False
    if cache is not None:
        signature=hashlib.sha256()
        for array in [endpoints[live],reference_lengths[live],live,vertices,inlet,outlet,
                      np.asarray([] if owners is None else owners,dtype=np.int64)]:
            signature.update(np.ascontiguousarray(array).tobytes())
            signature.update(str(array.shape).encode())
        key=(signature.hexdigest(),modes,network_solver,comm.py2f())
        reused=cache.get('key')==key
    if reused:
        network,owned,P,k,boundary,solver=cache['kernel']
        k['g']=(conductance*reference_lengths[live]/row_scale)[owned,None]
        put(boundary,boundary_values)
    else:
        if cache is not None:
            from mineral_lifecycle import clear_pressure
            clear_pressure(cache)
        network=d3.Network(vertex_ids=tuple(map(int,vertices)),edge_ids=tuple(map(int,live)),
                           endpoints=mapping[endpoints[live]],lengths=reference_lengths[live])
        dist=layout(network,comm=comm,owners=None if owners is None else np.asarray(owners)[live]);basis=d3.NetworkBasis(network,size=modes)
        owned=dist.network_partition.edges[comm.rank]
        p=dist.Field(name='p',bases=basis);P=dist.Field(name='P',bases=basis.vertices)
        tau1=dist.Field(name='tau1',bases=basis.edges);tau2=dist.Field(name='tau2',bases=basis.edges)
        k=dist.Field(name='k',bases=basis.edges);alpha=dist.Field(name='alpha',bases=basis.vertices)
        beta=dist.Field(name='beta',bases=basis.vertices);boundary=dist.Field(name='boundary',bases=basis.vertices)
        k['g']=(conductance*reference_lengths[live]/row_scale)[owned,None]
        put(alpha,boundary_mask);put(beta,~boundary_mask);put(boundary,boundary_values)
        R=basis.restriction;dx=basis.derivative
        lift=lambda a,i:d3.Lift(a,basis.interval.derivative_basis(2),i)
        namespace=dict(p=p,P=P,tau1=tau1,tau2=tau2,k=k,alpha=alpha,beta=beta,boundary=boundary,R=R,dx=dx,lift=lift)
        problem=d3.LBVP([p,P,tau1,tau2],namespace=namespace)
        problem.add_equation('-dx(k*dx(p))+lift(tau1,-1)+lift(tau2,-2)=0')
        problem.add_equation('endpoint(p)-R@P=0')
        problem.add_equation('alpha*P+beta*(R.T@(endpoint(k)*endpoint(p, deriv=1)))=boundary')
        solver=build(problem,network_solver)
        if cache is not None:
            cache.clear();cache.update(key=key,kernel=(network,owned,P,k,boundary,solver))
    setup=time.perf_counter()
    # The immutable topology/problem may be reused. Every new geometry state
    # explicitly regathers coefficients, assembles and refactors its matrix.
    # Only RHS-only pressure defect corrections reuse that new factorization.
    accumulated,original_rhs,residual,algebraic_polish=solve_pressure_system(solver,rebuild_matrices=reused)
    algebraic_polishes=[algebraic_polish]
    accumulated_low=[np.zeros_like(value) for value in accumulated]
    pv=get(P).ravel()
    # Reservoir values and their topologically proven constant-pressure
    # components are exact constraints. Reinsert them before differencing;
    # the original Kirchhoff balance is still checked at every interior node.
    boundary_adjustment=float(np.max(np.abs(pv[boundary_mask]-boundary_values[boundary_mask]),initial=0.))
    pv[boundary_mask]=boundary_values[boundary_mask]
    local_endpoints=network.endpoints
    residuals=[residual]
    def correction(rhs):
        nonlocal accumulated,accumulated_low
        rhs=rhs/row_scale;rhs[boundary_mask]=0.
        amplitude=float(np.max(np.abs(rhs),initial=0.))
        if amplitude==0:return np.zeros_like(rhs)
        put(boundary,rhs/amplitude)
        delta,_,auxiliary_residual,polish=solve_pressure_system(solver)
        residuals.append(auxiliary_residual);algebraic_polishes.append(polish)
        from mineral_topology import two_sum
        for index,value in enumerate(delta):
            update,error=two_sum(accumulated[index],value*amplitude)
            accumulated[index],accumulated_low[index]=two_sum(update,accumulated_low[index]+error)
        value=get(P).ravel()*amplitude;value[boundary_mask]=0.
        return value
    from mineral_topology import refine_pressure
    refined=refine_pressure(local_endpoints,conductance,pv,np.flatnonzero(physical_free),
                            np.flatnonzero(np.isin(vertices,inlet)),correction)
    combined_residual=algebraic_residual(solver,accumulated,original_rhs,accumulated_low) if len(residuals)>1 else residual
    unit_flow=refined['flow'];net=refined['net'];q0=refined['throughput']
    scale=total_flow/q0
    pressure[vertices]=(refined['high']+refined['low'])*scale
    pressure[np.isin(labels,np.setdiff1d(labels[inlet],labels[outlet]))]=scale
    flow[live]=unit_flow*scale
    free=physical_free
    flux_norm=np.bincount(local_endpoints.ravel(),weights=np.repeat(np.abs(unit_flow),2),minlength=len(vertices))
    conservation=float(np.max(np.abs(net[free]),initial=0.)/q0)
    pressure_hi=np.zeros(n);pressure_lo=np.zeros(n)
    pressure_hi[vertices]=refined['high'];pressure_lo[vertices]=refined['low']
    pressure_hi[np.isin(labels,np.setdiff1d(labels[inlet],labels[outlet]))]=1.
    result=dict(clogged=False,pressure=pressure,flow=flow,inlet_pressure=scale,
                conservation_residual=conservation,
                algebraic_residual=combined_residual,maximum_auxiliary_normalized_residual=max(residuals),
                unit_pressure_hi=pressure_hi,unit_pressure_lo=pressure_lo,
                pressure_refinement_iterations=refined['iterations'],water_before_refinement=refined['before'],
                algebraic_refinement=algebraic_polishes,
                immutable_pressure_graph_reused=reused,
                dirichlet_adjustment=boundary_adjustment,
                node_relative_balance=float(np.max(np.divide(np.abs(net[free]),flux_norm[free],
                    out=np.zeros(np.count_nonzero(free)),where=flux_norm[free]>0),initial=0.)),
                seconds=time.perf_counter()-start,setup_seconds=setup-start)
    if cache is None:
        from mineral_lifecycle import retire
        retire(solver.dist)
    return result


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
    tauB=dist.Field(name='tauB',bases=basis.edges);tauC=dist.Field(name='tauC',bases=basis.edges)
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
    Up=d3.CardinalMap(basis.vertices,basis.edges,
                     sparse.csr_matrix((np.ones(len(e)),(e,source)),shape=(len(e),len(vertices))),name='Up')
    L=d3.CardinalMap(basis.edges,basis.edges,sparse.diags(positive.astype(float)),name='L')
    H=d3.CardinalMap(basis.edges,basis.edges,sparse.diags((~positive).astype(float)),name='H')
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
    R=basis.restriction;dx=basis.derivative
    left=lambda a:d3.Interpolate(a,network.position_coord,'left')
    right=lambda a:d3.Interpolate(a,network.position_coord,'right')
    trace_up=lambda a:L@left(a)+H@right(a)
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
    lift=lambda a:d3.Lift(a,basis.interval.derivative_basis(1),-1)
    problem=d3.LBVP([B,C,VB,VC,tauB,tauC]+mixing_unknowns,namespace=locals())
    if fitted:
        # c=F(s)w makes dw/dx=0. Two modes suffice for the constant
        # amplitudes; sharp physical concentration layers are reconstructed
        # from F, with exact integrated reaction. This is not axial resolution
        # or extra physical pores, and applies only to uniform coefficients.
        problem.add_equation('dx(B)+lift(tauB)=0')
        problem.add_equation('dx(C)+lift(tauC)=0')
    else:
        problem.add_equation('Q*dx(B)+A1*B+lift(tauB)=0')
        problem.add_equation('Q*dx(C)-A1*B+A2*C+lift(tauC)=0')
    if concentration_map is None:
        problem.add_equation('trace_up(B)-Up@VB=0')
        problem.add_equation('trace_up(C)-Up@VC=0')
    else:
        problem.add_equation('trace_up(B)-mixed_b=0')
        problem.add_equation('trace_up(C)-mixed_c=0')
    if fitted:
        problem.add_equation('alpha*VB+beta*(VB-R.T@(incoming*TB*endpoint(B)))=prescribed')
        problem.add_equation('alpha*VC+beta*(VC-R.T@(incoming*(TC*endpoint(C)+TH*endpoint(B))))=0')
    else:
        problem.add_equation('alpha*VB+beta*(VB-R.T@(incoming*endpoint(B)))=prescribed')
        problem.add_equation('alpha*VC+beta*(VC-R.T@(incoming*endpoint(C)))=0')
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
