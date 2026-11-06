"""Conservative port-to-port solute transfer for the proposed model comparison.

For the diamond four-port geometry, the zero-mixing limit recovers Sharma et al.
(2023), equations 15--17, https://doi.org/10.1029/2023GL105183. Higher degree uses
an explicit additional closure: greedily match the smallest turning angle,
breaking equal-cost ties by stable physical port IDs. It is not a validated
continuum junction model. Mixing fraction one gives complete mixing exactly.

Directions point along fluid motion: incoming vectors point into the junction,
outgoing vectors away. Only strictly positive incident fluxes are supplied;
zero-flow ports are excluded by the caller. No hydraulic flow is rescaled.
Capacity-subtraction roundoff is assigned to the largest opposite port, with
stable ID ties, and accepted only under the unchanged per-port transfer audit.
Evolution/geometry integration and distributed qualification remain separate.
"""
import numpy as np


ROUND_OFF=128*np.finfo(float).eps


def mixing_settings(fraction):
    """Serializable physical model identity shared by histories and checkpoints."""
    fraction=float(fraction)
    if not np.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('Mixing fraction must lie in [0,1]')
    return dict(mixing_fraction=fraction,mixing_rule='complete' if fraction==1 else 'turning-angle-v1')


def _fluxes(incoming,outgoing):
    incoming=np.asarray(incoming,float);outgoing=np.asarray(outgoing,float)
    if (incoming.ndim!=1 or outgoing.ndim!=1 or not len(incoming) or not len(outgoing) or
        np.any(~np.isfinite(incoming)) or np.any(~np.isfinite(outgoing)) or
        np.any(incoming<=0) or np.any(outgoing<=0)):
        raise ValueError('Incident water fluxes must be finite positive vectors')
    total_in=np.sum(incoming,dtype=np.longdouble);total_out=np.sum(outgoing,dtype=np.longdouble)
    if abs(total_in-total_out)>ROUND_OFF*max(total_in,total_out):
        raise ValueError('Incoming and outgoing water fluxes are incompatible')
    return incoming,outgoing,total_in,total_out


def audit_transfer(matrix,incoming,outgoing):
    incoming,outgoing,_,_=_fluxes(incoming,outgoing);matrix=np.asarray(matrix,float)
    if matrix.shape!=(len(outgoing),len(incoming)) or np.any(~np.isfinite(matrix)) or np.any(matrix<0):
        raise ValueError('Transfer matrix must be finite, nonnegative and port-sized')
    wide=matrix.astype(np.longdouble)
    columns=np.sum(wide,axis=0)
    water=wide@incoming.astype(np.longdouble)
    column_error=float(np.max(np.abs(columns-1)))
    # Relative to each outgoing flux: a tiny outlet must preserve a uniform
    # concentration too, not merely satisfy a large network-wide denominator.
    water_error=float(np.max(np.abs(water-outgoing)/outgoing))
    if max(column_error,water_error)>ROUND_OFF:
        raise ArithmeticError('Port solute/water transfer constraints failed')
    return dict(maximum_column_sum_defect=column_error,maximum_relative_water_defect=water_error,
                minimum_transfer=float(np.min(matrix)),roundoff_tolerance=ROUND_OFF)


def transfer_matrix(incoming,outgoing,*,mixing_fraction=1.,incoming_vectors=None,outgoing_vectors=None,
                    incoming_ids=None,outgoing_ids=None):
    """Return T, with outgoing solute flux = T @ incoming solute flux.

    T[:,j] sums to one and T @ q_in = q_out to audited floating-point accuracy.
    A geometry-dependent coupling consumes available incoming/outgoing water in
    increasing turning-angle order. Fractional complete mixing is a convex
    combination, so conservation and uniform-state preservation remain exact
    algebraic properties. Ill-conditioned or incompatible numeric inputs fail.
    """
    incoming,outgoing,_,total_out=_fluxes(incoming,outgoing)
    if not np.isfinite(mixing_fraction) or not 0<=mixing_fraction<=1:
        raise ValueError('Mixing fraction must lie in [0,1]')
    full=np.broadcast_to(np.asarray(outgoing/total_out,float)[:,None],(len(outgoing),len(incoming))).copy()
    if mixing_fraction==1:
        audit_transfer(full,incoming,outgoing);return full
    vectors=[]
    for values,count in [(incoming_vectors,len(incoming)),(outgoing_vectors,len(outgoing))]:
        value=np.asarray(values,float)
        if value.shape!=(count,2) or not np.all(np.isfinite(value)):
            raise ValueError('Each physical port needs a finite two-dimensional direction')
        norm=np.linalg.norm(value,axis=1)
        if np.any(norm==0) or not np.all(np.isfinite(norm)):raise ValueError('Port direction must be nonzero')
        vectors.append(value/norm[:,None])
    identifiers=[]
    for values,count in [(incoming_ids,len(incoming)),(outgoing_ids,len(outgoing))]:
        value=np.asarray(values)
        if value.shape!=(count,) or value.dtype.kind not in 'iu' or np.any(value<0):
            raise ValueError('Stable nonnegative integer port IDs are required')
        identifiers.append(value)
    if len(np.unique(np.r_[identifiers[0],identifiers[1]]))!=len(incoming)+len(outgoing):
        raise ValueError('Incident physical port IDs must be distinct')
    cost=1-vectors[1]@vectors[0].T
    candidates=sorted((float(cost[i,j]),min(int(identifiers[1][i]),int(identifiers[0][j])),
        max(int(identifiers[1][i]),int(identifiers[0][j])),i,j)
        for i in range(len(outgoing)) for j in range(len(incoming)))
    remaining_in=incoming.astype(np.longdouble);remaining_out=outgoing.astype(np.longdouble)
    coupling=np.zeros((len(outgoing),len(incoming)),np.longdouble)
    for _,_,_,i,j in candidates:
        water=min(remaining_out[i],remaining_in[j])
        coupling[i,j]+=water;remaining_out[i]-=water;remaining_in[j]-=water
    # A small positive port can remain after a large capacity rounds to zero.
    # Keep that port's complete solute flux. Assign only these subtraction
    # remainders to the largest opposite port; this minimizes its relative
    # water perturbation and commutes with physical flow reversal. Never change
    # the hydraulic fluxes. The original relative column/water audit below
    # still rejects corrections exceeding its existing roundoff bound.
    largest_out=min(range(len(outgoing)),key=lambda i:(-outgoing[i],int(identifiers[1][i])))
    largest_in=min(range(len(incoming)),key=lambda j:(-incoming[j],int(identifiers[0][j])))
    coupling[largest_out,:]+=remaining_in
    coupling[:,largest_in]+=remaining_out
    reduced=np.asarray(coupling/incoming[None,:],float)
    matrix=(1-mixing_fraction)*reduced+mixing_fraction*full
    audit_transfer(matrix,incoming,outgoing)
    return matrix


def validate_network_map(matrix,endpoints,flow,inlet):
    """Validate an edge-to-edge map of outgoing to incoming concentration.

    At each junction M[i,j] = T[i,j]*abs(q[j])/abs(q[i]), with i an outgoing
    pore and j an incoming pore. Reservoir rows are zero and receive prescribed
    concentration separately. Rows/columns for exact zero-flow pores are zero.
    """
    from scipy import sparse
    endpoints=np.asarray(endpoints,np.int64);flow=np.asarray(flow,float)
    if endpoints.shape!=(len(flow),2) or np.any(~np.isfinite(flow)):
        raise ValueError('Invalid network flow input')
    matrix=sparse.csr_matrix(matrix,copy=True)
    if matrix.shape!=(len(flow),len(flow)) or np.any(~np.isfinite(matrix.data)) or np.any(matrix.data<0):
        raise ValueError('Concentration map must be finite, nonnegative and edge-sized')
    matrix.sum_duplicates();matrix.eliminate_zeros()
    source=np.where(flow>=0,endpoints[:,0],endpoints[:,1])
    target=np.where(flow>=0,endpoints[:,1],endpoints[:,0]);q=np.abs(flow)
    rows,cols=matrix.nonzero()
    if (np.any(q[rows]==0) or np.any(q[cols]==0) or np.any(source[rows]!=target[cols]) or
        np.any(np.isin(source[rows],inlet))):
        raise ValueError('Concentration map couples incompatible physical ports')
    live=q>0;prescribed=np.isin(source,inlet)
    row_sum=np.asarray(matrix.sum(axis=1)).ravel()
    if np.max(np.abs(row_sum-(live&~prescribed)),initial=0)>ROUND_OFF:
        raise ArithmeticError('Concentration map does not preserve uniform outgoing states')
    has_outgoing=np.unique(source[live])
    internal=live&np.isin(target,has_outgoing)&~np.isin(target,inlet)
    expected=np.where(internal,q,0.)
    delivered=np.asarray(matrix.T@q).ravel()
    error=np.divide(np.abs(delivered-expected),q,out=np.zeros_like(q),where=live)
    if np.max(error,initial=0)>ROUND_OFF:
        raise ArithmeticError('Concentration map does not conserve incident solute flux')
    return matrix


def network_concentration_map(geometry,flow,*,mixing_fraction=0.,return_audit=False):
    """Build outgoing concentration laws from physical periodic pore directions.

    Pore IDs remain the geometry's canonical edge indices. Port 2*edge is its
    first endpoint and port 2*edge+1 its second. Exact zero flows have no route;
    inlet reservoir rows are prescribed separately by the transport solver.
    Indexing is O(E log E), followed by local incident-port products, rather
    than a scan of every pore at every junction. This does not evolve geometry.
    """
    from scipy import sparse
    endpoints=np.asarray(geometry.endpoints);points=np.asarray(geometry.points,float)
    shift=np.asarray(geometry.edge_shift);flow=np.asarray(flow,float)
    inlet=np.asarray(geometry.inlet)
    count=len(flow);nodes=len(points)
    if (flow.ndim!=1 or endpoints.shape!=(count,2) or endpoints.dtype.kind not in 'iu' or
        points.shape!=(nodes,2) or shift.shape!=(count,) or shift.dtype.kind not in 'iu' or
        inlet.ndim!=1 or inlet.dtype.kind not in 'iu' or not np.isfinite(geometry.width) or geometry.width<=0 or
        np.any(~np.isfinite(flow)) or np.any(~np.isfinite(points)) or
        np.any(endpoints<0) or np.any(endpoints>=nodes) or np.any(inlet<0) or np.any(inlet>=nodes)):
        raise ValueError('Invalid physical pore geometry or flow')
    if not np.isfinite(mixing_fraction) or not 0<=mixing_fraction<=1:
        raise ValueError('Mixing fraction must lie in [0,1]')
    direction=points[endpoints[:,1]]-points[endpoints[:,0]]
    direction[:,0]+=shift*geometry.width
    direction*=np.sign(flow)[:,None]
    source=np.where(flow>0,endpoints[:,0],endpoints[:,1])
    target=np.where(flow>0,endpoints[:,1],endpoints[:,0])
    live=np.flatnonzero(flow!=0);q=np.abs(flow)
    outgoing=live[np.argsort(source[live],kind='stable')]
    incoming=live[np.argsort(target[live],kind='stable')]
    optr=np.r_[0,np.cumsum(np.bincount(source[live],minlength=nodes))]
    iptr=np.r_[0,np.cumsum(np.bincount(target[live],minlength=nodes))]
    reservoir=np.zeros(nodes,bool);reservoir[inlet]=True
    rows=[];cols=[];values=[]
    details=dict(mixing_fraction=float(mixing_fraction),junctions=0,zero_flow_pores=int(count-len(live)),
        maximum_column_sum_defect=0.,maximum_relative_water_defect=0.,roundoff_tolerance=ROUND_OFF,
        directions='Original centerline displacement with lateral periodic image shift, oriented along flow')
    for node in np.flatnonzero((np.diff(optr)>0)&~reservoir):
        out=outgoing[optr[node]:optr[node+1]];inc=incoming[iptr[node]:iptr[node+1]]
        if not len(inc):raise ArithmeticError('Non-reservoir outgoing flow has no incoming supply')
        transfer=transfer_matrix(q[inc],q[out],mixing_fraction=mixing_fraction,
            incoming_vectors=direction[inc],outgoing_vectors=direction[out],
            incoming_ids=2*inc+(flow[inc]>0),outgoing_ids=2*out+(flow[out]<0))
        local=audit_transfer(transfer,q[inc],q[out])
        details['junctions']+=1
        for name in ['maximum_column_sum_defect','maximum_relative_water_defect']:
            details[name]=max(details[name],local[name])
        concentration=np.asarray(transfer.astype(np.longdouble)*q[inc][None,:]/q[out][:,None],float)
        a,b=np.nonzero(concentration)
        rows.extend(out[a]);cols.extend(inc[b]);values.extend(concentration[a,b])
    matrix=sparse.csr_matrix((values,(rows,cols)),shape=(count,count))
    matrix=validate_network_map(matrix,endpoints,flow,inlet)
    return (matrix,details) if return_audit else matrix
