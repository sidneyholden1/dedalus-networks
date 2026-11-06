"""Independent uniform-pore reference for Budek & Szymczak (2025).

Lengths are in mean initial node-distance units, concentration in inlet B,
chi_A*nu_A = 1, k1 = 1, mu = 1.  Physical time is converted to the paper's
t' with 2*k1/d0.  This module is a comparator, not the Dedalus application.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve,splu


@dataclass(frozen=True)
class Chemistry:
    gamma: float = 1.0
    kappa: float = 1.0
    da: float = 0.5
    d0: float = 0.1
    g1: float = 1.0
    g2: float = 1.0
    k1: float = 1.0
    mu: float = 1.0

    def __post_init__(self):
        for key, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if min(self.da, self.d0, self.k1, self.mu) <= 0:
            raise ValueError("Positive Da, diameter, rate and viscosity required")

    def rates(self, diameter):
        diameter = np.asarray(diameter)
        # kappa is the INITIAL effective-rate ratio, not always k2/k1.
        k2 = self.kappa * self.k1 * (1 + self.g2) / (1 + self.g1)
        return (self.k1 / (1 + self.g1 * diameter / self.d0),
                k2 / (1 + self.g2 * diameter / self.d0))

    @property
    def mean_initial_flow(self):
        return np.pi * self.d0 * self.k1 / ((1 + self.g1) * self.da)

    def scaled_time(self, time):
        return 2 * self.k1 * time / self.d0


def pore_solution(a, b, inlet_b=1., inlet_c=0.):
    """Outlet B/C and integrated dissolution/precipitation in concentration units.

    a,b are the two nonnegative local Damkohler numbers. The equal-rate limit
    is evaluated without subtracting exponentials. Zero reaction is exact.
    A small-rate series avoids cancellation in the sequential-reaction amount.
    """
    a, b, cb, cc = np.broadcast_arrays(*map(lambda x: np.asarray(x, float),
                                          (a, b, inlet_b, inlet_c)))
    if any(np.any(~np.isfinite(x)) or np.any(x < 0) for x in (a, b, cb, cc)):
        raise ValueError("Finite nonnegative rates and concentrations required")
    delta = np.abs(a-b)
    phi = np.ones_like(delta)
    np.divide(-np.expm1(-delta), delta, out=phi, where=delta != 0)
    generated_c = cb * (a * np.exp(-np.minimum(a, b))) * phi
    consumed_b = cb * -np.expm1(-a)
    slow = np.minimum(a,b)
    # The completion probability is symmetric in the two waiting times.
    # Using the slower rate avoids subtracting O(1) quantities when only one
    # rate is tiny, e.g. a=1 and b=1e-20.
    precip_from_b = cb * (-np.expm1(-slow)-slow*np.exp(-slow)*phi)
    # Probability that both consecutive first-order reactions have occurred.
    small = np.maximum(a, b) < .05
    aa, bb = a[small], b[small]
    if aa.size:
        series = np.zeros_like(aa)
        factorial = 2.
        for m in range(16):
            if m:
                factorial *= m+2
            polynomial = sum(aa**k * bb**(m-k) for k in range(m+1))
            series += (-1.)**m * polynomial / factorial
        precip_from_b = np.array(precip_from_b, copy=True)
        precip_from_b[small] = cb[small] * aa * bb * series
    precip_from_b = np.where(b == 0, 0., precip_from_b)
    precip = precip_from_b + cc * -np.expm1(-b)
    tolerance = 16*np.finfo(float).eps*np.maximum(cb+cc, np.finfo(float).tiny)
    if np.any(precip < -tolerance):
        raise ArithmeticError("Negative integrated precipitation")
    cleanup = np.maximum(-precip, 0.)
    precip = np.maximum(precip, 0.)
    out_b = cb * np.exp(-a)
    out_c = cc * np.exp(-b) + generated_c
    out_c = np.where(b == 0, cc+consumed_b, out_c)
    return dict(b=out_b, c=out_c, dissolved=consumed_b, precipitated=precip,
                roundoff_cleanup=cleanup)


def front_profile(xi, *, kappa, gamma, initial_porosity, inverse_length=1.):
    """Appendix A2 traveling wave, including kappa=1 and the exhausted side.

    This is the constant-area continuum limit, not a random-network solution.
    No claim is made about the inconsistent printed front-speed formula.
    """
    x = np.maximum(np.asarray(xi, float), 0.) * inverse_length
    result = pore_solution(x, kappa*x)
    primary = (1-initial_porosity)*(-np.expm1(-x))
    secondary = gamma*(1-initial_porosity)*(result['b']+result['c'])
    return dict(c_B=result['b'], c_C=result['c'], v_A=primary, v_E=secondary,
                phi=1-primary-secondary)


def hydraulic_reference(endpoints, lengths, diameters, inlet, outlet,
                        total_flow, *, mu=1., active=None,reference_lengths=None):
    active=np.ones(len(endpoints),bool) if active is None else np.asarray(active,bool)
    if np.any((np.asarray(lengths)==0)&active):
        from mineral_topology import contract_zero_lengths
        reference_lengths=np.ones(len(endpoints)) if reference_lengths is None else reference_lengths
        return contract_zero_lengths(endpoints,reference_lengths,lengths,diameters,inlet,outlet,total_flow,
            active=active,positive_solver=lambda *a,**k:_hydraulic_positive(*a,mu=mu,**k))
    return _hydraulic_positive(endpoints,lengths,diameters,inlet,outlet,total_flow,mu=mu,active=active)


def _hydraulic_positive(endpoints, lengths, diameters, inlet, outlet,
                        total_flow, *, mu=1., active=None):
    """Direct graph-Laplacian solve with exact inactive-edge removal.

    Isolated components get a zero gauge. Dead ends connected only to inlet
    have constant inlet pressure and zero flow. Only components with both
    reservoirs transmit flow. No positive conductance floor is introduced.
    """
    endpoints = np.asarray(endpoints, np.int64)
    lengths, diameters = np.asarray(lengths, float), np.asarray(diameters, float)
    n = int(endpoints.max())+1
    inlet, outlet = np.asarray(inlet, np.int64), np.asarray(outlet, np.int64)
    if np.intersect1d(inlet, outlet).size or not inlet.size or not outlet.size:
        raise ValueError("Distinct nonempty reservoir sets required")
    active = np.ones(len(endpoints), bool) if active is None else np.asarray(active, bool)
    if not np.isfinite(total_flow) or total_flow <= 0:
        raise ValueError("Positive total flow required")
    if np.any(lengths[active] <= 0) or np.any(diameters[active] <= 0):
        raise ValueError("Active pores need positive length and diameter")
    u, v = endpoints[active].T
    adjacency = sparse.coo_matrix((np.ones(2*len(u)),
                                  (np.r_[u, v], np.r_[v, u])), shape=(n,n)).tocsr()
    _, labels = connected_components(adjacency)
    inc, outc = np.unique(labels[inlet]), np.unique(labels[outlet])
    transmitting = np.intersect1d(inc, outc)
    if not transmitting.size:
        return dict(clogged=True, pressure=np.zeros(n), flow=np.zeros(len(endpoints)),
                    pressure_residual=0., conservation_residual=0., inlet_pressure=None)
    live = active & np.isin(labels[endpoints[:,0]], transmitting)
    u, v = endpoints[live].T
    g = np.pi * diameters[live]**4 / (128*mu*lengths[live])
    lap = sparse.coo_matrix((np.r_[g,g,-g,-g],
                            (np.r_[u,v,u,v], np.r_[u,v,v,u])), shape=(n,n)).tocsr()
    pressure = np.zeros(n)
    pressure[np.isin(labels, np.setdiff1d(inc, outc))] = 1.
    pressure[inlet] = 1.
    from mineral_topology import reservoir_constraints
    constrained,values=reservoir_constraints(endpoints[live],n,inlet,outlet)
    pressure[constrained]=values[constrained]
    physical_free=np.flatnonzero(np.isin(labels,transmitting)&
                                ~np.isin(np.arange(n),np.r_[inlet,outlet]))
    free = np.flatnonzero(np.isin(labels, transmitting) & ~constrained)
    matrix = lap[free][:,free]
    rhs = -lap[free] @ pressure
    factor=splu(matrix.tocsc()) if len(free) else None
    pressure[free] = factor.solve(rhs) if len(free) else np.empty(0)
    def correction(rhs):
        value=np.zeros(n)
        if len(free):value[free]=factor.solve(rhs[free])
        return value
    from mineral_topology import refine_pressure
    refined=refine_pressure(endpoints[live],g,pressure,physical_free,inlet,correction)
    initial_flow=refined['throughput']
    scale = total_flow/initial_flow
    pressure = (refined['high']+refined['low'])*scale
    flow = np.zeros(len(endpoints))
    flow[live] = refined['flow']*scale
    net = refined['net']*scale
    residual = np.max(np.abs(net[physical_free]), initial=0.)
    denominator = max(np.max(np.abs(lap[physical_free]) @ np.abs(pressure), initial=0.),
                      np.finfo(float).tiny)
    return dict(clogged=False, pressure=pressure, flow=flow,
                inlet_pressure=scale, pressure_residual=residual/denominator,
                conservation_residual=residual/total_flow,
                unit_pressure_hi=refined['high'],unit_pressure_lo=refined['low'],
                pressure_refinement_iterations=refined['iterations'],water_before_refinement=refined['before'],
                reservoir_balance=abs(net[inlet].sum()+net[outlet].sum())/total_flow)


def transport_reference(endpoints, flow, lengths, diameters, pressure, inlet,
                        chemistry, *, dissolving=None,contraction_labels=None,potential_order=None,
                        factor_transport=True,concentration_map=None):
    """Independent topological transport sweep ordered by hydraulic potential.

    Pressure-driven positive flows form a directed acyclic graph. Signed
    orientation is recomputed on each call. Exact zero-flow pores are inert.
    """
    endpoints = np.asarray(endpoints, np.int64)
    flow, lengths, diameters = map(np.asarray, (flow, lengths, diameters))
    if concentration_map is not None:
        from mineral_mixing import validate_network_map
        concentration_map=validate_network_map(concentration_map,endpoints,flow,inlet)
    n, e = len(pressure), len(flow)
    upstream = np.where(flow >= 0, endpoints[:,0], endpoints[:,1])
    downstream = np.where(flow >= 0, endpoints[:,1], endpoints[:,0])
    q = np.abs(flow)
    k1, k2 = chemistry.rates(diameters)
    if dissolving is not None:
        access=np.asarray(dissolving,float)
        if np.any(~np.isfinite(access)) or np.any((access<0)|(access>1)):
            raise ValueError('Primary reactive fraction must lie in [0,1]')
        k1 = k1*access
    a = np.divide(np.pi*diameters*lengths*k1, q, out=np.zeros(e), where=q > 0)
    b = np.divide(np.pi*diameters*lengths*k2, q, out=np.zeros(e), where=q > 0)
    # The independent pore solution is linear in both incoming species.
    # Evaluate its rate-dependent coefficients once over all pores, rather
    # than repeating exponentials and small-rate series at every vertex.
    coefficients=pore_solution(a,b) if factor_transport else None
    c_transmission=np.exp(-b);c_consumed=-np.expm1(-b)
    order = np.argsort(upstream, kind='stable')
    ptr = np.r_[0, np.cumsum(np.bincount(upstream, minlength=n))]
    mass_b, mass_c, water = np.zeros(n), np.zeros(n), np.zeros(n)
    cb, cc, dissolved, precipitated, out_b, out_c = (np.zeros(e) for _ in range(6))
    node_b, node_c = np.zeros(n), np.zeros(n)
    is_inlet = np.zeros(n, bool); is_inlet[inlet] = True
    clean = 0.
    if potential_order is not None:node_order=np.asarray(potential_order,np.int64)
    elif contraction_labels is None:node_order=np.argsort(-np.asarray(pressure),kind='stable')
    else:
        from mineral_topology import flow_order
        node_order=flow_order(endpoints,flow,n)
    for node in node_order:
        if is_inlet[node]:
            node_b[node], node_c[node] = 1., 0.
        elif water[node] > 0:
            node_b[node], node_c[node] = mass_b[node]/water[node], mass_c[node]/water[node]
        edges = order[ptr[node]:ptr[node+1]]
        edges = edges[q[edges] > 0]
        if not len(edges):
            continue
        violation=np.asarray(pressure)[downstream[edges]]>=pressure[node]
        if contraction_labels is not None:
            violation &= np.asarray(contraction_labels)[downstream[edges]]!=contraction_labels[node]
        if potential_order is None and np.any(violation):
            raise ArithmeticError("Flow does not decrease hydraulic potential")
        if concentration_map is None or is_inlet[node]:
            cb[edges], cc[edges] = node_b[node], node_c[node]
        else:
            cb[edges]=np.asarray(concentration_map[edges]@out_b).ravel()
            cc[edges]=np.asarray(concentration_map[edges]@out_c).ravel()
        if factor_transport:
            r=dict(b=cb[edges]*coefficients['b'][edges],
                   c=cb[edges]*coefficients['c'][edges]+cc[edges]*c_transmission[edges],
                   dissolved=cb[edges]*coefficients['dissolved'][edges],
                   precipitated=cb[edges]*coefficients['precipitated'][edges]+cc[edges]*c_consumed[edges],
                   roundoff_cleanup=cb[edges]*coefficients['roundoff_cleanup'][edges])
            # Preserve the exactly non-precipitating branch of pore_solution.
            no_precipitation=b[edges]==0
            r['c'][no_precipitation]=cc[edges][no_precipitation]+r['dissolved'][no_precipitation]
        else:r = pore_solution(a[edges], b[edges], cb[edges], cc[edges])
        out_b[edges]=r['b'];out_c[edges]=r['c']
        dissolved[edges] = q[edges]*r['dissolved']
        precipitated[edges] = q[edges]*r['precipitated']
        clean += float(np.sum(q[edges]*r['roundoff_cleanup']))
        np.add.at(mass_b, downstream[edges], q[edges]*r['b'])
        np.add.at(mass_c, downstream[edges], q[edges]*r['c'])
        np.add.at(water, downstream[edges], q[edges])
    return dict(inlet_b=cb, inlet_c=cc, node_b=node_b, node_c=node_c,
                outlet_b=out_b,outlet_c=out_c,
                dissolved_rate=dissolved, precipitated_rate=precipitated,
                incoming_b=mass_b, incoming_c=mass_c, incoming_water=water,
                local_da1=a, local_da2=b, roundoff_cleanup=clean)
