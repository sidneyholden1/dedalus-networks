"""Event-aware finite-grain updates for the uniform-pore qualification model."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass
class State:
    diameter: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    active: np.ndarray
    time: float = 0.
    step: int = 0
    dissolved: float = 0.
    precipitated: float = 0.
    diameter_roundoff: float = 0.
    inventory_roundoff: float = 0.


def grain_rates(geometry,state,dissolved,precipitated,gamma,*,access_rule='remaining-grains'):
    """Share wall loss among A-bearing neighbors and gain among all neighbors.

    The one-exhausted-neighbor convention is an explicit reconstruction.
    It requires sensitivity qualification before published comparisons.
    """
    if access_rule!='remaining-grains':raise ValueError('Unqualified primary access rule')
    incidence=geometry.incidence
    access=incidence.multiply((state.primary>0)[:,None]).tocsr()
    count=np.asarray(access.sum(axis=0)).ravel()
    if np.any((count==0)&(np.asarray(dissolved)>0)):
        raise ArithmeticError('Dissolution occurs without an accessible primary grain')
    loss=np.asarray(access@np.divide(dissolved,count,out=np.zeros_like(dissolved),where=count>0)).ravel()
    gain=np.asarray(incidence@(gamma*np.asarray(precipitated)/geometry.edge_grain_count)).ravel()
    return loss,gain


def choose_step(geometry,state,lengths,dissolved,precipitated,*,gamma,dmin,
                max_dt,relative_diameter=.01,relative_grain=.02):
    if min(max_dt,relative_diameter,relative_grain,dmin)<=0:raise ValueError('Positive timestep controls required')
    loss,gain=grain_rates(geometry,state,dissolved,precipitated,gamma)
    zero=(np.asarray(lengths)==0)&state.active
    if np.any((np.asarray(lengths)<0)&state.active):raise ValueError('Negative hydraulic length')
    if np.any((np.asarray(dissolved)[zero]!=0)|(np.asarray(precipitated)[zero]!=0)):
        raise ArithmeticError('Reaction in a zero-area pore')
    rate=np.divide(2*(dissolved-gamma*precipitated),np.pi*lengths*state.diameter,
                   out=np.zeros_like(state.diameter),where=state.active&(np.asarray(lengths)>0))
    return limit_rates(geometry,state,rate,loss,gain,dmin=dmin,max_dt=max_dt,
                       relative_diameter=relative_diameter,relative_grain=relative_grain)


def limit_rates(geometry,state,rate,loss,gain,*,dmin,max_dt,relative_diameter,relative_grain):
    """Bound a proposed derivative using the ORIGINAL state and inventories."""
    diameter_bound=np.divide(relative_diameter*state.diameter,np.abs(rate),
                             out=np.full(len(rate),np.inf),where=rate!=0)
    exhaustion=np.divide(state.primary,loss,out=np.full(len(loss),np.inf),where=loss>0)
    # Control absolute changes relative to the original grain capacity.
    # A-relative bounds would approach pure-dissolution exhaustion forever.
    inventory_bound=np.divide(relative_grain*geometry.maximum_grain_volume,np.maximum(loss,gain),
                              out=np.full(len(loss),np.inf),where=np.maximum(loss,gain)>0)
    closure=np.divide(state.diameter-dmin,-rate,out=np.full(len(rate),np.inf),where=(rate<0)&state.active)
    dt=min(max_dt,float(diameter_bound.min()),float(exhaustion.min()),float(inventory_bound.min()),float(closure.min()))
    if not np.isfinite(dt) or dt<=0:raise ArithmeticError('No positive event-safe step')
    return dict(dt=dt,diameter_rate=rate,primary_loss=loss,secondary_gain=gain,
                exhaustion_times=exhaustion,closure_times=closure)


def midpoint_advance(geometry,state,observation,chemistry,observe_stage,*,dmin,max_dt,relative_step):
    """Explicit midpoint geometry update with finite-inventory stage checks.

    The first stage supplies an event-safe half step. The second-stage rates
    update the original state. If their full step would cross an event or the
    stated relative-change bound, shrink and recompute the midpoint. No failed
    trial is committed, and all trial solves are included in measured cost.
    Zero-step equations are identical to the Euler reconstruction.
    """
    first=observation['transport']
    initial=choose_step(geometry,state,observation['lengths'],first['dissolved_rate'],first['precipitated_rate'],
        gamma=chemistry.gamma,dmin=dmin,max_dt=max_dt,relative_diameter=relative_step,relative_grain=relative_step)
    dt=initial['dt']
    for trial in range(32):
        half=dict(initial,dt=dt/2)
        middle,_=advance(geometry,state,half,dissolved=first['dissolved_rate'],
            precipitated=first['precipitated_rate'],gamma=chemistry.gamma,dmin=dmin,
            resolve_clock_events=False)
        observed=observe_stage(middle)
        if observed['clogged']:raise ArithmeticError('A supposedly safe midpoint crossed pore closure')
        transport=observed['transport']
        derivative=choose_step(geometry,middle,observed['lengths'],transport['dissolved_rate'],
            transport['precipitated_rate'],gamma=chemistry.gamma,dmin=dmin,max_dt=max_dt,
            relative_diameter=relative_step,relative_grain=relative_step)
        update=limit_rates(geometry,state,derivative['diameter_rate'],derivative['primary_loss'],
            derivative['secondary_gain'],dmin=dmin,max_dt=max_dt,
            relative_diameter=relative_step,relative_grain=relative_step)
        if dt>update['dt']*(1+8*np.finfo(float).eps):
            # Reduce to the newly located limit and recompute the midpoint.
            # The half step remains inside the first-stage safe interval.
            dt=update['dt']
            continue
        update['dt']=dt
        result,event=advance(geometry,state,update,dissolved=transport['dissolved_rate'],
            precipitated=transport['precipitated_rate'],gamma=chemistry.gamma,dmin=dmin)
        return result,event,dict(stage_solves=trial+1,rejected_trials=trial)
    raise ArithmeticError('Midpoint failed to find a positive event-safe geometry step')


def exhaustion_events(state,update,primary,tolerance,*,resolve_clock_events):
    """Resolve only clock-indistinguishable exhaustion within the roundoff budget.

    A midpoint event can leave a vanishing positive inventory and keep taking
    steps smaller than one physical-clock ulp. Its cbrt length then makes the
    next pressure problem arbitrarily ill-conditioned. Complete that event only
    if both the proposed step and its remaining exhaustion time are unresolved
    by the clock, and the inventory is within the existing cleanup bound.
    Internal midpoint stages never use this boundary cleanup.
    """
    step=update['dt'];times=update['exhaustion_times']
    exact=times<=step*(1+8*np.finfo(float).eps)
    clock=np.zeros_like(exact)
    if resolve_clock_events and step>0 and state.time+step==state.time:
        clock=((state.primary>0)&(update['primary_loss']>0)&(step*update['primary_loss']>0)&
               (state.time+times==state.time)&(np.abs(primary)<=tolerance))
    return exact|clock,clock&~exact


def advance(geometry,state,update,*,dissolved,precipitated,gamma,dmin,resolve_clock_events=True):
    dt=update['dt'];primary=state.primary-dt*update['primary_loss']
    secondary=state.secondary+dt*update['secondary_gain']
    diameter=state.diameter+dt*update['diameter_rate']
    # Exact located events are snapped only within bounded roundoff. Record
    # that adjustment separately from the integrated chemical balance.
    tolerance=32*np.finfo(float).eps*np.maximum(geometry.maximum_grain_volume,state.primary)
    exhausted,clock_events=exhaustion_events(state,update,primary,tolerance,resolve_clock_events=resolve_clock_events)
    if np.any(primary < -tolerance) or np.any(np.abs(primary[exhausted])>tolerance[exhausted]):
        raise ArithmeticError('Inventory step crossed finite-mineral event')
    inventory_cleanup=float(np.sum(np.abs(primary[exhausted])))
    primary[exhausted]=0.
    # A shorter event-safe step can still round d exactly to d_min. Complete
    # that decreasing-pore event within the same bounded cleanup below;
    # otherwise the next closure-time bound is zero and evolution cannot move.
    rounded_boundary=(diameter==dmin)&(update['diameter_rate']<0)
    closing=state.active & ((update['closure_times']<=dt*(1+8*np.finfo(float).eps))|rounded_boundary)
    diameter_cleanup=0.
    if np.any(closing):
        if np.any(np.abs(diameter[closing]-dmin)>32*np.finfo(float).eps*np.maximum(state.diameter[closing],dmin)):
            raise ArithmeticError('Closure event location error')
        target=np.nextafter(dmin,0.) # enforces the paper's strict d < dmin test
        diameter_cleanup=float(np.sum(np.abs(diameter[closing]-target)))
        diameter[closing]=target
    if np.any(primary<0) or np.any(secondary<0) or np.any(diameter<=0):
        raise ArithmeticError('Invalid physical state')
    active=state.active & (diameter>=dmin)
    next_state=State(diameter,primary,secondary,active,state.time+dt,state.step+1,
                     state.dissolved+dt*float(np.sum(dissolved)),
                     state.precipitated+dt*gamma*float(np.sum(precipitated)),
                     state.diameter_roundoff+diameter_cleanup,state.inventory_roundoff+inventory_cleanup)
    return next_state,dict(exhausted_grains=np.flatnonzero(exhausted),clock_roundoff_grains=np.flatnonzero(clock_events),closed_pores=np.flatnonzero(closing),
                           dt=dt,inventory_cleanup=inventory_cleanup,diameter_cleanup=diameter_cleanup)


def spans(edges,mask,inlet,outlet,n):
    edges=np.asarray(edges)[mask]
    if not len(edges):return False
    u,v=edges.T
    adjacency=sparse.coo_matrix((np.ones(2*len(u)),(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    _,labels=connected_components(adjacency)
    return bool(np.intersect1d(labels[inlet],labels[outlet]).size)


def terminal_events(geometry,state,*,d0,grain_adjacency='shared-edge'):
    n=len(geometry.points)
    clogged=not spans(geometry.endpoints,state.active,geometry.inlet,geometry.outlet,n)
    dissolution=spans(geometry.endpoints,state.active&(state.diameter>2*d0),geometry.inlet,geometry.outlet,n)
    # Explicit dual graph; grains do not share pore-node connectivity objects.
    if grain_adjacency!='shared-edge':raise ValueError('Unqualified grain adjacency')
    eligible=state.secondary>state.primary/2
    connected=(geometry.incidence@geometry.incidence.T).tocoo()
    mask=(connected.row!=connected.col)&eligible[connected.row]&eligible[connected.col]
    g=len(eligible)
    graph=sparse.csr_matrix((np.ones(np.count_nonzero(mask)),(connected.row[mask],connected.col[mask])),shape=(g,g))
    _,labels=connected_components(graph)
    grain_in=eligible & np.any(np.isin(geometry.triangles,geometry.inlet),axis=1)
    grain_out=eligible & np.any(np.isin(geometry.triangles,geometry.outlet),axis=1)
    replacement=bool(np.intersect1d(labels[grain_in],labels[grain_out]).size)
    return dict(clogged=clogged,dissolution_breakthrough=dissolution,mineral_replacement_breakthrough=replacement)
