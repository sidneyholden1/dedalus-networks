"""Ownership contracts for the optional network partition preparation APIs."""
import builtins
import numpy as np
import pytest
import dedalus.public as d3
from dedalus.tools.network_partitioning import corridor_partition, mt_kahypar_partition


def graph():
    return d3.Network(endpoints=[(0,1),(1,2),(0,3),(3,4),(0,5),(5,6),
                                 (7,8),(8,9),(7,10),(10,11),(7,12),(12,13)],
                      lengths=np.ones(12), vertex_ids=range(15))


def test_corridors_preserve_graph_and_authoritative_owners():
    net=graph(); before=net.endpoints.copy()
    explicit=d3.NetworkPartition(net, np.zeros(net.num_edges, dtype=int),num_parts=6)
    part,info=corridor_partition(net,6)
    np.testing.assert_array_equal(part.edge_owners,np.repeat(np.arange(6),2))
    np.testing.assert_array_equal(net.endpoints,before)
    np.testing.assert_array_equal(explicit.edge_owners,np.zeros(12,dtype=int))
    assert info['anchor_count']==2 and info['estimated_imbalance']==1


@pytest.mark.parametrize('parts',[1,3,8])
def test_loops_parallel_edges_disconnected_and_idle_ranks(parts):
    net=d3.Network(endpoints=[(0,0),(0,1),(0,1),(3,4)],lengths=np.ones(4),vertex_ids=range(7))
    a,_=corridor_partition(net,parts,weights=[1,2,3,20])
    b,_=corridor_partition(net,parts,weights=[1,2,3,20])
    np.testing.assert_array_equal(a.edge_owners,b.edge_owners)
    assert np.all((a.edge_owners>=0)&(a.edge_owners<parts))


def test_empty_graph():
    net=d3.Network(endpoints=np.empty((0,2),dtype=int),lengths=[],vertex_ids=range(3))
    part,info=corridor_partition(net,4)
    assert len(part.edge_owners)==0 and info['blocks']==0


@pytest.mark.parametrize('parts',[0,-1,True,1.5])
def test_invalid_part_count(parts):
    with pytest.raises(ValueError):corridor_partition(graph(),parts)


@pytest.mark.parametrize('weights',[np.zeros(12),np.full(12,np.nan),[-1]*12,[1]*11,np.full(12,1e308)])
def test_invalid_work(weights):
    with pytest.raises(ValueError):corridor_partition(graph(),3,weights=weights)


def test_long_cycle_and_heavy_indivisible_edge():
    net=d3.Network(endpoints=[(i,(i+1)%100) for i in range(100)],lengths=np.ones(100))
    part,info=corridor_partition(net,4)
    np.testing.assert_array_equal(np.bincount(part.edge_owners),[25]*4)
    heavy=np.ones(100);heavy[0]=1000
    part,info=corridor_partition(net,4,weights=heavy)
    assert info['maximum_block_weight']<=1000


def test_external_dependency_is_optional(monkeypatch):
    original=builtins.__import__
    def unavailable(name,*args,**kwargs):
        if name=='mtkahypar':raise ImportError('intentional missing optional package')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',unavailable)
    assert len(corridor_partition(graph(),3)[0].edge_owners)==12
    with pytest.raises(ImportError,match='optional'):
        mt_kahypar_partition(graph(),3)
    assert len(d3.NetworkPartition.balanced(graph(),3).edge_owners)==12


@pytest.mark.parametrize('objective',['cut','km1','soed'])
def test_external_weighted_objectives(objective):
    pytest.importorskip('mtkahypar')
    net=graph();q=np.full(net.num_vertices,2,dtype=int);q[-1]=0
    part,info=mt_kahypar_partition(net,3,weights=np.arange(1,13),vertex_weights=q,
                                 objective=objective,seed=20261011)
    pins=[[] for _ in range(net.num_vertices)]
    for e,(u,v) in enumerate(net.endpoints):
        pins[u].append(e)
        if u!=v:pins[v].append(e)
    lam=np.array([len(set(part.edge_owners[x])) for x in pins])
    expected=dict(cut=int(q@(lam>1)),km1=int(q@np.maximum(lam-1,0)),soed=int(q@(lam*(lam>1))))
    assert info['package_metrics']==expected


def test_external_invalid_inputs():
    for options in [dict(epsilon=0),dict(objective='unsupported'),dict(seed=-1),
                    dict(vertex_weights=np.ones(15)*.5),dict(weights=[1e-20]+[1]*11)]:
        with pytest.raises(ValueError):mt_kahypar_partition(graph(),3,**options)
    with pytest.raises(ValueError,match='idle ranks'):mt_kahypar_partition(graph(),20)
