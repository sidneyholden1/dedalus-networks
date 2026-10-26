import numpy as np
import pytest
from mpi4py import MPI
import dedalus.public as d3


@pytest.mark.parametrize('root_last', [False, True])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_root_input_local_default(root_last, dtype):
    comm=MPI.COMM_WORLD
    root=comm.size-1 if root_last else 0
    endpoints=np.array([(0,1),(1,2),(2,0),(2,3),(1,1),(0,1)])
    lengths=np.array([.7,1.2,1.1,.9,.8,1.3])
    # Exercise idle ranks when run with four processes.
    owners=np.where(np.arange(6)%2==0,0,comm.size-1)
    graph=d3.Network(endpoints=endpoints,lengths=lengths) if comm.rank==root else None
    net=d3.Network.from_partition(graph,owners if comm.rank==root else None,comm=comm,root=root)
    dist=d3.Distributor(net,comm=comm,dtype=dtype)
    assert dist.network_vertex_storage=='owned_ghost'
    edges=np.flatnonzero(owners==comm.rank)
    np.testing.assert_array_equal(net.endpoints[edges],endpoints[edges])
    np.testing.assert_array_equal(net.lengths[edges],lengths[edges])
    assert net.endpoints.values.shape[0]==len(edges)
    basis=d3.NetworkBasis(net,size=24)
    u=dist.Field(name='u',bases=basis)
    U=dist.Field(name='U',bases=basis.vertices)
    t1=dist.Field(name='t1',bases=basis.edges)
    t2=dist.Field(name='t2',bases=basis.edges)
    f=dist.Field(name='f',bases=basis)
    g=dist.Field(name='g',bases=basis.vertices)
    value=1+.5j if dtype==np.complex128 else 1.
    f['g']=value;g['g']=value
    R,dx=basis.R,basis.derivative
    lift_basis=basis.interval.derivative_basis(2)
    lift=lambda a,n:d3.Lift(a,lift_basis,n)
    problem=d3.LBVP([u,U,t1,t2],namespace=locals())
    problem.add_equation('-dx(dx(u))+u+lift(t1,-1)+lift(t2,-2)=f')
    problem.add_equation('dtrace(u)-R@U=0')
    problem.add_equation('U+R.T@ntrace(u)=g')
    problem.build_solver(network_solver='dtn').solve()
    assert comm.allreduce(float(np.max(np.abs(u['g']-value),initial=0)),op=MPI.MAX)<1e-10
    integral=d3.Integrate(u).evaluate()['g'].item()
    assert abs(integral-value*lengths.sum())<1e-10


@pytest.mark.parametrize('fault',['owners','ids','root'])
def test_bad_root_input_is_collective(fault):
    comm=MPI.COMM_WORLD
    graph=None;owners=None
    if comm.rank==0:
        graph=d3.Network(endpoints=[(0,1),(1,2)],lengths=[1,1],
                         vertex_ids=('a','b','c') if fault=='ids' else None)
        owners=[0] if fault=='owners' else [0,comm.size-1]
    root=-1 if fault=='root' else 0
    with pytest.raises(RuntimeError) as err:
        d3.Network.from_partition(graph,owners,comm=comm,root=root)
    assert len(set(comm.allgather(str(err.value))))==1


def test_explicit_full_input_compatibility():
    net=d3.Network(endpoints=[(0,1),(1,2)],lengths=[1,1])
    comm=MPI.COMM_WORLD
    dist=d3.Distributor(net,comm=comm,dtype=np.float64,network_partition=[0,comm.size-1])
    assert dist.network_vertex_storage=='replicated'
    np.testing.assert_array_equal(net.endpoints,[[0,1],[1,2]])
