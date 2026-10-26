"""Independent sparse-reduction oracle and safe exchange-plan reuse."""
from mpi4py import MPI
import numpy as np
import pytest
import dedalus.public as d3
from dedalus.core.network import _distributed_label_max, _ResidentInterfaceDofs
from dedalus.core.network import NetworkVertexExchangePlan
from dedalus.core.network import _grouped_label_reduce


@pytest.mark.parametrize('empty', [False, True])
def test_grouped_sparse_reduction_with_multiple_leaders(empty):
    comm = MPI.COMM_WORLD
    # Two ranks per group exercises the inter-leader exchange with four ranks,
    # independently of whether the Slurm allocation spans physical nodes.
    group = comm.Split(comm.rank//2, key=comm.rank)
    labels = np.array([10**15, comm.rank % 2, 10**15], dtype=np.int64)
    values = np.array([[comm.rank+1, comm.rank], [4, 3], [2, 8]], dtype=np.int64)
    if empty or comm.rank == 0:
        labels, values = labels[:0], values[:0]
    try:
        actual = _grouped_label_reduce(comm, group, labels, values, ('sum', 'max'))
    finally:
        group.Free()
    oracle = {}
    for entries in comm.allgather(list(zip(labels.tolist(), values.tolist()))):
        for label, row in entries:
            before = oracle.setdefault(label, [0, 0])
            before[0] += row[0]
            before[1] = max(before[1], row[1])
    expected = np.array([oracle[int(label)] for label in labels], dtype=np.int64).reshape((-1, 2))
    np.testing.assert_array_equal(actual, expected)

@pytest.mark.parametrize('empty',[False,True])
def test_batched_sparse_reduction(empty):
    comm=MPI.COMM_WORLD
    labels=np.array([10**15,10**12+comm.rank,10**15],dtype=np.int64)
    values=np.array([[comm.rank+1,comm.rank,7],[3,4,1],[2,9,5]],dtype=np.int64)
    if empty or (comm.size>1 and comm.rank==0):labels,values=labels[:0],values[:0]
    actual=_distributed_label_max(comm,labels,values,operation=('sum','max','sum'))
    oracle={}
    for group in comm.allgather(list(zip(labels.tolist(),values.tolist()))):
        for label,row in group:
            before=oracle.setdefault(label,[0,0,0])
            before[0]+=row[0];before[1]=max(before[1],row[1]);before[2]+=row[2]
    expected=np.array([oracle[int(k)] for k in labels],dtype=np.int64).reshape((-1,3))
    np.testing.assert_array_equal(actual,expected)
    for col,op in enumerate(('sum','max','sum')):
        np.testing.assert_array_equal(actual[:,col],_distributed_label_max(comm,labels,values[:,col],operation=op))

@pytest.mark.parametrize('fault',[None,'partition','communicator'])
def test_reused_plan_exchange_and_collective_validation(fault):
    comm=MPI.COMM_WORLD
    net=d3.Network(endpoints=((0,1),(1,2)),lengths=(1,2))
    partition=d3.NetworkPartition(net,(0,comm.size-1),num_parts=comm.size)
    plan=NetworkVertexExchangePlan(partition,comm)
    keys=np.array([(int(v),k) for v in partition.resident_vertices[comm.rank] for k in (0,1)]+[(3,0)],dtype=np.int64)
    duplicate=None
    if fault=='partition' and comm.rank==0:
        object.__setattr__(plan,'partition',d3.NetworkPartition(net,(0,comm.size-1),num_parts=comm.size))
    if fault=='communicator':
        duplicate=comm.Dup();object.__setattr__(plan,'comm',duplicate)
    try:
        if fault:
            with pytest.raises(RuntimeError) as err:_ResidentInterfaceDofs(keys,partition,comm,exchange=plan)
            assert len(set(comm.allgather(str(err.value))))==1
        else:
            dofs=_ResidentInterfaceDofs(keys,partition,comm,exchange=plan)
            assert dofs.exchange is plan
            expected=(10*keys[:,0]+keys[:,1])*(1+2j)
            values=np.where(dofs.owned,expected,-999)
            dofs.forward(values)
            np.testing.assert_array_equal(values,expected)
            data=np.full((len(keys),2),comm.rank+1,dtype=complex)
            dofs.reduce_add(data)
            oracle={}
            for group in comm.allgather([(tuple(k),comm.rank+1) for k in keys]):
                for key,value in group:oracle[key]=oracle.get(key,0)+value
            np.testing.assert_array_equal(data[dofs.owned,0],[oracle[tuple(k)] for k in keys[dofs.owned]])
    finally:
        if duplicate is not None:duplicate.Free()
