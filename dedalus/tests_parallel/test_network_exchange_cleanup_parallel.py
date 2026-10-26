"""Boundary exchange against a gathered oracle, including idle ranks."""
import numpy as np
import pytest
from mpi4py import MPI
import dedalus.public as d3
from dedalus.core.network import NetworkVertexExchangePlan


def make_plan():
    comm = MPI.COMM_WORLD
    network = d3.Network(endpoints=((0,1),(0,2),(0,3),(3,4),(0,0),(0,1)),
                         lengths=(1,1,1,1,1,1))
    # Rank zero receives overlapping hub contributions from two neighbours;
    # rank two is idle in the four-rank test. Include loops and parallel edges.
    other = min(1, comm.size-1)
    owners = (0, other, comm.size-1, comm.size-1, 0, other)
    partition = d3.NetworkPartition(network, owners, num_parts=comm.size)
    return NetworkVertexExchangePlan(partition, comm)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('refresh', [False, True])
def test_changing_payloads_noncontiguous_and_additive_oracle(dtype, refresh):
    plan = make_plan()
    comm = plan.comm
    vertices = plan.resident
    for width in (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 1, 3):
        # Negative exchange axis and noncontiguous component storage.
        storage = np.full((2, len(vertices), 2*width), -999, dtype=dtype)
        data = storage[..., ::2]
        exact = (100*vertices[None,:,None] +
                 np.arange(2)[:,None,None]*10 + np.arange(width)[None,None,:]).astype(dtype)
        if dtype == np.complex128:
            exact += 1j*(exact+3)
        data[:,plan.owned_positions,:] = exact[:,plan.owned_positions,:]
        assert plan.forward(data, axis=-2) is data
        np.testing.assert_array_equal(data, exact)
        np.testing.assert_array_equal(storage[...,1::2], -999)

        data[...] = exact + comm.rank+1
        before = data.copy()
        gathered = comm.allgather((vertices.tolist(), before))
        oracle = {}
        for labels, values in gathered:
            for i, vertex in enumerate(labels):
                oracle[vertex] = oracle.get(vertex, np.zeros((2,width),dtype=dtype)) + values[:,i,:]
        assert plan.reduce_add(data, axis=-2, refresh=refresh) is data
        owned = set(plan.owned.tolist())
        for i, vertex in enumerate(vertices):
            target = oracle[vertex] if refresh or vertex in owned else before[:,i,:]
            np.testing.assert_array_equal(data[:,i,:], target)
        np.testing.assert_array_equal(storage[...,1::2], -999)
        assert len(plan._payload_layouts) <= 8
        assert all(not array.flags.writeable
                   for layout in plan._payload_layouts.values() for array in layout)


def test_scalar_payload_reuses_plan_across_dtypes():
    plan = make_plan()
    for dtype in (np.float64, np.complex128, np.float64):
        exact = (plan.resident+1).astype(dtype)
        if dtype == np.complex128:exact *= 1+2j
        data = np.full(plan.resident.size, -999, dtype=dtype)
        data[plan.owned_positions] = exact[plan.owned_positions]
        plan.forward(data)
        np.testing.assert_array_equal(data, exact)


@pytest.mark.parametrize('fault', ['shape', 'dtype'])
def test_payload_validation_remains_collective_after_cached_exchange(fault):
    plan = make_plan();comm = plan.comm
    if comm.size < 2:pytest.skip('requires distinct rank payloads')
    plan.forward(np.zeros((plan.resident.size,2)))
    width = 3 if fault=='shape' and comm.rank==1 else 2
    dtype = np.complex128 if fault=='dtype' and comm.rank==1 else np.float64
    with pytest.raises(RuntimeError) as error:
        plan.forward(np.zeros((plan.resident.size,width),dtype=dtype))
    assert len(set(comm.allgather(str(error.value))))==1
    # Failure did not invalidate the plan or skip a later collective.
    plan.forward(np.zeros((plan.resident.size,2)))
