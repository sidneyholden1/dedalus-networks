"""Grouped interface validation must fail collectively, including idle ranks."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.network import _ResidentInterfaceDofs
from dedalus.core.network import _distributed_label_max


@pytest.mark.parametrize("fault", [None, "duplicate", "gap", "count", "scalar"])
def test_grouped_resident_validation_is_collective(fault):
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four ranks, including two idle ranks")
    network = d3.Network(endpoints=((0, 1), (1, 2)), lengths=(1, 2))
    partition = d3.NetworkPartition(network, (0, comm.size - 1), num_parts=comm.size)
    keys = [(int(vertex), occurrence)
            for vertex in partition.resident_vertices[comm.rank]
            for occurrence in (0, 1)] + [(3, 0)]
    keys.reverse()
    if comm.rank == comm.size - 1:
        if fault == "duplicate":
            keys.append((1, 0))
        elif fault == "gap":
            keys[keys.index((1, 1))] = (1, 2)
        elif fault == "count":
            keys.remove((1, 1))
    if fault == "scalar" and comm.rank == 1:
        keys.remove((3, 0))
    keys = np.asarray(keys, dtype=np.int64).reshape((-1, 2))
    if fault is not None:
        with pytest.raises(RuntimeError, match="Resident Network") as error:
            _ResidentInterfaceDofs(keys, partition, comm)
        assert len(set(comm.allgather(str(error.value)))) == 1
    else:
        dofs = _ResidentInterfaceDofs(keys, partition, comm)
        expected = (10 * keys[:, 0] + keys[:, 1]).astype(complex) * (1 + 2j)
        values = np.where(dofs.owned, expected, -999)
        dofs.forward(values)
        assert np.array_equal(values, expected)
        dofs.reduce_add(np.full((len(keys), 2), comm.rank + 1, dtype=complex))


def test_sparse_label_reduction_with_large_labels_and_idle_rank():
    comm = MPI.COMM_WORLD
    labels = np.array([10**12 + comm.rank, 10**15], dtype=np.int64)
    values = np.array([comm.rank + 7, comm.rank + 1], dtype=np.int64)
    if comm.rank == 0:
        labels, values = labels[:0], values[:0]
    result = _distributed_label_max(comm, labels, values)
    expected = dict()
    for group in comm.allgather(list(zip(labels.tolist(), values.tolist()))):
        for label, value in group:
            expected[label] = max(value, expected.get(label, 0))
    np.testing.assert_array_equal(result, [expected[int(label)] for label in labels])
