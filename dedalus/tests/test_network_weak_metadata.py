"""Local topology maps and compact metadata retain the public numerical contract."""
import pickle

import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.network import _network_id_signature


def test_default_ids_match_explicit_ids_and_roundtrip():
    net = d3.Network(endpoints=((0, 1), (1, 2)), lengths=(1, 2))
    assert net.vertex_ids == (0, 1, 2)
    assert net.edge_ids == (0, 1)
    assert net.vertex_ids[-1] == 2
    assert net.vertex_ids[1:] == (1, 2)
    assert _network_id_signature(net.vertex_ids) == _network_id_signature((0, 1, 2))
    assert _network_id_signature(net.vertex_ids) != _network_id_signature((0, True, 2))
    restored = pickle.loads(pickle.dumps(net))
    assert restored.vertex_ids == net.vertex_ids
    partition = d3.NetworkPartition(net, (0, 2), num_parts=4)
    restored_partition = pickle.loads(pickle.dumps(partition))
    for name in ('edges', 'owned_vertices', 'resident_vertices', 'ghost_vertices'):
        for old, new in zip(getattr(partition, name), getattr(restored_partition, name)):
            np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize('bucket', [None, (0, 2)])
@pytest.mark.parametrize('selection', ['scrambled', 'empty'])
def test_local_topology_maps_match_dense_reference(bucket, selection):
    net = d3.Network(endpoints=((0, 1), (1, 1), (1, 2)), lengths=(.7, 2., 1.3))
    basis = d3.NetworkBasis(net, size=12, edge_indices=bucket)
    edges = basis.edge_indices
    n = len(edges)
    left = np.zeros((2*n, n)); right = left.copy()
    left[2*np.arange(n), np.arange(n)] = 1
    right[2*np.arange(n)+1, np.arange(n)] = 1
    restriction = np.zeros((2*n, 3))
    restriction[np.arange(2*n), net.endpoints[edges].ravel()] = 1
    inv = np.diag(1/net.lengths[edges])
    maps = [(basis.R, restriction), (basis._left_trace, left),
            (basis._right_trace, right), (basis.edge_to_ports, left+right),
            (basis._left_neumann, -left@inv), (basis._right_neumann, right@inv),
            (basis._inverse_lengths, inv)]
    for operator, expected in maps:
        for op, ref in [(operator, expected), (operator.T, expected.T)]:
            rows = np.array([ref.shape[0]-1, 0, 0]) if selection == 'scrambled' else np.array([], dtype=int)
            cols = np.array([ref.shape[1]-1, 0, 0])
            np.testing.assert_allclose(op.submatrix(rows, cols).toarray(), ref[np.ix_(rows, cols)])
            assert 'matrix' not in op.__dict__, 'Local selection materialized the global map'
        np.testing.assert_allclose(operator.matrix.toarray(), expected)
        assert not operator.matrix.data.flags.writeable


def test_partition_only_builds_requested_groups():
    endpoints = np.column_stack((np.arange(40), np.arange(1, 41)))
    net = d3.Network(endpoints=endpoints, lengths=np.ones(40))
    partition = d3.NetworkPartition(net, np.repeat(np.arange(10), 4))
    np.testing.assert_array_equal(partition.resident_vertices[3], np.arange(12, 17))
    assert set(partition.resident_vertices.cache) == {3}
    assert set(partition.incident_vertices.cache) == {3}
    assert set(partition.edges.cache) == {3}
    assert not partition.resident_vertices[3].flags.writeable
