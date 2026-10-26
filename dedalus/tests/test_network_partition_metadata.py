"""Canonical partition metadata for arbitrary authoritative edge owners."""
import numpy as np
import pytest
from dedalus.core.network import Network, NetworkPartition, _network_id_signature


@pytest.mark.parametrize("parts", [1, 2, 7, 33])
@pytest.mark.parametrize("kind", ["chain", "hub", "multigraph", "isolated"])
def test_partition_metadata_matches_incidence_definition(parts, kind):
    rng = np.random.default_rng(9126)
    if kind == "chain":
        endpoints = np.column_stack((np.arange(20), np.arange(1, 21)))
    elif kind == "hub":
        endpoints = np.column_stack((np.zeros(20, dtype=int), np.arange(1, 21)))
    elif kind == "isolated":
        endpoints = np.tile((0, 0), (20, 1))
    else:
        endpoints = np.array([(0, 0), (0, 1), (0, 1), *rng.integers(0, 16, (17, 2))])
    owners = rng.integers(0, parts, size=len(endpoints))
    original = owners.copy()
    network = Network(endpoints=endpoints, lengths=np.ones(len(endpoints)), vertex_ids=range(25))
    partition = NetworkPartition(network, owners, num_parts=parts)
    incidence = [{int(owners[e]) for e, uv in enumerate(endpoints) if v in uv} for v in range(25)]
    interior = [next(iter(p)) if len(p) == 1 else -1 for p in incidence]
    condition = [min(p) if p else v % parts for v, p in enumerate(incidence)]
    assert np.array_equal(partition.interior_vertex_owners, interior)
    assert np.array_equal(partition.vertex_equation_owners, condition)
    assert np.array_equal(partition.separator_vertices, [v for v, p in enumerate(incidence) if len(p) != 1])
    assert np.array_equal(owners, original)
    assert np.array_equal(partition.edge_owners, original)
    for part in range(parts):
        edges = {e for e, p in enumerate(owners) if p == part}
        owned = {v for v, p in enumerate(condition) if p == part}
        incident = {int(v) for e in edges for v in endpoints[e]}
        resident = owned | incident
        expected = dict(edges=edges, owned_vertices=owned, incident_vertices=incident,
                        resident_vertices=resident, ghost_vertices=resident-owned,
                        interior_vertices={v for v, p in enumerate(interior) if p == part})
        for name, values in expected.items():
            actual = getattr(partition, name)[part]
            assert np.array_equal(actual, sorted(values)), (part, name)
            assert not actual.flags.writeable
    owners[:] = 0
    assert np.array_equal(partition.edge_owners, original)


@pytest.mark.parametrize("left,right", [
    ((1, 2), (2, 1)), ((1,), (1.0,)), ((True,), (1,)),
    (((1, 2),), (1, 2)), (("1,2",), (1, 2)),
    (("a", "bc"), ("ab", "c")), ((None,), ("null",)),
    (("λ",), ("\\u03bb",)), ((0.0,), (-0.0,)),
])
def test_collective_id_signature_retains_json_distinctions(left, right):
    assert _network_id_signature(left) != _network_id_signature(right)


def test_collective_id_signature_normalization_and_fallback():
    native = (1, 1.5, True, ("λ", None))
    numpy = (np.int64(1), np.float64(1.5), np.bool_(True), ("λ", None))
    assert _network_id_signature(native) == _network_id_signature(numpy)
    assert len(_network_id_signature(native)[1]) == 16
    for value in (object(), float("nan"), float("inf"), 1 << 64, (object(),)):
        assert _network_id_signature((value,)) == ("canonical-index-only", None)
