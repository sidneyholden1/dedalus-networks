"""Frozen ownership from the original traversal, including weighted ties."""

import json
from pathlib import Path

import numpy as np
import pytest

from dedalus.core.network import Network, NetworkPartition
from scipy import sparse


REFERENCE = json.loads(
    Path(__file__).with_name("network_partition_reference.json").read_text()
)
FRONTIER_REFERENCE = json.loads(
    Path(__file__).with_name("network_partition_frontier_reference.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"] + FRONTIER_REFERENCE["cases"])
def test_connected_partition_preserves_reference_owners(case):
    network = Network(
        endpoints=case["endpoints"], lengths=np.ones(len(case["endpoints"])),
        vertex_ids=range(case["vertices"]),
    )
    partition = NetworkPartition.connected(network, case["parts"], case["weights"])
    assert np.array_equal(partition.edge_owners, case["owners"])


@pytest.mark.parametrize("parts", [2, 7, 31, 85])
@pytest.mark.parametrize("kind", ["chain", "hub", "multigraph", "disconnected"])
def test_bounded_seed_search_matches_unbounded_distances(monkeypatch, parts, kind):
    """Check pruning against full searches, including extreme weights and ties."""
    rng = np.random.default_rng(20260921)
    if kind == "chain":
        endpoints = np.column_stack((np.arange(80), np.arange(1, 81)))
    elif kind == "hub":
        endpoints = np.column_stack((np.zeros(80, dtype=int), np.arange(1, 81)))
    else:
        endpoints = rng.integers(0, 35, size=(80, 2))
        if kind == "disconnected":
            endpoints[40:] += 40
    network = Network(endpoints=endpoints, lengths=np.ones(len(endpoints)))
    weights = rng.choice([0., .001, 1., 1., 1e6], size=len(endpoints))
    dijkstra = sparse.csgraph.dijkstra
    limits = []

    def full_search(*args, **kwargs):
        limits.append(kwargs.pop("limit", np.inf))
        return dijkstra(*args, **kwargs)

    candidate = NetworkPartition.connected(network, parts, weights)
    with monkeypatch.context() as patch:
        patch.setattr(sparse.csgraph, "dijkstra", full_search)
        reference = NetworkPartition.connected(network, parts, weights)
    assert np.array_equal(candidate.edge_owners, reference.edge_owners)
    assert np.array_equal(candidate.separator_vertices, reference.separator_vertices)
    assert limits[:2] == [np.inf, np.inf]
    assert any(np.isfinite(radius) for radius in limits[2:]) or kind == "disconnected" and parts == 2
