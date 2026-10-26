"""Focused tests for package-neutral Network graph ingestion."""

import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core import network as network_core


class ProtocolGraph:
    def __init__(self, data):
        self.data = data

    def __dedalus_network__(self):
        return self.data


def test_protocol_accepts_minimal_mapping_and_snapshots_data():
    endpoints = np.array(((0, 1),), dtype=int)
    lengths = np.array((2.5,))
    network = d3.Network(
        ProtocolGraph(
            {
                "endpoints": endpoints,
                "lengths": lengths,
            }
        )
    )

    endpoints[0, 0] = 1
    lengths[0] = 9

    assert network.vertex_ids == (0, 1)
    assert network.edge_ids == (0,)
    assert np.array_equal(network.endpoints, ((0, 1),))
    assert np.array_equal(network.lengths, (2.5,))


@pytest.mark.parametrize("missing", ["endpoints", "lengths"])
def test_protocol_reports_missing_required_data(missing):
    data = {
        "endpoints": ((0, 1),),
        "lengths": (1,),
    }
    data.pop(missing)

    with pytest.raises(
        ValueError,
        match=rf"missing required.*{missing}",
    ):
        d3.Network(ProtocolGraph(data))


def test_protocol_attribute_must_be_callable():
    class Graph:
        __dedalus_network__ = {
            "endpoints": ((0, 1),),
            "lengths": (1,),
        }

    with pytest.raises(TypeError, match="must be a callable method"):
        d3.Network(Graph())


def test_protocol_network_preserves_or_explicitly_replaces_coordinates():
    snapshot = d3.Network(
        endpoints=((0, 1),),
        lengths=(2,),
        coordinate_names=("branch", "xi"),
    )
    wrapped = ProtocolGraph(snapshot)

    assert d3.Network(wrapped) is snapshot

    renamed = d3.Network(wrapped, coordinate_names=("edge", "s"))
    assert renamed is not snapshot
    assert renamed.names == ("edge", "s")
    assert np.array_equal(renamed.endpoints, snapshot.endpoints)
    assert np.array_equal(renamed.lengths, snapshot.lengths)

    directly_renamed = d3.Network(
        snapshot,
        coordinate_names=("arc", "reference"),
    )
    assert directly_renamed is not snapshot
    assert directly_renamed.names == ("arc", "reference")


def test_protocol_mapping_coordinates_are_preserved_or_overridden():
    wrapped = ProtocolGraph(
        {
            "endpoints": ((0, 1),),
            "lengths": (2,),
            "coordinate_names": ("branch", "xi"),
        }
    )

    preserved = d3.Network(wrapped)
    overridden = d3.Network(
        wrapped,
        coordinate_names=("edge", "s"),
    )

    assert preserved.names == ("branch", "xi")
    assert overridden.names == ("edge", "s")


def test_protocol_and_explicit_arrays_do_not_import_optional_packages(
    monkeypatch,
):
    def unexpected_import(name):
        raise AssertionError(f"unexpected optional import {name!r}")

    monkeypatch.setattr(
        network_core.importlib,
        "import_module",
        unexpected_import,
    )
    explicit = d3.Network(endpoints=((0, 1),), lengths=(1,))
    adapted = d3.Network(
        ProtocolGraph(
            {
                "endpoints": ((0, 1),),
                "lengths": (1,),
            }
        )
    )

    assert explicit.num_edges == adapted.num_edges == 1


def test_package_module_name_alone_does_not_select_an_adapter():
    impostor_type = type("Graph", (), {"__module__": "networkx.not_a_graph"})

    with pytest.raises(TypeError, match="Unsupported graph object"):
        d3.Network(impostor_type())


@pytest.mark.parametrize("directed", [False, True])
def test_networkx_multigraph_preserves_ports_keys_and_lengths(
    directed,
    monkeypatch,
):
    networkx = pytest.importorskip("networkx")
    graph_type = networkx.MultiDiGraph if directed else networkx.MultiGraph

    class DerivedGraph(graph_type):
        pass

    graph = DerivedGraph()
    graph.add_nodes_from(("right", ("junction", 1), "left", "isolated"))
    graph.add_edge(
        "right",
        ("junction", 1),
        key="wide",
        distance=2.5,
    )
    graph.add_edge(
        "right",
        ("junction", 1),
        key=("narrow", 2),
        distance=1.25,
    )
    graph.add_edge("left", "left", key="loop", distance=3.0)
    if directed:
        graph.add_edge(
            ("junction", 1),
            "right",
            key="reverse",
            distance=4.5,
        )

    records = tuple(graph.edges(keys=True, data=True))
    vertex_ids = tuple(graph.nodes)
    vertex_index = {
        vertex_id: index
        for index, vertex_id in enumerate(vertex_ids)
    }
    imported = []
    import_module = network_core.importlib.import_module

    def record_import(name):
        imported.append(name)
        return import_module(name)

    monkeypatch.setattr(
        network_core.importlib,
        "import_module",
        record_import,
    )
    network = d3.Network(graph, length="distance")

    assert imported == ["networkx"]
    assert network.vertex_ids == vertex_ids
    assert network.edge_ids == tuple(
        (tail, head, key)
        for tail, head, key, _ in records
    )
    assert np.array_equal(
        network.endpoints,
        tuple(
            (vertex_index[tail], vertex_index[head])
            for tail, head, _, _ in records
        ),
    )
    assert np.array_equal(
        network.lengths,
        tuple(data["distance"] for *_, data in records),
    )
    assert len(set(network.edge_ids)) == graph.number_of_edges()

    first = records[0]
    graph.edges[first[0], first[1], first[2]]["distance"] = 99
    assert 99 not in network.lengths


def test_networkx_missing_requested_length_is_actionable():
    networkx = pytest.importorskip("networkx")
    graph = networkx.MultiGraph()
    graph.add_edge("left", "right", key="branch", weight=2)

    with pytest.raises(
        ValueError,
        match=r"NetworkX edge.*'distance' length attribute",
    ):
        d3.Network(graph, length="distance")


def test_metricgraph_adapter_preserves_stable_snapshot(monkeypatch):
    metricgraph = pytest.importorskip("metricgraph")
    graph = metricgraph.MetricGraph(
        vertices=((0, 0), (2, 0)),
        edges=((1, 0), (1, 0)),
        lengths=(2, 3),
        vertex_ids=("left", "right"),
        groups={"boundary": (0, 1)},
    )
    imported = []
    import_module = network_core.importlib.import_module

    def record_import(name):
        imported.append(name)
        return import_module(name)

    monkeypatch.setattr(
        network_core.importlib,
        "import_module",
        record_import,
    )
    network = d3.Network(graph)

    assert imported == ["metricgraph"]
    assert network.vertex_ids == ("left", "right")
    assert network.edge_ids == (0, 1)
    assert np.array_equal(network.endpoints, ((1, 0), (1, 0)))
    assert np.array_equal(network.lengths, (2, 3))
    assert np.array_equal(network.vertex_groups["boundary"], (0, 1))

    graph.E.lengths = (4, 5)
    assert np.array_equal(network.lengths, (2, 3))
