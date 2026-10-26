"""Tests for immutable named vertex sets on Networks."""

import numpy as np
import pytest

import dedalus.public as d3


def _network(vertex_groups):
    return d3.Network(
        vertex_ids=("left", "middle", "right"),
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
        vertex_groups=vertex_groups,
    )


def test_vertex_groups_normalize_indices_masks_and_duplicates():
    indices = np.array((2, 0, 2))
    mask = np.array((False, True, False))
    network = _network(
        {
            "boundary": indices,
            "interior": mask,
            "integral_floats": (2.0, 0.0),
            "real_complex": (2 + 0j, 0 + 0j),
            "empty": (),
        }
    )

    indices[:] = 1
    mask[:] = False

    assert np.array_equal(network.vertex_groups["boundary"], (0, 2))
    assert np.array_equal(network.vertex_groups["interior"], (1,))
    assert np.array_equal(network.vertex_groups["integral_floats"], (0, 2))
    assert np.array_equal(network.vertex_groups["real_complex"], (0, 2))
    assert np.array_equal(network.vertex_groups["empty"], ())
    assert network.vertex_groups["boundary"].dtype == np.int64


def test_vertex_groups_are_immutable_snapshots():
    network = _network({"boundary": (0, 2)})
    group = network.vertex_groups["boundary"]

    assert not group.flags.writeable
    with pytest.raises(ValueError):
        group[0] = 1
    with pytest.raises(TypeError):
        network.vertex_groups["new"] = np.array((1,))


@pytest.mark.parametrize(
    "members,match",
    [
        pytest.param((0, 1.5), "finite integer", id="fractional"),
        pytest.param((0, np.nan), "finite integer", id="nan"),
        pytest.param((0, np.inf), "finite integer", id="infinite"),
        pytest.param((0, 1 + 1j), "finite integer", id="complex"),
        pytest.param(("0", "1"), "finite integer", id="strings"),
        pytest.param(((0, 1),), "one-dimensional", id="matrix"),
        pytest.param(1, "one-dimensional", id="scalar"),
        pytest.param((True, False), "must have shape", id="short-mask"),
        pytest.param((-1,), "invalid indices", id="negative"),
        pytest.param((3,), "invalid indices", id="out-of-range"),
    ],
)
def test_vertex_groups_reject_invalid_members(members, match):
    with pytest.raises(ValueError, match=match):
        _network({"bad": members})


def test_vertex_groups_reject_mixed_boolean_indices():
    with pytest.raises(ValueError, match="finite integer"):
        _network({"bad": (True, 2)})


def test_vertex_group_names_are_strings_without_normalized_collisions():
    with pytest.raises(TypeError, match="names must be strings"):
        _network({1: (0,), "1": (1,)})

    with pytest.raises(TypeError, match="vertex_groups must be a mapping"):
        _network((("boundary", (0, 2)),))


def test_protocol_vertex_groups_use_the_same_normalization():
    class Graph:
        def __dedalus_network__(self):
            return {
                "vertex_ids": ("left", "middle", "right"),
                "endpoints": ((0, 1), (1, 2)),
                "lengths": (1, 1),
                "vertex_groups": {
                    "boundary": (True, False, True),
                    "repeated": (2, 0, 2),
                },
            }

    network = d3.Network(Graph())

    assert np.array_equal(network.vertex_groups["boundary"], (0, 2))
    assert np.array_equal(network.vertex_groups["repeated"], (0, 2))


def test_vertex_group_indices_set_vertex_condition_fields():
    network = _network({"dirichlet": (0, 2)})
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)

    alpha["g"] = 0
    beta["g"] = 1
    dirichlet = network.vertex_groups["dirichlet"]
    alpha["g"][dirichlet] = 1
    beta["g"][dirichlet] = 0

    assert np.array_equal(alpha["g"].ravel(), (1, 0, 1))
    assert np.array_equal(beta["g"].ravel(), (0, 1, 0))


def test_metricgraph_vertex_groups_are_snapshotted():
    metricgraph = pytest.importorskip("metricgraph")
    graph = metricgraph.MetricGraph(
        vertices=((0, 0), (1, 0), (2, 0)),
        edges=((0, 1), (1, 2)),
        groups={"boundary": (2, 0, 2)},
    )
    network = d3.Network(graph)

    graph.groups["boundary"] = (1,)

    assert np.array_equal(network.vertex_groups["boundary"], (0, 2))
    assert not network.vertex_groups["boundary"].flags.writeable
