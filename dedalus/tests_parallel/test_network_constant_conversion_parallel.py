"""MPI tests for constant conversion onto distributed Network bases."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.evaluator import Evaluator


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("layout", ["c", "g"])
def test_distributed_network_convert_constant_cardinal(dtype, layout, monkeypatch):
    """Expand rank-zero constants onto every distributed Cardinal role."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(0.5, 1, 2),
    )
    owners = np.full(network.num_edges, comm.size - 1, dtype=int)
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=6)

    value = 2.5 if dtype == np.float64 else 2.5 - 0.75j
    source = dist.Field(name="source")
    # Replicated Network inputs are rank-zero authoritative.  Deliberately
    # poison every other replica to cover the synchronization boundary used by
    # direct FutureField evaluation.
    source["g"] = value if comm.rank == 0 else value + 100
    source.change_layout(layout)

    from dedalus.core.basis import ConvertConstantCardinal
    def forbid_global_matrix(*args):
        raise AssertionError("constant conversion allocated a global matrix")
    monkeypatch.setattr(ConvertConstantCardinal, '_full_matrix', forbid_global_matrix)

    targets = {
        "edges": basis.edges,
        "ports": basis.ports,
        "vertices": basis.vertices,
    }
    expected_sizes = {
        "edges": network.num_edges,
        "ports": network.num_ports,
        "vertices": network.num_vertices,
    }
    for role, target_basis in targets.items():
        converted = d3.Convert(source, target_basis).evaluate()
        assert converted.layout is dist.get_layout_object(layout)
        assert np.allclose(converted.data, value)
        assert np.allclose(
            converted.allgather_data(layout).ravel(),
            np.full(expected_sizes[role], value, dtype=dtype),
        )

    assert dist.network_partition.edges[0].size == 0
    assert dist.network_partition.edges[comm.size - 1].size == network.num_edges


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_root_vertex_map_and_handler_sync(dtype):
    """Synchronize root-only vertex/scalar inputs before map and task walks."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(0.5, 1, 2),
    )
    owners = np.full(network.num_edges, comm.size - 1, dtype=int)
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=6)

    vertex_values = np.arange(1, network.num_vertices + 1, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        vertex_values += 0.25j * vertex_values[::-1]
    U = dist.Field(name="U", bases=basis.vertices)
    if comm.rank == 0:
        U["g"] = vertex_values[:, None]
    else:
        U["c"] = -100

    restricted = (basis.restriction @ U).evaluate()
    owned_ports = np.flatnonzero(
        owners[network.port_edges] == comm.rank
    )
    expected_ports = vertex_values[network.endpoints.ravel()]
    assert np.allclose(restricted["g"].ravel(), expected_ports[owned_ports])
    assert np.allclose(U.allgather_data("g").ravel(), vertex_values)

    scalar = dist.Field(name="scalar")
    scalar["g"] = 3.5 if comm.rank == 0 else -200
    evaluator = Evaluator(dist, vars={})
    handler = evaluator.add_dictionary_handler()
    handler.add_task(
        d3.Convert(scalar, basis.edges),
        layout="g",
        name="converted",
    )
    evaluator.evaluate_handlers([handler])
    converted = handler["converted"]
    assert np.allclose(converted.data, 3.5)
    assert np.allclose(
        converted.allgather_data("g").ravel(),
        np.full(network.num_edges, 3.5, dtype=dtype),
    )

    profile = dist.Field(name="profile", bases=basis.interval)
    s = dist.local_grid(basis.interval)
    expected_profile = 1 + 2 * s
    if comm.rank == 0:
        profile["g"] = expected_profile
    else:
        profile["c"] = -300
    expanded_profile = d3.Convert(profile, basis.edges).evaluate()
    assert np.allclose(expanded_profile["g"], expected_profile)
    assert dist.network_partition.edges[0].size == 0


@pytest.mark.mpi(min_size=2)
def test_distributed_network_replica_order_validation():
    """Fail collectively before differently ordered broadcasts can cross."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=np.array((comm.size - 1,)),
    )
    first = dist.Field(name="first")
    second = dist.Field(name="second")
    fields = (first, second) if comm.rank == 0 else (second, first)
    with pytest.raises(RuntimeError, match="inconsistent replicated input"):
        dist.synchronize_network_replicas(fields)

    unnamed_first = dist.Field()
    unnamed_second = dist.Field()
    with pytest.raises(ValueError, match="unique names"):
        dist.synchronize_network_replicas((unnamed_first, unnamed_second))
