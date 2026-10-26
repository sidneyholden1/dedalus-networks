"""Serial contracts for owned/ghost Network vertex storage."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.network import NetworkVertexExchangePlan


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="Serial owned/ghost contracts are covered here.",
)


def test_network_partition_vertex_storage_sets():
    network = d3.Network(
        vertex_ids=range(5),
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(1, 1, 1),
    )
    partition = d3.NetworkPartition(
        network,
        edge_owners=(3, 1, 3),
        num_parts=4,
    )

    expected_owned = ((4,), (1, 2), (), (0, 3))
    expected_incident = ((), (1, 2), (), (0, 1, 2, 3))
    expected_resident = ((4,), (1, 2), (), (0, 1, 2, 3))
    expected_ghost = ((), (), (), (1, 2))
    for part in range(4):
        assert np.array_equal(
            partition.owned_vertices[part],
            expected_owned[part],
        )
        assert np.array_equal(
            partition.incident_vertices[part],
            expected_incident[part],
        )
        assert np.array_equal(
            partition.resident_vertices[part],
            expected_resident[part],
        )
        assert np.array_equal(
            partition.ghost_vertices[part],
            expected_ghost[part],
        )
        assert not partition.resident_vertices[part].flags.writeable
    assert np.array_equal(
        partition.vertex_equation_owners,
        (3, 1, 1, 3, 0),
    )
    assert np.array_equal(partition.separator_vertices, (1, 2, 4))


def test_network_vertex_storage_option_validation_and_auto_policy():
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))

    with pytest.raises(ValueError, match="requires a network_partition"):
        d3.Distributor(network, network_vertex_storage="owned_ghost")
    with pytest.raises(ValueError, match="must be 'replicated'"):
        d3.Distributor(network, network_vertex_storage="invalid")

    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(0,),
        network_vertex_storage="auto",
    )
    assert dist.requested_network_vertex_storage == "auto"
    assert dist.network_vertex_storage == "owned_ghost"
    assert dist.network_vertex_exchange_plan is not None


def test_network_vertex_exchange_plan_is_immutable():
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    partition = d3.NetworkPartition(network, (0,), num_parts=1)
    plan = NetworkVertexExchangePlan(partition, MPI.COMM_SELF)

    with pytest.raises(AttributeError, match="immutable"):
        plan.rank = 1
    with pytest.raises(ValueError):
        plan.owned[0] = 10
