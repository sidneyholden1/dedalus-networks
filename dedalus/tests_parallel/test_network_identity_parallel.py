"""MPI validation tests for Network semantic identity."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("storage", ["replicated", "distributed"])
@pytest.mark.parametrize(
    "mismatch",
    ["vertex_ids", "edge_ids", "coordinate_names", "vertex_groups"],
)
def test_network_identity_mismatch_fails_collectively(storage, mismatch):
    """Reject rank-local semantic identity before allocating fields."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    vertex_ids = ("left", "right")
    edge_ids = ("edge",)
    coordinate_names = ("e", "s")
    vertex_groups = {"boundary": (0,)}
    if comm.rank != 0:
        if mismatch == "vertex_ids":
            vertex_ids = ("west", "east")
        elif mismatch == "edge_ids":
            edge_ids = ("branch",)
        elif mismatch == "coordinate_names":
            coordinate_names = ("branch", "position")
        else:
            vertex_groups = {"boundary": (1,)}
    network = d3.Network(
        vertex_ids=vertex_ids,
        edge_ids=edge_ids,
        endpoints=((0, 1),),
        lengths=(1,),
        vertex_groups=vertex_groups,
        coordinate_names=coordinate_names,
    )
    partition = None if storage == "replicated" else (comm.size - 1,)
    stage = (
        "Distributed Network topology validation"
        if storage == "distributed"
        else "Network topology validation"
    )
    with pytest.raises(RuntimeError, match=stage):
        d3.Distributor(
            network,
            dtype=np.float64,
            network_partition=partition,
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("storage", ["replicated", "distributed"])
def test_nonportable_network_ids_keep_canonical_fallback(storage):
    """Opaque rank-local IDs are never serialized for identity validation."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    class OpaqueID:
        pass

    network = d3.Network(
        vertex_ids=(OpaqueID(), OpaqueID()),
        edge_ids=(OpaqueID(),),
        endpoints=((0, 1),),
        lengths=(1,),
    )
    partition = None if storage == "replicated" else (comm.size - 1,)
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=partition,
    )

    assert dist.network is network
    assert dist.distributed_network == (storage == "distributed")
