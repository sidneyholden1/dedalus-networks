"""MPI tests for gathering replicated Network vertex fields."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.mpi(min_size=2)
def test_network_vertex_gather_uses_rank_zero_for_nonzero_root():
    """Return the authoritative rank-zero replica on any requested root."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(1, 1, 1),
    )
    owners = np.full(network.num_edges, comm.size - 1, dtype=int)
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=6)
    vertex = dist.Field(name="vertex", bases=basis.vertices)

    authoritative = np.arange(network.num_vertices, dtype=np.float64) + 0.25
    rank_values = authoritative + 100 * comm.rank
    vertex["g"] = rank_values.reshape(vertex["g"].shape)

    requested_root = comm.size - 1
    gathered = vertex.gather_data(root=requested_root, layout="g")
    if comm.rank == requested_root:
        assert np.array_equal(gathered.ravel(), authoritative)
    else:
        assert gathered is None
