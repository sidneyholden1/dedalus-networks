"""Metadata disagreement must fail on every rank before a ghost exchange."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.field import synchronize_network_vertex_inputs


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("change", ["name", "order", "elements", "duplicate"])
def test_vertex_input_identity_is_checked_at_each_boundary(change):
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")
    network = d3.Network(endpoints=((0, 1), (1, 2)), lengths=(1, 2))
    dist = d3.Distributor(
        network, dtype=np.float64,
        network_partition=(comm.size - 1, comm.size - 1),
        network_vertex_storage="owned_ghost",
    )
    basis = d3.NetworkBasis(network, size=6)
    first = dist.Field(name="first", bases=basis.vertices)
    second = dist.Field(name="second", bases=basis.vertices)
    fields = (first, second)
    synchronize_network_vertex_inputs(dist, fields)
    original = basis.vertices._network_elements
    try:
        if change == "duplicate":
            second.name = first.name
        elif comm.rank == comm.size - 1:
            if change == "name":
                first.name = "renamed"
            elif change == "order":
                fields = fields[::-1]
            else:
                basis.vertices._network_elements = original[::-1].copy()
        error_type = ValueError if change == "duplicate" else RuntimeError
        match = "unique names" if change == "duplicate" else "inconsistent.*input fields"
        with pytest.raises(error_type, match=match) as error:
            synchronize_network_vertex_inputs(dist, fields)
        assert len(set(comm.allgather(str(error.value)))) == 1
    finally:
        basis.vertices._network_elements = original
    assert dist.network_partition.resident_vertices[0].size == 0


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_batched_vertex_values_refresh_with_idle_root(dtype):
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip('four ranks required')
    network = d3.Network(endpoints=((0, 1), (1, 2)), lengths=(1, 2))
    dist = d3.Distributor(network, dtype=dtype,
                          network_partition=(1, comm.size-1),
                          network_vertex_storage='owned_ghost')
    basis = d3.NetworkBasis(network, size=6)
    fields = [dist.Field(name='a', bases=basis.vertices),
              dist.TensorField(network, name='b', bases=basis.vertices)]
    plan = dist.network_vertex_exchange_plan
    factor = 1 + .375j if np.issubdtype(dtype, np.complexfloating) else 1
    for step in range(3):
        for i, field in enumerate(fields):
            field.require_coeff_space()
            view = np.moveaxis(field.data, len(field.tensorsig), 0)
            view[...] = -999
            for pos in plan.owned_positions:
                view[pos] = factor*(step+1)*(i+1)*(plan.resident[pos]+1)
        synchronize_network_vertex_inputs(dist, fields)
        for i, field in enumerate(fields):
            view = np.moveaxis(field.data, len(field.tensorsig), 0)
            expected = factor*(step+1)*(i+1)*(plan.resident+1)
            np.testing.assert_allclose(view, np.broadcast_to(
                expected.reshape((-1,)+(1,)*(view.ndim-1)), view.shape))
