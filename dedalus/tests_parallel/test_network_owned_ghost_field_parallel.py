"""MPI Field, evaluation, and restart tests for owned/ghost vertices."""

import pathlib
import shutil
import tempfile

import h5py
from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.evaluator import Evaluator


def _setup(dtype, owners):
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(1, 1.5, 2),
    )
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=np.asarray(owners, dtype=int),
        network_vertex_storage="owned_ghost",
    )
    basis = d3.NetworkBasis(network, size=5)
    field = dist.TensorField(
        network,
        name="U",
        bases=basis.vertices,
    )
    return network, dist, basis, field


def _global_values(field, offset=0):
    shape = (
        tuple(coordsys.dim for coordsys in field.tensorsig)
        + field.layout.global_shape(field.domain, field.scales)
    )
    values = np.full(shape, offset + 0.125, dtype=np.float64)
    for axis, indices in enumerate(
        np.meshgrid(
            *(np.arange(size) for size in shape),
            indexing="ij",
        )
    ):
        values += (axis + 1) * indices
    values = values.astype(field.dtype)
    if np.issubdtype(field.dtype, np.complexfloating):
        values += 1j * (0.25 * values + 3)
    return values


def _vertex_owner_offsets(field, values, scale=100):
    owners = field.dist.network_partition.vertex_equation_owners
    shape = (1,) * len(field.tensorsig) + (owners.size, 1)
    return values + scale * owners.reshape(shape)


def _local_global_data(field, global_data):
    elements = field.layout.local_elements(field.domain, field.scales)
    index = (slice(None),) * len(field.tensorsig) + np.ix_(*elements)
    return global_data[index]


def _corrupt_ghosts(field, value):
    positions = field.dist.network_vertex_exchange_plan.ghost_positions
    index = [slice(None)] * field.data.ndim
    index[len(field.tensorsig)] = positions
    field.data[tuple(index)] = value


def _shared_tempdir(comm):
    path = (
        tempfile.mkdtemp(prefix="dedalus-network-owned-ghost-")
        if comm.rank == 0
        else None
    )
    return pathlib.Path(comm.bcast(path, root=0))


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_owned_ghost_field_global_assembly_loads_norms_and_evaluation(dtype):
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    _, dist, _, U = _setup(dtype, (3, 1, 3))
    base = _global_values(U)

    # Deliberately pass a different global array on every rank.  Owners retain
    # their authoritative values and set_global_data refreshes every ghost.
    U.set_global_data(base + 100 * comm.rank)
    expected = _vertex_owner_offsets(U, base)
    assert np.allclose(U.data, _local_global_data(U, expected))

    # Global assembly and norms count owners exactly once, even if a caller
    # has temporarily dirtied a resident ghost value.
    _corrupt_ghosts(U, -12345 + 17j if U.is_complex else -12345)
    assert np.allclose(U.allgather_data(), expected)
    gather_root = 2
    gathered = U.gather_data(root=gather_root)
    if comm.rank == gather_root:
        assert np.allclose(gathered, expected)
    else:
        assert gathered is None
    assert np.allclose(
        U.allreduce_data_norm(order=2),
        np.linalg.norm(expected.ravel()),
    )
    assert np.allclose(
        U.allreduce_data_norm(order=np.inf),
        np.max(np.abs(expected)),
    )

    # Future evaluation refreshes owned/ghost leaves before doing arithmetic.
    result = (2 * U).evaluate()
    assert np.allclose(result.data, 2 * _local_global_data(U, expected))

    # Evaluator entry points do the same even when the task itself is a Field.
    _corrupt_ghosts(U, -54321 + 9j if U.is_complex else -54321)
    evaluator = Evaluator(dist, {})
    handler = evaluator.add_dictionary_handler()
    handler.add_task(U, name="U", layout="c")
    evaluator.evaluate_handlers([handler], iteration=1)
    assert np.allclose(U.data, _local_global_data(U, expected))

    # Both direct global loaders refresh ghosts after applying a rank-local
    # callback, so the final resident copies follow the owner transformation.
    for load in (
        U.load_from_global_coeff_data,
        U.load_from_global_grid_data,
    ):
        load(base, func=lambda local: local + 100 * comm.rank)
        assert np.allclose(U.data, _local_global_data(U, expected))
        assert np.allclose(U.allgather_data(), expected)

    assert dist.network_partition.resident_vertices[0].size == 0
    assert dist.network_partition.resident_vertices[2].size == 0


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_multiple_vertex_interval_inputs_normalize_layouts_and_scales(dtype):
    """Evaluation safely normalizes distinct resident leaf layouts."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(1, 1.5, 2),
    )
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=np.array((3, 1, 3)),
        network_vertex_storage="owned_ghost",
    )
    basis = d3.NetworkBasis(network, size=5, dealias=2)
    A = dist.Field(
        name="vertex_profile_A",
        bases=(basis.vertices, basis.interval),
    )
    B = dist.Field(
        name="vertex_profile_B",
        bases=(basis.vertices, basis.interval),
    )

    vertices = np.arange(network.num_vertices)[:, None]
    modes = np.arange(basis.size)[None, :]
    expected_A = 1.25 + 2 * vertices + 0.125 * modes
    expected_B = -0.75 + 0.5 * vertices - 0.2 * modes
    expected_A = expected_A.astype(dtype)
    expected_B = expected_B.astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        expected_A += 1j * (0.3 * expected_A + 1)
        expected_B += 1j * (-0.4 * expected_B + 2)

    partition = dist.network_partition
    resident = partition.resident_vertices[comm.rank]
    owned = partition.owned_vertices[comm.rank]
    owned_positions = np.searchsorted(resident, owned)
    sentinel = -1234 + 23j if A.is_complex else -1234
    for field, expected in ((A, expected_A), (B, expected_B)):
        local = np.full(field["c"].shape, sentinel, dtype=dtype)
        local[owned_positions] = expected[owned]
        field["c"] = local

    # Give the two same-shaped inputs independent, rank-dependent states.
    # Before the fix, one direction raised MPI truncation and the reverse
    # direction silently copied an uninitialized receive-buffer tail.
    if comm.rank == 1:
        A.change_scales((1, 2))
        A.require_grid_space()
    if comm.rank == 3:
        B.change_scales((1, 3))
        B.require_grid_space()

    result = (2 * A - 3 * B).evaluate()
    expected_result = 2 * expected_A - 3 * expected_B

    for field, expected in ((A, expected_A), (B, expected_B)):
        assert field.layout is dist.coeff_layout
        assert field.scales == field.domain.dealias
        assert np.allclose(field.data, expected[resident], atol=2e-12)
        assert np.allclose(field.allgather_data("c"), expected, atol=2e-12)
    assert np.allclose(result.data, expected_result[resident], atol=3e-12)
    assert np.allclose(
        result.allgather_data("c"),
        expected_result,
        atol=3e-12,
    )
    assert dist.network_partition.resident_vertices[0].size == 0
    assert dist.network_partition.resident_vertices[2].size == 0


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_owned_ghost_hdf5_restart_after_repartition(dtype):
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    network, write_dist, write_basis, U = _setup(dtype, (3, 1, 3))
    expected = _global_values(U, offset=20)
    U.set_global_data(expected)
    _corrupt_ghosts(U, -9999 + 11j if U.is_complex else -9999)

    output_path = _shared_tempdir(comm)
    try:
        evaluator = Evaluator(write_dist, {})
        output = evaluator.add_file_handler(
            output_path,
            parallel="gather",
            iter=1,
            max_writes=1,
        )
        output.add_task(U, name="U", layout="g")
        evaluator.evaluate_handlers(
            [output],
            iteration=7,
            wall_time=1,
            sim_time=0.5,
            timestep=0.125,
        )
        comm.Barrier()

        metadata_error = None
        if comm.rank == 0:
            try:
                with h5py.File(output.current_file, mode="r") as file:
                    task = file["tasks/U"]
                    assert np.allclose(task[-1], expected)
                    metadata_path = task.attrs["network_basis_metadata"]
                    assert metadata_path in file
                    assert file[metadata_path].attrs["role"] == "vertices"
                    assert np.array_equal(
                        file[metadata_path + "/canonical_elements"][:],
                        np.arange(network.num_vertices),
                    )
            except Exception as error:
                metadata_error = repr(error)
        metadata_error = comm.bcast(metadata_error, root=0)
        assert metadata_error is None, metadata_error

        _, read_dist, _, restored = _setup(dtype, (1, 3, 1))
        restored["c"] = -777
        with h5py.File(output.current_file, mode="r") as file:
            restored.load_from_hdf5(file, -1, task="U")

        assert np.allclose(
            restored.data,
            _local_global_data(restored, expected),
        )
        assert np.allclose(restored.allgather_data(), expected)
        assert not np.array_equal(
            write_dist.network_partition.edge_owners,
            read_dist.network_partition.edge_owners,
        )
        assert read_dist.network_partition.resident_vertices[0].size == 0
        assert read_dist.network_partition.resident_vertices[2].size == 0
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
        comm.Barrier()
