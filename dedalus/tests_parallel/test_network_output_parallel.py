"""Parallel HDF5 output tests for whole-edge Network storage."""

import pathlib
import shutil
import tempfile

import h5py
from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.evaluator import Evaluator, H5GatherFileHandler


def _network_and_owners(comm):
    """Build a partition with an edge-free root and irregular active ranks."""
    num_edges = 7
    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(num_edges)),
        lengths=np.linspace(0.5, 2.0, num_edges),
    )
    if comm.size == 2:
        owners = np.ones(num_edges, dtype=int)
    else:
        # Ranks zero and (for four ranks) two are idle.  Both active ranks own
        # noncontiguous global edge and port indices.
        owners = np.where(np.arange(num_edges) % 2, 1, comm.size - 1)
    return network, owners


def _values(elements, offset, dtype):
    shape = tuple(axis.size for axis in elements)
    values = np.full(shape, offset, dtype=np.float64)
    for axis, indices in enumerate(np.meshgrid(*elements, indexing="ij")):
        values += (100 if axis == 0 else 1) * indices
    values = values.astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * (0.5 * values + 0.25)
    return values


def _shared_tempdir(comm):
    path = (
        tempfile.mkdtemp(prefix="dedalus-network-output-")
        if comm.rank == 0
        else None
    )
    return pathlib.Path(comm.bcast(path, root=0))


def _strings(values):
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.atleast_1d(values)
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("output_layout", ["g", "c"])
def test_distributed_network_gather_output(dtype, output_layout):
    """Gather edge, port, and replicated vertex data by global indices."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network, owners = _network_and_owners(comm)
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    assert dist.network_partition.edges[0].size == 0
    if comm.size > 2:
        for rank in (1, comm.size - 1):
            rank_edges = dist.network_partition.edges[rank]
            assert rank_edges.size > 1
            assert np.any(np.diff(rank_edges) != 1)
        for rank in range(2, comm.size - 1):
            assert dist.network_partition.edges[rank].size == 0
    basis = d3.NetworkBasis(network, size=6)
    fields = {
        "edge_interval": dist.Field(name="u", bases=basis),
        "edge": dist.Field(name="edge", bases=basis.edges),
        "port": dist.Field(name="port", bases=basis.ports),
        "vertex": dist.Field(name="vertex", bases=basis.vertices),
    }

    expected = {}
    for offset, (name, field) in enumerate(fields.items(), start=1):
        field.change_layout(output_layout)
        local_elements = field.layout.local_elements(field.domain, field.scales)
        field[output_layout] = _values(local_elements, 10 * offset, dtype)
        global_elements = tuple(
            np.arange(size) for size in field.global_shape
        )
        expected[name] = _values(
            global_elements,
            10 * offset,
            dtype,
        )

    output_path = _shared_tempdir(comm)
    try:
        evaluator = Evaluator(dist, {})
        # Exercise both the distributed-Network default and an explicit gather
        # request without multiplying the dtype/layout test matrix.
        parallel = None if output_layout == "g" else "gather"
        output = evaluator.add_file_handler(
            output_path,
            parallel=parallel,
            iter=1,
            max_writes=2,
        )
        assert isinstance(output, H5GatherFileHandler)
        for name, field in fields.items():
            output.add_task(
                field,
                layout=output_layout,
                name=name,
                scales=1,
            )

        evaluator.evaluate_handlers(
            [output],
            iteration=3,
            wall_time=2.0,
            sim_time=1.5,
            timestep=0.25,
        )

        # Only root touches the gathered file.  Catch I/O errors before the
        # broadcast so a failed assertion cannot strand the other ranks.
        payload = None
        if comm.rank == 0:
            try:
                with h5py.File(output.current_file, mode="r") as file:
                    payload = {
                        "error": None,
                        "writes": int(file.attrs["writes"]),
                        "iteration": int(file["scales/iteration"][-1]),
                        "network_schema": file["network"].attrs["schema"],
                        "endpoints": file["network/endpoints"][:],
                        "lengths": file["network/lengths"][:],
                        "tasks": {
                            name: file[f"tasks/{name}"][-1]
                            for name in fields
                        },
                        "roles": {
                            name: _strings(
                                file[f"tasks/{name}"].attrs[
                                    "network_axis_roles"
                                ]
                            )
                            for name in fields
                        },
                    }
            except Exception as error:  # pragma: no cover - MPI failure path
                payload = {
                    "error": f"{type(error).__name__}: {error}",
                }
        payload = comm.bcast(payload, root=0)
        assert payload["error"] is None
        assert payload["writes"] == 1
        assert payload["iteration"] == 3
        assert payload["network_schema"] == "dedalus.network"
        assert np.array_equal(payload["endpoints"], network.endpoints)
        assert np.array_equal(payload["lengths"], network.lengths)
        assert payload["roles"] == {
            "edge_interval": ("edges", "position"),
            "edge": ("edges", "constant"),
            "port": ("ports", "constant"),
            "vertex": ("vertices", "constant"),
        }
        for name in fields:
            assert payload["tasks"][name].shape == expected[name].shape
            assert np.allclose(
                payload["tasks"][name],
                expected[name],
                atol=5e-13,
                rtol=5e-13,
            )
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
        comm.Barrier()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("parallel", ["virtual", "mpio"])
def test_distributed_network_rejects_slice_based_output(parallel):
    """Noncontiguous Network storage cannot use slice-based HDF5 writers."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    network, owners = _network_and_owners(comm)
    dist = d3.Distributor(network, network_partition=owners)
    evaluator = Evaluator(dist, {})
    with pytest.raises(
        NotImplementedError,
        match="supports only parallel='gather'",
    ):
        evaluator.add_file_handler("unused-network-output", parallel=parallel)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("parallel", [None, "gather"])
def test_replicated_network_output_remains_rejected(parallel):
    """Do not silently write duplicate replicated-symbolic MPI fields."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(network)
    evaluator = Evaluator(dist, {})
    with pytest.raises(
        NotImplementedError,
        match="replicated-symbolic network backend",
    ):
        evaluator.add_file_handler("unused-network-output", parallel=parallel)
