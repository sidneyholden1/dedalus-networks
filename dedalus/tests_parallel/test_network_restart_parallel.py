"""Parallel restart tests for whole-edge Network storage."""

import pathlib
import shutil
import tempfile

import h5py
from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.evaluator import Evaluator


def _shared_tempdir(comm):
    path = (
        tempfile.mkdtemp(prefix="dedalus-network-restart-")
        if comm.rank == 0
        else None
    )
    return pathlib.Path(comm.bcast(path, root=0))


def _write_and_read_owners(comm, num_edges):
    """Keep root idle while changing noncontiguous ownership when possible."""
    if comm.size == 2:
        owners = np.ones(num_edges, dtype=int)
        return owners, owners.copy()
    edges = np.arange(num_edges)
    write_owners = np.where(edges % 2, 1, comm.size - 1)
    read_owners = np.where(edges % 3, comm.size - 1, 1)
    return write_owners, read_owners


def _field_set(dist, network, basis):
    """Cover tensor components and every distributed Network storage role."""
    return {
        "edge_interval": dist.TensorField(
            network,
            name="edge_interval",
            bases=basis,
        ),
        "edge": dist.VectorField(
            network,
            name="edge",
            bases=basis.edges,
        ),
        "port": dist.TensorField(
            network,
            name="port",
            bases=basis.ports,
        ),
        "vertex": dist.VectorField(
            network,
            name="vertex",
            bases=basis.vertices,
        ),
    }


def _global_values(field, layout, offset):
    tensor_shape = tuple(coordsys.dim for coordsys in field.tensorsig)
    spatial_shape = layout.global_shape(field.domain, field.scales)
    shape = tensor_shape + spatial_shape
    values = np.full(shape, 10 * offset + 0.125, dtype=np.float64)
    for axis, indices in enumerate(np.meshgrid(
        *(np.arange(size) for size in shape),
        indexing="ij",
    )):
        values += (axis + 1) * (10 ** (len(shape) - axis - 1)) * indices
    values = values.astype(field.dtype)
    if np.issubdtype(field.dtype, np.complexfloating):
        values += 1j * (0.5 * values + offset)
    return values


def _local_index(field):
    elements = field.layout.local_elements(field.domain, field.scales)
    return (slice(None),) * len(field.tensorsig) + np.ix_(*elements)


@pytest.mark.mpi(min_size=4)
def test_heterogeneous_bucket_restart_after_repartition():
    """Reload independent dense bucket tasks on a changed whole-edge split."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(6)),
        lengths=np.linspace(0.5, 1.75, 6),
    )
    write_owners = np.array((comm.size - 1, 1, 2, comm.size - 1, 1, 2))
    read_owners = np.array((1, comm.size - 1, 2, 1, comm.size - 1, 2))

    def field_set(owners):
        dist = d3.Distributor(
            network,
            dtype=np.float64,
            network_partition=owners,
        )
        bases = d3.NetworkBasisSet(
            network,
            {
                "chebyshev": {
                    "edge_indices": (0, 2, 4),
                    "size": 7,
                },
                "legendre": {
                    "edge_indices": (1, 3, 5),
                    "size": 10,
                    "interval_basis": d3.Legendre,
                },
            },
        )
        fields = {
            "u_cheb": dist.Field(name="u_cheb", bases=bases["chebyshev"]),
            "u_leg": dist.Field(name="u_leg", bases=bases["legendre"]),
            "tau_cheb": dist.Field(
                name="tau_cheb",
                bases=bases["chebyshev"].edges,
            ),
            "ports_leg": dist.Field(
                name="ports_leg",
                bases=bases["legendre"].ports,
            ),
            "vertices": dist.Field(name="vertices", bases=bases.vertices),
        }
        return dist, bases, fields

    write_dist, write_bases, write_fields = field_set(write_owners)
    expected = {}
    for offset, (name, field) in enumerate(write_fields.items(), start=1):
        field.change_layout("g")
        expected[name] = _global_values(field, field.layout, offset)
        field.set_global_data(expected[name])

    output_path = _shared_tempdir(comm)
    try:
        evaluator = Evaluator(write_dist, {})
        output = evaluator.add_file_handler(
            output_path,
            parallel="gather",
            iter=1,
            max_writes=1,
        )
        for name, field in write_fields.items():
            output.add_task(field, name=name, layout="g")
        evaluator.evaluate_handlers([output], iteration=4, timestep=0.25)
        comm.Barrier()

        if comm.rank == 0:
            with h5py.File(output.current_file, mode="r") as file:
                assert file["tasks/u_cheb"].shape == (1, 3, 7)
                assert file["tasks/u_leg"].shape == (1, 3, 10)
                cheb_ref = file["tasks/u_cheb"].attrs[
                    "network_basis_metadata"
                ]
                leg_ref = file["tasks/u_leg"].attrs[
                    "network_basis_metadata"
                ]
                assert cheb_ref != leg_ref
                assert np.array_equal(
                    file[cheb_ref + "/edge_indices"][:],
                    write_bases["chebyshev"].edge_indices,
                )
                assert np.array_equal(
                    file[leg_ref + "/edge_indices"][:],
                    write_bases["legendre"].edge_indices,
                )
        comm.Barrier()

        read_dist, _, read_fields = field_set(read_owners)
        with h5py.File(output.current_file, mode="r") as file:
            for name, field in read_fields.items():
                field.load_from_hdf5(file, -1, task=name)

        for name, field in read_fields.items():
            assert np.allclose(
                field.allgather_data(),
                expected[name],
                atol=8e-13,
                rtol=8e-13,
            )
        assert read_dist.network_partition.edges[0].size == 0
        assert not np.array_equal(write_owners, read_owners)
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
        comm.Barrier()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("output_layout", ["g", "c"])
def test_distributed_network_hdf5_restart_repartition(dtype, output_layout):
    """Load gathered restarts by current edge indices after repartitioning."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    num_edges = 7
    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(num_edges)),
        lengths=np.linspace(0.5, 2.0, num_edges),
    )
    write_owners, read_owners = _write_and_read_owners(comm, num_edges)
    assert not np.any(write_owners == 0)
    assert not np.any(read_owners == 0)
    if comm.size > 2:
        assert not np.array_equal(write_owners, read_owners)

    write_dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=write_owners,
    )
    write_basis = d3.NetworkBasis(network, size=6)
    write_fields = _field_set(write_dist, network, write_basis)
    expected = {}
    for offset, (name, field) in enumerate(write_fields.items(), start=1):
        field.change_layout(output_layout)
        expected[name] = _global_values(field, field.layout, offset)
        field.set_global_data(expected[name])

    output_path = _shared_tempdir(comm)
    try:
        evaluator = Evaluator(write_dist, {})
        output = evaluator.add_file_handler(
            output_path,
            parallel="gather",
            iter=1,
            max_writes=1,
        )
        for name, field in write_fields.items():
            output.add_task(field, layout=output_layout, name=name, scales=1)
        evaluator.evaluate_handlers(
            [output],
            iteration=11,
            wall_time=3.0,
            sim_time=2.5,
            timestep=0.125,
        )
        comm.Barrier()

        read_dist = d3.Distributor(
            network,
            dtype=dtype,
            network_partition=read_owners,
        )
        read_basis = d3.NetworkBasis(network, size=6)
        read_fields = _field_set(read_dist, network, read_basis)
        assert read_dist.network_partition.edges[0].size == 0
        if comm.size > 2:
            # At least root and one interior rank are empty, while active ranks
            # receive noncontiguous edge and port selections.
            assert read_dist.network_partition.edges[2].size == 0
            for rank in (1, comm.size - 1):
                rank_edges = read_dist.network_partition.edges[rank]
                assert rank_edges.size
                assert np.any(np.diff(rank_edges) != 1)

        with h5py.File(output.current_file, mode="r") as file:
            for name, field in read_fields.items():
                field.load_from_hdf5(file, -1, task=name)

        expected_layout = read_dist.get_layout_object(output_layout)
        for name, field in read_fields.items():
            assert field.layout is expected_layout
            local_expected = expected[name][_local_index(field)]
            assert field.data.shape == local_expected.shape
            assert np.allclose(
                field.data,
                local_expected,
                atol=5e-13,
                rtol=5e-13,
            )
            assert np.allclose(
                field.allgather_data(),
                expected[name],
                atol=5e-13,
                rtol=5e-13,
            )

        # Ownership may change, but the graph represented by the canonical
        # indices may not.  Verify that every rank rejects altered geometry.
        incompatible_lengths = network.lengths.copy()
        incompatible_lengths[-1] += 0.25
        incompatible_network = d3.Network(
            endpoints=network.endpoints,
            lengths=incompatible_lengths,
        )
        incompatible_dist = d3.Distributor(
            incompatible_network,
            dtype=dtype,
            network_partition=read_owners,
        )
        incompatible_basis = d3.NetworkBasis(incompatible_network, size=6)
        incompatible_field = incompatible_dist.Field(
            name="edge_interval",
            bases=incompatible_basis,
        )
        rejected = False
        with h5py.File(output.current_file, mode="r") as file:
            try:
                incompatible_field.load_from_hdf5(
                    file,
                    -1,
                    task="edge_interval",
                )
            except ValueError as error:
                rejected = (
                    "Network restart metadata validation failed" in str(error)
                    and "lengths" in str(error)
                )
        assert all(comm.allgather(rejected))
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
        comm.Barrier()


def _heat_solver(network, owners):
    """Build the implicit heat-mode problem used for restart integration."""
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=12)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    alpha["g"] = np.array((1, 0, 1)).reshape(alpha["g"].shape)
    beta["g"] = np.array((0, 1, 0)).reshape(beta["g"].shape)
    R = basis.restriction
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.IVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "dt(u) - dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = 0")
    solver = problem.build_solver(d3.SBDF1)
    return dist, basis, solver, u, U


def _two_hidden_interface_solver(network, owners, interface_order):
    """Build an IVP whose two managed vertex fields have stable identities."""
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    v = dist.Field(name="v", bases=basis)
    tau_u1 = dist.Field(name="tau_u1", bases=basis.edges)
    tau_u2 = dist.Field(name="tau_u2", bases=basis.edges)
    tau_v1 = dist.Field(name="tau_v1", bases=basis.edges)
    tau_v2 = dist.Field(name="tau_v2", bases=basis.edges)
    edge_fields = {"u": u, "v": v}
    interfaces = [edge_fields[name] for name in interface_order]
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.IVP(
        [u, v, tau_u1, tau_u2, tau_v1, tau_v2],
        network_interfaces=interfaces,
        namespace=locals(),
    )
    problem.add_equation(
        "dt(u) - dx(dx(u)) + lift(tau_u1,-1) + lift(tau_u2,-2) = 0"
    )
    problem.add_equation("continuity(u) = 0")
    problem.add_equation("vertex_value(u) = 0")
    problem.add_equation(
        "dt(v) - dx(dx(v)) + lift(tau_v1,-1) + lift(tau_v2,-2) = 0"
    )
    problem.add_equation("continuity(v) = 0")
    problem.add_equation("vertex_value(v) = 0")
    solver = problem.build_solver(d3.SBDF1, network_solver="dtn")
    hidden = {
        edge.name: vertex
        for edge, vertex in problem.network_interfaces
    }
    return dist, basis, solver, edge_fields, hidden


@pytest.mark.mpi(min_size=4)
def test_hidden_interfaces_checkpoint_roundtrip_reordered_after_repartition():
    """Restore managed U fields by semantic names, not registration order."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    num_edges = 6
    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(num_edges)),
        lengths=np.linspace(0.75, 1.5, num_edges),
    )
    write_owners = np.array((comm.size - 1, 1, 2, comm.size - 1, 1, 2))
    read_owners = np.array((1, comm.size - 1, 2, 1, comm.size - 1, 2))
    (
        write_dist,
        write_basis,
        write_solver,
        write_edges,
        write_hidden,
    ) = _two_hidden_interface_solver(
        network,
        write_owners,
        ("u", "v"),
    )

    expected_edges = {}
    canonical_edges, s = write_basis.local_grids(write_dist, scales=(1, 1))
    for offset, name in enumerate(("u", "v"), start=1):
        write_edges[name]["g"] = (
            100 * offset + 10 * canonical_edges + (offset + 1) * s
        )
        global_edges, global_s = write_basis.global_grids(
            write_dist,
            scales=(1, 1),
        )
        expected_edges[name] = (
            100 * offset + 10 * global_edges + (offset + 1) * global_s
        )

    expected_vertices = {
        "u": np.linspace(1.0, 2.5, network.num_vertices),
        "v": np.linspace(-3.0, -1.5, network.num_vertices),
    }
    for name, vertex in write_hidden.items():
        if comm.rank == 0:
            vertex["g"] = expected_vertices[name][:, None]
        else:
            vertex["g"] = -999

    output_path = _shared_tempdir(comm)
    try:
        checkpoint = write_solver.evaluator.add_file_handler(
            output_path,
            parallel="gather",
            iter=1,
            max_writes=1,
        )
        checkpoint.add_tasks(write_solver.state)
        write_solver.evaluate_handlers([checkpoint], dt=0.125)
        comm.Barrier()

        (
            read_dist,
            _,
            read_solver,
            read_edges,
            read_hidden,
        ) = _two_hidden_interface_solver(
            network,
            read_owners,
            ("v", "u"),
        )
        write_names = {
            name: field.name for name, field in write_hidden.items()
        }
        read_names = {name: field.name for name, field in read_hidden.items()}
        assert write_names == read_names
        assert [
            edge.name for edge, _ in read_solver.problem.network_interfaces
        ] == ["v", "u"]

        write_number, loaded_dt = read_solver.load_state(
            checkpoint.current_file
        )
        assert write_number == 1
        assert loaded_dt == pytest.approx(0.125)
        for name, field in read_edges.items():
            assert np.allclose(
                field.allgather_data("g"),
                expected_edges[name],
                atol=8e-13,
                rtol=8e-13,
            )
        for name, vertex in read_hidden.items():
            assert np.allclose(
                vertex["g"].ravel(),
                expected_vertices[name],
                atol=8e-13,
                rtol=8e-13,
            )
        assert read_dist.network_partition.edges[0].size == 0
        assert not np.array_equal(write_owners, read_owners)
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
        comm.Barrier()


@pytest.mark.mpi(min_size=2)
def test_distributed_network_solver_restart_after_repartition():
    """Resume an implicit solve from a gathered checkpoint on new owners."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    if comm.size == 2:
        write_owners = np.array((0, 1))
        read_owners = np.array((1, 0))
    else:
        write_owners = np.array((comm.size - 1, 1))
        read_owners = np.array((1, comm.size - 1))

    write_dist, write_basis, write_solver, write_u, write_U = _heat_solver(
        network,
        write_owners,
    )
    write_edges = write_dist.network_partition.edges[comm.rank]
    local_x = write_basis.local_physical_grid(write_dist)
    edge_starts = np.array((0, 1))
    initial_global = np.vstack(
        (
            np.sin(np.pi * write_dist.local_grid(write_basis.interval) / 3),
            np.sin(
                np.pi
                * (1 + 2 * write_dist.local_grid(write_basis.interval))
                / 3
            ),
        )
    )
    write_u["g"] = np.sin(
        np.pi * (edge_starts[write_edges, None] + local_x) / 3
    )
    initial_vertices = np.array((0, np.sin(np.pi / 3), 0))
    write_U["g"] = initial_vertices.reshape(write_U["g"].shape)

    first_dt = 0.01
    write_solver.step(first_dt)
    eigenvalue = (np.pi / 3) ** 2
    first_decay = 1 / (1 + first_dt * eigenvalue)

    output_path = _shared_tempdir(comm)
    try:
        checkpoint = write_solver.evaluator.add_file_handler(
            output_path,
            parallel="gather",
            iter=1,
            max_writes=1,
        )
        checkpoint.add_tasks(write_solver.state)
        write_solver.evaluate_handlers([checkpoint], dt=first_dt)
        comm.Barrier()

        read_dist, read_basis, read_solver, read_u, read_U = _heat_solver(
            network,
            read_owners,
        )
        write_number, loaded_dt = read_solver.load_state(
            checkpoint.current_file
        )
        assert write_number == 1
        assert loaded_dt == first_dt
        assert read_solver.iteration == 1
        assert read_solver.sim_time == pytest.approx(first_dt)
        assert np.allclose(
            read_u.allgather_data("g"),
            first_decay * initial_global,
            atol=4e-10,
        )
        assert np.allclose(
            read_U["g"].ravel(),
            first_decay * initial_vertices,
            atol=4e-10,
        )

        second_dt = 0.02
        read_solver.step(second_dt)
        second_decay = first_decay / (1 + second_dt * eigenvalue)
        assert np.allclose(
            read_u.allgather_data("g"),
            second_decay * initial_global,
            atol=6e-10,
        )
        assert np.allclose(
            read_U["g"].ravel(),
            second_decay * initial_vertices,
            atol=6e-10,
        )
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
        comm.Barrier()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("storage", ["replicated", "distributed"])
@pytest.mark.parametrize("failure", ["metadata", "missing_task"])
def test_network_restart_validation_error_is_collective(storage, failure):
    """Propagate a rank-local metadata failure before reading field data."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    owners = np.array((comm.size - 1,))
    write_dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    write_basis = d3.NetworkBasis(network, size=6)
    source = write_dist.Field(name="u", bases=write_basis)
    source["g"] = 1

    output_path = _shared_tempdir(comm)
    try:
        evaluator = Evaluator(write_dist, {})
        output = evaluator.add_file_handler(
            output_path,
            parallel="gather",
            iter=1,
            max_writes=1,
        )
        output.add_task(source, name="u", layout="g")
        evaluator.evaluate_handlers([output], iteration=0)
        comm.Barrier()

        rank_path = output_path / f"rank-{comm.rank}.h5"
        shutil.copy2(output.current_file, rank_path)
        if comm.rank == comm.size - 1:
            with h5py.File(rank_path, mode="r+") as file:
                if failure == "metadata":
                    file["network/lengths"][0] = 2
                else:
                    del file["tasks/u"]
        comm.Barrier()

        read_partition = owners if storage == "distributed" else None
        read_dist = d3.Distributor(
            network,
            dtype=np.float64,
            network_partition=read_partition,
        )
        read_basis = d3.NetworkBasis(network, size=6)
        restored = read_dist.Field(name="restored", bases=read_basis)
        expected_rank = comm.size - 1
        error_type = ValueError if failure == "metadata" else KeyError
        detail = "lengths" if failure == "metadata" else "u"
        with h5py.File(rank_path, mode="r") as file:
            with pytest.raises(
                error_type,
                match=(
                    "Parallel Network restart validation failed on "
                    f"rank {expected_rank}.*{detail}"
                ),
            ):
                restored.load_from_hdf5(file, -1, task="u")
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(output_path, ignore_errors=True)
        comm.Barrier()
