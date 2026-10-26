"""Test self-describing HDF5 output for metric networks."""

import json

import h5py
from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.core.evaluator import Evaluator, _setup_network_basis_metadata


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="Network output metadata is covered here in serial.",
)


class _OpaqueID:
    """Hashable ID which deliberately has no portable HDF5 encoding."""

    def __init__(self, label):
        self.label = label

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return isinstance(other, _OpaqueID) and self.label == other.label


def _decode_json_id(value):
    """Decode JSON arrays recursively as tuples, matching Network ID types."""
    value = json.loads(value)
    if isinstance(value, list):
        return tuple(_decode_json_value(item) for item in value)
    return value


def _decode_json_value(value):
    if isinstance(value, list):
        return tuple(_decode_json_value(item) for item in value)
    return value


def _strings(value):
    """Normalize scalar or array HDF5 strings to ordinary Python strings."""
    values = np.atleast_1d(value)
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    )


def _attached_scale(dataset, axis):
    scales = list(dataset.dims[axis].values())
    assert len(scales) == 1
    return scales[0]


def _portable_network():
    """E == V network including parallel edges, a loop, and an isolated vertex."""
    return d3.Network(
        vertex_ids=("alpha-α", ("branch", np.int64(2)), None, np.int64(7)),
        edge_ids=("loop", ("parallel", 0), ("parallel", 1), ("tail", True)),
        endpoints=((0, 1), (0, 1), (1, 1), (2, 0)),
        lengths=(0.5, 2.0, 1.25, 3.0),
        coordinate_names=("branch", "s"),
    )


def _build_output(tmp_path, network, *, max_writes=None):
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    evaluator = Evaluator(dist, {})
    handler = evaluator.add_file_handler(
        tmp_path / "snapshots",
        parallel="gather",
        mode="overwrite",
        max_writes=max_writes,
    )
    return dist, basis, evaluator, handler


def test_network_metadata_round_trip_and_physical_position(tmp_path):
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    physical_position = basis.global_physical_grid(dist)
    u = dist.Field(name="u", bases=basis)
    u["g"] = physical_position
    handler.add_task(u, name="u", layout="g")

    evaluator.evaluate_handlers([handler], iteration=0)

    with h5py.File(handler.current_file, "r") as file:
        metadata = file["network"]
        assert metadata.attrs["schema"] == "dedalus.network"
        assert metadata.attrs["schema_version"] == 1
        assert _strings(metadata.attrs["coordinate_names"]) == network.names
        assert metadata.attrs["num_vertices"] == network.num_vertices
        assert metadata.attrs["num_edges"] == network.num_edges
        assert metadata.attrs["num_ports"] == network.num_ports
        assert metadata.attrs["edge_orientation"] == "tail_to_head"
        assert (
            metadata.attrs["position_semantics"]
            == "normalized_reference_from_tail"
        )
        assert (
            metadata.attrs["stored_position_scale"]
            == "normalized_reference_from_tail"
        )
        assert metadata.attrs["position_map"] == "length_times_reference"
        assert metadata.attrs["port_order"] == "edge_major_tail_head"
        assert np.array_equal(metadata["vertex_index"][:], np.arange(4))
        assert np.array_equal(metadata["edge_index"][:], np.arange(4))
        assert tuple(metadata["endpoint_slot"].asstr()[:]) == ("tail", "head")
        assert np.array_equal(metadata["endpoints"][:], network.endpoints)
        assert np.array_equal(metadata["lengths"][:], network.lengths)
        assert _attached_scale(metadata["endpoints"], 0).name == metadata["edge_index"].name
        assert _attached_scale(metadata["endpoints"], 1).name == metadata["endpoint_slot"].name
        assert _attached_scale(metadata["lengths"], 0).name == metadata["edge_index"].name

        assert metadata.attrs["vertex_ids_encoding"] == "json-v1"
        assert metadata.attrs["edge_ids_encoding"] == "json-v1"
        assert metadata["vertex_ids"].attrs["encoding"] == "json-v1"
        assert metadata["edge_ids"].attrs["encoding"] == "json-v1"
        vertex_ids = tuple(
            _decode_json_id(value) for value in metadata["vertex_ids"].asstr()[:]
        )
        edge_ids = tuple(
            _decode_json_id(value) for value in metadata["edge_ids"].asstr()[:]
        )

        restored = d3.Network(
            vertex_ids=vertex_ids,
            edge_ids=edge_ids,
            endpoints=metadata["endpoints"][:],
            lengths=metadata["lengths"][:],
            coordinate_names=_strings(metadata.attrs["coordinate_names"]),
        )
        assert restored.vertex_ids == network.vertex_ids
        assert restored.edge_ids == network.edge_ids
        assert np.array_equal(restored.endpoints, network.endpoints)
        assert np.array_equal(restored.lengths, network.lengths)
        assert restored.names == network.names

        task = file["tasks/u"]
        reference_position = _attached_scale(task, 2)[:]
        reconstructed_position = network.lengths[:, None] * reference_position[None, :]
        assert np.allclose(reconstructed_position, physical_position)
        assert np.allclose(task[0], reconstructed_position)


def test_task_axis_roles_disambiguate_equal_cardinal_sizes(tmp_path):
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    edge = dist.Field(name="edge", bases=basis.edges)
    vertex = dist.Field(name="vertex", bases=basis.vertices)
    port = dist.Field(name="port", bases=basis.ports)
    u = dist.Field(name="u", bases=basis)
    edge["g"] = np.arange(network.num_edges)[:, None]
    vertex["g"] = np.arange(network.num_vertices)[:, None]
    port["g"] = np.arange(network.num_ports)[:, None]
    u["g"] = np.arange(network.num_edges * basis.size).reshape(
        network.num_edges, basis.size
    )
    handler.add_task(edge, name="edge", layout="g")
    handler.add_task(vertex, name="vertex", layout="g")
    handler.add_task(port, name="port", layout="g")
    handler.add_task(u, name="edge_interval", layout="g")
    handler.add_task(u, name="edge_interval_coeff", layout="c")

    evaluator.evaluate_handlers([handler], iteration=0)

    with h5py.File(handler.current_file, "r") as file:
        tasks = file["tasks"]
        expected_roles = {
            "edge": ("edges", "constant"),
            "vertex": ("vertices", "constant"),
            "port": ("ports", "constant"),
            "edge_interval": ("edges", "position"),
            "edge_interval_coeff": ("edges", "position"),
        }
        for name, roles in expected_roles.items():
            assert tasks[name].attrs["network_metadata"] == "/network"
            assert _strings(tasks[name].attrs["network_axis_roles"]) == roles

        # Canonical values can coincide when E == V, but their dimension
        # scales retain distinct edge/vertex semantics.
        edge_scale = _attached_scale(tasks["edge"], 1)
        vertex_scale = _attached_scale(tasks["vertex"], 1)
        assert edge_scale.name != vertex_scale.name
        assert tasks["edge"].dims[1].label == "edge"
        assert tasks["vertex"].dims[1].label == "vertex"
        assert np.array_equal(edge_scale[:], vertex_scale[:])
        assert np.array_equal(tasks["edge_interval"].attrs["grid_space"], (True, True))
        assert np.array_equal(
            tasks["edge_interval_coeff"].attrs["grid_space"],
            (False, False),
        )


@pytest.mark.parametrize(
    "mismatch, expected_detail",
    [
        ("schema", "schema"),
        ("count", "num_edges"),
        ("endpoints", "endpoints"),
        ("lengths", "lengths"),
        ("semantics", "edge_orientation"),
    ],
)
def test_network_restart_rejects_incompatible_metadata(
    tmp_path,
    mismatch,
    expected_detail,
):
    """Replicated restart validates topology, geometry, and conventions."""
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    u = dist.Field(name="u", bases=basis)
    u["g"] = 1
    handler.add_task(u, name="u", layout="g")
    evaluator.evaluate_handlers([handler], iteration=0)

    with h5py.File(handler.current_file, "r+") as file:
        metadata = file["network"]
        if mismatch == "schema":
            metadata.attrs["schema"] = "not.dedalus.network"
        elif mismatch == "count":
            metadata.attrs["num_edges"] = network.num_edges + 1
        elif mismatch == "endpoints":
            metadata["endpoints"][0, 0] = 2
        elif mismatch == "lengths":
            metadata["lengths"][0] += 0.25
        elif mismatch == "semantics":
            metadata.attrs["edge_orientation"] = "head_to_tail"

    restored = dist.Field(name="restored", bases=basis)
    with h5py.File(handler.current_file, "r") as file:
        with pytest.raises(
            ValueError,
            match=(
                "Network restart metadata validation failed.*"
                f"{expected_detail}"
            ),
        ):
            restored.load_from_hdf5(file, 0, task="u")


@pytest.mark.parametrize(
    "id_kind",
    ["vertex", "edge"],
)
def test_network_restart_rejects_different_portable_ids(tmp_path, id_kind):
    """Canonical indices do not hide changed portable graph identities."""
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    u = dist.Field(name="u", bases=basis)
    u["g"] = 1
    handler.add_task(u, name="u", layout="g")
    evaluator.evaluate_handlers([handler], iteration=0)

    vertex_ids = network.vertex_ids
    edge_ids = network.edge_ids
    if id_kind == "vertex":
        vertex_ids = ("changed",) + vertex_ids[1:]
    else:
        edge_ids = ("changed",) + edge_ids[1:]
    active_network = d3.Network(
        vertex_ids=vertex_ids,
        edge_ids=edge_ids,
        endpoints=network.endpoints,
        lengths=network.lengths,
        coordinate_names=network.names,
    )
    active_dist = d3.Distributor(active_network, dtype=np.float64)
    active_basis = d3.NetworkBasis(active_network, size=basis.size)
    restored = active_dist.Field(name="restored", bases=active_basis)

    with h5py.File(handler.current_file, "r") as file:
        with pytest.raises(
            ValueError,
            match=f"Network restart metadata validation failed.*{id_kind}_ids",
        ):
            restored.load_from_hdf5(file, 0, task="u")


@pytest.mark.parametrize(
    "malformation, expected_detail",
    [
        ("canonical_with_object", "canonical-index-only.*vertex_ids.*exists"),
        ("missing_encoding_attribute", "missing attribute.*vertex_ids_encoding"),
        ("unsupported_encoding", "vertex_ids_encoding.*unsupported"),
        ("missing_dataset", "missing dataset.*vertex_ids"),
        ("group", "vertex_ids.*not an HDF5 dataset"),
        ("scalar", "vertex_ids.*shape.*expected"),
        ("wrong_length", "vertex_ids.*shape.*expected"),
        ("nonstring", "vertex_ids.*must contain strings"),
        ("missing_dataset_encoding", "vertex_ids.*missing its encoding"),
        ("invalid_dataset_encoding", "vertex_ids.*invalid encoding"),
        ("invalid_json", "vertex_ids.*not valid json-v1"),
        ("invalid_utf8", "vertex_ids.*cannot be read as strings"),
    ],
)
def test_network_restart_rejects_malformed_id_metadata(
    tmp_path,
    malformation,
    expected_detail,
):
    """Malformed semantic-ID objects fail through the validation boundary."""
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    u = dist.Field(name="u", bases=basis)
    u["g"] = 1
    handler.add_task(u, name="u", layout="g")
    evaluator.evaluate_handlers([handler], iteration=0)

    with h5py.File(handler.current_file, "r+") as file:
        metadata = file["network"]
        dataset_name = "vertex_ids"
        if malformation == "canonical_with_object":
            metadata.attrs["vertex_ids_encoding"] = "canonical-index-only"
        elif malformation == "missing_encoding_attribute":
            del metadata.attrs["vertex_ids_encoding"]
        elif malformation == "unsupported_encoding":
            metadata.attrs["vertex_ids_encoding"] = "pickle-v1"
        elif malformation == "missing_dataset":
            del metadata[dataset_name]
        elif malformation == "group":
            del metadata[dataset_name]
            metadata.create_group(dataset_name)
        elif malformation in ("scalar", "wrong_length", "nonstring", "invalid_utf8"):
            del metadata[dataset_name]
            if malformation == "scalar":
                replacement = metadata.create_dataset(
                    dataset_name,
                    data='"alpha-α"',
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            elif malformation == "wrong_length":
                replacement = metadata.create_dataset(
                    dataset_name,
                    data=np.asarray(('"alpha-α"',), dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            elif malformation == "nonstring":
                replacement = metadata.create_dataset(
                    dataset_name,
                    data=np.arange(network.num_vertices),
                )
            else:
                replacement = metadata.create_dataset(
                    dataset_name,
                    data=np.asarray(
                        (b"\xff", b'"branch"', b"null", b"7"),
                        dtype="S16",
                    ),
                )
            replacement.attrs["encoding"] = "json-v1"
        elif malformation == "missing_dataset_encoding":
            del metadata[dataset_name].attrs["encoding"]
        elif malformation == "invalid_dataset_encoding":
            metadata[dataset_name].attrs["encoding"] = "yaml-v1"
        elif malformation == "invalid_json":
            metadata[dataset_name][0] = "not-json"

    restored = dist.Field(name="restored", bases=basis)
    with h5py.File(handler.current_file, "r") as file:
        with pytest.raises(
            ValueError,
            match=(
                "Network restart metadata validation failed.*"
                f"{expected_detail}"
            ),
        ):
            restored.load_from_hdf5(file, 0, task="u")


def test_network_restart_compares_json_ids_semantically(tmp_path):
    """Equivalent valid JSON spellings preserve portable restart identity."""
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    u = dist.Field(name="u", bases=basis)
    u["g"] = np.arange(network.num_edges * basis.size).reshape(
        network.num_edges,
        basis.size,
    )
    handler.add_task(u, name="u", layout="g")
    evaluator.evaluate_handlers([handler], iteration=0)

    with h5py.File(handler.current_file, "r+") as file:
        file["network/vertex_ids"][0] = '  "alpha-\\u03b1"  '

    restored = dist.Field(name="restored", bases=basis)
    with h5py.File(handler.current_file, "r") as file:
        restored.load_from_hdf5(file, 0, task="u")
    assert np.allclose(restored["g"], u["g"])


def test_network_restart_rejects_cardinal_shape_and_role_mismatches(tmp_path):
    """Network task roles and indivisible cardinal sizes must match exactly."""
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    edge = dist.Field(name="edge", bases=basis.edges)
    edge["g"] = np.arange(network.num_edges)[:, None]
    handler.add_task(edge, name="edge", layout="g")
    evaluator.evaluate_handlers([handler], iteration=0)

    with h5py.File(handler.current_file, "r+") as file:
        source = file["tasks/edge"]
        malformed = file["tasks"].create_dataset(
            "malformed_edge",
            shape=(1, network.num_edges + 1, 1),
            dtype=np.float64,
        )
        for name in ("grid_space", "network_metadata", "network_axis_roles"):
            malformed.attrs[name] = source.attrs[name]

    with h5py.File(handler.current_file, "r") as file:
        with pytest.raises(
            ValueError,
            match="Network restart task validation failed.*edges.*size",
        ):
            edge.load_from_hdf5(file, 0, task="malformed_edge")

        # E == V for this graph, so shape alone cannot distinguish these axes.
        vertex = dist.Field(name="vertex", bases=basis.vertices)
        with pytest.raises(
            ValueError,
            match="Network restart task validation failed.*axis roles",
        ):
            vertex.load_from_hdf5(file, 0, task="edge")


def test_unsupported_network_ids_degrade_to_canonical_indices(tmp_path):
    network = d3.Network(
        vertex_ids=(_OpaqueID("left"), _OpaqueID("right")),
        edge_ids=("portable-edge",),
        endpoints=((0, 1),),
        lengths=(2.0,),
    )
    dist, basis, evaluator, handler = _build_output(tmp_path, network)
    u = dist.Field(name="u", bases=basis)
    u["g"] = np.arange(basis.size)[None, :]
    handler.add_task(u, name="u", layout="g")

    evaluator.evaluate_handlers([handler], iteration=0)

    with h5py.File(handler.current_file, "r") as file:
        metadata = file["network"]
        assert metadata.attrs["vertex_ids_encoding"] == "canonical-index-only"
        assert "vertex_ids" not in metadata
        assert metadata.attrs["edge_ids_encoding"] == "json-v1"
        assert tuple(metadata["edge_ids"].asstr()[:]) == ('"portable-edge"',)
        assert np.array_equal(metadata["vertex_index"][:], (0, 1))

    # The file intentionally contains no semantic vertex IDs, so another
    # opaque-ID graph with the same canonical topology remains restartable.
    active_network = d3.Network(
        vertex_ids=(_OpaqueID("west"), _OpaqueID("east")),
        edge_ids=("portable-edge",),
        endpoints=network.endpoints,
        lengths=network.lengths,
    )
    active_dist = d3.Distributor(active_network, dtype=np.float64)
    active_basis = d3.NetworkBasis(active_network, size=basis.size)
    restored = active_dist.Field(name="restored", bases=active_basis)
    with h5py.File(handler.current_file, "r") as file:
        restored.load_from_hdf5(file, 0, task="u")
    assert np.allclose(restored["g"], u["g"])


def test_network_metadata_is_repeated_after_file_rollover(tmp_path):
    network = _portable_network()
    dist, basis, evaluator, handler = _build_output(
        tmp_path,
        network,
        max_writes=1,
    )
    edge = dist.Field(name="edge", bases=basis.edges)
    edge["g"] = np.arange(network.num_edges)[:, None]
    handler.add_task(edge, name="edge", layout="g")

    evaluator.evaluate_handlers([handler], iteration=0)
    first_file = handler.current_file
    edge["g"] += 1
    evaluator.evaluate_handlers([handler], iteration=1)
    second_file = handler.current_file

    assert first_file != second_file
    for set_number, path in enumerate((first_file, second_file), start=1):
        with h5py.File(path, "r") as file:
            assert file.attrs["set_number"] == set_number
            assert file["network"].attrs["schema"] == "dedalus.network"
            assert np.array_equal(file["network/endpoints"][:], network.endpoints)
            assert np.array_equal(file["network/lengths"][:], network.lengths)
            assert _strings(file["tasks/edge"].attrs["network_axis_roles"]) == (
                "edges",
                "constant",
            )


def _heterogeneous_output(tmp_path):
    """Write two dense, differently discretized edge buckets."""
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3), (3, 4)),
        lengths=(0.5, 1.25, 0.75, 1.5),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    bases = d3.NetworkBasisSet(
        network,
        {
            "chebyshev": {"edge_indices": (0, 2), "size": 8},
            "legendre": {
                "edge_indices": (1, 3),
                "size": 11,
                "interval_basis": d3.Legendre,
            },
        },
    )
    cheb = bases["chebyshev"]
    leg = bases["legendre"]
    u_cheb = dist.Field(name="u_cheb", bases=cheb)
    u_leg = dist.Field(name="u_leg", bases=leg)
    tau_cheb = dist.Field(name="tau_cheb", bases=cheb.edges)
    ports_leg = dist.Field(name="ports_leg", bases=leg.ports)
    vertices = dist.Field(name="vertices", bases=bases.vertices)
    u_cheb["g"] = np.arange(16).reshape(2, 8)
    u_leg["g"] = 100 + np.arange(22).reshape(2, 11)
    tau_cheb["g"] = np.asarray((3, 7))[:, None]
    ports_leg["g"] = np.arange(4)[:, None]
    vertices["g"] = np.arange(network.num_vertices)[:, None]

    evaluator = Evaluator(dist, {})
    handler = evaluator.add_file_handler(
        tmp_path / "heterogeneous",
        parallel="gather",
        mode="overwrite",
    )
    for field in (u_cheb, u_leg, tau_cheb, ports_leg, vertices):
        handler.add_task(field, name=field.name, layout="g")
    evaluator.evaluate_handlers([handler], iteration=0)
    return network, dist, bases, handler


def test_heterogeneous_tasks_have_hashed_discretizations_and_canonical_scales(
    tmp_path,
):
    """Store one dense task per bucket with canonical edge/port labels."""
    network, _, bases, handler = _heterogeneous_output(tmp_path)

    with h5py.File(handler.current_file, "r") as file:
        tasks = file["tasks"]
        assert tasks["u_cheb"].shape == (1, 2, 8)
        assert tasks["u_leg"].shape == (1, 2, 11)
        assert np.allclose(tasks["u_cheb"][0], np.arange(16).reshape(2, 8))
        assert np.allclose(
            tasks["u_leg"][0],
            100 + np.arange(22).reshape(2, 11),
        )

        refs = {
            name: _strings(task.attrs["network_basis_metadata"])[0]
            for name, task in tasks.items()
        }
        assert refs["u_cheb"] == refs["tau_cheb"]
        assert refs["u_cheb"] != refs["u_leg"]
        assert refs["ports_leg"] != refs["u_leg"]
        assert refs["vertices"] not in {
            refs["u_cheb"],
            refs["u_leg"],
            refs["ports_leg"],
        }

        expected = {
            "u_cheb": ("edges", [0, 2], [0, 2], 8, -0.5),
            "u_leg": ("edges", [1, 3], [1, 3], 11, 0.0),
            "tau_cheb": ("edges", [0, 2], [0, 2], 8, -0.5),
            "ports_leg": ("ports", [1, 3], [2, 3, 6, 7], 11, 0.0),
            "vertices": (
                "vertices",
                list(range(network.num_edges)),
                list(range(network.num_vertices)),
                network.num_vertices,
                None,
            ),
        }
        for name, (role, edges, elements, interval_size, jacobi_a) in (
            expected.items()
        ):
            group = file[refs[name]]
            descriptor = json.loads(group.attrs["descriptor_json"])
            assert group.name.startswith("/network/discretizations/")
            assert group.name.rsplit("/", 1)[-1] == group.attrs["digest"]
            assert descriptor["role"] == role
            assert descriptor["edge_indices"] == edges
            assert descriptor["canonical_elements"] == elements
            assert np.array_equal(group["edge_indices"][:], edges)
            assert np.array_equal(group["canonical_elements"][:], elements)
            if jacobi_a is None:
                assert descriptor["interval"] is None
            else:
                options = descriptor["interval"]["options"]
                assert options["size"] == interval_size
                assert options["a"] == jacobi_a
                assert options["b"] == jacobi_a

        assert np.array_equal(
            _attached_scale(tasks["u_cheb"], 1)[:],
            bases["chebyshev"].edge_indices,
        )
        assert np.array_equal(
            _attached_scale(tasks["u_leg"], 1)[:],
            bases["legendre"].edge_indices,
        )
        assert np.array_equal(
            _attached_scale(tasks["ports_leg"], 1)[:],
            bases["legendre"].port_indices,
        )
        assert np.array_equal(
            _attached_scale(tasks["vertices"], 1)[:],
            np.arange(network.num_vertices),
        )


@pytest.mark.parametrize(
    "mismatch,match",
    [
        ("subset", "edge_indices"),
        ("family", "interval"),
        ("resolution", "interval"),
    ],
)
def test_bucket_restart_rejects_subset_family_and_resolution_mismatch(
    tmp_path,
    mismatch,
    match,
):
    """Compare saved bucket contents even when task roles/shapes coincide."""
    network, _, bases, handler = _heterogeneous_output(tmp_path)
    if mismatch == "subset":
        active_basis = d3.NetworkBasis(
            network,
            size=8,
            edge_indices=(1, 3),
        )
    elif mismatch == "family":
        active_basis = d3.NetworkBasis(
            network,
            size=8,
            edge_indices=(0, 2),
            interval_basis=d3.Legendre,
        )
    else:
        active_basis = d3.NetworkBasis(
            network,
            size=9,
            edge_indices=(0, 2),
        )
    active_dist = d3.Distributor(network, dtype=np.float64)
    restored = active_dist.Field(name="restored", bases=active_basis)

    with h5py.File(handler.current_file, "r") as file:
        with pytest.raises(
            ValueError,
            match=(
                "Network restart discretization validation failed.*"
                f"{match}"
            ),
        ):
            restored.load_from_hdf5(file, 0, task="u_cheb")

    matching = active_dist.Field(name="matching", bases=bases["chebyshev"])
    with h5py.File(handler.current_file, "r") as file:
        matching.load_from_hdf5(file, 0, task="u_cheb")
    assert np.allclose(matching["g"], np.arange(16).reshape(2, 8))


def test_legacy_restart_is_limited_to_full_edge_basis(tmp_path):
    """Accept old full-edge files but never guess a bucket subset."""
    network = _portable_network()
    dist, full_basis, evaluator, full_handler = _build_output(tmp_path, network)
    full = dist.Field(name="full", bases=full_basis)
    full["g"] = np.arange(network.num_edges * full_basis.size).reshape(
        network.num_edges,
        full_basis.size,
    )
    full_handler.add_task(full, name="full", layout="g")
    evaluator.evaluate_handlers([full_handler], iteration=0)
    with h5py.File(full_handler.current_file, "r+") as file:
        del file["tasks/full"].attrs["network_basis_metadata"]
    restored_full = dist.Field(name="restored_full", bases=full_basis)
    with h5py.File(full_handler.current_file, "r") as file:
        restored_full.load_from_hdf5(file, 0, task="full")
    assert np.allclose(restored_full["g"], full["g"])

    network, _, bases, bucket_handler = _heterogeneous_output(tmp_path)
    with h5py.File(bucket_handler.current_file, "r+") as file:
        del file["tasks/u_cheb"].attrs["network_basis_metadata"]
    bucket_dist = d3.Distributor(network, dtype=np.float64)
    restored_bucket = bucket_dist.Field(
        name="restored_bucket",
        bases=bases["chebyshev"],
    )
    with h5py.File(bucket_handler.current_file, "r") as file:
        with pytest.raises(ValueError, match="only for a full-edge basis"):
            restored_bucket.load_from_hdf5(file, 0, task="u_cheb")


def test_discretization_digest_reuse_checks_group_contents(tmp_path):
    """Never silently reuse a corrupted or colliding digest group."""
    _, dist, bases, handler = _heterogeneous_output(tmp_path)
    probe = dist.Field(name="probe", bases=bases["chebyshev"])
    with h5py.File(handler.current_file, "r+") as file:
        ref = _strings(
            file["tasks/u_cheb"].attrs["network_basis_metadata"]
        )[0]
        file[ref + "/edge_indices"][0] = 1
        with pytest.raises(RuntimeError, match="collision or content mismatch"):
            _setup_network_basis_metadata(
                file,
                probe.domain,
            )
