"""MPI layout and operator tests for owned/ghost Network vertices."""

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3


def _setup(dtype):
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(1, 1.5, 2),
    )
    owners = np.array((3, 1, 3), dtype=int)
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
        network_vertex_storage="owned_ghost",
    )
    basis = d3.NetworkBasis(network, size=5)
    return comm, network, owners, dist, basis


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_layout_sets_include_idle_ranks():
    comm, network, owners, dist, basis = _setup(np.float64)
    partition = dist.network_partition

    if comm.rank == 1:
        expected_owned = np.array((1, 2))
        expected_resident = np.array((1, 2))
        expected_ghost = np.array((), dtype=int)
    elif comm.rank == 3:
        expected_owned = np.array((0, 3))
        expected_resident = np.arange(4)
        expected_ghost = np.array((1, 2))
    else:
        expected_owned = np.array((), dtype=int)
        expected_resident = np.array((), dtype=int)
        expected_ghost = np.array((), dtype=int)

    assert np.array_equal(partition.owned_vertices[comm.rank], expected_owned)
    assert np.array_equal(
        partition.resident_vertices[comm.rank],
        expected_resident,
    )
    assert np.array_equal(partition.ghost_vertices[comm.rank], expected_ghost)
    assert np.all(np.isin(expected_ghost, partition.separator_vertices))

    U = dist.Field(bases=basis.vertices)
    for layout in dist.layouts:
        assert np.array_equal(
            layout.local_elements(U.domain, U.scales)[0],
            expected_resident,
        )
        assert layout.local_shape(U.domain, U.scales) == (
            expected_resident.size,
            1,
        )
    assert dist.network_axis_uses_owned_ghost(U.domain)
    assert not dist.network_axis_is_replicated(U.domain)
    assert np.array_equal(
        dist.network_vertex_canonical_elements(U.domain),
        expected_resident,
    )


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_network_vertex_exchange_forward_and_reduce(dtype):
    comm, network, owners, dist, basis = _setup(dtype)
    plan = dist.network_vertex_exchange_plan
    partition = dist.network_partition
    resident = plan.resident

    def vertex_blocks(vertices):
        values = np.empty((2, vertices.size, 3), dtype=dtype)
        for component in range(2):
            for column in range(3):
                values[component, :, column] = (
                    100 * component + 10 * vertices + column + 1
                )
        if np.issubdtype(dtype, np.complexfloating):
            values += 1j * (0.25 * values + 2)
        return values

    data = np.full((2, resident.size, 3), -999, dtype=dtype)
    data[:, plan.owned_positions, :] = vertex_blocks(plan.owned)
    plan.forward(data, axis=1)
    assert np.allclose(data, vertex_blocks(resident))

    contribution = np.empty_like(data)
    for component in range(2):
        for column in range(3):
            contribution[component, :, column] = (
                (comm.rank + 1)
                * (resident + 1)
                * (component + 1)
                + column
            )
    if np.issubdtype(dtype, np.complexfloating):
        contribution += 1j * (0.5 * contribution + comm.rank)
    plan.reduce_add(contribution, axis=1)

    expected = np.zeros_like(contribution)
    for local_index, vertex in enumerate(resident):
        parts = [
            part
            for part, part_resident in enumerate(partition.resident_vertices)
            if vertex in part_resident
        ]
        for component in range(2):
            for column in range(3):
                real = sum(
                    (part + 1) * (vertex + 1) * (component + 1) + column
                    for part in parts
                )
                expected[component, local_index, column] = real
                if np.issubdtype(dtype, np.complexfloating):
                    expected[component, local_index, column] += 1j * sum(
                        0.5
                        * (
                            (part + 1)
                            * (vertex + 1)
                            * (component + 1)
                            + column
                        )
                        + part
                        for part in parts
                    )
    assert np.allclose(contribution, expected)


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_owned_ghost_restriction_and_transpose(dtype):
    comm, network, owners, dist, basis = _setup(dtype)
    partition = dist.network_partition
    resident = partition.resident_vertices[comm.rank]
    owned = partition.owned_vertices[comm.rank]
    owned_positions = np.searchsorted(resident, owned)
    owned_ports = np.flatnonzero(owners[network.port_edges] == comm.rank)

    vertex_values = np.vstack(
        (
            np.arange(network.num_vertices) + 1,
            10 * np.arange(network.num_vertices) + 3,
        )
    ).astype(dtype)
    port_values = np.vstack(
        (
            np.arange(network.num_ports) + 0.5,
            2 * np.arange(network.num_ports) + 1.5,
        )
    ).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        vertex_values += 1j * (0.5 * vertex_values + 1)
        port_values += 1j * (0.25 * port_values - 2)

    U = dist.VectorField(network, name="U", bases=basis.vertices)
    local_vertices = np.full(U["c"].shape, -999, dtype=dtype)
    local_vertices[:, owned_positions, 0] = vertex_values[:, owned]
    U["c"] = local_vertices
    restricted = (basis.R @ U).evaluate()
    assert np.allclose(
        restricted["c"][:, :, 0],
        vertex_values[:, network.endpoints.ravel()[owned_ports]],
    )

    q = dist.VectorField(network, name="q", bases=basis.ports)
    q["c"] = port_values[:, owned_ports, None]
    summed = (basis.R.T @ q).evaluate()
    expected_sums = np.stack(
        [network.restriction_matrix.T @ values for values in port_values]
    )
    assert np.allclose(
        summed["c"][:, :, 0],
        expected_sums[:, resident],
    )


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_vertex_map_and_reduction_count_vertices_once():
    comm, network, owners, dist, basis = _setup(np.float64)
    partition = dist.network_partition
    resident = partition.resident_vertices[comm.rank]
    owned = partition.owned_vertices[comm.rank]
    owned_positions = np.searchsorted(resident, owned)
    values = np.arange(network.num_vertices, dtype=np.float64) + 1

    U = dist.Field(name="U", bases=basis.vertices)
    local_values = np.full(U["c"].shape, -999.0)
    local_values[owned_positions, 0] = values[owned]
    U["c"] = local_values

    direct = d3.CardinalMap(
        basis.vertices,
        basis.vertices,
        sparse.diags(
            np.arange(network.num_vertices, dtype=np.float64) + 2
        ),
        name="vertex_diagonal",
    )
    mapped = (direct @ U).evaluate()
    expected_mapped = (np.arange(network.num_vertices) + 2) * values
    assert np.allclose(mapped["c"].ravel(), expected_mapped[resident])

    integral_operator = d3.Integrate(U, network.edge_coord)
    local_matrix = integral_operator.subspace_matrix(dist.coeff_layout)
    expected_weights = np.isin(resident, owned).astype(np.float64)[None, :]
    assert np.array_equal(local_matrix, expected_weights)
    assert integral_operator.network_matrix_additive
    integral = integral_operator.evaluate()
    assert np.allclose(integral["c"], values.sum())


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_owned_ghost_interpolate_selects_every_canonical_vertex(dtype):
    """Cardinal interpolation selects owner data and replicates the scalar."""
    comm, network, owners, dist, basis = _setup(dtype)
    partition = dist.network_partition
    resident = partition.resident_vertices[comm.rank]
    owned = partition.owned_vertices[comm.rank]
    owned_positions = np.searchsorted(resident, owned)

    values = (3 * np.arange(network.num_vertices) + 0.75).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * (0.5 * values - 2)

    U = dist.Field(name="U_interpolate", bases=basis.vertices)
    local_values = np.full(
        U["c"].shape,
        -999 + 17j if U.is_complex else -999,
        dtype=dtype,
    )
    local_values[owned_positions, 0] = values[owned]
    U["c"] = local_values

    for vertex, expected in enumerate(values):
        selected = d3.Interpolate(
            U,
            network.edge_coord,
            vertex,
        ).evaluate()
        assert selected.domain.bases == ()
        assert selected["c"].shape == (1, 1)
        assert np.allclose(selected["c"], expected)
        replicas = comm.allgather(selected["c"].item())
        assert np.allclose(replicas, expected)

    # Vertices 1 and 2 are owned by rank 1 and ghosted on rank 3, so this
    # explicitly exercises selection of a value with both owner and ghost
    # resident copies.
    assert np.array_equal(partition.ghost_vertices[3], (1, 2))


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_rejects_nonlocal_vertex_cardinal_map():
    comm, network, owners, dist, basis = _setup(np.float64)
    rows = np.arange(network.num_ports)
    columns = (network.endpoints.ravel() + 2) % network.num_vertices
    nonlocal_map = d3.CardinalMap(
        basis.vertices,
        basis.ports,
        sparse.csr_matrix(
            (np.ones(network.num_ports), (rows, columns)),
            shape=(network.num_ports, network.num_vertices),
        ),
        name="nonlocal_vertex_to_port",
    )
    U = dist.Field(name="U", bases=basis.vertices)
    U["c"] = 0

    with pytest.raises(
        NotImplementedError,
        match="outside the partition's resident endpoint halo",
    ):
        (nonlocal_map @ U).evaluate()
