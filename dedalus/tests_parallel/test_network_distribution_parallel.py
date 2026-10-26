"""Parallel storage tests for whole-edge Network layouts."""

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3
from dedalus.core import subsystems as core_subsystems


def _test_values(shape, dtype, rank):
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    values += 100 * rank + 1
    values = values.astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * (values + 0.5)
    return values


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("partition_kind", ["interleaved", "last_owner"])
def test_whole_edge_network_storage(dtype, partition_kind):
    """Keep complete edge intervals together under arbitrary ownership."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    num_edges = 6
    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(num_edges)),
        lengths=np.linspace(0.5, 2.0, num_edges),
    )
    last_rank = comm.size - 1
    if partition_kind == "interleaved":
        owners = np.where(np.arange(num_edges) % 2, 0, last_rank)
    else:
        owners = np.full(num_edges, last_rank)

    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=7, dealias=3 / 2)
    owned_edges = np.flatnonzero(owners == comm.rank)
    owned_ports = np.flatnonzero(owners[network.port_edges] == comm.rank)
    all_vertices = np.arange(network.num_vertices)

    assert dist.distributed_network
    assert dist.layout_comm.rank == comm.rank
    assert dist.layout_comm.size == comm.size
    assert np.array_equal(dist.network_partition.edge_owners, owners)

    role_fields = (
        (dist.Field(bases=basis.edges), owned_edges),
        (dist.Field(bases=basis.ports), owned_ports),
        (dist.Field(bases=basis.vertices), all_vertices),
    )
    for field, expected_elements in role_fields:
        for layout in dist.layouts:
            local_elements = layout.local_elements(field.domain, field.scales)
            assert np.array_equal(local_elements[0], expected_elements)
            assert layout.local_shape(field.domain, field.scales) == (
                expected_elements.size,
                1,
            )

        coefficients = _test_values(field["c"].shape, dtype, comm.rank)
        field["c"] = coefficients
        field.require_grid_space()
        field.require_coeff_space()
        assert np.array_equal(field["c"], coefficients)

        global_values = _test_values(field.global_shape, dtype, 0)
        field.set_global_data(global_values)
        assert np.array_equal(field.allgather_data(), global_values)
        gathered = field.gather_data(root=0)
        if comm.rank == 0:
            assert np.array_equal(gathered, global_values)
        else:
            assert gathered is None
        assert np.isclose(
            field.allreduce_data_norm(),
            np.linalg.norm(global_values.ravel()),
        )
        assert np.isclose(
            field.allreduce_data_max(),
            np.max(np.abs(global_values)),
        )

        if (
            expected_elements.size > 1
            and np.any(np.diff(expected_elements) != 1)
        ):
            with pytest.raises(NotImplementedError, match="local_elements"):
                dist.coeff_layout.slices(field.domain, field.scales)

    u = dist.Field(name="u", bases=basis)
    coefficient_shape = u["c", basis.dealias].shape
    assert coefficient_shape == (owned_edges.size, basis.size)
    coefficients = _test_values(coefficient_shape, dtype, comm.rank)
    u["c"] = coefficients
    u.require_grid_space()
    expected_grid_size = basis.interval.global_grid(
        dist,
        scale=basis.dealias[1],
    ).shape[1]
    assert u["g"].shape == (owned_edges.size, expected_grid_size)
    u.require_coeff_space()
    assert np.allclose(u["c"], coefficients, atol=5e-13, rtol=5e-13)

    edge_index, s = dist.local_grids(basis, scales=basis.dealias)
    reference_grid = basis.interval.global_grid(
        dist,
        scale=basis.dealias[1],
    )
    physical_x = basis.local_physical_grid(
        dist,
        scale=basis.dealias[1],
    )
    assert np.array_equal(edge_index.ravel(), owned_edges)
    assert np.array_equal(s, reference_grid)
    assert physical_x.shape == (owned_edges.size, expected_grid_size)
    assert np.allclose(
        physical_x,
        network.lengths[owned_edges, None] * reference_grid,
    )

    # Seeded random fields follow one canonical global stream regardless of
    # noncontiguous ownership, complex storage, or idle ranks.
    random_seed = 8675309
    u.fill_random(layout="g", seed=random_seed, distribution="uniform")
    random_shape = u.global_shape
    rng = np.random.default_rng(random_seed)
    if np.issubdtype(dtype, np.complexfloating):
        random_data = rng.uniform(size=random_shape + (2,))
        expected_random = random_data[..., 0] + 1j * random_data[..., 1]
    else:
        expected_random = rng.uniform(size=random_shape)
    assert np.allclose(u["g"], expected_random[owned_edges])
    assert np.allclose(u.allgather_data("g"), expected_random)

    port_random = dist.Field(bases=basis.ports)
    port_random.fill_random(
        layout="g",
        seed=random_seed,
        distribution="uniform",
    )
    rng = np.random.default_rng(random_seed)
    if np.issubdtype(dtype, np.complexfloating):
        random_data = rng.uniform(size=port_random.global_shape + (2,))
        expected_ports = random_data[..., 0] + 1j * random_data[..., 1]
    else:
        expected_ports = rng.uniform(size=port_random.global_shape)
    assert np.allclose(port_random["g"], expected_ports[owned_ports])

    vertex_random = dist.Field(bases=basis.vertices)
    vertex_random.fill_random(
        layout="g",
        seed=random_seed,
        distribution="uniform",
    )
    rng = np.random.default_rng(random_seed)
    if np.issubdtype(dtype, np.complexfloating):
        random_data = rng.uniform(size=vertex_random.global_shape + (2,))
        expected_vertices = random_data[..., 0] + 1j * random_data[..., 1]
    else:
        expected_vertices = rng.uniform(size=vertex_random.global_shape)
    assert np.allclose(vertex_random["g"], expected_vertices)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_cardinal_maps(dtype):
    """Apply trace and incidence maps without replicating edge fields."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    num_edges = 6
    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(num_edges)),
        lengths=np.linspace(0.5, 2.0, num_edges),
    )
    owners = np.where(
        np.arange(num_edges) % 2,
        0,
        comm.size - 1,
    )
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=8)
    owned_edges = dist.network_partition.edges[comm.rank]
    owned_ports = np.flatnonzero(owners[network.port_edges] == comm.rank)

    intercept = np.arange(num_edges, dtype=np.float64) + 1
    slope = np.arange(num_edges, dtype=np.float64) + 2
    if np.issubdtype(dtype, np.complexfloating):
        intercept = intercept + 1j * (intercept + 0.25)
        slope = slope - 1j * (slope + 0.5)
    u = dist.Field(name="u", bases=basis)
    s = dist.local_grid(basis.interval)
    u["g"] = (
        intercept[owned_edges, None]
        + slope[owned_edges, None] * s
    )

    trace_values = d3.dtrace(u).evaluate()
    global_trace = np.column_stack((intercept, intercept + slope)).ravel()
    assert np.array_equal(
        trace_values.local_elements()[0],
        owned_ports,
    )
    assert np.allclose(
        trace_values["g"].ravel(),
        global_trace[owned_ports],
    )
    assert np.allclose(trace_values.allgather_data("g").ravel(), global_trace)

    normal_values = d3.ntrace(u).evaluate()
    physical_slope = slope / network.lengths
    global_normal = np.column_stack(
        (-physical_slope, physical_slope),
    ).ravel()
    assert np.allclose(
        normal_values["g"].ravel(),
        global_normal[owned_ports],
    )

    derivative = basis.derivative(u).evaluate()
    assert np.allclose(
        derivative["g"],
        physical_slope[owned_edges, None],
    )

    U = dist.Field(name="U", bases=basis.vertices)
    vertex_values = np.arange(network.num_vertices, dtype=np.float64) + 3
    if np.issubdtype(dtype, np.complexfloating):
        vertex_values = vertex_values + 2j * vertex_values
    U["g"] = vertex_values.reshape(U["g"].shape)
    restricted = (basis.restriction @ U).evaluate()
    expected_restricted = vertex_values[network.endpoints.ravel()]
    assert np.allclose(
        restricted["g"].ravel(),
        expected_restricted[owned_ports],
    )

    vertex_sums = (basis.restriction.T @ trace_values).evaluate()
    expected_sums = network.restriction_matrix.T @ global_trace
    assert np.allclose(vertex_sums["g"].ravel(), expected_sums)

    # A generic map whose rows depend on remotely owned edges falls back to an
    # indexed source gather while retaining local output ownership.
    columns = (np.arange(num_edges) + 1) % num_edges
    shift_matrix = sparse.csr_matrix(
        (np.ones(num_edges), (np.arange(num_edges), columns)),
        shape=(num_edges, num_edges),
    )
    shift = d3.CardinalMap(
        basis.edges,
        basis.edges,
        shift_matrix,
        name="shift_edges",
    )
    edge_values = dist.Field(bases=basis.edges)
    edge_values.set_global_data(intercept[:, None])
    shifted = (shift @ edge_values).evaluate()
    expected_shifted = intercept[columns]
    assert np.allclose(
        shifted["g"].ravel(),
        expected_shifted[owned_edges],
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_global_reductions(dtype):
    """Integrate and average owned edge data without gathering the field."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=(comm.size - 1, 0),
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    unit = dtype(1 + 0.25j) if np.issubdtype(dtype, np.complexfloating) else dtype(1)
    u["g"] = unit

    integral = d3.Integrate(u, network).evaluate()
    average = d3.Average(u, network).evaluate()
    edge_integral = d3.Integrate(u, network.edge_coord).evaluate()
    assert np.allclose(integral.allgather_data("g"), 3 * unit)
    assert np.allclose(average.allgather_data("g"), unit)
    assert np.allclose(edge_integral.allgather_data("g"), 3 * unit)
    assert np.allclose(
        u.allreduce_L2_norm(normalize_volume=False),
        np.sqrt(3) * np.abs(unit),
    )
    assert np.allclose(u.allreduce_L2_norm(), np.abs(unit))

    # Replicated vertex data must contribute once, not once per MPI rank.
    U = dist.Field(name="U", bases=basis.vertices)
    vertex_values = np.array((1, 2, 5), dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        vertex_values += 1j * np.array((0.5, 1, 2), dtype=dtype)
    if comm.rank == 0:
        U["g"] = vertex_values.reshape(U["g"].shape)
    else:
        U["g"] = (vertex_values + 100).reshape(U["g"].shape)
    vertex_integral = d3.Integrate(U, network.edge_coord).evaluate()
    vertex_average = d3.Average(U, network.edge_coord).evaluate()
    assert np.allclose(
        vertex_integral.allgather_data("g"),
        vertex_values.sum(),
    )
    assert np.allclose(
        vertex_average.allgather_data("g"),
        vertex_values.mean(),
    )

    port = dist.Field(name="port", bases=basis.ports)
    port_values = np.arange(1, network.num_ports + 1, dtype=dtype)
    port.set_global_data(port_values[:, None])
    port_integral = d3.Integrate(port, network.edge_coord).evaluate()
    port_average = d3.Average(port, network.edge_coord).evaluate()
    assert np.allclose(
        port_integral.allgather_data("g"),
        port_values.sum(),
    )
    assert np.allclose(
        port_average.allgather_data("g"),
        port_values.mean(),
    )


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_explicit_edge_first_reductions(dtype):
    """Keep a reduced position field replicated through the second reduction."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    lengths = np.array((0.5, 1.0, 1.5, 2.0))
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3), (3, 4)),
        lengths=lengths,
    )
    owners = np.array((comm.size - 1, 1, comm.size - 1, 2))
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=12)
    u = dist.Field(name="u", bases=basis)
    owned_edges = dist.network_partition.edges[comm.rank]
    s = dist.local_grid(basis.interval)
    amplitudes = np.arange(1, network.num_edges + 1, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        amplitudes += 0.25j * amplitudes[::-1]
    u["g"] = amplitudes[owned_edges, None] * (1 + s)

    edge_profile = d3.Integrate(u, network.edge_coord).evaluate()
    edge_first = d3.Integrate(
        edge_profile,
        network.position_coord,
    ).evaluate()
    edge_average = d3.Average(u, network.edge_coord).evaluate()
    average_edge_first = d3.Average(
        edge_average,
        network.position_coord,
    ).evaluate()
    weighted_amplitude = np.dot(lengths, amplitudes)

    # The position-only intermediates are complete on every rank, including
    # the idle root, and agree with the serial result above.
    assert edge_profile["g"].shape == (1, basis.size)
    assert np.allclose(edge_profile["g"], weighted_amplitude * (1 + s))
    assert np.allclose(edge_first["g"], 1.5 * weighted_amplitude)
    assert np.allclose(edge_first["g"], d3.Integrate(u).evaluate()["g"])
    assert np.allclose(
        average_edge_first["g"],
        1.5 * weighted_amplitude / lengths.sum(),
    )
    assert np.allclose(average_edge_first["g"], d3.Average(u).evaluate()["g"])
    assert dist.network_partition.edges[0].size == 0

    # Gathering and norms use rank zero as the authoritative copy for any
    # replicated position-only field.
    expected_profile = weighted_amplitude * (1 + s)
    if comm.rank != 0:
        edge_profile["g"] += 100
    assert np.allclose(edge_profile.allgather_data("g"), expected_profile)
    assert np.allclose(
        edge_profile.allreduce_data_max("g"),
        np.max(np.abs(expected_profile)),
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_chain_poisson(dtype):
    """Solve an LBVP while retaining only owned edge fields on each rank."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    owners = np.array((comm.size - 1, 0))
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)
    alpha["g"] = np.array((1, 0, 1)).reshape(alpha["g"].shape)
    beta["g"] = np.array((0, 1, 0)).reshape(beta["g"].shape)
    if comm.rank == 0:
        g["g"] = np.array((1, 0, 0)).reshape(g["g"].shape)
    else:
        g["g"] = np.array((11, 13, 17)).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = g")
    solver = problem.build_solver()
    assert solver.matrix_assembly_root is None
    solver.solve()

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    owned_edges = dist.network_partition.edges[comm.rank]
    assert np.array_equal(factorization.owned_edges, owned_edges)
    assert factorization.edge_response_rows_local == (
        owned_edges.size * (basis.size + 2)
    )

    edge_index, _ = dist.local_grids(basis)
    x = basis.local_physical_grid(dist)
    edge_starts = np.array((0, 1))
    expected_local = 1 - (edge_starts[owned_edges, None] + x) / 3
    assert np.array_equal(edge_index.ravel(), owned_edges)
    assert np.allclose(u["g"], expected_local, atol=2e-12)
    assert np.allclose(U["g"].ravel(), (1, 2 / 3, 0), atol=2e-12)

    expected_global = np.vstack(
        (
            1 - dist.local_grid(basis.interval) / 3,
            (2 / 3) * (1 - dist.local_grid(basis.interval)),
        )
    )
    assert np.allclose(u.allgather_data("g"), expected_global, atol=2e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_nonzero_constant_rhs(dtype):
    """Broadcast constants onto edge, port, and vertex equation domains."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=(0.5, 1, 2),
    )
    # Keep rank zero, and every rank except the last, free of edge data.  This
    # exposes any accidental reliance on the rank-zero-owned constant field.
    owners = np.full(network.num_edges, comm.size - 1, dtype=int)
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)

    if np.issubdtype(dtype, np.complexfloating):
        forcing = 2 - 0.75j
        trace_jump = -0.25 + 0.125j
        boundary_value = 1 + 0.5j
    else:
        forcing = 2.0
        trace_jump = -0.25
        boundary_value = 1.0

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = trace_jump")
    problem.add_equation("U = boundary_value")
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    reference = dist.local_grid(basis.interval)
    physical_x = network.lengths[:, None] * reference
    expected = (
        boundary_value
        + trace_jump
        + forcing
        * physical_x
        * (network.lengths[:, None] - physical_x)
        / 2
    )
    assert dist.network_partition.edges[0].size == 0
    assert np.allclose(u.allgather_data("g"), expected, atol=2e-11)
    assert np.allclose(U["g"], boundary_value, atol=2e-11)
    assert np.allclose(
        d3.dtrace(u).evaluate().allgather_data("g"),
        boundary_value + trace_jump,
        atol=2e-11,
    )


@pytest.mark.mpi(min_size=2)
def test_distributed_network_edge_coefficient_rebuild():
    """Rebuild local edge NCC blocks without changing field ownership."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    owners = np.array((comm.size - 1, 0))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=7)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    kappa = dist.Field(name="kappa", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)
    alpha["g"] = np.array((1, 0, 1)).reshape(alpha["g"].shape)
    beta["g"] = np.array((0, 1, 0)).reshape(beta["g"].shape)
    g["g"] = np.array((1, 0, 0)).reshape(g["g"].shape)
    kappa.set_global_data(np.array((1, 4))[:, None])

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(kappa*dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ (dtrace(kappa)*ntrace(u))) = g"
    )
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(U["g"].ravel(), (1, 0.2, 0), atol=2e-12)

    kappa.set_global_data(np.array((4, 1))[:, None])
    solver.solve(rebuild_matrices=True)
    assert np.allclose(U["g"].ravel(), (1, 0.8, 0), atol=2e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_spatially_varying_diffusion(dtype):
    """Build spatial NCCs from owned edges, including on idle ranks."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    owners = np.array((comm.size - 1, 0))
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=10)

    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    kappa = dist.Field(name="kappa", bases=basis)
    forcing = dist.Field(name="forcing", bases=basis)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    boundary_data = dist.Field(name="boundary_data", bases=basis.vertices)

    owned_edges = dist.network_partition.edges[comm.rank]
    local_x = basis.local_physical_grid(dist)
    edge_starts = np.array((0, 1))
    physical_x = edge_starts[owned_edges, None] + local_x
    expected_local = physical_x**2
    kappa["g"] = 1 + physical_x
    forcing["g"] = -2 - 4*physical_x
    alpha["g"] = np.array((1, 0, 1)).reshape(alpha["g"].shape)
    beta["g"] = np.array((0, 1, 0)).reshape(beta["g"].shape)
    boundary_data["g"] = np.array((0, 0, 9)).reshape(
        boundary_data["g"].shape
    )

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(kappa*dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "= forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ (dtrace(kappa)*ntrace(u))) "
        "= boundary_data"
    )
    solver = problem.build_solver()
    solver.solve()

    flux = R.T @ (d3.dtrace(kappa) * d3.ntrace(u))
    assert u["g"].shape == expected_local.shape
    assert np.allclose(u["g"], expected_local, atol=3e-11)
    assert np.allclose(U["g"].ravel(), (0, 1, 9), atol=3e-11)
    assert np.allclose(
        flux.evaluate()["g"].ravel(),
        (0, 0, 24),
        atol=3e-10,
    )

    s = dist.local_grid(basis.interval)
    expected_global = np.vstack((s**2, (1 + 2*s) ** 2))
    assert np.allclose(
        u.allgather_data("g"),
        expected_global,
        atol=3e-11,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_coupled_pressure_flux_system(dtype):
    """Condense a coupled first-order edge system with signed port fluxes."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    owners = np.array((comm.size - 1, 0))
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=10)

    pressure = dist.Field(name="pressure", bases=basis)
    flux = dist.Field(name="flux", bases=basis)
    vertex_pressure = dist.Field(
        name="vertex_pressure",
        bases=basis.vertices,
    )
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    boundary_data = dist.Field(
        name="boundary_data",
        bases=basis.vertices,
    )
    port_sign = dist.Field(name="port_sign", bases=basis.ports)

    alpha["g"] = np.array((1, 0, 1)).reshape(alpha["g"].shape)
    beta["g"] = np.array((0, 1, 0)).reshape(beta["g"].shape)
    boundary_data["g"] = np.array((1, 0, 4)).reshape(
        boundary_data["g"].shape
    )
    port_sign.set_global_data(network.port_signs[:, None])

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(1)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP(
        [pressure, flux, vertex_pressure, tau_1, tau_2],
        namespace=locals(),
    )
    problem.add_equation("flux + dx(pressure) = 0")
    problem.add_equation(
        "dx(flux) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(pressure) - R @ vertex_pressure = 0")
    problem.add_equation(
        "alpha*vertex_pressure "
        "+ beta*(R.T @ (port_sign*dtrace(flux))) = boundary_data"
    )
    solver = problem.build_solver()
    solver.solve()

    owned_edges = dist.network_partition.edges[comm.rank]
    local_x = basis.local_physical_grid(dist)
    edge_starts = np.array((0, 1))
    physical_x = edge_starts[owned_edges, None] + local_x
    assert np.allclose(pressure["g"], 1 + physical_x, atol=3e-11)
    assert np.allclose(flux["g"], -np.ones_like(physical_x), atol=3e-11)
    assert np.allclose(
        vertex_pressure["g"].ravel(),
        (1, 2, 4),
        atol=3e-11,
    )
    vertex_flux = R.T @ (port_sign * d3.dtrace(flux))
    assert np.allclose(
        vertex_flux.evaluate()["g"].ravel(),
        (1, 0, -1),
        atol=3e-11,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_network_mixed_interface_blocks(dtype):
    """Sum replicated and owned-edge terms in one vertex equation."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition=(comm.size - 1, 0),
    )
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    w = dist.Field(name="w", bases=basis.ports)
    g = dist.Field(name="g", bases=basis.vertices)
    w["g"] = np.ones(w["g"].shape, dtype=dtype)
    g["g"] = np.array((4, 13, 20), dtype=dtype).reshape(g["g"].shape)

    R = basis.restriction
    A = d3.CardinalMap(
        basis.vertices,
        basis.vertices,
        sparse.csr_matrix(
            [
                [2, 1, 0],
                [0, 3, 1],
                [0, -1, 4],
            ],
            dtype=dtype,
        ),
        name="A",
    )
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "A @ U + R.T @ ntrace(u) + R.T @ (w*(R @ U)) = g"
    )
    solver = problem.build_solver()
    solver.solve()

    expected_U = np.array((1, 2, 4), dtype=dtype)
    assert np.allclose(U["g"].ravel(), expected_U, atol=2e-12)

    owned_edges = dist.network_partition.edges[comm.rank]
    s = dist.local_grid(basis.interval)
    expected_global_u = np.vstack((1 + s, 2 + 2*s)).astype(dtype)
    assert np.allclose(u["g"], expected_global_u[owned_edges], atol=2e-12)
    assert np.allclose(u.allgather_data("g"), expected_global_u, atol=2e-12)

    # With four or more ranks, this ownership deliberately leaves idle ranks
    # between the two active ranks.
    if comm.size > 2 and 0 < comm.rank < comm.size - 1:
        assert owned_edges.size == 0
        assert u["g"].shape == (0, basis.size)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("cross_owner", [False, True], ids=["local", "remote"])
def test_distributed_network_implicit_cardinal_map_ownership(cross_owner):
    """Accept owner-local maps and collectively reject remote LHS entries."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(comm.size - 1, 0),
    )
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)

    map_matrix = sparse.eye(network.num_edges, format="lil")
    if cross_owner:
        map_matrix[0, 1] = 0.5
    edge_map = d3.CardinalMap(
        basis.edges,
        basis.edges,
        map_matrix.tocsr(),
        name="edge_map",
    )
    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + edge_map @ u "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = 0")
    solver = problem.build_solver()

    if cross_owner:
        with pytest.raises(
            RuntimeError,
            match="CardinalMap ownership validation.*crosses Network owners",
        ):
            solver.solve()
    else:
        solver.solve()
        assert np.allclose(u["g"], 0, atol=2e-12)
        assert np.allclose(U["g"], 0, atol=2e-12)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "failure_stage",
    ["ncc", "expression", "finalization"],
)
def test_distributed_network_matrix_assembly_errors_are_collective(
    monkeypatch,
    failure_stage,
):
    """Turn a one-rank assembly fault into the same error on every rank."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    failing_rank = comm.size - 1
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(failing_rank, 0),
    )
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = 0")
    solver = problem.build_solver()

    marker = f"deliberate rank-local {failure_stage} failure"
    target_expression = problem.eqs[0]["L"]
    if failure_stage == "ncc":
        expression_type = type(target_expression)
        original = expression_type.gather_ncc_coeffs

        def fail_on_one_rank(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if self is target_expression and comm.rank == failing_rank:
                raise ValueError(marker)
            return result

        monkeypatch.setattr(
            expression_type,
            "gather_ncc_coeffs",
            fail_on_one_rank,
        )
        expected_stage = "Distributed Network NCC gathering"
    elif failure_stage == "expression":
        expression_type = type(target_expression)
        original = expression_type.expression_matrices

        def fail_on_one_rank(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if self is target_expression and comm.rank == failing_rank:
                raise ValueError(marker)
            return result

        monkeypatch.setattr(
            expression_type,
            "expression_matrices",
            fail_on_one_rank,
        )
        expected_stage = "Distributed Network expression-matrix construction"
    else:
        original = core_subsystems.Subproblem.build_matrices

        def fail_on_one_rank(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if comm.rank == failing_rank:
                raise ValueError(marker)
            return result

        monkeypatch.setattr(
            core_subsystems.Subproblem,
            "build_matrices",
            fail_on_one_rank,
        )
        expected_stage = "Distributed Network subproblem matrix finalization"

    with pytest.raises(RuntimeError) as caught:
        solver.solve()
    message = str(caught.value)
    assert expected_stage in message
    assert f"failed on network rank {failing_rank}" in message
    assert marker in message
    assert comm.allgather(message) == [message] * comm.size

    # Four-rank runs exercise both active owners and intervening idle ranks.
    if comm.size > 2 and 0 < comm.rank < failing_rank:
        assert dist.network_partition.edges[comm.rank].size == 0


@pytest.mark.mpi(min_size=2)
def test_distributed_network_rejects_cross_partition_direct_map():
    """Reject direct maps coupling interiors owned by separate ranks."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(comm.size - 1, 0),
    )
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    R = basis.restriction
    cross_partition = d3.CardinalMap(
        basis.vertices,
        basis.vertices,
        sparse.csr_matrix(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ]
        ),
        name="cross_partition",
    )
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("cross_partition @ U + R.T @ ntrace(u) = 0")
    solver = problem.build_solver()
    with pytest.raises(
        RuntimeError,
        match="Subgraph structure validation",
    ):
        solver.solve()


@pytest.mark.mpi(min_size=2)
def test_distributed_network_requires_dtn_solver():
    """Reject independent local solves for a collectively coupled problem."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(comm.size - 1,),
    )
    basis = d3.NetworkBasis(network, size=5)
    U = dist.Field(name="U", bases=basis.vertices)
    problem = d3.LBVP([U], namespace=locals())
    problem.add_equation("U = 0")
    with pytest.raises(ValueError, match="require network_solver='dtn'"):
        problem.build_solver(network_solver=None)


@pytest.mark.mpi(min_size=2)
def test_distributed_network_root_owns_no_edge():
    """Allow rank zero to own only the authoritative vertex data."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    owners = (comm.size - 1,)
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)
    g["g"] = np.array((1, 0)).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = g")
    solver = problem.build_solver()
    solver.solve()

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    assert factorization.owned_edges.size == int(
        comm.rank == comm.size - 1
    )
    if comm.rank == 0:
        assert u["g"].shape == (0, basis.size)
    assert factorization.schur_matrix.shape == (0, 0)
    s = dist.local_grid(basis.interval)
    expected_local = np.empty_like(u["g"])
    if comm.rank == comm.size - 1:
        expected_local[:] = 1 - s
    assert np.allclose(u["g"], expected_local)
    assert np.allclose(U["g"].ravel(), (1, 0))

    subproblem = solver.subproblems[0]
    input_size = subproblem.L_min.shape[1]
    if comm.rank == 0:
        input_size -= 1
    invalid_input = np.zeros(input_size)
    with pytest.raises(
        RuntimeError,
        match="matrix-application validation",
    ):
        solver.apply_subproblem_matrix(
            subproblem.L_min,
            invalid_input,
            subproblem=subproblem,
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("timestepper", [d3.SBDF1, d3.RK111, d3.RK222])
def test_distributed_network_heat_step(timestepper):
    """Advance implicit stages without gathering distributed edge state."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    owners = np.array((comm.size - 1, 0))
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
    w = dist.Field(name="w", bases=basis.ports)
    degree = dist.Field(name="degree", bases=basis.vertices)
    alpha["g"] = np.array((1, 0, 1)).reshape(alpha["g"].shape)
    beta["g"] = np.array((0, 1, 0)).reshape(beta["g"].shape)
    w["g"] = 1
    degree["g"] = network.degree.reshape(degree["g"].shape)

    owned_edges = dist.network_partition.edges[comm.rank]
    x = basis.local_physical_grid(dist)
    edge_starts = np.array((0, 1))
    initial_u = np.sin(
        np.pi * (edge_starts[owned_edges, None] + x) / 3
    )
    initial_U = np.array((0, np.sin(np.pi / 3), 0))
    u["g"] = initial_u
    if comm.rank == 0:
        U["g"] = initial_U.reshape(U["g"].shape)
    else:
        U["g"] = np.array((4, 5, 6)).reshape(U["g"].shape)

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
    problem.add_equation(
        "alpha*U + beta*(R.T @ ntrace(u)) "
        "+ R.T @ (w*(R @ U)) - degree*U = 0"
    )
    solver = problem.build_solver(timestepper)
    timestep = 0.01
    solver.step(timestep)

    eigenvalue = (np.pi / 3) ** 2

    def amplification(step):
        z = step * eigenvalue
        if timestepper is d3.RK222:
            gamma = (2 - np.sqrt(2)) / 2
            stage_one = 1 / (1 + gamma * z)
            return (1 - z * (1 - gamma) * stage_one) / (1 + gamma * z)
        return 1 / (1 + z)

    decay = amplification(timestep)
    assert np.allclose(u["g"], decay * initial_u, atol=3e-10)
    assert np.allclose(U["g"].ravel(), decay * initial_U, atol=3e-10)

    second_timestep = 0.02
    solver.step(second_timestep)
    decay *= amplification(second_timestep)
    assert np.allclose(u["g"], decay * initial_u, atol=4e-10)
    assert np.allclose(U["g"].ravel(), decay * initial_U, atol=4e-10)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("timestepper", [d3.SBDF1, d3.RK222])
def test_distributed_network_explicit_reaction_step(timestepper):
    """Evaluate a nonzero explicit edge RHS with idle MPI ranks."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(comm.size - 1,),
    )
    basis = d3.NetworkBasis(network, size=16)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)

    owned_edges = dist.network_partition.edges[comm.rank]
    x = basis.local_physical_grid(dist)
    initial_u = np.sin(np.pi * x)
    u["g"] = initial_u
    U["g"] = 0

    R = basis.restriction
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    reaction_rate = 0.7
    problem = d3.IVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "dt(u) - dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "= reaction_rate*u"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = 0")
    solver = problem.build_solver(timestepper)

    timestep = 0.01
    solver.step(timestep)

    diffusion = np.pi**2
    z = timestep * diffusion
    q = timestep * reaction_rate
    if timestepper is d3.SBDF1:
        amplification = (1 + q) / (1 + z)
    else:
        gamma = (2 - np.sqrt(2)) / 2
        delta = 1 - 1 / (2 * gamma)
        stage_one = (1 + gamma*q) / (1 + gamma*z)
        amplification = (
            1
            + delta*q
            + ((1 - delta)*q - (1 - gamma)*z) * stage_one
        ) / (1 + gamma*z)

    # The first evaluator output is the explicit reaction term.  Its global
    # norm must be nonzero even though every rank except the last owns no edge.
    explicit_rhs = solver.F[0]
    assert explicit_rhs.allreduce_data_norm(layout="g") > 0
    if comm.rank == comm.size - 1:
        expected_owned_edges = np.array((0,))
    else:
        expected_owned_edges = np.array([], dtype=int)
    assert np.array_equal(owned_edges, expected_owned_edges)
    if comm.rank != comm.size - 1:
        assert u["g"].shape == (0, basis.size)
        assert explicit_rhs["g"].shape == (0, basis.size)

    implicit_only = 1 / (1 + z)
    assert not np.isclose(amplification, implicit_only, atol=1e-6)
    assert np.allclose(u["g"], amplification * initial_u, atol=3e-11)
    assert np.allclose(U["g"], 0, atol=3e-11)


@pytest.mark.mpi(min_size=2)
def test_distributed_network_topology_mismatch_fails_collectively():
    """Reject incompatible rank-local topology before allocating fields."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    if comm.rank == 0:
        endpoints = ((0, 1), (1, 2))
    else:
        endpoints = ((0, 2), (2, 1))
    network = d3.Network(endpoints=endpoints, lengths=(1, 1))

    with pytest.raises(
        RuntimeError,
        match="Distributed Network topology validation",
    ):
        d3.Distributor(
            network,
            dtype=np.float64,
            network_partition=(0, comm.size - 1),
        )


@pytest.mark.mpi(min_size=2)
def test_distributed_network_mode_mismatch_fails_collectively():
    """All ranks must opt into whole-edge storage together."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    partition = (0,) if comm.rank == 0 else None

    with pytest.raises(RuntimeError, match="network_partition on every MPI rank"):
        d3.Distributor(
            network,
            dtype=np.float64,
            network_partition=partition,
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("mismatch", ["size", "dealias", "actual_basis"])
def test_distributed_network_basis_mismatch_fails_collectively(mismatch):
    """Reject rank-dependent interval configurations before assembly."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(comm.size - 1,),
    )
    size = 6 + comm.rank if mismatch == "size" else 6
    dealias = 1 + comm.rank / 4 if mismatch == "dealias" else 1
    basis = d3.NetworkBasis(network, size=size, dealias=dealias)
    if mismatch == "actual_basis":
        interval = basis.interval.derivative_basis(comm.rank % 2)
        u_bases = (basis.edges, interval)
    else:
        u_bases = basis
    u = dist.Field(name="u", bases=u_bases)
    U = dist.Field(name="U", bases=basis.vertices)
    problem = d3.LBVP([u, U], namespace=locals())
    problem.add_equation("u = 0")
    problem.add_equation("U = 0")

    with pytest.raises(
        RuntimeError,
        match="Network discretization validation",
    ):
        problem.build_solver()
