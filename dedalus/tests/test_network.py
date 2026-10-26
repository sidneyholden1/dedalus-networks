"""Test metric-network normalization and spectral traces."""

import itertools
import copy
import pickle

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3
from dedalus.core import network as network_core


def build_chain():
    return d3.Network(
        vertex_ids=("left", "junction", "right"),
        edge_ids=("short", "long"),
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )


def test_owned_edge_factorization_uses_packed_rows():
    """Owned block solves never allocate unowned global edge rows."""

    class DenseFactorization:
        def __init__(self, matrix, solver):
            self.matrix = matrix.toarray()

        def solve(self, rhs):
            return np.linalg.solve(self.matrix, rhs)

    class Solver:
        entry_cutoff = 1e-12

    blocks = (
        np.array([[2.0, 0.0], [0.0, 4.0]]),
        np.array([[3.0, 1.0], [0.0, 2.0]]),
    )
    labels = np.array((0, 1, 0, 1))
    edge_columns = (np.flatnonzero(labels == 0), np.flatnonzero(labels == 1))
    matrix = sparse.lil_matrix((4, 4))
    for columns, block in zip(edge_columns, blocks):
        matrix[np.ix_(columns, columns)] = block
    matrix = matrix.tocsr()
    factorization = network_core._EdgeBlockFactorization(
        matrix,
        labels,
        labels,
        DenseFactorization,
        Solver(),
        owned_edges=(1,),
    )

    rhs = np.arange(1.0, 9.0).reshape(4, 2)
    solution = factorization.solve(rhs)
    assert np.array_equal(factorization.local_columns, edge_columns[1])
    assert solution.shape == (2, 2)
    assert np.allclose(
        solution,
        np.linalg.solve(blocks[1], rhs[edge_columns[1]]),
    )

    coupling = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
            ]
        )
    )
    response = factorization.response(coupling)
    assert response.shape == (2, 3)
    assert np.allclose(
        response.toarray(),
        np.linalg.solve(
            blocks[1],
            coupling[edge_columns[1]].toarray(),
        ),
    )
    gathered = network_core._allgatherv_packed_rows(
        solution,
        (solution.shape[0],),
        MPI.COMM_SELF,
    )
    assert np.array_equal(gathered, solution)


def test_network_normalization_and_ports():
    network = d3.Network(
        vertex_ids=("a", "b"),
        edge_ids=("loop", "first", "second"),
        endpoints=((0, 0), (0, 1), (0, 1)),
        lengths=(3, 1, 2),
    )

    assert network.num_vertices == 2
    assert network.num_edges == 3
    assert network.num_ports == 6
    assert np.array_equal(network.degree, (4, 2))
    assert np.array_equal(
        network.restriction_matrix.toarray(),
        [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]],
    )
    assert not network.endpoints.flags.writeable
    assert not network.lengths.flags.writeable


def test_network_identity_equality_and_hash():
    first = d3.Network(endpoints=((0, 1),), lengths=(1,))
    second = d3.Network(endpoints=((0, 1),), lengths=(2,))

    assert first == first
    assert first != second
    assert hash(first) == hash(first)


def test_network_is_frozen_and_pickleable():
    network = d3.Network(
        vertex_ids=("left", "right"),
        edge_ids=("edge",),
        endpoints=((0, 1),),
        lengths=(2,),
        vertex_groups={"boundary": (0, 1)},
        coordinate_names=("edge", "s"),
    )
    restored = pickle.loads(pickle.dumps(network))
    copied = copy.copy(network)
    deepcopied = copy.deepcopy(network)

    assert restored is not network
    assert restored.names == ("edge", "s")
    assert restored.vertex_ids == network.vertex_ids
    assert restored.edge_ids == network.edge_ids
    assert np.array_equal(restored.endpoints, network.endpoints)
    assert np.array_equal(restored.lengths, network.lengths)
    assert np.array_equal(restored.vertex_groups["boundary"], (0, 1))
    assert copied is not network
    assert deepcopied is not network
    assert np.array_equal(copied.lengths, network.lengths)
    assert np.array_equal(deepcopied.endpoints, network.endpoints)
    with pytest.raises(AttributeError, match="immutable"):
        network.lengths = np.ones(1)
    with pytest.raises(TypeError):
        network.vertex_groups["interior"] = np.array([], dtype=int)


def test_empty_network_with_isolated_vertices():
    network = d3.Network(
        vertex_ids=("first", "second"),
        endpoints=(),
        lengths=(),
    )
    partition = d3.NetworkPartition.connected(network, 3)

    assert network.num_edges == 0
    assert network.num_vertices == 2
    assert network.restriction_matrix.shape == (0, 2)
    assert np.array_equal(network.degree, (0, 0))
    assert partition.edge_owners.size == 0
    assert np.array_equal(partition.separator_vertices, (0, 1))

    with pytest.raises(ValueError, match="requires at least one edge"):
        d3.NetworkBasis(network, size=6)


def test_network_coordinate_names_must_be_distinct():
    with pytest.raises(ValueError, match="must be distinct"):
        d3.Network(
            endpoints=((0, 1),),
            lengths=(1,),
            coordinate_names=("x", "x"),
        )


def test_network_basis_is_a_pickleable_bundle():
    basis = d3.NetworkBasis(build_chain(), size=6)
    restored = pickle.loads(pickle.dumps(basis))
    dist = d3.Distributor(restored.network, dtype=np.float64)
    u = dist.Field(name="u", bases=restored)

    assert not isinstance(basis, tuple)
    assert tuple(basis) == (basis.edges, basis.interval)
    assert restored.size == basis.size
    assert restored.network.num_edges == basis.network.num_edges
    assert u.domain.bases == (restored.edges, restored.interval)


def test_network_basis_reuses_canonical_bundle_and_maps():
    network = build_chain()
    first = d3.NetworkBasis(network, size=6)
    identical = d3.NetworkBasis(network, size=6)
    equivalent = d3.NetworkBasis(
        network=network,
        size=np.int64(6),
        interval_basis=d3.Chebyshev,
        dealias=[np.float64(1)],
        library="fftw_dct",
    )

    assert identical is first
    assert equivalent is first
    assert equivalent.edges is first.edges
    assert equivalent.vertices is first.vertices
    assert equivalent.ports is first.ports
    assert equivalent.interval is first.interval
    assert equivalent.restriction is first.restriction
    assert first.interval_dealias == (1.0,)
    assert first.library == first.interval.library == "fftw_dct"

    dist = d3.Distributor(network, dtype=np.float64)
    u = dist.Field(bases=equivalent)
    endpoint_values = first.dtrace(u)
    assert endpoint_values.domain.get_basis(network.edge_coord) is first.ports


def test_network_basis_normalizes_interval_factory_defaults():
    network = build_chain()
    implicit = d3.NetworkBasis(
        network,
        size=6,
        interval_basis=d3.Legendre,
    )
    explicit = d3.NetworkBasis(
        network,
        size=6,
        interval_basis=d3.Legendre,
        dealias=(1,),
        library="matrix",
    )

    assert explicit is implicit
    assert implicit.interval_dealias == (1.0,)
    assert implicit.library == implicit.interval.library == "matrix"


def test_network_basis_cache_preserves_network_identity():
    first_network = build_chain()
    equivalent_snapshot = build_chain()
    first = d3.NetworkBasis(first_network, size=6)
    different = d3.NetworkBasis(equivalent_snapshot, size=6)

    assert equivalent_snapshot is not first_network
    assert np.array_equal(equivalent_snapshot.endpoints, first_network.endpoints)
    assert np.array_equal(equivalent_snapshot.lengths, first_network.lengths)
    assert different is not first
    assert different.edges is not first.edges


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"size": 7}, id="size"),
        pytest.param({"size": 6, "dealias": 3 / 2}, id="dealias"),
        pytest.param({"size": 6, "library": "matrix"}, id="library"),
        pytest.param(
            {"size": 6, "interval_basis": d3.Legendre},
            id="interval-basis",
        ),
    ],
)
def test_network_basis_keeps_different_discretizations_distinct(options):
    network = build_chain()
    reference = d3.NetworkBasis(network, size=6)
    different = d3.NetworkBasis(network, **options)

    assert different is not reference
    assert different.edges is not reference.edges


@pytest.mark.parametrize(
    "select_basis",
    [
        pytest.param(lambda basis: basis, id="bundle"),
        pytest.param(lambda basis: basis.edges, id="edges"),
        pytest.param(lambda basis: basis.interval, id="interval"),
        pytest.param(
            lambda basis: basis.interval.derivative_basis(1),
            id="derivative-interval",
        ),
    ],
)
def test_network_basis_must_match_distributor(select_basis):
    """Coordinate names cannot disguise a different Network geometry."""
    dist_network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    basis_network = d3.Network(endpoints=((0, 1),), lengths=(2,))
    dist = d3.Distributor(dist_network, dtype=np.float64)
    basis = d3.NetworkBasis(basis_network, size=6)

    with pytest.raises(ValueError, match="must belong to the Network"):
        dist.Field(bases=select_basis(basis))


def test_network_domain_accepts_local_derivative_basis_but_rejects_mixed_pair():
    dist_network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    other_network = d3.Network(endpoints=((0, 1),), lengths=(2,))
    dist = d3.Distributor(dist_network, dtype=np.float64)
    basis = d3.NetworkBasis(dist_network, size=6)
    other_basis = d3.NetworkBasis(other_network, size=6)

    derivative_basis = basis.interval.derivative_basis(2)
    field = dist.Field(bases=(basis.edges, derivative_basis))
    assert field.domain.bases == (basis.edges, derivative_basis)

    with pytest.raises(ValueError, match="must belong to the Network"):
        dist.Field(bases=(basis.edges, other_basis.interval))


def test_network_derivative_and_ntrace_of_edge_constant_are_zero():
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    edge_constant = dist.Field(bases=basis.edges)

    assert basis.dx(edge_constant) == 0
    assert d3.ntrace(edge_constant) == 0


def test_network_partition_separator():
    network = build_chain()
    partition = d3.NetworkPartition(network, (0, 1), num_parts=2)

    assert np.array_equal(partition.edge_owners, (0, 1))
    assert np.array_equal(partition.interior_vertex_owners, (0, -1, 1))
    assert np.array_equal(partition.vertex_equation_owners, (0, 0, 1))
    assert np.array_equal(partition.separator_vertices, (1,))
    assert np.array_equal(partition.interior_vertices[0], (0,))
    assert np.array_equal(partition.interior_vertices[1], (2,))


@pytest.mark.parametrize("owners", [(0.9, 1), (np.nan, 1), (0, 1.2)])
def test_network_partition_rejects_noninteger_owners(owners):
    with pytest.raises(ValueError, match="integer-valued"):
        d3.NetworkPartition(build_chain(), owners, num_parts=2)


def test_connected_partition_reduces_chain_separator():
    network = d3.Network(
        endpoints=tuple((vertex, vertex + 1) for vertex in range(8)),
        lengths=np.ones(8),
    )
    connected = d3.NetworkPartition.connected(network, 2)
    balanced_edges = d3.NetworkPartition.balanced(network, 2)

    assert connected.separator_vertices.size == 1
    assert all(np.all(np.diff(edges) == 1) for edges in connected.edges)
    assert balanced_edges.separator_vertices.size == 7


def test_connected_partition_handles_hub_without_line_graph():
    num_edges = 1024
    network = d3.Network(
        endpoints=tuple((0, leaf + 1) for leaf in range(num_edges)),
        lengths=np.ones(num_edges),
    )
    partition = d3.NetworkPartition.connected(network, 4)
    serial = d3.NetworkPartition.connected(network, 1)

    assert [edges.size for edges in partition.edges] == [256] * 4
    assert np.array_equal(partition.separator_vertices, (0,))
    assert np.array_equal(serial.edge_owners, np.zeros(num_edges))
    assert serial.separator_vertices.size == 0


@pytest.mark.parametrize("num_parts", [2, 4, 8])
def test_connected_partition_balances_binary_tree(num_parts):
    """Spread seeds from tree extremities before growing connected regions."""
    num_edges = 512
    network = d3.Network(
        endpoints=tuple(
            ((child - 1) // 2, child)
            for child in range(1, num_edges + 1)
        ),
        lengths=np.ones(num_edges),
    )
    partition = d3.NetworkPartition.connected(network, num_parts)
    repeated = d3.NetworkPartition.connected(network, num_parts)
    counts = np.array([edges.size for edges in partition.edges])

    assert counts.max() - counts.min() <= 2
    assert partition.separator_vertices.size <= num_parts - 1
    assert np.array_equal(partition.edge_owners, repeated.edge_owners)

    for owned_edges in partition.edges:
        remaining = set(owned_edges.tolist())
        connected = {remaining.pop()}
        vertices = set(network.endpoints[next(iter(connected))])
        while remaining:
            adjacent = {
                edge
                for edge in remaining
                if vertices.intersection(network.endpoints[edge])
            }
            assert adjacent
            connected.update(adjacent)
            remaining.difference_update(adjacent)
            for edge in adjacent:
                vertices.update(network.endpoints[edge])


def test_network_distributor_retains_parallel_communicator():
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)

    assert dist.comm == MPI.COMM_WORLD
    assert dist.layout_comm == MPI.COMM_SELF
    assert dist.network_comm == MPI.COMM_WORLD


@pytest.mark.parametrize("lengths", [(0, 1), (np.inf, 1), (np.nan, 1)])
def test_network_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError, match="positive and finite"):
        d3.Network(endpoints=((0, 1), (1, 2)), lengths=lengths)


def test_network_protocol_adapter():
    class Graph:
        def __dedalus_network__(self):
            return {
                "vertex_ids": (10, 20),
                "edge_ids": ("edge",),
                "endpoints": ((0, 1),),
                "lengths": (2.5,),
            }

    network = d3.Network(Graph())
    assert network.vertex_ids == (10, 20)
    assert network.edge_ids == ("edge",)
    assert np.array_equal(network.lengths, (2.5,))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_network_traces(dtype):
    network = build_chain()
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=5)
    u = dist.Field(name="u", bases=basis)

    s = dist.local_grid(basis.interval)
    u["g"] = np.array([[2], [5]], dtype=dtype) + np.array(
        [[3], [-4]], dtype=dtype
    ) * s

    values = d3.dtrace(u).evaluate()
    derivatives = d3.ntrace(u).evaluate()

    assert values.domain.get_basis(network.edge_coord) is basis.ports
    assert np.allclose(values["g"].ravel(), [2, 5, 5, 1])
    assert np.allclose(derivatives["g"].ravel(), [-3, 3, 2, -2])


def test_network_restriction_map():
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=4)
    U = dist.Field(name="U", bases=basis.vertices)
    U["g"] = np.array([1, 2, 4]).reshape(U["g"].shape)

    port_values = (basis.restriction @ U).evaluate()
    vertex_sums = (basis.restriction.T @ port_values).evaluate()

    assert np.allclose(port_values["g"].ravel(), [1, 2, 2, 4])
    assert np.allclose(vertex_sums["g"].ravel(), [1, 4, 4])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_network_replicated_input_evaluation_boundary(dtype):
    """Exercise direct scalar and vertex evaluation through Network storage."""
    network = build_chain()
    dist = d3.Distributor(
        network,
        dtype=dtype,
        network_partition="balanced_edges",
    )
    basis = d3.NetworkBasis(network, size=4)

    scalar = dist.Field(name="scalar")
    value = 2.5 if dtype == np.float64 else 2.5 - 0.75j
    scalar["g"] = value
    converted = d3.Convert(scalar, basis.edges).evaluate()
    assert np.allclose(converted["g"], value)

    U = dist.Field(name="U", bases=basis.vertices)
    vertex_values = np.arange(1, network.num_vertices + 1, dtype=dtype)
    U["g"] = vertex_values[:, None]
    restricted = (basis.restriction @ U).evaluate()
    assert np.allclose(
        restricted["g"].ravel(),
        vertex_values[network.endpoints.ravel()],
    )


def test_cardinal_map_copies_and_freezes_sparse_matrix():
    network = build_chain()
    basis = d3.NetworkBasis(network, size=4)
    matrix = sparse.eye(network.num_vertices, format="csr")
    cardinal_map = d3.CardinalMap(
        basis.vertices,
        basis.vertices,
        matrix,
    )
    transpose = cardinal_map.T

    matrix.data[:] = 2
    assert np.array_equal(cardinal_map.matrix.toarray(), np.eye(3))
    assert np.array_equal(transpose.matrix.toarray(), np.eye(3))
    with pytest.raises(ValueError, match="read-only"):
        cardinal_map.matrix.data[:] = 2


DTN_MATRIX_OPTIONS = [
    pytest.param(
        "dtn",
        {
            "bc_top": bc_top,
            "tau_left": tau_left,
            "interleave_components": interleave_components,
        },
        id=(
            f"dtn-bc_top={bc_top}-tau_left={tau_left}-"
            f"interleave={interleave_components}"
        ),
    )
    for bc_top, tau_left, interleave_components in itertools.product(
        (False, True),
        repeat=3,
    )
]


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "network_solver,matrix_options",
    [pytest.param(None, {}, id="direct"), *DTN_MATRIX_OPTIONS],
)
def test_network_chain_poisson_lbvp(dtype, network_solver, matrix_options):
    """Solve a two-edge chain using only symbolic network equations."""
    network = build_chain()
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=6)

    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)

    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)
    g["g"] = np.array([1, 0, 0]).reshape(g["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ ntrace(u)) = g"
    )
    solver = problem.build_solver(
        network_solver=network_solver,
        **matrix_options,
    )
    solver.solve()

    s = dist.local_grid(basis.interval)
    expected_u = np.concatenate(
        (
            1 - s / 3,
            (2 / 3) * (1 - s),
        ),
        axis=0,
    )
    assert np.allclose(u["g"], expected_u)
    assert np.allclose(U["g"].ravel(), [1, 2 / 3, 0])
    assert np.allclose(d3.dtrace(u).evaluate()["g"].ravel(), [1, 2 / 3, 2 / 3, 0])
    assert np.allclose(
        (R.T @ d3.ntrace(u)).evaluate()["g"].ravel(),
        [1 / 3, 0, -1 / 3],
    )
    if network_solver == "dtn":
        factorization = next(iter(solver.subproblem_matsolvers.values()))
        assert factorization.schur_matrix.shape == (network.num_vertices,) * 2
        assert sparse.issparse(factorization.edge_response)
        assert len(factorization.edge_solver.blocks) == network.num_edges
        assert {
            rows.size
            for rows, columns, block_solver in factorization.edge_solver.blocks.values()
        } == {basis.size + 2}
        g["g"] *= 2
        solver.solve()
        assert np.allclose(U["g"].ravel(), [2, 4 / 3, 0])


@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_network_heterogeneous_diffusion(network_solver):
    """Balance weighted flux for edgewise non-constant coefficients."""
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    kappa = dist.Field(name="kappa", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    g = dist.Field(name="g", bases=basis.vertices)
    kappa["g"] = np.array([1, 4]).reshape(kappa["g"].shape)
    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)
    g["g"] = np.array([1, 0, 0]).reshape(g["g"].shape)

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
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    s = dist.local_grid(basis.interval)
    expected_u = np.concatenate(
        (1 - (2 / 3) * s, (1 / 3) * (1 - s)),
        axis=0,
    )
    assert np.allclose(u["g"], expected_u)
    assert np.allclose(U["g"].ravel(), [1, 1 / 3, 0])
    flux = d3.dtrace(kappa) * d3.ntrace(u)
    assert np.allclose((R.T @ flux).evaluate()["g"].ravel(), [2 / 3, 0, -2 / 3])


def test_network_dtn_multiple_fields():
    """Condense a two-component edge system with two vertex fields."""
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    v = dist.Field(name="v", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    V = dist.Field(name="V", bases=basis.vertices)
    u_tau_1 = dist.Field(name="u_tau_1", bases=basis.edges)
    u_tau_2 = dist.Field(name="u_tau_2", bases=basis.edges)
    v_tau_1 = dist.Field(name="v_tau_1", bases=basis.edges)
    v_tau_2 = dist.Field(name="v_tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    u_data = dist.Field(name="u_data", bases=basis.vertices)
    v_data = dist.Field(name="v_data", bases=basis.vertices)
    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)
    u_data["g"] = np.array([1, 0, 0]).reshape(u_data["g"].shape)
    v_data["g"] = np.array([0, 0, 1]).reshape(v_data["g"].shape)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    variables = [
        u,
        v,
        U,
        V,
        u_tau_1,
        u_tau_2,
        v_tau_1,
        v_tau_2,
    ]
    problem = d3.LBVP(variables, namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(u_tau_1,-1) + lift(u_tau_2,-2) = 0"
    )
    problem.add_equation(
        "-dx(dx(v)) + lift(v_tau_1,-1) + lift(v_tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("dtrace(v) - R @ V = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = u_data")
    problem.add_equation("alpha*V + beta*(R.T @ ntrace(v)) = v_data")
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    s = dist.local_grid(basis.interval)
    expected_u = np.concatenate((1 - s / 3, (2 / 3) * (1 - s)), axis=0)
    expected_v = np.concatenate((s / 3, 1 / 3 + (2 / 3) * s), axis=0)
    assert np.allclose(u["g"], expected_u)
    assert np.allclose(v["g"], expected_v)
    assert np.allclose(U["g"].ravel(), [1, 2 / 3, 0])
    assert np.allclose(V["g"].ravel(), [0, 1 / 3, 1])


@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_parallel_edges_and_self_loop(network_solver):
    network = d3.Network(
        vertex_ids=("left", "right", "loop"),
        edge_ids=("short", "long", "loop"),
        endpoints=((0, 1), (0, 1), (2, 2)),
        lengths=(1, 2, 3),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=6)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    g = dist.Field(name="g", bases=basis.vertices)
    g["g"] = np.array([1, 0, 2]).reshape(g["g"].shape)
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
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()

    s = dist.local_grid(basis.interval)
    expected = np.concatenate((1 - s, 1 - s, 2 * np.ones_like(s)), axis=0)
    assert np.allclose(u["g"], expected)
    assert np.allclose(U["g"].ravel(), [1, 0, 2])
    assert np.allclose(
        (R.T @ d3.ntrace(u)).evaluate()["g"].ravel(),
        [1.5, -1.5, 0],
    )


@pytest.mark.parametrize("network_solver", [None, "dtn"])
@pytest.mark.parametrize("timestepper", [d3.SBDF1, d3.RK111])
def test_network_heat_ivp_step(network_solver, timestepper):
    """Apply one backward-Euler heat step through the network DtN solver."""
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=12)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    alpha["g"] = np.array([1, 0, 1]).reshape(alpha["g"].shape)
    beta["g"] = np.array([0, 1, 0]).reshape(beta["g"].shape)

    s = dist.local_grid(basis.interval)
    initial_u = np.concatenate(
        (
            np.sin(np.pi * s / 3),
            np.sin(np.pi * (1 + 2 * s) / 3),
        ),
        axis=0,
    )
    initial_U = np.array([0, np.sin(np.pi / 3), 0])
    u["g"] = initial_u
    U["g"] = initial_U.reshape(U["g"].shape)

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
    solver = problem.build_solver(
        timestepper,
        network_solver=network_solver,
    )
    timestep = 0.01
    solver.step(timestep)

    decay = 1 / (1 + timestep * (np.pi / 3) ** 2)
    assert np.allclose(u["g"], decay * initial_u, atol=2e-10)
    assert np.allclose(U["g"].ravel(), decay * initial_U, atol=2e-10)
