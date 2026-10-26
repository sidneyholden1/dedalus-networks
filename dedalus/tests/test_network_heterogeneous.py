"""Tests for heterogeneous Network edge-discretization buckets."""

import pickle

import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3


def build_chain(lengths=(0.5, 1.25, 0.75, 1.5)):
    return d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3), (3, 4)),
        lengths=lengths,
    )


def test_network_basis_subset_normalization_cache_maps_and_grids():
    network = build_chain()
    basis = d3.NetworkBasis(
        network,
        size=7,
        edge_indices=(3.0, np.int64(1)),
    )
    reordered = d3.NetworkBasis(
        network,
        size=7,
        edge_indices=(1, 3),
    )
    full = d3.NetworkBasis(network, size=7)
    explicit_full = d3.NetworkBasis(
        network,
        size=7,
        edge_indices=range(network.num_edges),
    )

    assert reordered is basis
    assert explicit_full is full
    assert np.array_equal(basis.edge_indices, (1, 3))
    assert np.array_equal(basis.port_indices, (2, 3, 6, 7))
    assert np.array_equal(basis.edges.weights, network.lengths[[1, 3]])
    assert basis.shape == (2, 7)
    assert basis.restriction.matrix.shape == (4, network.num_vertices)
    assert np.array_equal(
        basis.restriction.matrix.toarray(),
        network.restriction_matrix[basis.port_indices].toarray(),
    )
    assert basis.edges._network is network
    assert basis.ports._network is network
    assert basis.vertices._network is network
    assert np.array_equal(basis.edges._network_elements, (1, 3))
    assert np.array_equal(basis.ports._network_elements, (2, 3, 6, 7))
    assert not hasattr(basis.vertices, "_network_basis")

    dist = d3.Distributor(network, dtype=np.float64)
    edge_grid, s = basis.local_grids(dist, scales=(1, 1))
    assert np.array_equal(edge_grid.ravel(), (1, 3))
    assert np.allclose(
        basis.local_physical_grid(dist),
        network.lengths[[1, 3], None] * s,
    )

    u = dist.Field(name="u", bases=basis)
    u["g"] = edge_grid + 2 * s
    assert np.allclose(
        basis.dtrace(u).evaluate()["g"].ravel(),
        (1, 3, 3, 5),
    )
    expected_normal = np.array(
        (-2 / network.lengths[1], 2 / network.lengths[1],
         -2 / network.lengths[3], 2 / network.lengths[3])
    )
    assert np.allclose(
        basis.ntrace(u).evaluate()["g"].ravel(),
        expected_normal,
    )

    restored = pickle.loads(pickle.dumps(basis))
    assert np.array_equal(restored.edge_indices, basis.edge_indices)
    assert np.array_equal(restored.port_indices, basis.port_indices)


@pytest.mark.parametrize(
    "edge_indices,error",
    [
        ((), ValueError),
        ((0, 0), ValueError),
        ((-1,), ValueError),
        ((4,), ValueError),
        ((True,), TypeError),
        ((1.5,), ValueError),
        ((np.inf,), ValueError),
        (("1",), TypeError),
    ],
)
def test_network_basis_subset_validation(edge_indices, error):
    with pytest.raises(error, match="edge_indices"):
        d3.NetworkBasis(
            build_chain(),
            size=6,
            edge_indices=edge_indices,
        )


def test_network_basis_set_exact_cover_mapping_and_costs():
    network = build_chain()
    bases = d3.NetworkBasisSet(
        network,
        {
            "coarse": {"edge_indices": (0, 2), "size": 5},
            "fine": {
                "edge_indices": (1, 3),
                "size": 9,
                "interval_basis": d3.Legendre,
            },
        },
    )

    assert tuple(bases) == ("coarse", "fine")
    assert tuple(bases.buckets) == ("coarse", "fine")
    assert bases["coarse"].vertices is bases["fine"].vertices
    assert bases.vertices is bases["coarse"].vertices
    assert bases["coarse"].interval_basis is d3.Chebyshev
    assert bases["fine"].interval_basis is d3.Legendre
    assert np.array_equal(bases.edge_costs, (5**3, 9**3, 5**3, 9**3))
    assert not bases.edge_costs.flags.writeable
    with pytest.raises(TypeError):
        bases.buckets["other"] = bases["coarse"]
    with pytest.raises(AttributeError, match="immutable"):
        bases.vertices = None

    dist = d3.Distributor(network, dtype=np.float64)
    with pytest.raises(TypeError, match="select one bucket"):
        dist.Field(name="ragged", bases=bases)

    restored = pickle.loads(pickle.dumps(bases))
    assert tuple(restored) == tuple(bases)
    assert np.array_equal(restored.edge_costs, bases.edge_costs)


@pytest.mark.parametrize(
    "buckets,match",
    [
        (
            {"a": {"edge_indices": (0, 1), "size": 5}},
            "missing canonical edges",
        ),
        (
            {
                "a": {"edge_indices": (0, 1, 2), "size": 5},
                "b": {"edge_indices": (2, 3), "size": 7},
            },
            "overlapping canonical edges",
        ),
        (
            {"": {"edge_indices": (0, 1, 2, 3), "size": 5}},
            "nonempty strings",
        ),
    ],
)
def test_network_basis_set_validation(buckets, match):
    with pytest.raises((TypeError, ValueError), match=match):
        d3.NetworkBasisSet(build_chain(), buckets)


def test_network_bucket_ncc_identity_and_cross_bucket_rejection():
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)
    first = d3.NetworkBasis(network, size=6, edge_indices=(0, 2))
    second = d3.NetworkBasis(network, size=6, edge_indices=(1, 3))
    u = dist.Field(name="u", bases=first)
    coefficient = dist.Field(name="coefficient", bases=first)
    other = dist.Field(name="other", bases=second)
    u["g"] = 2
    coefficient["g"] = 3

    assert np.allclose((coefficient * u).evaluate()["g"], 6)
    with pytest.raises((TypeError, NotImplementedError, ValueError)):
        other * u


def test_mixed_network_buckets_linear_chain_solve():
    lengths = np.array((0.5, 1.25, 0.75, 1.5))
    network = build_chain(lengths)
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
    U = dist.Field(name="U", bases=bases.vertices)
    tau_c1 = dist.Field(name="tau_c1", bases=cheb.edges)
    tau_c2 = dist.Field(name="tau_c2", bases=cheb.edges)
    tau_l1 = dist.Field(name="tau_l1", bases=leg.edges)
    tau_l2 = dist.Field(name="tau_l2", bases=leg.edges)
    g = dist.Field(name="g", bases=bases.vertices)
    g["g"] = np.array((2, 0, 0, 0, 5))[:, None]

    D = d3.CardinalMap(
        bases.vertices,
        bases.vertices,
        sparse.diags((1.0, 0, 0, 0, 1)),
        name="D",
    )
    K = d3.CardinalMap(
        bases.vertices,
        bases.vertices,
        sparse.diags((0.0, 1, 1, 1, 0)),
        name="K",
    )
    lift_c_basis = cheb.interval.derivative_basis(2)
    lift_l_basis = leg.interval.derivative_basis(2)
    lift_c = lambda A, n: d3.Lift(A, lift_c_basis, n)
    lift_l = lambda A, n: d3.Lift(A, lift_l_basis, n)

    problem = d3.LBVP(
        [u_cheb, u_leg, U, tau_c1, tau_c2, tau_l1, tau_l2],
        network_interfaces=[(u_cheb, U), (u_leg, U)],
        namespace=locals(),
    )
    problem.add_equation(
        "-cheb.dx(cheb.dx(u_cheb)) + lift_c(tau_c1,-1) "
        "+ lift_c(tau_c2,-2) = 0"
    )
    problem.add_equation(
        "-leg.dx(leg.dx(u_leg)) + lift_l(tau_l1,-1) "
        "+ lift_l(tau_l2,-2) = 0"
    )
    problem.add_equation("continuity(u_cheb) = 0")
    problem.add_equation("continuity(u_leg) = 0")
    problem.add_equation(
        "D @ U + K @ (cheb.R.T @ cheb.ntrace(u_cheb) "
        "+ leg.R.T @ leg.ntrace(u_leg)) = g"
    )
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    offsets = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    slope = 3 / lengths.sum()
    for field, basis in ((u_cheb, cheb), (u_leg, leg)):
        canonical_edges, s = basis.local_grids(dist, scales=(1, 1))
        x = offsets[canonical_edges.astype(int)] + basis.local_physical_grid(dist)
        assert np.allclose(field["g"], 2 + slope * x, atol=3e-11)
    assert np.allclose(
        U["g"].ravel(),
        2 + slope * np.concatenate(([0], np.cumsum(lengths))),
        atol=3e-11,
    )
    assert problem.vertex_field(u_cheb) is U
    assert problem.vertex_field(u_leg) is U


def test_automatic_interfaces_reject_heterogeneous_physical_vertex_split():
    network = build_chain()
    dist = d3.Distributor(network, dtype=np.float64)
    first = d3.NetworkBasis(network, size=5, edge_indices=(0, 2))
    second = d3.NetworkBasis(network, size=7, edge_indices=(1, 3))
    u_first = dist.Field(name="u_first", bases=first)
    u_second = dist.Field(name="u_second", bases=second)

    with pytest.raises(ValueError, match="shared U"):
        d3.LBVP(
            [u_first, u_second],
            network_interfaces=[u_first, u_second],
        )
