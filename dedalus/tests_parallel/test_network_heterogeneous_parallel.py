"""MPI tests for heterogeneous Network edge-discretization buckets."""

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3


@pytest.mark.mpi(min_size=4)
def test_distributed_mixed_network_buckets_shared_vertices():
    """Solve interleaved mixed-resolution/family buckets with an idle root."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    lengths = np.array((0.5, 1.25, 0.75, 1.5))
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3), (3, 4)),
        lengths=lengths,
    )
    owners = np.array((comm.size - 1, 1, comm.size - 1, 2))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
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

    # Local storage indices remain bucket-logical, while metadata and grids
    # map them back to canonical Network edge/port labels.
    for basis in bases.values():
        local_edges = dist.coeff_layout.local_elements(
            basis.edges.domain(dist),
            scales=1,
        )[0]
        expected_edges = np.flatnonzero(
            owners[basis.edge_indices] == comm.rank
        )
        assert np.array_equal(local_edges, expected_edges)

        local_ports = dist.coeff_layout.local_elements(
            basis.ports.domain(dist),
            scales=1,
        )[0]
        expected_ports = np.flatnonzero(
            owners[network.port_edges[basis.port_indices]] == comm.rank
        )
        assert np.array_equal(local_ports, expected_ports)

        canonical_edges, _ = basis.local_grids(dist, scales=(1, 1))
        assert np.array_equal(
            canonical_edges.ravel(),
            basis.edge_indices[expected_edges],
        )

    probe_U = dist.Field(name="probe_U", bases=bases.vertices)
    probe_values = np.arange(network.num_vertices, dtype=np.float64) + 2
    probe_U["g"] = probe_values[:, None] if comm.rank == 0 else -100
    for basis in bases.values():
        restricted = (basis.R @ probe_U).evaluate()
        local_ports = dist.grid_layout.local_elements(
            basis.ports.domain(dist),
            scales=1,
        )[0]
        canonical_ports = basis.port_indices[local_ports]
        expected = probe_values[network.endpoints.ravel()[canonical_ports]]
        assert np.allclose(restricted["g"].ravel(), expected)

    u_cheb = dist.Field(name="u_cheb", bases=cheb)
    u_leg = dist.Field(name="u_leg", bases=leg)
    U = dist.Field(name="U", bases=bases.vertices)
    tau_c1 = dist.Field(name="tau_c1", bases=cheb.edges)
    tau_c2 = dist.Field(name="tau_c2", bases=cheb.edges)
    tau_l1 = dist.Field(name="tau_l1", bases=leg.edges)
    tau_l2 = dist.Field(name="tau_l2", bases=leg.edges)
    g = dist.Field(name="g", bases=bases.vertices)
    if comm.rank == 0:
        g["g"] = np.array((2, 0, 0, 0, 5), dtype=np.float64)[:, None]
    else:
        g["g"] = -200

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
        canonical_edges, _ = basis.local_grids(dist, scales=(1, 1))
        x = (
            offsets[canonical_edges.astype(int)]
            + basis.local_physical_grid(dist)
        )
        assert np.allclose(field["g"], 2 + slope * x, atol=4e-11)
    assert np.allclose(
        U["g"].ravel(),
        2 + slope * np.concatenate(([0], np.cumsum(lengths))),
        atol=4e-11,
    )
    assert dist.network_partition.edges[0].size == 0


@pytest.mark.mpi(min_size=2)
def test_distributed_bucket_subset_mismatch_fails_collectively():
    """Include canonical subset membership in discretization signatures."""
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
        network_partition=(comm.size - 1, comm.size - 1),
    )
    selected = (0,) if comm.rank == 0 else (1,)
    basis = d3.NetworkBasis(network, size=6, edge_indices=selected)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    problem = d3.LBVP([u, U], namespace=locals())
    problem.add_equation("u = 0")
    problem.add_equation("U = 0")

    with pytest.raises(RuntimeError, match="Network discretization validation"):
        problem.build_solver()
