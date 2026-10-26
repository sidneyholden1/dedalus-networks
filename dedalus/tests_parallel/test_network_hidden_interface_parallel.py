"""MPI regression for a problem-managed Network vertex interface."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.tests import network_dynamic_vertex_manufactured as manufactured


@pytest.mark.mpi(min_size=4)
def test_hidden_dynamic_interface_with_edge_idle_root():
    """Lower a hidden vertex field before a four-rank DtN solve."""
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip("regression uses exactly four MPI ranks")

    network = manufactured.build_network()
    owners = np.array((3, 1))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=18)
    u = dist.Field(name="u", bases=basis)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    mass = dist.Field(name="mass", bases=basis.vertices)
    damping = dist.Field(name="damping", bases=basis.vertices)

    s = dist.local_grid(basis.interval)
    initial = manufactured.mode_data(network, s, np.float64)
    owned_edges = dist.network_partition.edges[comm.rank]
    u["g"] = initial["edge"][owned_edges]
    mass["g"] = manufactured.VERTEX_MASS.reshape(mass["g"].shape)
    damping["g"] = manufactured.VERTEX_DAMPING.reshape(
        damping["g"].shape
    )

    kappa = manufactured.KAPPA
    R = basis.restriction
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.IVP(
        [u, tau_1, tau_2],
        network_interfaces=[u],
        namespace=locals(),
    )
    U = problem.vertex_field(u)
    U["g"] = initial["vertex"].reshape(U["g"].shape)
    problem.add_equation(
        "dt(u) - kappa*dx(dx(u)) "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("continuity(u) = 0")
    problem.add_equation(
        "mass*dt(vertex_value(u)) + damping*vertex_value(u) "
        "+ kappa*(R.T @ ntrace(u)) = 0"
    )

    solver = problem.build_solver(d3.SBDF1)
    assert solver.network_solver == "dtn"
    timestep = 0.025
    solver.step(timestep)
    amplification = 1 / (1 + timestep * manufactured.DECAY_RATE)

    assert np.array_equal(
        owned_edges,
        np.flatnonzero(owners == comm.rank),
    )
    if comm.rank in (0, 2):
        assert owned_edges.size == 0
        assert u["g"].shape == (0, basis.size)

    factorization = solver.subproblems[0].LHS_solver
    assert np.array_equal(factorization.owned_edges, owned_edges)
    assert factorization.schur_matrix.shape == (1, 1)

    expected_u = amplification * initial["edge"]
    expected_U = amplification * initial["vertex"]
    assert np.allclose(u["g"], expected_u[owned_edges], atol=4e-10)
    assert np.allclose(u.allgather_data("g"), expected_u, atol=4e-10)
    assert np.allclose(U["g"].ravel(), expected_U, atol=4e-10)

    vertex_flux = (kappa * (R.T @ d3.ntrace(u))).evaluate()
    assert np.allclose(
        vertex_flux["g"].ravel(),
        amplification * initial["vertex_flux"],
        atol=4e-10,
    )
