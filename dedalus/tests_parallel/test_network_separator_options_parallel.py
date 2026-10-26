"""MPI agreement checks for public Network separator solver options."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


def _build_problem(comm):
    network = d3.Network(endpoints=((0, 1),), lengths=(1.0,))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=np.array((comm.size - 1,)),
    )
    basis = d3.NetworkBasis(network, size=4)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    R = basis.restriction
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, n: d3.Lift(field, lift_basis, n)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation("u + lift(tau_1,-1) + lift(tau_2,-2) = 0")
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U = 0")
    return problem


@pytest.mark.mpi(min_size=2)
def test_network_separator_invalid_option_fails_collectively():
    comm = MPI.COMM_WORLD
    problem = _build_problem(comm)
    restart = 0 if comm.rank == 0 else 8

    with pytest.raises(
        RuntimeError,
        match="Network separator option validation failed on network rank 0",
    ):
        problem.build_solver(network_separator_restart=restart)


@pytest.mark.mpi(min_size=2)
def test_network_separator_option_mismatch_fails_collectively():
    comm = MPI.COMM_WORLD
    problem = _build_problem(comm)
    restart = 7 if comm.rank == 0 else 8

    with pytest.raises(
        RuntimeError,
        match="Network separator option agreement failed on network rank 1",
    ):
        problem.build_solver(network_separator_restart=restart)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("invalid", [True, False])
def test_network_interior_option_errors_are_collective(invalid):
    comm = MPI.COMM_WORLD
    problem = _build_problem(comm)
    method = ("invalid" if invalid else "two_stage") if comm.rank == 0 else "coupled"
    stage = "validation" if invalid else "agreement"
    with pytest.raises(RuntimeError, match=f"Network interior option {stage} failed"):
        problem.build_solver(network_interior_elimination=method)
