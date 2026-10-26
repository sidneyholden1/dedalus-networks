"""MPI tests for nonlinear Network DtN solves."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("storage", ["whole_edge", "replicated"])
def test_parallel_network_nlbvp_cubic(dtype, storage):
    """Rebuild and condense nonlinear Jacobians with irregular ownership."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    num_edges = 5
    network = d3.Network(
        endpoints=tuple((edge, edge + 1) for edge in range(num_edges)),
        lengths=np.linspace(0.5, 2, num_edges),
    )
    owners = np.where(
        np.arange(num_edges) % 2,
        1,
        comm.size - 1,
    )
    if storage == "whole_edge":
        dist = d3.Distributor(
            network,
            dtype=dtype,
            network_partition=owners,
        )
        solver_options = {}
    else:
        dist = d3.Distributor(network, dtype=dtype)
        solver_options = {"network_partition": owners}
    basis = d3.NetworkBasis(network, size=10)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    edge_forcing = dist.Field(name="edge_forcing", bases=basis)
    vertex_forcing = dist.Field(
        name="vertex_forcing",
        bases=basis.vertices,
    )

    target = 1.0 if dtype == np.float64 else 1.0 + 0.25j
    edge_forcing["g"] = target**3
    vertex_forcing["g"] = target**3
    u["g"] = 0.7 * target
    U["g"] = 0.7 * target

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.NLBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + u*u*u + lift(tau_1,-1) + lift(tau_2,-2) "
        "= edge_forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U*U*U = vertex_forcing")
    solver = problem.build_solver(**solver_options)
    for _ in range(7):
        solver.newton_iteration()

    assert solver.network_solver == "dtn"
    if storage == "whole_edge":
        assert solver.matrix_assembly_root is None
        assert dist.network_partition.edges[0].size == 0
        if comm.size > 2:
            assert dist.network_partition.edges[2].size == 0
            assert np.any(np.diff(dist.network_partition.edges[1]) != 1)
            assert np.any(
                np.diff(dist.network_partition.edges[comm.size - 1]) != 1
            )
    else:
        assert solver.matrix_assembly_root == 0
        assert not dist.distributed_network
    assert np.allclose(u.allgather_data("g"), target, atol=3e-11)
    assert np.allclose(U["g"], target, atol=3e-11)
