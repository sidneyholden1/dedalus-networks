"""MPI coverage for Network global scalar constraints."""

from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_distributed_neumann_poisson_global_gauge(dtype):
    """Solve with a global gauge while rank zero and another rank own no edge."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    lengths = np.array((0.5, 1.0, 1.5))
    offsets = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    total_length = lengths.sum()
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=lengths,
    )
    owners = np.array((comm.size - 1, 1, comm.size - 1))
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
    tau_gauge = dist.Field(name="tau_gauge")
    forcing = dist.Field(name="forcing", bases=basis)

    owned_edges = dist.network_partition.edges[comm.rank]
    physical_x = basis.local_physical_grid(dist)
    global_x = offsets[owned_edges, None] + physical_x
    forcing["g"] = global_x

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP(
        [u, U, tau_1, tau_2, tau_gauge],
        namespace=locals(),
    )
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "+ tau_gauge = forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = 0")
    # Exercise the explicitly requested edge-first order.  The intermediate
    # position field must be replicated so every rank's matrix contribution
    # reaches the final scalar row.
    problem.add_equation(
        "integ(integ(u, network.edge_coord), "
        "network.position_coord) = 0"
    )
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    expected_local = (
        total_length * global_x**2 / 4
        - global_x**3 / 6
        - total_length**3 / 24
    )
    vertex_x = np.concatenate(([0.0], np.cumsum(lengths)))
    expected_vertices = (
        total_length * vertex_x**2 / 4
        - vertex_x**3 / 6
        - total_length**3 / 24
    )
    assert dist.network_partition.edges[0].size == 0
    assert dist.network_partition.edges[2].size == 0
    assert np.allclose(u["g"], expected_local, atol=3e-11)
    assert np.allclose(U["g"].ravel(), expected_vertices, atol=3e-11)
    assert np.allclose(tau_gauge["g"], total_length / 2, atol=3e-11)
    assert np.allclose(d3.Integrate(u).evaluate()["g"], 0, atol=3e-11)

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    # The global gauge row touches all four vertex labels after edge
    # condensation, so they join the synthetic scalar on the separator.
    assert factorization.schur_matrix.shape == (5, 5)


@pytest.mark.mpi(min_size=4)
def test_distributed_heat_ivp_global_gauge_with_idle_root():
    """Retain a global gauge in a four-rank implicit timestepper stage."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(comm.size - 1,),
    )
    basis = d3.NetworkBasis(network, size=20)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    tau_gauge = dist.Field(name="tau_gauge")

    s = dist.local_grid(basis.interval)
    if comm.rank == comm.size - 1:
        u["g"] = np.cos(np.pi * s)
    if comm.rank == 0:
        U["g"] = np.array((1, -1)).reshape(U["g"].shape)
    else:
        U["g"] = -100

    R = basis.restriction
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.IVP(
        [u, U, tau_1, tau_2, tau_gauge],
        namespace=locals(),
    )
    problem.add_equation(
        "dt(u) - dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "+ tau_gauge = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = 0")
    problem.add_equation("integ(u) = 0")
    solver = problem.build_solver(d3.SBDF1, network_solver="dtn")
    timestep = 0.01
    solver.step(timestep)

    amplification = 1 / (1 + timestep * np.pi**2)
    if comm.rank == comm.size - 1:
        assert np.allclose(
            u["g"],
            amplification * np.cos(np.pi * s),
            atol=2e-11,
        )
    else:
        assert u["g"].shape[0] == 0
    assert np.allclose(
        U["g"].ravel(),
        amplification * np.array((1, -1)),
        atol=2e-11,
    )
    assert np.allclose(tau_gauge["g"], 0, atol=2e-11)
    assert np.allclose(d3.Integrate(u).evaluate()["g"], 0, atol=2e-12)
    factorization = solver.subproblems[0].LHS_solver
    # Both endpoint labels are coupled to the global gauge row.
    assert factorization.schur_matrix.shape == (3, 3)


@pytest.mark.mpi(min_size=4)
def test_distributed_vertex_average_gauge_promotes_interior_vertices():
    """Move vertices touched by a global row onto the shared separator."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    lengths = np.ones(3)
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=lengths,
    )
    owners = np.array((1, 1, comm.size - 1))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=owners,
    )
    basis = d3.NetworkBasis(network, size=10)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    tau_gauge = dist.Field(name="tau_gauge")
    forcing = dist.Field(name="forcing", bases=basis)

    owned_edges = dist.network_partition.edges[comm.rank]
    local_x = basis.local_physical_grid(dist)
    global_x = owned_edges[:, None] + local_x
    forcing["g"] = global_x

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP(
        [u, U, tau_1, tau_2, tau_gauge],
        namespace=locals(),
    )
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "+ tau_gauge = forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = 0")
    # Vertex 1 is interior to rank 1 before this global row promotes it (and
    # every other touched vertex) onto the shared separator.  Average also
    # exercises reinitialization through Average.new_operand.
    problem.add_equation("ave(U) = 0")
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    raw = 3 * global_x**2 / 4 - global_x**3 / 6
    constant = -9 / 8
    expected_vertices = (
        3 * np.arange(4, dtype=np.float64) ** 2 / 4
        - np.arange(4, dtype=np.float64) ** 3 / 6
        + constant
    )
    assert np.allclose(u["g"], raw + constant, atol=3e-11)
    assert np.allclose(U["g"].ravel(), expected_vertices, atol=3e-11)
    assert np.allclose(tau_gauge["g"], 1.5, atol=3e-11)
    assert np.allclose(d3.Average(U).evaluate()["g"], 0, atol=3e-11)
    assert dist.network_partition.edges[0].size == 0
    assert dist.network_partition.interior_vertex_owners[1] == 1

    factorization = next(iter(solver.subproblem_matsolvers.values()))
    assert factorization.schur_matrix.shape == (5, 5)


@pytest.mark.mpi(min_size=4)
def test_distributed_nlbvp_global_gauge_with_idle_root():
    """Retain the global row and column while rebuilding Newton Jacobians."""
    comm = MPI.COMM_WORLD
    if comm.size < 4:
        pytest.skip("requires at least four MPI ranks")

    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition=(comm.size - 1,),
    )
    basis = d3.NetworkBasis(network, size=20)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    tau_gauge = dist.Field(name="tau_gauge")
    forcing = dist.Field(name="forcing", bases=basis)

    s = dist.local_grid(basis.interval)
    if comm.rank == comm.size - 1:
        expected = np.cos(np.pi * s)
        u["g"] = 0.8 * expected
        forcing["g"] = np.pi**2 * expected + expected**3
    if comm.rank == 0:
        U["g"] = 0.8 * np.array((1, -1)).reshape(U["g"].shape)
    else:
        U["g"] = -100

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.NLBVP(
        [u, U, tau_1, tau_2, tau_gauge],
        namespace=locals(),
    )
    problem.add_equation(
        "-dx(dx(u)) + u*u*u + lift(tau_1,-1) + lift(tau_2,-2) "
        "+ tau_gauge = forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = 0")
    problem.add_equation("integ(u) = 0")
    solver = problem.build_solver(network_solver="dtn")
    for _ in range(6):
        solver.newton_iteration()

    if comm.rank == comm.size - 1:
        assert np.allclose(u["g"], np.cos(np.pi * s), atol=2e-11)
    else:
        assert u["g"].shape[0] == 0
    assert np.allclose(U["g"].ravel(), (1, -1), atol=2e-11)
    assert np.allclose(tau_gauge["g"], 0, atol=2e-11)
    assert np.allclose(d3.Integrate(u).evaluate()["g"], 0, atol=2e-12)
