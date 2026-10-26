"""Four-rank acceptance contracts for owned/ghost Network solvers."""

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3


def _require_four_ranks():
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip("owned/ghost solver contract uses exactly four MPI ranks")
    return comm


def _distributor(network, owners, dtype=np.float64):
    return d3.Distributor(
        network,
        dtype=dtype,
        network_partition=np.asarray(owners, dtype=int),
        network_vertex_storage="owned_ghost",
    )


def _set_vertex_data(field, values):
    values = np.asarray(values, dtype=field.dtype)
    field.set_global_data(values[:, None])


def _assert_unique_vertex_ownership(dist):
    comm = dist.layout_comm
    partition = dist.network_partition
    local_owned = partition.owned_vertices[comm.rank]
    owned = comm.allgather(local_owned)
    owned = np.concatenate(owned) if owned else np.empty(0, dtype=int)
    assert np.array_equal(np.sort(owned), np.arange(dist.network.num_vertices))
    assert np.unique(owned).size == owned.size
    assert np.all(
        np.isin(
            partition.ghost_vertices[comm.rank],
            partition.separator_vertices,
        )
    )


def _assert_vertex_state(field, expected=None, atol=1e-10):
    """Check resident allocation and every ghost against its owner value."""
    field.change_layout("g")
    dist = field.dist
    comm = dist.layout_comm
    partition = dist.network_partition
    canonical = dist.network_vertex_canonical_elements(field.domain)
    resident = partition.resident_vertices[comm.rank]
    ghost = partition.ghost_vertices[comm.rank]
    local = field["g"].ravel().copy()
    gathered = field.allgather_data("g").ravel()

    assert dist.network_vertex_storage == "owned_ghost"
    assert np.array_equal(canonical, resident)
    assert field["g"].shape == (resident.size, 1)
    assert np.allclose(local, gathered[canonical], atol=atol)
    if ghost.size:
        ghost_positions = np.searchsorted(canonical, ghost)
        assert np.allclose(
            local[ghost_positions],
            gathered[ghost],
            atol=atol,
        )
    if expected is not None:
        assert np.allclose(gathered, expected, atol=atol)
    _assert_unique_vertex_ownership(dist)
    return gathered


def _build_non_diagonal_lbvp(dtype=np.float64):
    comm = _require_four_ranks()
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    owners = np.array((3, 1))
    dist = _distributor(network, owners, dtype=dtype)
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    w = dist.Field(name="w", bases=basis.ports)
    g = dist.Field(name="g", bases=basis.vertices)
    w["g"] = 1
    _set_vertex_data(g, (4, 13, 20))

    R = basis.R
    A = d3.CardinalMap(
        basis.vertices,
        basis.vertices,
        sparse.csr_matrix(
            [[2, 1, 0], [0, 3, 1], [0, -1, 4]],
            dtype=dtype,
        ),
        name="topology_local_A",
    )
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "A @ U + R.T @ ntrace(u) + R.T @ (w*(R @ U)) = g"
    )
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()
    return comm, network, owners, dist, basis, solver, u, U


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_owned_ghost_explicit_u_non_diagonal_lbvp_idle_ranks(dtype):
    """Solve with owner-local off-diagonal vertex rows and two idle ranks."""
    comm, network, owners, dist, basis, solver, u, U = (
        _build_non_diagonal_lbvp(dtype)
    )
    s = basis.interval.global_grid(dist, scale=1).ravel()
    expected_u = np.vstack((1 + s, 2 + 2 * s)).astype(dtype)
    expected_U = np.array((1, 2, 4), dtype=dtype)

    assert solver.network_solver == "dtn"
    assert dist.network_partition.edges[0].size == 0
    assert dist.network_partition.edges[2].size == 0
    assert np.allclose(u.allgather_data("g"), expected_u, atol=3e-12)
    _assert_vertex_state(U, expected_U, atol=3e-12)
    if comm.rank in (0, 2):
        assert u["g"].shape == (0, basis.size)
        assert U["g"].shape[0] == 0
    if comm.rank == 3:
        assert np.array_equal(dist.network_partition.ghost_vertices[3], (1,))


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_matrix_application_and_canonical_ownership():
    """Apply the assembled local matrix and recover the evaluated LBVP RHS."""
    comm, network, owners, dist, basis, solver, u, U = (
        _build_non_diagonal_lbvp()
    )
    subproblem = solver.subproblems[0]
    state = subproblem.gather_inputs(solver.state).copy()
    rhs = subproblem.gather_outputs(solver.F).copy()
    applied = solver.apply_subproblem_matrix(
        subproblem.L_min,
        state,
        subproblem=subproblem,
    )

    assert applied.shape == rhs.shape
    assert np.allclose(applied, rhs, atol=2e-10)
    assert subproblem.L_min.shape[1] == state.size
    assert subproblem.L_min.shape[0] == rhs.size
    _assert_unique_vertex_ownership(dist)
    _assert_vertex_state(U, (1, 2, 4), atol=3e-12)


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_interpolate_in_scalar_subproblem_equation():
    """Use an owner-selected vertex value in a coupled scalar equation."""
    comm = _require_four_ranks()
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1),
    )
    dist = _distributor(network, owners=(3, 1))
    basis = d3.NetworkBasis(network, size=8)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    scalar = dist.Field(name="scalar")
    w = dist.Field(name="w", bases=basis.ports)
    g = dist.Field(name="g", bases=basis.vertices)
    w["g"] = 1
    _set_vertex_data(g, (4, 13, 20))

    R = basis.R
    A = d3.CardinalMap(
        basis.vertices,
        basis.vertices,
        sparse.csr_matrix(
            [[2, 1, 0], [0, 3, 1], [0, -1, 4]],
            dtype=np.float64,
        ),
        name="topology_local_A_with_selector",
    )
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    selected = d3.Interpolate(U, network.edge_coord, 1)

    problem = d3.LBVP(
        [u, U, tau_1, tau_2, scalar],
        namespace=locals(),
    )
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "A @ U + R.T @ ntrace(u) + R.T @ (w*(R @ U)) = g"
    )
    problem.add_equation((selected + scalar, 7))
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    _assert_vertex_state(U, (1, 2, 4), atol=3e-12)
    assert np.allclose(scalar.allgather_data("c"), 5, atol=3e-12)
    assert np.allclose(selected.evaluate()["c"], 2, atol=3e-12)
    # Vertex 1 is owned on rank 1 and is a ghost on rank 3, while ranks 0 and
    # 2 have no resident Network data at all.
    assert dist.network_partition.vertex_equation_owners[1] == 1
    assert 1 in dist.network_partition.ghost_vertices[3]
    if comm.rank in (0, 2):
        assert U["c"].shape[0] == 0


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_hidden_u_lbvp_idle_ranks():
    """Lower continuity/vertex_value onto a hidden resident interface field."""
    comm = _require_four_ranks()
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1.7),
    )
    owners = np.array((3, 1))
    dist = _distributor(network, owners)
    basis = d3.NetworkBasis(network, size=10)
    u = dist.Field(name="u", bases=basis)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    forcing = dist.Field(name="forcing", bases=basis)
    boundary = dist.Field(name="boundary", bases=basis.vertices)
    target = 1.25
    forcing["g"] = target
    _set_vertex_data(boundary, np.full(network.num_vertices, target))

    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.LBVP(
        [u, tau_1, tau_2],
        network_interfaces=[u],
        namespace=locals(),
    )
    U = problem.vertex_field(u)
    problem.add_equation(
        "-dx(dx(u)) + u + lift(tau_1,-1) + lift(tau_2,-2) "
        "= forcing"
    )
    problem.add_equation("continuity(u) = 0")
    problem.add_equation("vertex_value(u) = boundary")
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    assert U is problem.variables[-1]
    assert np.allclose(u.allgather_data("g"), target, atol=2e-11)
    _assert_vertex_state(
        U,
        np.full(network.num_vertices, target),
        atol=2e-11,
    )
    if comm.rank in (0, 2):
        assert u["g"].shape == (0, basis.size)


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_heterogeneous_buckets_share_one_u():
    """Condense Chebyshev and Legendre buckets against one resident U."""
    comm = _require_four_ranks()
    lengths = np.array((0.5, 1.25, 0.75, 1.5))
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3), (3, 4)),
        lengths=lengths,
    )
    owners = np.array((3, 1, 3, 2))
    dist = _distributor(network, owners)
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
    _set_vertex_data(g, (2, 0, 0, 0, 5))

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
    lift_c = lambda field, order: d3.Lift(field, lift_c_basis, order)
    lift_l = lambda field, order: d3.Lift(field, lift_l_basis, order)
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
        s = basis.interval.global_grid(dist, scale=1).ravel()
        canonical_edges = basis.edge_indices
        x = offsets[canonical_edges, None] + lengths[canonical_edges, None] * s
        expected = 2 + slope * x
        assert np.allclose(field.allgather_data("g"), expected, atol=4e-11)
    expected_U = 2 + slope * np.concatenate(([0], np.cumsum(lengths)))
    _assert_vertex_state(U, expected_U, atol=4e-11)
    assert problem.vertex_field(u_cheb) is U
    assert problem.vertex_field(u_leg) is U
    assert dist.network_partition.edges[0].size == 0


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_pure_neumann_mean_zero_gauge():
    """Promote a global gauge without replicating persistent vertex state."""
    comm = _require_four_ranks()
    lengths = np.array((0.5, 1.0, 1.5))
    offsets = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    total_length = lengths.sum()
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3)),
        lengths=lengths,
    )
    owners = np.array((3, 1, 3))
    dist = _distributor(network, owners)
    basis = d3.NetworkBasis(network, size=10)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    tau_gauge = dist.Field(name="tau_gauge")
    forcing = dist.Field(name="forcing", bases=basis)
    owned_edges = dist.network_partition.edges[comm.rank]
    local_x = offsets[owned_edges, None] + basis.local_physical_grid(dist)
    forcing["g"] = local_x

    R = basis.R
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
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
    problem.add_equation(
        "integ(integ(u, network.edge_coord), "
        "network.position_coord) = 0"
    )
    solver = problem.build_solver(network_solver="dtn")
    solver.solve()

    s = basis.interval.global_grid(dist, scale=1).ravel()
    global_x = offsets[:, None] + lengths[:, None] * s
    expected_u = (
        total_length * global_x**2 / 4
        - global_x**3 / 6
        - total_length**3 / 24
    )
    vertex_x = np.concatenate(([0.0], np.cumsum(lengths)))
    expected_U = (
        total_length * vertex_x**2 / 4
        - vertex_x**3 / 6
        - total_length**3 / 24
    )
    assert np.allclose(u.allgather_data("g"), expected_u, atol=3e-11)
    _assert_vertex_state(U, expected_U, atol=3e-11)
    assert np.allclose(tau_gauge["g"], total_length / 2, atol=3e-11)
    assert np.allclose(d3.Integrate(u).evaluate()["g"], 0, atol=3e-11)
    assert dist.network_partition.edges[0].size == 0
    assert dist.network_partition.edges[2].size == 0


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_heat_ivp_step():
    """Advance an implicit heat mode through resident interface matrices."""
    comm = _require_four_ranks()
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    owners = np.array((3, 1))
    dist = _distributor(network, owners)
    basis = d3.NetworkBasis(network, size=12)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    degree = dist.Field(name="degree", bases=basis.vertices)
    w = dist.Field(name="w", bases=basis.ports)
    _set_vertex_data(alpha, (1, 0, 1))
    _set_vertex_data(beta, (0, 1, 0))
    _set_vertex_data(degree, network.degree)
    w["g"] = 1

    s = basis.interval.global_grid(dist, scale=1).ravel()
    starts = np.array((0.0, 1.0))
    global_u = np.sin(
        np.pi
        * (starts[:, None] + network.lengths[:, None] * s)
        / 3
    )
    owned_edges = dist.network_partition.edges[comm.rank]
    u["g"] = global_u[owned_edges]
    initial_U = np.array((0, np.sin(np.pi / 3), 0))
    _set_vertex_data(U, initial_U)

    R = basis.R
    dx = basis.derivative
    dt = d3.TimeDerivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.IVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "dt(u) - dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ ntrace(u)) "
        "+ R.T @ (w*(R @ U)) - degree*U = 0"
    )
    solver = problem.build_solver(d3.SBDF1, network_solver="dtn")
    assert solver.total_modes == (
        network.num_edges * (basis.size + 2) + network.num_vertices
    )
    timestep = 0.01
    solver.step(timestep)

    amplification = 1 / (1 + timestep * (np.pi / 3) ** 2)
    assert np.allclose(
        u.allgather_data("g"),
        amplification * global_u,
        atol=3e-10,
    )
    _assert_vertex_state(
        U,
        amplification * initial_U,
        atol=3e-10,
    )
    if comm.rank in (0, 2):
        assert u["g"].shape == (0, basis.size)


@pytest.mark.mpi(min_size=4)
def test_owned_ghost_nlbvp_cubic_rebuilds_resident_jacobian():
    """Refresh ghost perturbations across repeated nonlinear Jacobian solves."""
    comm = _require_four_ranks()
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 1.5),
    )
    owners = np.array((3, 1))
    dist = _distributor(network, owners)
    basis = d3.NetworkBasis(network, size=10)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    edge_forcing = dist.Field(name="edge_forcing", bases=basis)
    vertex_forcing = dist.Field(name="vertex_forcing", bases=basis.vertices)
    target = 1.2
    edge_forcing["g"] = target**3
    _set_vertex_data(
        vertex_forcing,
        np.full(network.num_vertices, target**3),
    )
    u["g"] = 0.7 * target
    _set_vertex_data(U, np.full(network.num_vertices, 0.7 * target))

    R = basis.R
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.NLBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + u*u*u + lift(tau_1,-1) + lift(tau_2,-2) "
        "= edge_forcing"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("U*U*U = vertex_forcing")
    solver = problem.build_solver(network_solver="dtn")
    for _ in range(7):
        solver.newton_iteration()

    assert np.allclose(u.allgather_data("g"), target, atol=3e-11)
    _assert_vertex_state(
        U,
        np.full(network.num_vertices, target),
        atol=3e-11,
    )
    assert len(problem.perturbations) == len(problem.variables)


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_owned_ghost_targeted_right_evp_and_set_state(dtype):
    """Run collective Arnoldi and scatter a right mode into resident fields."""
    comm = _require_four_ranks()
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(np.pi / 2, np.pi / 2),
    )
    owners = np.array((3, 1))
    dist = _distributor(network, owners, dtype=dtype)
    basis = d3.NetworkBasis(network, size=14)
    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    eigenvalue = dist.Field(name="eigenvalue")
    _set_vertex_data(alpha, (1, 0, 1))
    _set_vertex_data(beta, (0, 1, 0))

    R = basis.R
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda field, order: d3.Lift(field, lift_basis, order)
    problem = d3.EVP(
        [u, U, tau_1, tau_2],
        eigenvalue,
        namespace=locals(),
    )
    problem.add_equation(
        "eigenvalue*u + dx(dx(u)) "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("alpha*U + beta*(R.T @ ntrace(u)) = 0")
    solver = problem.build_solver(
        bc_top=True,
        tau_left=True,
        network_solver="dtn",
    )
    solver.solve_sparse(
        solver.subproblems[0],
        N=3,
        target=0.8,
        ncv=26,
        tol=2e-7,
    )

    assert np.allclose(solver.eigenvalues.real, (1, 4, 9), atol=2e-6)
    assert np.all(np.abs(solver.eigenvalues.imag) < 2e-7)
    assert np.all(solver.eigenvalue_residuals < 2e-7)
    solver.set_state(0)

    gathered_u = u.allgather_data("g")
    gathered_U = _assert_vertex_state(U, atol=2e-7)
    assert np.all(np.isfinite(gathered_u))
    assert np.linalg.norm(gathered_u) > 0
    assert np.all(np.isfinite(gathered_U))
    assert np.max(np.abs(gathered_U[[0, 2]])) < (
        2e-6 * np.max(np.abs(gathered_U))
    )
    if comm.rank in (0, 2):
        assert u["g"].shape == (0, basis.size)
