"""Tests for problem-managed Network vertex interface fields."""

import numpy as np
import pytest

import dedalus.public as d3
from dedalus.tests import network_dynamic_vertex_manufactured as manufactured


def _chain_problem_fields(dtype, size=12):
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1.0, 1.7),
    )
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=size)
    u = dist.Field(name="u", bases=basis)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    return network, dist, basis, u, tau_1, tau_2


def _solve_constant_lbvp(dtype, network_solver, interface_kind):
    _, dist, basis, u, tau_1, tau_2 = _chain_problem_fields(dtype)
    forcing = dist.Field(name="forcing", bases=basis)
    boundary = dist.Field(name="boundary", bases=basis.vertices)
    target = np.asarray(
        1.25 if dtype == np.float64 else 1.25 - 0.4j,
        dtype=dtype,
    ).item()
    forcing["g"] = target
    boundary["g"] = target

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    if interface_kind == "hidden":
        problem = d3.LBVP(
            [u, tau_1, tau_2],
            network_interfaces=[u],
            namespace=locals(),
        )
        U = problem.vertex_field(u)
        continuity_equation = "continuity(u) = 0"
        vertex_equation = "vertex_value(u) = boundary"
    elif interface_kind == "bound":
        U = dist.Field(name="U", bases=basis.vertices)
        problem = d3.LBVP(
            [u, U, tau_1, tau_2],
            network_interfaces=[(u, U)],
            namespace=locals(),
        )
        continuity_equation = "continuity(u) = 0"
        vertex_equation = "vertex_value(u) = boundary"
    else:
        U = dist.Field(name="U", bases=basis.vertices)
        problem = d3.LBVP(
            [u, U, tau_1, tau_2],
            namespace=locals(),
        )
        continuity_equation = "dtrace(u) - R @ U = 0"
        vertex_equation = "U = boundary"

    problem.add_equation(
        "-dx(dx(u)) + u + lift(tau_1,-1) + lift(tau_2,-2) "
        "= forcing"
    )
    problem.add_equation(continuity_equation)
    problem.add_equation(vertex_equation)
    solver = problem.build_solver(network_solver=network_solver)
    solver.solve()
    return problem, u["g"].copy(), U["g"].ravel().copy(), target


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_hidden_interface_lbvp_matches_explicit(dtype, network_solver):
    explicit, explicit_u, explicit_U, target = _solve_constant_lbvp(
        dtype,
        network_solver,
        "explicit",
    )
    hidden, hidden_u, hidden_U, _ = _solve_constant_lbvp(
        dtype,
        network_solver,
        "hidden",
    )

    assert len(explicit.variables) == 4
    assert len(hidden.variables) == 4
    edge_field, vertex_field = hidden.network_interfaces[0]
    assert hidden.vertex_value(edge_field) is vertex_field
    assert vertex_field is hidden.variables[-1]
    assert hidden.local_namespace["continuity"].__self__ is hidden
    assert hidden.local_namespace["vertex_value"].__self__ is hidden
    assert np.allclose(hidden_u, target, atol=2e-11)
    assert np.allclose(hidden_U, target, atol=2e-11)
    assert np.allclose(hidden_u, explicit_u, atol=2e-11)
    assert np.allclose(hidden_U, explicit_U, atol=2e-11)


def test_explicit_interface_binding_uses_problem_helpers():
    problem, u, U, target = _solve_constant_lbvp(
        np.float64,
        "dtn",
        "bound",
    )
    edge_field, vertex_field = problem.network_interfaces[0]

    assert problem.vertex_field(edge_field) is vertex_field
    assert not problem._hidden_network_interface_fields
    assert np.allclose(u, target, atol=2e-11)
    assert np.allclose(U, target, atol=2e-11)


@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_hidden_interface_dynamic_vertex_ivp(network_solver):
    dtype = np.float64
    network = manufactured.build_network()
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=18)
    u = dist.Field(name="u", bases=basis)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    mass = dist.Field(name="mass", bases=basis.vertices)
    damping = dist.Field(name="damping", bases=basis.vertices)

    s = dist.local_grid(basis.interval)
    initial = manufactured.mode_data(network, s, dtype)
    u["g"] = initial["edge"]
    mass["g"] = manufactured.VERTEX_MASS.reshape(mass["g"].shape)
    damping["g"] = manufactured.VERTEX_DAMPING.reshape(damping["g"].shape)

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
    assert problem.eqs[2]["M"]

    solver = problem.build_solver(d3.SBDF1, network_solver=network_solver)
    timestep = 0.025
    solver.step(timestep)
    amplification = 1 / (1 + timestep * manufactured.DECAY_RATE)

    assert np.allclose(
        u["g"],
        amplification * initial["edge"],
        atol=3e-10,
    )
    assert np.allclose(
        U["g"].ravel(),
        amplification * initial["vertex"],
        atol=3e-10,
    )


@pytest.mark.parametrize("network_solver", [None, "dtn"])
def test_hidden_interface_nlbvp(network_solver):
    _, dist, basis, u, tau_1, tau_2 = _chain_problem_fields(
        np.float64,
        size=10,
    )
    edge_forcing = dist.Field(name="edge_forcing", bases=basis)
    vertex_forcing = dist.Field(
        name="vertex_forcing",
        bases=basis.vertices,
    )
    target = 1.2
    edge_forcing["g"] = target**3
    vertex_forcing["g"] = target**3
    u["g"] = 0.8 * target

    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.NLBVP(
        [u, tau_1, tau_2],
        network_interfaces=[u],
        namespace=locals(),
    )
    U = problem.vertex_field(u)
    U["g"] = 0.8 * target
    problem.add_equation(
        "-dx(dx(u)) + u*u*u + lift(tau_1,-1) + lift(tau_2,-2) "
        "= edge_forcing"
    )
    problem.add_equation("continuity(u) = 0")
    problem.add_equation(
        "vertex_value(u)*vertex_value(u)*vertex_value(u) "
        "= vertex_forcing"
    )
    solver = problem.build_solver(network_solver=network_solver)
    for _ in range(7):
        solver.newton_iteration()

    assert len(problem.perturbations) == len(problem.variables) == 4
    assert np.allclose(u["g"], target, atol=2e-11)
    assert np.allclose(U["g"], target, atol=2e-11)


def test_network_interface_validation_and_helper_errors():
    _, dist, basis, u, tau_1, tau_2 = _chain_problem_fields(np.float64)
    other = dist.Field(name="other", bases=basis)

    with pytest.raises(ValueError, match="included in the problem variable"):
        d3.LBVP([u, tau_1, tau_2], network_interfaces=[other])
    with pytest.raises(ValueError, match="only once"):
        d3.LBVP([u, tau_1, tau_2], network_interfaces=[u, u])
    with pytest.raises(ValueError, match="interval coordinate"):
        d3.LBVP([tau_1], network_interfaces=[tau_1])
    with pytest.raises(TypeError, match="edge Field"):
        d3.LBVP([u], network_interfaces=[object()])

    problem = d3.LBVP(
        [u, other, tau_1, tau_2],
        network_interfaces=[u],
    )
    with pytest.raises(ValueError, match="not registered"):
        problem.vertex_field(other)
    with pytest.raises(TypeError, match="registered edge Field"):
        problem.continuity(1)


def test_network_interface_reserved_names_and_explicit_basis_validation():
    _, dist, basis, u, tau_1, tau_2 = _chain_problem_fields(np.float64)
    reserved = dist.Field(name="__network_vertex_0", bases=basis.edges)
    wrong_vertex = dist.Field(name="wrong_vertex", bases=basis.ports)
    U = dist.Field(name="U", bases=basis.vertices)

    with pytest.raises(ValueError, match="reserved"):
        d3.LBVP(
            [u, tau_1, tau_2, reserved],
            network_interfaces=[u],
        )
    with pytest.raises(ValueError, match="matching NetworkBasis vertex"):
        d3.LBVP(
            [u, wrong_vertex, tau_1, tau_2],
            network_interfaces=[(u, wrong_vertex)],
        )
    with pytest.raises(ValueError, match="included in the problem variable"):
        d3.LBVP(
            [u, tau_1, tau_2],
            network_interfaces=[(u, U)],
        )


def test_hidden_interface_names_are_stable_under_registration_reordering():
    """Bind checkpoint field names to semantic edge names, not list order."""
    _, dist, basis, u, tau_1, tau_2 = _chain_problem_fields(np.float64)
    v = dist.Field(name="v", bases=basis)
    first = d3.LBVP(
        [u, v, tau_1, tau_2],
        network_interfaces=[u, v],
    )
    second = d3.LBVP(
        [u, v, tau_1, tau_2],
        network_interfaces=[v, u],
    )

    first_names = {
        edge.name: vertex.name
        for edge, vertex in first.network_interfaces
    }
    second_names = {
        edge.name: vertex.name
        for edge, vertex in second.network_interfaces
    }
    assert first_names == second_names
    assert first_names["u"] != first_names["v"]
    assert all(
        name.startswith("__network_vertex_")
        for name in first_names.values()
    )


def test_automatic_hidden_interface_requires_a_semantic_edge_name():
    """Unnamed fields need an explicit U to define stable restart identity."""
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=8)
    unnamed = dist.Field(bases=basis)

    with pytest.raises(ValueError, match="nonempty edge-field name"):
        d3.LBVP([unnamed], network_interfaces=[unnamed])


def test_ivp_build_evp_recreates_hidden_network_interface():
    """Keep managed U semantics and helpers when linearizing an IVP."""
    _, dist, basis, u, tau_1, tau_2 = _chain_problem_fields(
        np.complex128,
        size=10,
    )
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    beta = dist.Field(name="beta", bases=basis.vertices)
    alpha["g"] = np.array((1, 0, 1))[:, None]
    beta["g"] = np.array((0, 1, 0))[:, None]

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
    source_U = problem.vertex_field(u)
    problem.add_equation(
        "dt(u) - dx(dx(u)) "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("continuity(u) = 0")
    problem.add_equation(
        "alpha*vertex_value(u) + beta*(R.T @ ntrace(u)) = 0"
    )

    evp = problem.build_EVP()
    delta_u = next(var for var in evp.variables if var.name == "δu")
    delta_U = evp.vertex_field(delta_u)

    assert len(evp.variables) == len(problem.variables)
    assert evp.network_interfaces == ((delta_u, delta_U),)
    assert evp._hidden_network_interface_fields == (delta_U,)
    assert delta_U is evp.variables[-1]
    assert delta_U is not source_U
    assert delta_U.name.startswith("__network_vertex_")
    assert evp.local_namespace["continuity"].__self__ is evp
    assert evp.local_namespace["vertex_value"].__self__ is evp

    # Building and solving verifies that every occurrence of the source
    # hidden field in the already-parsed IVP equations was rebound too.
    solver = evp.build_solver()
    solver.solve_dense(solver.subproblems[0])
    assert np.any(np.isfinite(solver.eigenvalues))


def test_ivp_build_evp_preserves_explicit_network_interface():
    """Carry an explicit (u, U) binding onto the perturbation fields."""
    _, dist, basis, u, tau_1, tau_2 = _chain_problem_fields(np.float64)
    U = dist.Field(name="U", bases=basis.vertices)
    dt = d3.TimeDerivative
    problem = d3.IVP(
        [u, U, tau_1, tau_2],
        network_interfaces=[(u, U)],
        namespace=locals(),
    )
    problem.add_equation("dt(u) = 0")
    problem.add_equation("continuity(u) = 0")

    evp = problem.build_EVP()
    delta_u = next(var for var in evp.variables if var.name == "δu")
    delta_U = next(var for var in evp.variables if var.name == "δU")

    assert evp.network_interfaces == ((delta_u, delta_U),)
    assert evp.vertex_field(delta_u) is delta_U
    assert not evp._hidden_network_interface_fields
    assert evp.local_namespace["continuity"].__self__ is evp
    assert evp.local_namespace["vertex_value"].__self__ is evp
