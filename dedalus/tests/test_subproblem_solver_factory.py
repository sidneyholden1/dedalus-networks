"""Tests for custom subproblem solver factories."""

import numpy as np
import pytest

import dedalus.public as d3


class RecordingFactory:
    """Record factory calls while delegating to the configured matsolver."""

    def __init__(self):
        self.calls = []

    def __call__(self, matrix, solver, subproblem):
        self.calls.append((matrix, solver, subproblem))
        return solver.matsolver(matrix, solver)


def test_lbvp_subproblem_solver_factory():
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.float64)
    u = dist.Field(name="u")
    forcing = dist.Field(name="forcing")
    forcing["g"] = 3

    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("u = forcing")
    factory = RecordingFactory()
    solver = problem.build_solver(subproblem_solver_factory=factory)

    solver.solve()
    assert np.allclose(u["g"], 3)
    assert len(factory.calls) == 1
    matrix, called_solver, subproblem = factory.calls[0]
    assert called_solver is solver
    assert subproblem in solver.subproblems
    assert matrix is subproblem.L_min

    solver.solve()
    assert len(factory.calls) == 1
    solver.solve(rebuild_matrices=True)
    assert len(factory.calls) == 2


@pytest.mark.parametrize("timestepper", ["SBDF1", "RK111"])
def test_ivp_subproblem_solver_factory(timestepper):
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.float64)
    u = dist.Field(name="u")
    u["g"] = 1

    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) + u = 0")
    factory = RecordingFactory()
    solver = problem.build_solver(
        timestepper,
        subproblem_solver_factory=factory,
    )

    solver.step(0.1)
    assert np.allclose(u["g"], 1 / 1.1)
    assert len(factory.calls) == 1
    matrix, called_solver, subproblem = factory.calls[0]
    assert called_solver is solver
    assert subproblem in solver.subproblems
    assert matrix is subproblem.LHS

    solver.step(0.1)
    assert len(factory.calls) == 1


def test_nlbvp_subproblem_solver_factory():
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.float64)
    u = dist.Field(name="u")
    forcing = dist.Field(name="forcing")
    u["g"] = 1
    forcing["g"] = 4

    problem = d3.NLBVP([u], namespace=locals())
    problem.add_equation("u*u = forcing")
    factory = RecordingFactory()
    solver = problem.build_solver(subproblem_solver_factory=factory)

    for _ in range(5):
        solver.newton_iteration()
    assert np.allclose(u["g"], 2)
    assert len(factory.calls) == 5
    assert all(call[1] is solver for call in factory.calls)
    assert all(call[2] in solver.subproblems for call in factory.calls)


def test_subproblem_solver_factory_must_be_callable():
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.float64)
    u = dist.Field(name="u")
    forcing = dist.Field(name="forcing")
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("u = forcing")

    with pytest.raises(TypeError, match="must be callable"):
        problem.build_solver(subproblem_solver_factory=object())
