"""Public solver options for distributed Network separator systems."""

import numpy as np
import pytest

import dedalus.public as d3


SEPARATOR_DEFAULTS = {
    "network_interior_elimination": "coupled",
    "network_separator_method": "auto",
    "network_separator_direct_max_size": 256,
    "network_separator_restart": 40,
    "network_separator_maxiter": None,
    "network_separator_rtol": 1e-10,
    "network_separator_atol": 0.0,
    "network_separator_preconditioner": "jacobi",
}

SEPARATOR_OVERRIDES = {
    "network_interior_elimination": "two_stage",
    "network_separator_method": "gmres",
    "network_separator_direct_max_size": 0,
    "network_separator_restart": 7,
    "network_separator_maxiter": 19,
    "network_separator_rtol": 2e-9,
    "network_separator_atol": 3e-13,
    "network_separator_preconditioner": "none",
}


def _build_solver(problem_type, **solver_options):
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.complex128)
    u = dist.Field(name="u")

    if problem_type == "lbvp":
        problem = d3.LBVP([u], namespace=locals())
        problem.add_equation("u = 1")
        return problem.build_solver(**solver_options)
    if problem_type == "nlbvp":
        problem = d3.NLBVP([u], namespace=locals())
        problem.add_equation("u*u = 1")
        return problem.build_solver(**solver_options)
    if problem_type == "ivp":
        problem = d3.IVP([u], namespace=locals())
        problem.add_equation("dt(u) + u = 0")
        return problem.build_solver(d3.SBDF1, **solver_options)
    if problem_type == "evp":
        eigenvalue = dist.Field(name="eigenvalue")
        problem = d3.EVP([u], eigenvalue, namespace=locals())
        problem.add_equation("eigenvalue*u + u = 0")
        return problem.build_solver(**solver_options)
    raise AssertionError(f"Unknown problem type {problem_type!r}.")


@pytest.mark.parametrize("problem_type", ("lbvp", "nlbvp", "ivp", "evp"))
@pytest.mark.parametrize(
    "options",
    (SEPARATOR_DEFAULTS, SEPARATOR_OVERRIDES),
    ids=("defaults", "overrides"),
)
def test_network_separator_options_reach_every_solver(problem_type, options):
    build_options = {} if options is SEPARATOR_DEFAULTS else options
    solver = _build_solver(problem_type, **build_options)

    for name, expected in options.items():
        assert getattr(solver, name) == expected


@pytest.mark.parametrize(
    "name,value,error,match",
    (
        (
            "network_interior_elimination",
            None,
            TypeError,
            "network_interior_elimination must be a string",
        ),
        (
            "network_interior_elimination",
            "auto",
            ValueError,
            "network_interior_elimination must be",
        ),
        (
            "network_separator_method",
            None,
            TypeError,
            "network_separator_method must be a string",
        ),
        (
            "network_separator_method",
            "direct",
            ValueError,
            "network_separator_method must be",
        ),
        (
            "network_separator_direct_max_size",
            1.5,
            TypeError,
            "network_separator_direct_max_size must be an integer",
        ),
        (
            "network_separator_direct_max_size",
            -1,
            ValueError,
            "network_separator_direct_max_size must be nonnegative",
        ),
        (
            "network_separator_restart",
            True,
            TypeError,
            "network_separator_restart must be an integer",
        ),
        (
            "network_separator_restart",
            0,
            ValueError,
            "network_separator_restart must be positive",
        ),
        (
            "network_separator_maxiter",
            2.5,
            TypeError,
            "network_separator_maxiter must be an integer",
        ),
        (
            "network_separator_maxiter",
            0,
            ValueError,
            "network_separator_maxiter must be positive",
        ),
        (
            "network_separator_rtol",
            "1e-10",
            TypeError,
            "network_separator_rtol must be a real number",
        ),
        (
            "network_separator_rtol",
            np.inf,
            ValueError,
            "network_separator_rtol must be finite and nonnegative",
        ),
        (
            "network_separator_atol",
            -1.0,
            ValueError,
            "network_separator_atol must be finite and nonnegative",
        ),
        (
            "network_separator_preconditioner",
            "ilu",
            ValueError,
            "network_separator_preconditioner must be",
        ),
    ),
)
def test_network_separator_options_validate(name, value, error, match):
    with pytest.raises(error, match=match):
        _build_solver("lbvp", **{name: value})


def test_network_separator_options_do_not_change_ordinary_solve():
    solver = _build_solver("lbvp", **SEPARATOR_OVERRIDES)
    solver.solve()
    assert np.allclose(solver.state[0]["g"], 1)
