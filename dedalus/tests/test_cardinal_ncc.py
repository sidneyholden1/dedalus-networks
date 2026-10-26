"""Test non-constant coefficients over Cardinal bases."""

import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_cardinal_ncc_preceding_chebyshev(dtype):
    """A coefficient may vary over Cardinal entries but be constant in x."""
    ncoord = d3.Coordinate("n")
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor((ncoord, xcoord), dtype=dtype)
    nbasis = d3.CardinalBasis(ncoord, size=3)
    xbasis = d3.Chebyshev(xcoord, size=4, bounds=(0, 1))

    u = dist.Field(name="u", bases=(nbasis, xbasis))
    coefficient = dist.Field(name="coefficient", bases=nbasis)
    forcing = dist.Field(name="forcing", bases=(nbasis, xbasis))

    values = np.array([1, 2, 4], dtype=dtype)
    coefficient["g"] = values.reshape(coefficient["g"].shape)
    forcing["g"] = 1

    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("coefficient*u = forcing")
    solver = problem.build_solver()
    solver.solve()

    expected = np.broadcast_to((1 / values)[:, None], u["g"].shape)
    assert np.allclose(u["g"], expected)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_cardinal_and_chebyshev_ncc(dtype):
    """A separate last-axis NCC expansion is used at every Cardinal node."""
    ncoord = d3.Coordinate("n")
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor((ncoord, xcoord), dtype=dtype)
    nbasis = d3.CardinalBasis(ncoord, size=3)
    xbasis = d3.Chebyshev(xcoord, size=6, bounds=(0, 1))

    u = dist.Field(name="u", bases=(nbasis, xbasis))
    coefficient = dist.Field(
        name="coefficient",
        bases=(nbasis, xbasis),
    )
    forcing = dist.Field(name="forcing", bases=(nbasis, xbasis))

    x = dist.local_grid(xbasis)
    offsets = np.array([1, 2, 4], dtype=dtype)[:, None]
    slopes = np.array([1, -0.5, 2], dtype=dtype)[:, None]
    expected = np.array([2, -1, 0.5], dtype=dtype)[:, None]
    coefficient["g"] = offsets + slopes * x
    forcing["g"] = coefficient["g"] * expected

    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("coefficient*u = forcing")
    solver = problem.build_solver()
    solver.solve()

    assert np.allclose(u["g"], expected)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_cardinal_ncc_on_cardinal_basis(dtype):
    """Cardinal fields multiply pointwise in assembled equations."""
    coord = d3.Coordinate("n")
    dist = d3.Distributor(coord, dtype=dtype)
    basis = d3.CardinalBasis(coord, size=3)

    u = dist.Field(name="u", bases=basis)
    coefficient = dist.Field(name="coefficient", bases=basis)
    forcing = dist.Field(name="forcing", bases=basis)

    values = np.array([1, 2, 4], dtype=dtype)
    coefficient["g"] = values
    forcing["g"] = 1

    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("coefficient*u = forcing")
    solver = problem.build_solver()
    solver.solve()

    assert np.allclose(u["g"], 1 / values)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_cardinal_ncc_with_two_coupled_prefix_axes(dtype):
    """Cardinal block scaling preserves axis order with coupled prefixes."""
    tcoord = d3.Coordinate("t")
    ncoord = d3.Coordinate("n")
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor((tcoord, ncoord, xcoord), dtype=dtype)
    tbasis = d3.CardinalBasis(tcoord, size=2)
    nbasis = d3.CardinalBasis(ncoord, size=3)
    xbasis = d3.Chebyshev(xcoord, size=4, bounds=(0, 1))

    u = dist.Field(name="u", bases=(tbasis, nbasis, xbasis))
    time_coefficient = dist.Field(name="time_coefficient", bases=tbasis)
    edge_coefficient = dist.Field(name="edge_coefficient", bases=nbasis)
    forcing = dist.Field(name="forcing", bases=(tbasis, nbasis, xbasis))

    time_values = np.array([1, 10], dtype=dtype)
    edge_values = np.array([2, 4, 8], dtype=dtype)
    time_coefficient["g"] = time_values.reshape(time_coefficient["g"].shape)
    edge_coefficient["g"] = edge_values.reshape(edge_coefficient["g"].shape)
    forcing["g"] = 1

    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation(
        "time_coefficient*u + edge_coefficient*u = forcing"
    )
    solver = problem.build_solver()
    solver.solve()

    expected = 1 / (time_values[:, None, None] + edge_values[None, :, None])
    expected = np.broadcast_to(expected, u["g"].shape)
    assert np.allclose(u["g"], expected)
