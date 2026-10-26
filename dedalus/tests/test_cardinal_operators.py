"""Test Cardinal basis operators: ConvertConstant, Interpolate, Integrate, Average."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


N_range = [5, 10]
dtype_range = [np.float64, np.complex128]


@CachedMethod
def build_cardinal(N, dtype):
    c = d3.Coordinate('n')
    dist = d3.Distributor(c, dtype=dtype)
    b = d3.CardinalBasis(c, size=N)
    n = dist.local_grid(b, scale=1)
    return c, dist, b, n


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_cardinal_convert_constant(N, dtype, layout):
    """Test conversion from constant to Cardinal basis (broadcasts scalar to all N entries)."""
    c, dist, b, n = build_cardinal(N, dtype)
    f = dist.Field()
    f['g'] = 3
    f.change_layout(layout)
    g = d3.Convert(f, b).evaluate()
    assert np.allclose(g['g'], 3 * np.ones(N))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('index', [0, 2, -1])
def test_cardinal_interpolate(N, dtype, index):
    """Test Interpolate extracts the value at the given integer index."""
    c, dist, b, n = build_cardinal(N, dtype)
    f = dist.Field(bases=b)
    f.fill_random('g')
    g = d3.Interpolate(f, c, index).evaluate()
    assert np.allclose(g['g'], f['g'][index])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_cardinal_integrate(N, dtype):
    """Test Integrate computes the discrete sum over all entries."""
    c, dist, b, n = build_cardinal(N, dtype)
    f = dist.Field(bases=b)
    f.fill_random('g')
    g = d3.Integrate(f, c).evaluate()
    assert np.allclose(g['g'], f['g'].sum())


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_cardinal_integrate_constant(N, dtype):
    """Test Integrate of a uniform field equals N * value."""
    c, dist, b, n = build_cardinal(N, dtype)
    f = dist.Field(bases=b)
    f['g'] = 3
    g = d3.Integrate(f, c).evaluate()
    assert np.allclose(g['g'], 3 * N)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_cardinal_average(N, dtype):
    """Test Average computes the mean over all entries."""
    c, dist, b, n = build_cardinal(N, dtype)
    f = dist.Field(bases=b)
    f.fill_random('g')
    g = d3.Average(f, c).evaluate()
    assert np.allclose(g['g'], f['g'].mean())


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_cardinal_average_constant(N, dtype):
    """Test Average of a uniform field returns that value."""
    c, dist, b, n = build_cardinal(N, dtype)
    f = dist.Field(bases=b)
    f['g'] = 7
    g = d3.Average(f, c).evaluate()
    assert np.allclose(g['g'], 7)


@pytest.mark.parametrize('dtype', dtype_range)
def test_weighted_cardinal_measure(dtype):
    c = d3.Coordinate('n')
    dist = d3.Distributor(c, dtype=dtype)
    weights = np.array([1, 2, 4], dtype=float)
    basis = d3.CardinalBasis(c, size=3, weights=weights)
    f = dist.Field(bases=basis)
    f['g'] = np.array([2, 3, 5], dtype=dtype)

    integral = d3.Integrate(f, c).evaluate()
    average = d3.Average(f, c).evaluate()
    expected_integral = np.dot(weights, f['g'])

    assert basis.volume == weights.sum()
    assert np.allclose(integral['g'], expected_integral)
    assert np.allclose(average['g'], expected_integral / weights.sum())
    assert not basis.weights.flags.writeable


@pytest.mark.parametrize(
    'weights, message',
    [
        ([1, 2], 'shape'),
        ([1, np.inf, 2], 'finite and nonnegative'),
        ([1, -1, 2], 'finite and nonnegative'),
        ([0, 0, 0], 'positive sum'),
    ],
)
def test_cardinal_rejects_invalid_weights(weights, message):
    coord = d3.Coordinate('n')
    with pytest.raises(ValueError, match=message):
        d3.CardinalBasis(coord, size=3, weights=weights)


def test_cardinal_grid_shape_in_product_domain():
    """Cardinal grids broadcast along their distributor axis."""
    ncoord = d3.Coordinate("n")
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor((ncoord, xcoord), dtype=np.float64)
    nbasis = d3.CardinalBasis(ncoord, size=3)
    xbasis = d3.Chebyshev(xcoord, size=4, bounds=(0, 1))

    n, x = dist.local_grids(nbasis, xbasis)

    assert n.shape == (3, 1)
    assert x.shape == (1, 4)
    assert np.array_equal(n.ravel(), np.arange(3))
