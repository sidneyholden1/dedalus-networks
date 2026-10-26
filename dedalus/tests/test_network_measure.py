"""Physical grid and measure tests for metric networks."""

import numpy as np

import dedalus.public as d3


def test_network_reference_and_physical_grids():
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=12)
    edge, s = dist.local_grids(basis)
    component_edge = dist.local_grid(basis.edges)
    reference = dist.local_grid(basis.interval)
    global_edge, global_s = basis.global_grids(dist, scales=(1, 1))
    local_x = basis.local_physical_grid(dist)
    global_x = basis.global_physical_grid(dist)

    assert network.names == ("e", "s")
    assert np.array_equal(edge[:, 0], (0, 1))
    assert np.array_equal(edge, component_edge)
    assert np.array_equal(s, reference)
    assert np.array_equal(global_edge, edge)
    assert np.array_equal(global_s, s)
    assert s.shape == (1, basis.size)
    assert local_x.shape == (network.num_edges, basis.size)
    assert np.allclose(local_x[0], reference[0])
    assert np.allclose(local_x[1], 2 * reference[0])
    assert np.array_equal(global_x, local_x)
    assert np.array_equal(basis.physical_grid(dist), local_x)


def test_network_reference_derivative_and_physical_measure():
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=12)
    one = dist.Field(name="one", bases=basis)
    x = dist.Field(name="x", bases=basis)
    one["g"] = 1
    x["g"] = basis.local_physical_grid(dist)

    reference_derivative = d3.Differentiate(
        x,
        network.position_coord,
    ).evaluate()
    physical_derivative = basis.derivative(x).evaluate()
    reference_integral = d3.Integrate(one, "s").evaluate()
    integral = d3.Integrate(one).evaluate()
    explicit_integral = d3.Integrate(one, network).evaluate()
    average = d3.Average(one).evaluate()

    assert np.allclose(
        reference_derivative["g"],
        network.lengths[:, None],
    )
    assert np.allclose(physical_derivative["g"], 1)
    assert np.allclose(reference_integral["g"], 1)
    assert np.allclose(integral["g"], 3)
    assert np.allclose(explicit_integral["g"], 3)
    assert np.allclose(average["g"], 1)
    assert one.domain.volume == network.total_length == 3


def test_network_polynomial_integrals_and_norm():
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=12)
    u = dist.Field(name="u", bases=basis)
    x = basis.local_physical_grid(dist)
    u["g"] = x

    integral = d3.Integrate(u).evaluate()
    mean_square = d3.Average(u*u).evaluate()
    l2_norm = u.allreduce_L2_norm(normalize_volume=False)

    expected_integral = (1**2 + 2**2) / 2
    expected_square_integral = (1**3 + 2**3) / 3
    assert np.allclose(integral["g"], expected_integral)
    assert np.allclose(mean_square["g"], expected_square_integral / 3)
    assert np.allclose(l2_norm, np.sqrt(expected_square_integral))


def test_network_explicit_edge_first_reductions():
    """Match the canonical reduction when edges are reduced first."""
    lengths = np.array((0.5, 1.0, 1.5, 2.0))
    network = d3.Network(
        endpoints=((0, 1), (1, 2), (2, 3), (3, 4)),
        lengths=lengths,
    )
    dist = d3.Distributor(network, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=12)
    u = dist.Field(name="u", bases=basis)
    s = dist.local_grid(basis.interval)
    amplitudes = np.arange(1, network.num_edges + 1)
    u["g"] = amplitudes[:, None] * (1 + s)

    edge_profile = d3.Integrate(u, network.edge_coord).evaluate()
    edge_first = d3.Integrate(
        edge_profile,
        network.position_coord,
    ).evaluate()
    edge_average = d3.Average(u, network.edge_coord).evaluate()
    average_edge_first = d3.Average(
        edge_average,
        network.position_coord,
    ).evaluate()
    weighted_amplitude = np.dot(lengths, amplitudes)

    assert np.allclose(edge_profile["g"], weighted_amplitude * (1 + s))
    assert np.allclose(edge_first["g"], 1.5 * weighted_amplitude)
    assert np.allclose(edge_first["g"], d3.Integrate(u).evaluate()["g"])
    assert np.allclose(
        average_edge_first["g"],
        1.5 * weighted_amplitude / lengths.sum(),
    )
    assert np.allclose(average_edge_first["g"], d3.Average(u).evaluate()["g"])
