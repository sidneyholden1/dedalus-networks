"""Manufactured mode shared by dynamic Network-vertex IVP tests."""

import numpy as np

import dedalus.public as d3


ENDPOINTS = np.array(((0, 1), (2, 1)))
LENGTHS = np.array((1.0, 2.0))
VERTEX_X = np.array((0.0, 1.0, 3.0))
KAPPA = 1.3
WAVENUMBER = 0.7
PHASE = 0.2
DECAY_RATE = KAPPA * WAVENUMBER**2
VERTEX_MASS = np.array((1.4, 1.7, 2.1))


def build_network():
    """Return an unequal two-edge chain with one reversed orientation."""
    return d3.Network(
        vertex_ids=("left", "junction", "right"),
        edge_ids=("short", "reversed-long"),
        endpoints=ENDPOINTS,
        lengths=LENGTHS,
    )


def _port_fluxes(amplitude):
    """Evaluate outward diffusive flux at every oriented edge port."""
    vertex_derivative = (
        -amplitude
        * WAVENUMBER
        * np.sin(WAVENUMBER * VERTEX_X + PHASE)
    )
    directions = (
        VERTEX_X[ENDPOINTS[:, 1]] - VERTEX_X[ENDPOINTS[:, 0]]
    ) / LENGTHS
    local_tail_derivative = vertex_derivative[ENDPOINTS[:, 0]] * directions
    local_head_derivative = vertex_derivative[ENDPOINTS[:, 1]] * directions
    outward_derivative = np.column_stack(
        (-local_tail_derivative, local_head_derivative)
    ).ravel()
    return KAPPA * outward_derivative


UNIT_VERTEX_MODE = np.cos(WAVENUMBER * VERTEX_X + PHASE)
UNIT_PORT_FLUX = _port_fluxes(1.0)
UNIT_VERTEX_FLUX = np.zeros(VERTEX_X.size)
np.add.at(UNIT_VERTEX_FLUX, ENDPOINTS.ravel(), UNIT_PORT_FLUX)
VERTEX_DAMPING = (
    VERTEX_MASS * DECAY_RATE - UNIT_VERTEX_FLUX / UNIT_VERTEX_MODE
)


def mode_data(network, reference_grid, dtype):
    """Evaluate one exact generalized eigenmode of the PDE--ODE system."""
    if np.issubdtype(dtype, np.complexfloating):
        amplitude = np.asarray(1 + 0.45j, dtype=dtype).item()
    else:
        amplitude = np.asarray(1, dtype=dtype).item()
    s = np.asarray(reference_grid)
    tail_x = VERTEX_X[network.endpoints[:, 0], None]
    head_x = VERTEX_X[network.endpoints[:, 1], None]
    physical_x = tail_x + (head_x - tail_x) * s
    edge_mode = amplitude * np.cos(WAVENUMBER * physical_x + PHASE)
    vertex_mode = amplitude * UNIT_VERTEX_MODE
    port_flux = _port_fluxes(amplitude)
    vertex_flux = network.restriction_matrix.T @ port_flux
    return {
        "edge": edge_mode.astype(dtype),
        "vertex": vertex_mode.astype(dtype),
        "port_flux": port_flux.astype(dtype),
        "vertex_flux": vertex_flux.astype(dtype),
    }
