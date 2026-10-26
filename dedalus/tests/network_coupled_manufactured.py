"""Manufactured data shared by coupled Network regression tests."""

import numpy as np

import dedalus.public as d3


KU = 1.4 + 0.1j
KV = 0.9 - 0.08j
EDGE_MATRIX = np.array(
    (
        (1.3 + 0.25j, -0.45 + 0.2j),
        (0.35 - 0.3j, 0.95 + 0.15j),
    )
)
VALUE_MATRIX = np.array(
    (
        (1.15 - 0.2j, -0.3 + 0.12j),
        (0.4 + 0.18j, 0.85 - 0.1j),
    )
)
FLUX_MATRIX = np.array(
    (
        (0.65 + 0.1j, 0.22 - 0.14j),
        (-0.18 + 0.09j, 1.05 + 0.16j),
    )
)
U_VALUES = np.array((1.1 + 0.35j, -0.45 + 0.7j, 0.8 - 0.25j))
V_VALUES = np.array((-0.3 + 0.9j, 1.2 - 0.4j, 0.25 + 0.55j))
U_BUMPS = np.array((0.4 - 0.18j, -0.28 + 0.32j))
V_BUMPS = np.array((-0.22 + 0.27j, 0.36 + 0.11j))


def build_network():
    """Return two unequal edges with the second orientation reversed."""
    return d3.Network(
        vertex_ids=("left", "junction", "right"),
        edge_ids=("short", "reversed-long"),
        endpoints=((0, 1), (2, 1)),
        lengths=(0.9, 1.7),
    )


def manufactured_data(network, reference_grid):
    """Evaluate a continuous quadratic solution and its exact forcing."""
    s = np.asarray(reference_grid)
    endpoints = network.endpoints
    lengths = network.lengths
    u_ends = U_VALUES[endpoints]
    v_ends = V_VALUES[endpoints]

    u = (
        u_ends[:, :1]
        + (u_ends[:, 1:] - u_ends[:, :1]) * s
        + U_BUMPS[:, None] * s * (1 - s)
    )
    v = (
        v_ends[:, :1]
        + (v_ends[:, 1:] - v_ends[:, :1]) * s
        + V_BUMPS[:, None] * s * (1 - s)
    )
    u_xx = -2 * U_BUMPS / lengths**2
    v_xx = -2 * V_BUMPS / lengths**2
    forcing_u = (
        -KU * u_xx[:, None]
        + EDGE_MATRIX[0, 0] * u
        + EDGE_MATRIX[0, 1] * v
    )
    forcing_v = (
        -KV * v_xx[:, None]
        + EDGE_MATRIX[1, 0] * u
        + EDGE_MATRIX[1, 1] * v
    )

    u_delta = u_ends[:, 1] - u_ends[:, 0]
    v_delta = v_ends[:, 1] - v_ends[:, 0]
    u_normal = np.column_stack(
        (
            -(u_delta + U_BUMPS) / lengths,
            (u_delta - U_BUMPS) / lengths,
        )
    ).ravel()
    v_normal = np.column_stack(
        (
            -(v_delta + V_BUMPS) / lengths,
            (v_delta - V_BUMPS) / lengths,
        )
    ).ravel()
    u_port_flux = KU * u_normal
    v_port_flux = KV * v_normal
    u_vertex_flux = network.restriction_matrix.T @ u_port_flux
    v_vertex_flux = network.restriction_matrix.T @ v_port_flux
    boundary_u = (
        VALUE_MATRIX[0, 0] * U_VALUES
        + VALUE_MATRIX[0, 1] * V_VALUES
        + FLUX_MATRIX[0, 0] * u_vertex_flux
        + FLUX_MATRIX[0, 1] * v_vertex_flux
    )
    boundary_v = (
        VALUE_MATRIX[1, 0] * U_VALUES
        + VALUE_MATRIX[1, 1] * V_VALUES
        + FLUX_MATRIX[1, 0] * u_vertex_flux
        + FLUX_MATRIX[1, 1] * v_vertex_flux
    )
    return {
        "u": u,
        "v": v,
        "forcing_u": forcing_u,
        "forcing_v": forcing_v,
        "u_port_flux": u_port_flux,
        "v_port_flux": v_port_flux,
        "u_vertex_flux": u_vertex_flux,
        "v_vertex_flux": v_vertex_flux,
        "boundary_u": boundary_u,
        "boundary_v": boundary_v,
    }
