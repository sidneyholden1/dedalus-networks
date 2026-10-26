"""Deterministic synthetic metric graphs for JCP benchmark campaigns."""

from __future__ import annotations

import math

import numpy as np

import dedalus.public as d3


GRAPH_FAMILIES = (
    "chain",
    "comb",
    "binary_tree",
    "star",
    "cycle",
    "grid",
    "irregular",
)
METRICS = ("uniform", "heterogeneous")


def _grid_capacity(num_vertices: int, width: int) -> int:
    """Return the number of lattice edges in a row-major partial grid."""
    rows = math.ceil(num_vertices / width)
    horizontal = num_vertices - rows
    vertical = max(0, num_vertices - width)
    return horizontal + vertical


def _grid_endpoints(num_edges: int) -> list[tuple[int, int]]:
    """Build a connected, near-square grid with exactly ``num_edges``."""
    if num_edges < 1:
        raise ValueError("A grid requires at least one edge.")
    best = None
    max_width = max(2, math.ceil(math.sqrt(num_edges)) + 2)
    for width in range(2, max_width + 1):
        if _grid_capacity(num_edges + 1, width) < num_edges:
            continue
        low = 2
        high = num_edges + 1
        while low < high:
            middle = (low + high) // 2
            if _grid_capacity(middle, width) >= num_edges:
                high = middle
            else:
                low = middle + 1
        num_vertices = low
        height = math.ceil(num_vertices / width)
        score = (num_vertices, abs(width - height), max(width, height))
        if best is None or score < best[0]:
            best = (score, num_vertices, width)
    if best is None:
        raise RuntimeError("Unable to construct the requested grid.")

    _, num_vertices, width = best
    tree = []
    for vertex in range(1, num_vertices):
        if vertex % width:
            tree.append((vertex - 1, vertex))
        else:
            tree.append((vertex - width, vertex))
    tree_set = {tuple(sorted(edge)) for edge in tree}

    extras = []
    for vertex in range(num_vertices):
        if vertex % width:
            edge = (vertex - 1, vertex)
            if edge not in tree_set:
                extras.append(edge)
        if vertex >= width:
            edge = (vertex - width, vertex)
            if edge not in tree_set:
                extras.append(edge)
    required = num_edges - len(tree)
    if required < 0 or required > len(extras):
        raise RuntimeError("Grid construction did not bracket the edge count.")
    return tree + extras[:required]


def _irregular_endpoints(
    num_edges: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Build a seeded connected graph with heterogeneous vertex degrees."""
    if num_edges < 2:
        raise ValueError("An irregular graph requires at least two edges.")
    num_vertices = max(3, min(num_edges + 1, round(0.6 * num_edges) + 1))
    rng = np.random.default_rng(seed)
    endpoints = []
    existing = set()

    # A random recursive tree guarantees connectivity and naturally creates
    # a broad degree distribution without requiring an external graph package.
    for vertex in range(1, num_vertices):
        parent = int(np.floor(vertex * rng.random() ** 1.8))
        edge = (parent, vertex)
        endpoints.append(edge)
        existing.add(edge)

    attempts = 0
    max_attempts = max(1000, 50 * num_edges)
    while len(endpoints) < num_edges and attempts < max_attempts:
        tail, head = rng.choice(num_vertices, size=2, replace=False)
        edge = tuple(sorted((int(tail), int(head))))
        attempts += 1
        if edge in existing:
            continue
        existing.add(edge)
        endpoints.append(edge)
    if len(endpoints) != num_edges:
        raise RuntimeError("Unable to generate enough distinct irregular edges.")
    return endpoints


def graph_endpoints(
    graph_family: str,
    num_edges: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return deterministic endpoints with exactly ``num_edges`` rows."""
    if num_edges <= 0:
        raise ValueError("Synthetic graphs require a positive edge count.")
    if graph_family == "chain":
        # Avoid a temporary Python tuple and two Python integers per edge.
        endpoints = np.column_stack((np.arange(num_edges), np.arange(1, num_edges + 1)))
    elif graph_family == "comb":
        # A chain with one single-edge branch at each spine vertex. Interleave
        # teeth and spine edges so contiguous edge intervals remain connected.
        spine_edges = num_edges // 2
        endpoints = []
        for vertex in range(spine_edges):
            endpoints.extend([(vertex, spine_edges+1+vertex), (vertex, vertex+1)])
        if num_edges % 2:
            endpoints.append((spine_edges, 2*spine_edges+1))
    elif graph_family == "binary_tree":
        endpoints = [
            ((child - 1) // 2, child)
            for child in range(1, num_edges + 1)
        ]
    elif graph_family == "star":
        endpoints = [(0, edge + 1) for edge in range(num_edges)]
    elif graph_family == "cycle":
        if num_edges < 3:
            raise ValueError("The cycle graph requires at least three edges.")
        endpoints = [
            (vertex, (vertex + 1) % num_edges)
            for vertex in range(num_edges)
        ]
    elif graph_family == "grid":
        endpoints = _grid_endpoints(num_edges)
    elif graph_family == "irregular":
        endpoints = _irregular_endpoints(num_edges, seed)
    else:
        raise ValueError(f"Unknown graph family {graph_family!r}.")
    array = np.asarray(endpoints, dtype=np.int64)
    if array.shape != (num_edges, 2):
        raise RuntimeError("Synthetic graph did not have the requested size.")
    return array


def metric_lengths(
    num_edges: int,
    *,
    metric: str,
    seed: int = 0,
) -> np.ndarray:
    """Return uniform or smooth deterministic heterogeneous edge lengths."""
    if metric == "uniform":
        return np.ones(num_edges, dtype=np.float64)
    if metric != "heterogeneous":
        raise ValueError(f"Unknown metric {metric!r}.")
    edge = np.arange(num_edges, dtype=np.float64) + 1
    phase = 0.173 * (seed % 997)
    lengths = (
        0.85
        + 0.28 * np.sin(0.73 * edge + phase)
        + 0.17 * np.cos(1.31 * edge - 0.5 * phase)
    )
    if np.any(lengths <= 0):
        raise RuntimeError("Heterogeneous metric produced a nonpositive length.")
    return lengths


def build_network(
    graph_family: str,
    num_edges: int,
    *,
    metric: str = "uniform",
    seed: int = 0,
) -> d3.Network:
    """Construct a deterministic Dedalus metric Network."""
    endpoints = graph_endpoints(graph_family, num_edges, seed=seed)
    lengths = metric_lengths(num_edges, metric=metric, seed=seed)
    return d3.Network(endpoints=endpoints, lengths=lengths)
