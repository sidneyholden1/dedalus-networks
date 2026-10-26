"""Connected geometric edge regions, balancing caller-supplied work weights.

A heuristic, not an optimal cut. The embedding guides recursive bisection;
small disconnected fragments are attached to adjacent anchored regions.
"""
import numpy as np
from scipy import sparse


def spatial_owners(network, num_parts, positions, weights=None):
    parts = int(num_parts)
    if parts != num_parts or not 1 <= parts <= network.num_edges:
        raise ValueError('Require 1 <= integer num_parts <= number of edges')
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != network.num_vertices or not positions.shape[1] or not np.isfinite(positions).all():
        raise ValueError('Supply finite embedding coordinates for every vertex')
    weights = np.ones(network.num_edges) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (network.num_edges,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Work weights must be finite and positive for every edge')
    endpoints = network.endpoints
    centers = positions[endpoints].mean(axis=1)
    owners = np.empty(network.num_edges, dtype=np.int64)
    def split(ids, first, count):
        if count == 1:
            owners[ids] = first
            return
        axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
        order = ids[np.lexsort((ids, centers[ids, axis]))]
        prefix = np.cumsum(weights[order])
        left = count//2
        proposed = np.searchsorted(prefix, prefix[-1]*left/count)+1
        # Leave at least one edge in each eventual region, even for a heavy edge.
        options = np.unique(np.clip([proposed-1, proposed], left, len(order)-(count-left)))
        cut = int(min(options, key=lambda k: abs(prefix[k-1]-prefix[-1]*left/count)))
        split(order[:cut], first, left)
        split(order[cut:], first+left, count-left)
    split(np.arange(network.num_edges), 0, parts)
    # Identify a connected anchor for each region before moving any fragments.
    anchors = np.zeros(network.num_edges, dtype=bool)
    pending = []
    ordered = np.argsort(owners, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(owners, minlength=parts))]
    for part in range(parts):
        edges = ordered[offsets[part]:offsets[part+1]]
        vertices, inverse = np.unique(endpoints[edges], return_inverse=True)
        mapped = inverse.reshape(-1, 2)
        graph = sparse.coo_matrix((np.ones(len(edges)), (mapped[:, 0], mapped[:, 1])), shape=(len(vertices), len(vertices))).tocsr()
        number, labels = sparse.csgraph.connected_components(graph, directed=False)
        edge_labels = labels[mapped[:, 0]]
        component_weights = np.bincount(edge_labels, weights=weights[edges], minlength=number)
        anchor = int(np.argmax(component_weights))
        anchors[edges[edge_labels == anchor]] = True
        for component in range(number):
            if component != anchor:
                pending.append(edges[edge_labels == component])
    incident = sparse.coo_matrix((np.ones(2*network.num_edges, dtype=np.int8),
                                  (endpoints.ravel(), np.repeat(np.arange(network.num_edges), 2))),
                                 shape=(network.num_vertices, network.num_edges)).tocsr()
    loads = np.bincount(owners, weights=weights, minlength=parts)
    average = weights.sum()/parts
    # Connected components always attach to an already-connected target. A
    # pass with no progress identifies an unsupported disconnected input.
    while pending:
        remaining = []
        for edges in pending:
            contacts = {}
            for vertex in np.unique(endpoints[edges]):
                neighbors = incident.indices[incident.indptr[vertex]:incident.indptr[vertex+1]]
                for part in np.unique(owners[neighbors[anchors[neighbors]]]):
                    contacts[int(part)] = contacts.get(int(part), 0)+1
            if not contacts:
                remaining.append(edges)
                continue
            old = int(owners[edges[0]])
            weight = float(weights[edges].sum())
            target = min(contacts, key=lambda p: (max(0., (loads[p]+weight)/average-1.1), -contacts[p], loads[p], p))
            loads[old] -= weight
            loads[target] += weight
            owners[edges] = target
            anchors[edges] = True
        if len(remaining) == len(pending):
            raise ValueError('Cannot make connected spatial parts for this disconnected graph')
        pending = remaining
    return owners
