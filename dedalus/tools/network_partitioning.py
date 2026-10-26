"""Optional corridor and Mt-KaHyPar preparation for explicit whole-edge ownership.

Call on one preparation process. The numerical solver never imports or invokes
Mt-KaHyPar implicitly. Existing NetworkPartition methods remain independent.
"""
import importlib.metadata
import operator
import threading
import time

import numpy as np

_LOCK = threading.Lock()
_INITIALIZER = None
_THREADS = None


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive integer')
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be a positive integer') from error
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _integer_work(weights, count):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (count,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('weights must contain one positive finite value per edge')
    total = float(weights.sum())
    if not np.isfinite(total):
        raise ValueError('The total edge weight exceeds the supported floating-point range')
    scale = min(1000 / float(weights.min()), (2**30 - 1) / total)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Edge weights cannot be represented safely')
    integers = np.maximum(1, np.rint(weights * scale)).astype(np.int64)
    error = float(np.max(np.abs(integers / scale / weights - 1)))
    if integers.sum() >= 2**31 or error > .01:
        raise ValueError('Weight conversion would overflow or change an edge weight by over 1%')
    return integers, dict(scale=scale, maximum_relative_rounding_error=error)


def mt_kahypar_partition(network, num_parts, *, weights=None, vertex_weights=None,
                        objective='km1', epsilon=.05, preset='default', seed=0,
                        threads=1):
    """Return ``(NetworkPartition, metadata)`` from an optional external package.

    Each network edge is an indivisible work object; each physical vertex is
    an incidence hyperedge. ``weights`` estimates edge work. Integer
    ``vertex_weights`` estimates scalar interface multiplicity (default one).
    These objectives are surrogates: constraints can retain additional unknowns.

    Supported objectives are ``cut``, ``km1`` and ``soed``; presets are
    ``default`` and ``quality``. Disconnected parts are permitted. Imbalance
    is a package target and may be infeasible for indivisible heavy edges.
    The package's process-global worker pool has a fixed thread count after
    the first call through this adapter. Use a fresh process to change it.

    A full graph is required, normally on MPI root during input preparation.
    Save the resulting owners with graph provenance, then pass them explicitly
    to NetworkPartition/Distributor. No default partition policy is changed.
    Missing optional dependencies or invalid inputs raise an actionable error;
    existing ``connected``, ``spatial`` and ``balanced`` methods remain available.
    """
    from ..core.network import Network, NetworkPartition
    global _INITIALIZER, _THREADS
    tick = time.perf_counter()
    network = Network(network)
    if getattr(network, '_local_input', False):
        raise ValueError('Prepare the external partition from a full graph before distributing shards')
    parts = _positive_integer(num_parts, 'num_parts')
    threads = _positive_integer(threads, 'threads')
    if network.num_edges < parts:
        raise ValueError('Mt-KaHyPar requires at least one edge per requested part; use an existing partitioner for idle ranks')
    if objective not in ('cut', 'km1', 'soed') or preset not in ('default', 'quality'):
        raise ValueError('Unsupported Mt-KaHyPar objective or preset')
    if not np.isfinite(epsilon) or epsilon <= 0 or epsilon >= 1:
        raise ValueError('epsilon must be finite and between zero and one')
    try:
        seed = operator.index(seed)
    except TypeError as error:
        raise ValueError('seed must be a nonnegative integer') from error
    if seed < 0 or seed >= 2**31:
        raise ValueError('seed must lie in [0, 2**31)')
    work = np.ones(network.num_edges) if weights is None else np.asarray(weights, dtype=float)
    integer_work, conversion = _integer_work(work, network.num_edges)
    q = np.ones(network.num_vertices, dtype=np.int64) if vertex_weights is None else np.asarray(vertex_weights)
    if (q.shape != (network.num_vertices,) or q.dtype.kind not in 'iu'
            or np.any(q < 0) or sum(map(int, q)) >= 2**31):
        raise ValueError('vertex_weights must be nonnegative integers in the supported range')
    try:
        import mtkahypar as package
    except ImportError as error:
        raise ImportError('Mt-KaHyPar is optional. Install a compatible mtkahypar package in a separate preparation environment, or use NetworkPartition.connected/spatial/balanced.') from error
    nets = [[] for _ in range(network.num_vertices)]
    for edge, (u, v) in enumerate(network.endpoints):
        nets[u].append(edge)
        if u != v:
            nets[v].append(edge)
    selected = [i for i, net in enumerate(nets) if net and q[i] > 0]
    with _LOCK:
        if _INITIALIZER is None:
            package.set_seed(seed)
            _INITIALIZER = package.initialize(threads)
            _THREADS = threads
        elif threads != _THREADS:
            raise ValueError('Mt-KaHyPar thread count is already fixed in this process; use a fresh preparation process')
        package.set_seed(seed)
        context = _INITIALIZER.context_from_preset(getattr(package.PresetType, preset.upper()))
        context.set_partitioning_parameters(parts, epsilon, getattr(package.Objective, objective.upper()))
        hypergraph = _INITIALIZER.create_hypergraph(
            context, network.num_edges, len(selected), [nets[i] for i in selected],
            integer_work.tolist(), [int(q[i]) for i in selected])
        built = time.perf_counter()
        result = hypergraph.partition(context)
        partitioned = time.perf_counter()
        owners = np.array([result.block_id(i) for i in range(network.num_edges)], dtype=np.int64)
        metrics = {name: int(getattr(result, name)()) for name in ('cut', 'km1', 'soed')}
    partition = NetworkPartition(network, owners, num_parts=parts)
    loads = np.bincount(owners, weights=work, minlength=parts)
    info = dict(package='mtkahypar', version=importlib.metadata.version('mtkahypar'),
                objective=objective, preset=preset, seed=seed, threads=threads,
                epsilon=epsilon, num_parts=parts, weight_conversion=conversion,
                package_metrics=metrics, estimated_loads=loads.tolist(),
                estimated_imbalance=float(loads.max()/loads.mean()),
                conversion_and_import_seconds=built-tick,
                partition_seconds=partitioned-built,
                total_seconds=time.perf_counter()-tick)
    return partition, info


def corridor_partition(network, num_parts, *, weights=None):
    """Return a junction-coherent corridor partition and preparation metadata.

    Degree-two corridors form indivisible assignment blocks, capped at mean
    part work (or the heaviest edge when larger). Blocks are ordered by their
    highest-degree junction, then by canonical edge order. Centered cumulative
    work assigns contiguous blocks to ranks. This keeps most ranks incident
    to few junctions while balancing coefficient-and-tau work.

    Recommended only for corridor-rich hub graphs validated by the partition
    cost study. It is not a universal replacement for spatial/connected methods.
    Supply positive per-edge work, e.g. F*(N+2) for the tested second-order PDE.
    Sparse factorization fill is topology dependent and is not predicted exactly.
    Load imbalance can reach an additive largest-block error; tight percentage
    balance is not guaranteed for heavy indivisible edges. Idle ranks are valid.

    Preparation requires a full graph and has no external partition dependency.
    It changes only explicit edge ownership, never equations or graph topology.
    Returns ``(NetworkPartition, metadata)``; call on MPI root and broadcast
    ``partition.edge_owners`` before constructing rank-local distributors.
    """
    from ..core.network import Network, NetworkPartition
    import hashlib
    tick = time.perf_counter()
    network = Network(network)
    if getattr(network, '_local_input', False):
        raise ValueError('Prepare corridor ownership from the full graph before distributing shards')
    parts = _positive_integer(num_parts, 'num_parts')
    work = np.ones(network.num_edges) if weights is None else np.asarray(weights, dtype=float)
    if (work.shape != (network.num_edges,) or np.any(~np.isfinite(work))
            or np.any(work <= 0) or not np.isfinite(work.sum())):
        raise ValueError('weights must contain positive finite work for every edge with a finite sum')
    if not network.num_edges:
        return NetworkPartition(network, np.empty(0, dtype=np.int64), num_parts=parts), dict(
            algorithm='corridor-junction-prefix', version=1, num_parts=parts, blocks=0,
            total_seconds=time.perf_counter()-tick)
    pins = [[] for _ in range(network.num_vertices)]
    degree = np.bincount(network.endpoints.ravel(), minlength=network.num_vertices)
    for edge, (u, v) in enumerate(network.endpoints):
        pins[u].append(edge)
        if u != v:
            pins[v].append(edge)
    parent = np.arange(network.num_edges)
    block_work = work.copy()
    counts = np.ones(network.num_edges, dtype=int)
    def root(edge):
        while parent[edge] != edge:
            parent[edge] = parent[parent[edge]]
            edge = parent[edge]
        return edge
    capacity = max(float(work.max()), float(work.sum()/parts))
    for vertex, edges in enumerate(pins):
        if degree[vertex] != 2 or len(edges) != 2:
            continue
        a, b = map(root, edges)
        if a == b or block_work[a]+block_work[b] > capacity:
            continue
        if counts[a] < counts[b]:
            a, b = b, a
        parent[b] = a
        block_work[a] += block_work[b]
        counts[a] += counts[b]
    roots = np.array([root(edge) for edge in range(network.num_edges)])
    unique, inverse = np.unique(roots, return_inverse=True)
    totals = np.bincount(inverse, weights=work)
    anchors = np.full(len(unique), network.num_vertices, dtype=int)
    anchor_degree = np.full(len(unique), -1, dtype=int)
    for edge, ends in enumerate(network.endpoints):
        block = inverse[edge]
        for vertex in ends:
            if (degree[vertex] > anchor_degree[block]
                    or (degree[vertex] == anchor_degree[block] and vertex < anchors[block])):
                anchors[block] = vertex
                anchor_degree[block] = degree[vertex]
    order = np.lexsort((unique, anchors))
    centers = np.cumsum(totals[order]) - .5*totals[order]
    chosen = np.empty(len(unique), dtype=np.int64)
    chosen[order] = np.minimum(parts-1, (centers/(work.sum()/parts)).astype(np.int64))
    owners = chosen[inverse]
    partition = NetworkPartition(network, owners, num_parts=parts)
    loads = np.bincount(owners, weights=work, minlength=parts)
    info = dict(algorithm='corridor-junction-prefix', version=1, num_parts=parts,
                blocks=len(unique), contractions=network.num_edges-len(unique),
                block_capacity=capacity, maximum_block_weight=float(totals.max()),
                anchor_count=len(np.unique(anchors)), estimated_loads=loads.tolist(),
                estimated_imbalance=float(loads.max()/loads.mean()),
                owners_sha256=hashlib.sha256(owners.tobytes()).hexdigest(),
                total_seconds=time.perf_counter()-tick)
    return partition, info
