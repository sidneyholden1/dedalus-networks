"""Metric-network data, bases, and trace operators."""

from collections.abc import Mapping, Sequence
import hashlib
import heapq
import importlib
import json
from types import MappingProxyType

from mpi4py import MPI
from .network_storage import take_elements, CanonicalIndices, LocalGraphTable

import numpy as np
from scipy import sparse

from .basis import CardinalBasis, CardinalMap, Chebyshev
from .coords import Coordinate, CoordinateSystem
from .field import Operand
from . import operators
from ..tools.cache import CachedClass, CachedAttribute
from ..tools.collective_linalg import CollectiveSparseSolver
from ..tools.collective_metadata import CollectiveMetadata


__all__ = [
    "Network",
    "NetworkPartition",
    "NetworkBasis",
    "NetworkBasisSet",
]


def _portable_network_id(value):
    """Normalize a conservative subset of network IDs for JSON storage."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if -(1 << 63) <= value < (1 << 63):
            return value
        raise OverflowError("integer network ID is outside signed 64-bit range")
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        raise ValueError("floating-point network ID must be finite")
    if isinstance(value, tuple):
        return tuple(_portable_network_id(item) for item in value)
    raise TypeError(f"unsupported network ID type {type(value).__name__}")


def _encode_network_ids(values):
    """Encode portable network IDs individually, or return ``None``."""
    try:
        return tuple(
            json.dumps(
                _portable_network_id(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            for value in values
        )
    except (TypeError, ValueError, OverflowError):
        return None


class _CanonicalIDs(Sequence):
    """Immutable default integer IDs with tuple-compatible value equality."""

    __slots__ = ('_size',)

    def __init__(self, size):
        object.__setattr__(self, '_size', int(size))

    def __setattr__(self, name, value):
        raise AttributeError('Canonical IDs are immutable')

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(range(self._size))

    def __getitem__(self, index):
        value = range(self._size)[index]
        return tuple(value) if isinstance(index, slice) else value

    def __eq__(self, other):
        if isinstance(other, _CanonicalIDs):
            return self._size == other._size
        if not isinstance(other, (tuple, range)):
            return NotImplemented
        return len(other) == self._size and all(a == b for a, b in zip(self, other))

    def __reduce__(self):
        return (_CanonicalIDs, (self._size,))


def _network_id_signature(values):
    """Return a compact signature for portable IDs or their fallback mode."""
    # Collective-only format: encode the ordered sequence in one JSON call.
    # Output/restart IDs retain their individual json-v1 representation.
    if isinstance(values, _CanonicalIDs):
        return ('canonical-integer-range-v1', len(values))
    # Explicit integer IDs must agree with implicit default IDs across ranks.
    if (all(type(value) is int or isinstance(value, np.integer) for value in values)
            and all(value == index for index, value in enumerate(values))):
        return ('canonical-integer-range-v1', len(values))
    try:
        normalized = tuple(_portable_network_id(value) for value in values)
        payload = json.dumps(
            normalized, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError):
        return ("canonical-index-only", None)
    return ("json-sequence-v2", hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest())


def _network_vertex_groups_signature(groups):
    """Return a compact signature for normalized named vertex sets."""
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(groups):
        name_payload = name.encode("utf-8")
        indices = np.asarray(groups[name], dtype=np.int64)
        digest.update(len(name_payload).to_bytes(8, "little"))
        digest.update(name_payload)
        digest.update(indices.size.to_bytes(8, "little"))
        digest.update(indices.tobytes())
    return (len(groups), digest.digest())


def _canonicalize_network_id_json(value):
    """Validate and canonically re-encode one JSON network ID."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError("JSON network IDs must be strings")

    def reject_constant(constant):
        raise ValueError(f"non-finite JSON constant {constant!r}")

    decoded = json.loads(value, parse_constant=reject_constant)

    def lists_to_tuples(item):
        if isinstance(item, list):
            return tuple(lists_to_tuples(element) for element in item)
        return item

    normalized = _portable_network_id(lists_to_tuples(decoded))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _readonly_array(values, dtype, *, ndim=None):
    array = np.array(values, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional array, got shape {array.shape}.")
    array.flags.writeable = False
    return array


def _require_unique(values, kind):
    if isinstance(values, _CanonicalIDs):
        return
    try:
        unique = set(values)
    except TypeError as error:
        raise TypeError(f"Network {kind} IDs must be hashable.") from error
    if len(unique) != len(values):
        raise ValueError(f"Network {kind} IDs must be unique.")


def _normalize_vertex_group(indices, num_vertices, name):
    """Normalize a named vertex set from indices or a Boolean mask."""
    raw = np.asarray(indices)
    if raw.ndim != 1:
        raise ValueError(
            f"Vertex group {name!r} must be one-dimensional."
        )
    if raw.dtype.kind == "b":
        if raw.shape != (num_vertices,):
            raise ValueError(
                f"Boolean vertex group {name!r} must have shape "
                f"({num_vertices},)."
            )
        group = np.flatnonzero(raw)
    else:
        contains_boolean = any(
            isinstance(value, (bool, np.bool_))
            for value in np.asarray(indices, dtype=object).flat
        )
        kind = raw.dtype.kind
        valid = kind in "iufc" and not contains_boolean
        if valid and kind in "fc":
            valid = np.all(np.isfinite(raw))
            values = raw.real if kind == "c" else raw
            if kind == "c":
                valid = valid and np.all(raw.imag == 0)
            valid = valid and np.all(values == np.floor(values))
        else:
            values = raw
        if valid and raw.size:
            limits = np.iinfo(np.int64)
            valid = np.all(values >= limits.min) and np.all(
                values <= limits.max
            )
        if not valid:
            raise ValueError(
                f"Vertex group {name!r} must contain finite integer "
                "indices or be a Boolean mask."
            )
        group = values.astype(np.int64)
    group = np.unique(group)
    if group.size and (group.min() < 0 or group.max() >= num_vertices):
        raise ValueError(f"Vertex group {name!r} contains invalid indices.")
    return _readonly_array(group, np.int64, ndim=1)


def _is_optional_graph_instance(graph, module_name, class_name):
    """Check an optional graph type without adding a runtime dependency."""
    source_modules = {
        base.__module__.partition(".")[0]
        for base in type(graph).__mro__
    }
    if module_name not in source_modules:
        return False
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return False
        raise
    graph_class = getattr(module, class_name, None)
    return isinstance(graph_class, type) and isinstance(graph, graph_class)


def _metricgraph_data(graph):
    graph.validate()
    return {
        "vertex_ids": tuple(graph.V.info["ids"]),
        "edge_ids": tuple(edge.id for edge in graph.E),
        "endpoints": np.asarray(graph.E.endpoints),
        "lengths": np.asarray(graph.E.lengths),
        "vertex_groups": {
            name: np.asarray(graph.groups[name])
            for name in graph.groups
        },
    }


def _networkx_data(graph, length):
    vertex_ids = tuple(graph.nodes)
    vertex_index = {vertex: i for i, vertex in enumerate(vertex_ids)}
    edge_ids = []
    endpoints = []
    lengths = []

    if graph.is_multigraph():
        edge_records = graph.edges(keys=True, data=True)
        for tail, head, key, data in edge_records:
            edge_ids.append((tail, head, key))
            endpoints.append((vertex_index[tail], vertex_index[head]))
            try:
                lengths.append(data[length])
            except KeyError as error:
                raise ValueError(
                    f"NetworkX edge {(tail, head, key)!r} is missing "
                    f"the {length!r} length attribute."
                ) from error
    else:
        edge_records = graph.edges(data=True)
        for tail, head, data in edge_records:
            edge_ids.append((tail, head))
            endpoints.append((vertex_index[tail], vertex_index[head]))
            try:
                lengths.append(data[length])
            except KeyError as error:
                raise ValueError(
                    f"NetworkX edge {(tail, head)!r} is missing "
                    f"the {length!r} length attribute."
                ) from error

    return {
        "vertex_ids": vertex_ids,
        "edge_ids": tuple(edge_ids),
        "endpoints": endpoints,
        "lengths": lengths,
    }


def _adapt_graph(graph, length):
    protocol = getattr(graph, "__dedalus_network__", None)
    if protocol is not None:
        if not callable(protocol):
            raise TypeError("__dedalus_network__ must be a callable method.")
        data = protocol()
        if isinstance(data, Network):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("__dedalus_network__() must return a mapping or Network.")
        data = dict(data)
        missing = tuple(
            name
            for name in ("endpoints", "lengths")
            if data.get(name) is None
        )
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ValueError(
                "__dedalus_network__() mapping is missing required "
                f"non-None item(s): {names}."
            )
        return data

    if _is_optional_graph_instance(graph, "metricgraph", "MetricGraph"):
        return _metricgraph_data(graph)
    if _is_optional_graph_instance(graph, "networkx", "Graph"):
        return _networkx_data(graph, length)

    raise TypeError(
        "Unsupported graph object. Provide a NetworkX graph, a metricgraph "
        "MetricGraph, an object implementing __dedalus_network__(), or explicit "
        "vertex_ids/endpoints/lengths arrays."
    )


def _rebuild_network(
    vertex_ids,
    edge_ids,
    endpoints,
    lengths,
    vertex_groups,
    coordinate_names,
):
    """Reconstruct a normalized network during unpickling."""
    return Network(
        vertex_ids=vertex_ids,
        edge_ids=edge_ids,
        endpoints=endpoints,
        lengths=lengths,
        vertex_groups=vertex_groups,
        coordinate_names=coordinate_names,
    )


class Network(CoordinateSystem):
    """Immutable metric-network snapshot and Dedalus coordinate collection.

    The primary constructor accepts a supported graph object.  Explicit arrays
    provide the package-neutral low-level interface.  The second coordinate
    is a normalized reference position on every edge, independent of the
    physical edge lengths.

    Parameters
    ----------
    graph : Network or graph-like, optional
        A NetworkX graph, ``metricgraph.MetricGraph``, or object implementing
        ``__dedalus_network__()``.  The protocol must return a :class:`Network`
        or a mapping containing the explicit constructor data below.
    vertex_ids : sequence of hashable, optional
        External vertex identifiers.  If omitted in the explicit interface,
        consecutive integer identifiers are inferred from ``endpoints``.
    edge_ids : sequence of hashable, optional
        External edge identifiers, one per edge.  Defaults to consecutive
        integers in the explicit interface.
    endpoints : array-like, shape (num_edges, 2), optional
        Integer indices into ``vertex_ids`` giving each oriented edge's tail
        and head.  Required with ``lengths`` when ``graph`` is omitted.
    lengths : array-like, shape (num_edges,), optional
        Positive, finite physical edge lengths.  Required with ``endpoints``
        when ``graph`` is omitted.
    vertex_groups : mapping, optional
        Named groups of vertex-table indices or Boolean vertex masks.
    length : str, optional
        NetworkX edge attribute containing physical length (default:
        ``"length"``).
    coordinate_names : 2-tuple of str, optional
        Names for the edge-index and reference-position coordinates (default:
        ``("e", "s")``).

    Attributes
    ----------
    edge_coord, position_coord : Coordinate
        Dedalus coordinates for the edge index and normalized edge position.
    endpoints : ndarray, shape (num_edges, 2)
        Read-only canonical endpoint table.
    lengths : ndarray, shape (num_edges,)
        Read-only physical edge lengths.
    restriction_matrix : scipy.sparse.csr_matrix
        Vertex-to-port restriction matrix in canonical port order.
    port_edges, port_signs : ndarray
        Edge indices and outward signs for the canonical endpoint ports.
    degree : ndarray, shape (num_vertices,)
        Number of incident ports at each vertex.

    Notes
    -----
    The constructor snapshots graph topology and geometry.  Edge orientation
    only fixes the ordering and outward signs of endpoint data; it does not
    change orientation-invariant equations.
    """

    _dedalus_replicated_distributor = True

    def __new__(
        cls,
        graph=None,
        *,
        vertex_ids=None,
        edge_ids=None,
        endpoints=None,
        lengths=None,
        vertex_groups=None,
        length="length",
        coordinate_names=None,
    ):
        explicit = any(
            value is not None
            for value in (vertex_ids, edge_ids, endpoints, lengths, vertex_groups)
        )
        if (
            isinstance(graph, cls)
            and not explicit
            and coordinate_names is None
        ):
            return graph
        if graph is not None:
            if explicit:
                raise TypeError("Pass either a graph object or explicit network arrays.")
            data = graph if isinstance(graph, cls) else _adapt_graph(graph, length)
            if isinstance(data, cls):
                if coordinate_names is None:
                    return data
                vertex_ids = data.vertex_ids
                edge_ids = data.edge_ids
                endpoints = data.endpoints
                lengths = data.lengths
                vertex_groups = data.vertex_groups
            else:
                if coordinate_names is None:
                    coordinate_names = data.get("coordinate_names")
                vertex_ids = data.get("vertex_ids")
                edge_ids = data.get("edge_ids")
                endpoints = data.get("endpoints")
                lengths = data.get("lengths")
                vertex_groups = data.get("vertex_groups")
        elif endpoints is None or lengths is None:
            raise TypeError("Explicit networks require endpoints and lengths.")

        if coordinate_names is None:
            coordinate_names = ("e", "s")

        endpoints_array = np.asarray(endpoints)
        if endpoints_array.shape == (0,):
            endpoints_array = endpoints_array.reshape((0, 2))
        if endpoints_array.ndim != 2 or endpoints_array.shape[1:] != (2,):
            raise ValueError("Network endpoints must have shape (num_edges, 2).")
        if not np.issubdtype(endpoints_array.dtype, np.integer):
            if not np.all(np.equal(endpoints_array, np.floor(endpoints_array))):
                raise ValueError("Network endpoints must contain integer indices.")
        endpoints_array = _readonly_array(endpoints_array, np.int64, ndim=2)
        num_edges = endpoints_array.shape[0]

        if vertex_ids is None:
            num_vertices = int(endpoints_array.max()) + 1 if num_edges else 0
            vertex_ids = _CanonicalIDs(num_vertices)
        else:
            vertex_ids = vertex_ids if isinstance(vertex_ids, _CanonicalIDs) else tuple(vertex_ids)
            num_vertices = len(vertex_ids)
        if edge_ids is None:
            edge_ids = _CanonicalIDs(num_edges)
        else:
            edge_ids = edge_ids if isinstance(edge_ids, _CanonicalIDs) else tuple(edge_ids)

        if len(edge_ids) != num_edges:
            raise ValueError("Network edge_ids must contain one ID per edge.")
        _require_unique(vertex_ids, "vertex")
        _require_unique(edge_ids, "edge")

        if endpoints_array.size:
            if endpoints_array.min() < 0 or endpoints_array.max() >= num_vertices:
                raise ValueError("Network endpoint indices are outside the vertex table.")

        lengths_array = _readonly_array(lengths, np.float64, ndim=1)
        if lengths_array.shape != (num_edges,):
            raise ValueError("Network lengths must contain one value per edge.")
        invalid_lengths = np.flatnonzero(
            ~np.isfinite(lengths_array) | (lengths_array <= 0)
        )
        if invalid_lengths.size:
            details = [
                (edge_ids[i], float(lengths_array[i]))
                for i in invalid_lengths
            ]
            raise ValueError(
                f"Network edge lengths must be positive and finite; invalid {details}."
            )

        if vertex_groups is None:
            group_items = ()
        elif isinstance(vertex_groups, Mapping):
            group_items = vertex_groups.items()
        else:
            raise TypeError("Network vertex_groups must be a mapping.")
        groups = {}
        for name, indices in group_items:
            if not isinstance(name, str):
                raise TypeError("Network vertex group names must be strings.")
            if name in groups:
                raise ValueError(
                    f"Network vertex group name {name!r} is duplicated."
                )
            groups[name] = _normalize_vertex_group(
                indices,
                num_vertices,
                name,
            )

        try:
            edge_name, position_name = coordinate_names
        except (TypeError, ValueError) as error:
            raise ValueError("coordinate_names must contain exactly two names.") from error
        edge_name = str(edge_name)
        position_name = str(position_name)
        if edge_name == position_name:
            raise ValueError("Network coordinate names must be distinct.")
        obj = object.__new__(cls)
        edge_coord = Coordinate(edge_name)
        position_coord = Coordinate(position_name)
        obj.names = (edge_coord.name, position_coord.name)
        obj.coords = (edge_coord, position_coord)
        obj.dim = 2
        obj.curvilinear = False
        obj.default_nonconst_groups = (1, 1)
        obj.edge_coord = edge_coord
        obj.position_coord = position_coord
        obj.vertex_ids = vertex_ids
        obj.edge_ids = edge_ids
        obj.endpoints = endpoints_array
        obj.lengths = lengths_array
        obj.vertex_groups = MappingProxyType(groups)
        obj.num_vertices = num_vertices
        obj.num_edges = num_edges
        obj.num_ports = 2 * num_edges
        obj.total_length = float(lengths_array.sum())
        obj.source_type = None if graph is None else type(graph)

        # Every discretization bucket over this Network shares one full-vertex
        # basis.  Store only direct Network metadata on it: attaching a parent
        # NetworkBasis would be ambiguous as soon as multiple edge buckets use
        # different interval families or resolutions.
        if num_vertices:
            vertices_basis = CardinalBasis(edge_coord, size=num_vertices)
            vertices_basis._network = obj
            vertices_basis._network_role = "vertices"
            vertices_basis._network_elements = CanonicalIndices(num_vertices)
        else:
            vertices_basis = None
        obj._vertices_basis = vertices_basis
        obj._frozen = True
        return obj

    def __init__(self, *args, **kwargs):
        # Normalization is performed in __new__ so Network(existing_network)
        # can preserve object identity.
        pass

    @classmethod
    def from_partition(cls, graph=None, partition=None, *, comm=None, root=0):
        """Build a local-input computational graph from a root-supplied partition.

        Only ``root`` supplies a full Network (or accepted graph input) and
        preassigned edge owners. All ranks call this collective constructor.
        The result stores local edges and resident vertices, with owned/ghost
        vertex storage selected automatically by Distributor. Use prepared
        edge shards with ``from_local`` to avoid root-side input preparation.
        The current local-input restrictions also apply to this constructor.
        """
        from .network_local import from_partition
        return from_partition(cls, graph=graph, partition=partition,
                              comm=MPI.COMM_WORLD if comm is None else comm,
                              root=root)

    @classmethod
    def from_local(cls, *, comm, num_vertices, num_edges, edge_indices,
                   endpoints, lengths, coordinate_names=('e', 's')):
        """Collectively construct a graph from disjoint, preassigned edge shards.

        Canonical integer labels must cover [0, num_edges) exactly once and
        endpoints must cover [0, num_vertices). Each rank supplies only its
        owned edges. Isolated vertices and explicit partial basis buckets are
        not currently supported by this optional input path.
        """
        from .network_local import from_local
        return from_local(cls, comm=comm, num_vertices=num_vertices,
                          num_edges=num_edges, edge_indices=edge_indices,
                          endpoints=endpoints, lengths=lengths,
                          coordinate_names=coordinate_names)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("Network snapshots are immutable.")
        object.__setattr__(self, name, value)

    @property
    def restriction_matrix(self):
        if '_restriction_matrix' not in self.__dict__:
            ports = self.endpoints.ravel()
            matrix = sparse.csr_matrix((np.ones(self.num_ports),
                                        (np.arange(self.num_ports), ports)),
                                       shape=(self.num_ports, self.num_vertices))
            for array in (matrix.data, matrix.indices, matrix.indptr):
                array.flags.writeable = False
            object.__setattr__(self, '_restriction_matrix', matrix)
        return self._restriction_matrix

    @property
    def port_edges(self):
        if '_port_edges' not in self.__dict__:
            object.__setattr__(self, '_port_edges', _readonly_array(
                np.repeat(np.arange(self.num_edges), 2), np.int64))
        return self._port_edges

    @property
    def port_signs(self):
        if '_port_signs' not in self.__dict__:
            object.__setattr__(self, '_port_signs', _readonly_array(
                np.tile((-1., 1.), self.num_edges), np.float64))
        return self._port_signs

    @property
    def degree(self):
        if '_degree' not in self.__dict__:
            object.__setattr__(self, '_degree', _readonly_array(
                np.bincount(self.endpoints.ravel(), minlength=self.num_vertices), np.int64))
        return self._degree

    def __len__(self):
        return len(self.coords)

    def __iter__(self):
        return iter(self.coords)

    def __dedalus_separable_coords__(self):
        """Return the stable reduction order for separable operators.

        Reducing the interval coordinate before the distributed edge index
        retains one edgewise contribution per owner until the final global
        reduction.  The reverse order is mathematically equivalent in serial
        but would prematurely place an interval-only intermediate on rank
        zero in a whole-edge layout.
        """
        return (self.position_coord, self.edge_coord)

    def __repr__(self):
        return (
            f"Network(num_vertices={self.num_vertices}, "
            f"num_edges={self.num_edges})"
        )

    def __eq__(self, other):
        """Compare network coordinate collections by identity.

        Network objects carry topology and geometry in addition to their two
        coordinates.  Tuple's elementwise equality would therefore make two
        different networks with the same coordinate names compare equal.
        """
        return self is other

    def __ne__(self, other):
        return self is not other

    __hash__ = object.__hash__

    def __reduce__(self):
        return (
            _rebuild_network,
            (
                self.vertex_ids,
                self.edge_ids,
                self.endpoints,
                self.lengths,
                dict(self.vertex_groups),
                self.names,
            ),
        )


class _LazyPartitionGroups(Sequence):
    """Immutable per-part index arrays, constructed only when requested."""

    def __init__(self, size, build):
        self.size, self.build, self.cache = size, build, {}

    def __len__(self):
        return self.size

    def __getitem__(self, part):
        if isinstance(part, slice):
            return tuple(self[i] for i in range(self.size)[part])
        if part < 0:
            part += self.size
        if not 0 <= part < self.size:
            raise IndexError(part)
        if part not in self.cache:
            self.cache[part] = _readonly_array(self.build(part), np.int64, ndim=1)
        return self.cache[part]


class NetworkPartition:
    """Assignment of whole network edges to parallel subdomains.

    A vertex is interior to a part when every incident edge has the same
    owner.  Vertices incident to multiple owners form the separator for a
    second, subgraph-level Schur complement.  The parallel DtN solver first
    eliminates owned edges and then eliminates these rank-interior vertices.

    Parameters
    ----------
    network : Network or graph-like
        Network to partition.
    edge_owners : array-like, shape (num_edges,)
        Nonnegative owner index for every edge.
    num_parts : int, optional
        Total number of parts.  Defaults to one more than the largest owner.

    Attributes
    ----------
    edge_owners : ndarray, shape (num_edges,)
        Read-only owner table in canonical edge order.
    edges : tuple of ndarray
        Canonical edge indices owned by each part.
    interior_vertices : tuple of ndarray
        Vertices whose incident edges all belong to each part.
    separator_vertices : ndarray
        Vertices incident to multiple parts, together with isolated vertices.
    vertex_equation_owners : ndarray, shape (num_vertices,)
        Deterministic part responsible for each vertex equation.
    owned_vertices, incident_vertices, resident_vertices, ghost_vertices : tuple
        Canonical vertex indices owned by, incident to, stored on, and ghosted
        by each part.  Resident storage is the union of owned and incident
        vertices.
    """

    def __init__(self, network, edge_owners, *, num_parts=None):
        network = Network(network)
        raw_edge_owners = np.asarray(edge_owners)
        if raw_edge_owners.ndim != 1:
            raise ValueError(
                "A network partition requires one owner per edge."
            )
        try:
            numeric_owners = np.asarray(
                raw_edge_owners,
                dtype=np.complex128,
            )
        except (TypeError, ValueError, OverflowError):
            numeric_owners = None
        if numeric_owners is not None and np.any(~np.isfinite(numeric_owners)):
            raise ValueError(
                "Network partition owners must be integer-valued."
            )
        try:
            edge_owners = _readonly_array(
                raw_edge_owners,
                np.int64,
                ndim=1,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Network partition owners must be integer-valued."
            ) from error
        integer_valued = (
            numeric_owners is not None
            and np.all(numeric_owners.imag == 0)
            and np.all(numeric_owners.real == edge_owners)
        )
        if not integer_valued:
            raise ValueError(
                "Network partition owners must be integer-valued."
            )
        if edge_owners.shape != (network.num_edges,):
            raise ValueError("A network partition requires one owner per edge.")
        if edge_owners.size and edge_owners.min() < 0:
            raise ValueError("Network partition owners must be nonnegative.")
        required_parts = int(edge_owners.max()) + 1 if edge_owners.size else 1
        if num_parts is None:
            num_parts = required_parts
        num_parts = int(num_parts)
        if num_parts <= 0 or required_parts > num_parts:
            raise ValueError(
                "Network partition owners must lie in [0, num_parts)."
            )

        # Min/max incident owners identify interiors and separators without
        # allocating a Python set for every vertex on every rank. Repeated
        # endpoints (self-loops and parallel edges) are intentional reductions.
        minimum_owner = np.full(network.num_vertices, num_parts, dtype=np.int64)
        maximum_owner = np.full(network.num_vertices, -1, dtype=np.int64)
        for endpoints in network.endpoints.T:
            np.minimum.at(minimum_owner, endpoints, edge_owners)
            np.maximum.at(maximum_owner, endpoints, edge_owners)
        vertex_owners = np.where(minimum_owner == maximum_owner, minimum_owner, -1)
        condition_owners = minimum_owner.copy()
        isolated = np.flatnonzero(maximum_owner < 0)
        condition_owners[isolated] = isolated % num_parts
        separator = np.flatnonzero(vertex_owners < 0)

        def owner_groups(owners):
            # Stable grouping avoids a full graph scan for every part, while
            # retaining canonical sorted indices and empty/idle partitions.
            if np.all(owners[1:] >= owners[:-1]):
                starts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1]
                ends = np.r_[starts[1:], owners.size]
                slices = {int(owners[begin]): (int(begin), int(end))
                          for begin, end in zip(starts, ends) if begin < owners.size and owners[begin] >= 0}
                return _LazyPartitionGroups(num_parts, lambda part:
                    np.arange(*slices.get(part, (0, 0)), dtype=np.int64))
            labels, order, bounds = _group_label_indices(owners)
            slices = {int(label): slice(int(begin), int(end))
                      for label, begin, end in zip(labels, bounds[:-1], bounds[1:])
                      if label >= 0}
            return _LazyPartitionGroups(num_parts, lambda part:
                                        order[slices.get(part, slice(0, 0))])

        edges = owner_groups(edge_owners)
        owned_vertices = owner_groups(condition_owners)
        incident_vertices = _LazyPartitionGroups(num_parts, lambda part:
            np.unique(network.endpoints[edges[part]].ravel()))
        resident_vertices = _LazyPartitionGroups(num_parts, lambda part:
            np.union1d(owned_vertices[part], incident_vertices[part]))
        ghost_vertices = _LazyPartitionGroups(num_parts, lambda part:
            np.setdiff1d(resident_vertices[part], owned_vertices[part], assume_unique=True))

        self.network = network
        self.num_parts = num_parts
        self.edge_owners = edge_owners
        self.interior_vertex_owners = _readonly_array(
            vertex_owners,
            np.int64,
            ndim=1,
        )
        self.vertex_equation_owners = _readonly_array(
            condition_owners,
            np.int64,
            ndim=1,
        )
        self.separator_vertices = _readonly_array(
            separator,
            np.int64,
            ndim=1,
        )
        self.edges = edges
        self.interior_vertices = owner_groups(vertex_owners)
        self.owned_vertices = owned_vertices
        self.incident_vertices = incident_vertices
        self.resident_vertices = resident_vertices
        self.ghost_vertices = ghost_vertices

    @CachedAttribute
    def _resident_incidence(self):
        """Sparse membership table for immutable rank/vertex halo pairs."""
        counts = np.fromiter(map(len, self.resident_vertices), dtype=np.int64)
        indptr = np.concatenate(([0], np.cumsum(counts)))
        indices = np.concatenate(self.resident_vertices)
        return sparse.csr_matrix(
            (np.ones(indices.size, dtype=bool), indices, indptr),
            shape=(self.num_parts, self.network.num_vertices),
        )

    def _contains_resident_pairs(self, owners, vertices):
        """Check paired halo membership without scanning all pairs per rank."""
        if not np.size(vertices):
            return True
        return bool(np.all(np.asarray(self._resident_incidence[owners, vertices])))

    @classmethod
    def spatial(cls, network, num_parts, positions, weights=None):
        """Build connected spatial regions with estimated edge-work balance.

        ``positions`` is a finite ``(num_vertices, dimension)`` array in the
        network's vertex order. Positive ``weights`` estimate work per edge;
        the default balances edge counts. Use coefficient counts when edge
        resolutions differ, or lengths when the resolution policy makes work
        proportional to length.

        Recursive geometric bisection encourages small spatial interfaces.
        Disconnected fragments attach to adjacent connected regions, favoring
        shared boundary vertices and balanced work. This is a deterministic
        heuristic, not an optimal cut or a guarantee of a particular imbalance.
        Heavy indivisible edges and topology can limit the achievable balance.

        Run on a full graph during offline input preparation, then supply the
        resulting ownership with ``Network.from_partition`` or local shards.
        User-supplied partitions and the existing defaults remain authoritative.
        """
        from .network_partitioning import spatial_owners
        network = Network(network)
        if getattr(network, '_local_input', False):
            raise ValueError('Spatial partitioning requires the full graph during offline preparation')
        owners = spatial_owners(network, num_parts, positions, weights)
        return cls(network, owners, num_parts=int(num_parts))

    @classmethod
    def balanced(cls, network, num_parts, weights=None):
        """Balance indivisible edge work without considering connectivity.

        Parameters
        ----------
        network : Network or graph-like
            Network to partition.
        num_parts : int
            Number of parts.
        weights : array-like, shape (num_edges,), optional
            Nonnegative estimate of the work on each edge (default: uniform).

        Returns
        -------
        NetworkPartition
            Deterministic greedy edge assignment.
        """
        network = Network(network)
        num_parts = int(num_parts)
        if num_parts <= 0:
            raise ValueError("num_parts must be positive.")
        if weights is None:
            weights = np.ones(network.num_edges)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (network.num_edges,):
            raise ValueError("Partition weights must contain one value per edge.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Partition weights must be finite and nonnegative.")

        owners = np.empty(network.num_edges, dtype=np.int64)
        loads = np.zeros(num_parts, dtype=np.float64)
        counts = np.zeros(num_parts, dtype=np.int64)
        order = np.argsort(-weights, kind="stable")
        for edge in order:
            owner = min(
                range(num_parts),
                key=lambda part: (loads[part], counts[part], part),
            )
            owners[edge] = owner
            loads[owner] += weights[edge]
            counts[owner] += 1
        return cls(network, owners, num_parts=num_parts)

    @classmethod
    def connected(cls, network, num_parts, weights=None):
        """Grow weighted regions through vertex incidence.

        This deterministic heuristic spreads seeds through implicit edge
        adjacency and grows the least-loaded available region.  Seed distances
        use the sparse bipartite edge/vertex incidence graph, and growth uses
        vertex incidence lists. Neither materializes the line graph, which
        would require quadratic storage on hub graphs. It is
        deliberately lightweight; externally partitioned edge owners (for
        example from METIS) can be passed to the solver for higher-quality
        cuts.

        Parameters
        ----------
        network : Network or graph-like
            Network to partition.
        num_parts : int
            Number of parts.
        weights : array-like, shape (num_edges,), optional
            Nonnegative estimate of the work on each edge (default: uniform).

        Returns
        -------
        NetworkPartition
            Connectivity-aware edge assignment.
        """
        network = Network(network)
        num_parts = int(num_parts)
        if num_parts <= 0:
            raise ValueError("num_parts must be positive.")
        if weights is None:
            weights = np.ones(network.num_edges)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (network.num_edges,):
            raise ValueError("Partition weights must contain one value per edge.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Partition weights must be finite and nonnegative.")
        if not network.num_edges:
            return cls(network, np.empty(0, dtype=int), num_parts=num_parts)
        if num_parts == 1:
            return cls(
                network,
                np.zeros(network.num_edges, dtype=int),
                num_parts=1,
            )

        incident_edges = [[] for _ in range(network.num_vertices)]
        for edge, (tail, head) in enumerate(network.endpoints):
            incident_edges[int(tail)].append(edge)
            if head != tail:
                incident_edges[int(head)].append(edge)

        # Every edge-to-edge hop is two hops in this bipartite graph. The
        # compiled traversal preserves the original distances (including
        # disconnected components, loops and parallel edges) with O(E + V)
        # storage, rather than constructing a potentially dense line graph.
        edge_nodes = np.repeat(np.arange(network.num_edges), 2)
        vertex_nodes = network.num_edges + network.endpoints.ravel()
        node_count = network.num_edges + network.num_vertices
        incidence_graph = sparse.csr_matrix(
            (np.ones(2 * edge_nodes.size),
             (np.concatenate((edge_nodes, vertex_nodes)),
              np.concatenate((vertex_nodes, edge_nodes)))),
            shape=(node_count, node_count),
        )

        def edge_distances(seed, limit=np.inf):
            distances = sparse.csgraph.dijkstra(
                incidence_graph, indices=seed, directed=True, unweighted=True,
                limit=limit,
            )
            return distances[:network.num_edges] / 2

        def farthest_edge(distances, candidates):
            # candidates are in canonical order: argmax keeps the lowest
            # edge index when both distance and weight tie.
            values = distances[candidates]
            farthest = candidates[values == values.max()]
            return int(farthest[np.argmax(weights[farthest])])

        # Begin at an approximate diameter endpoint rather than at an arbitrary
        # high-weight edge.  On trees this avoids pairing a central seed with a
        # leaf, which can strand one region long before it reaches its target
        # load.  Subsequent farthest-point seeds also spread across disconnected
        # components before subdividing an already seeded component.
        num_seeds = min(num_parts, network.num_edges)
        anchor = int(np.argmax(weights))
        anchor_distances = edge_distances(anchor)
        connected_edges = np.flatnonzero(np.isfinite(anchor_distances))
        first_seed = farthest_edge(anchor_distances, connected_edges)
        seeds = [int(first_seed)]
        seeded = np.zeros(network.num_edges, dtype=bool)
        seeded[seeds[0]] = True
        minimum_distance = edge_distances(seeds[0])
        while len(seeds) < num_seeds:
            candidates = np.flatnonzero(~seeded)
            seed = farthest_edge(minimum_distance, candidates)
            seeds.append(seed)
            seeded[seed] = True
            # The selected seed has the largest current nearest-seed distance.
            # Beyond that radius, a new distance cannot improve any edge's
            # existing minimum. Incidence distances are twice edge distances.
            # An unseeded component has infinite radius and remains unbounded.
            minimum_distance = np.minimum(
                minimum_distance,
                edge_distances(seed, limit=2 * minimum_distance[seed]),
            )

        owners = np.full(network.num_edges, -1, dtype=np.int64)
        loads = np.zeros(num_parts, dtype=np.float64)
        counts = np.zeros(num_parts, dtype=np.int64)
        frontiers = [[] for _ in range(num_parts)]
        frontier_members = [set() for _ in range(num_parts)]
        expanded_vertices = [set() for _ in range(num_parts)]
        unowned = [(-weights[edge], edge) for edge in range(network.num_edges)]
        heapq.heapify(unowned)

        def add_frontier(part, edge):
            if owners[edge] < 0 and edge not in frontier_members[part]:
                frontier_members[part].add(edge)
                heapq.heappush(frontiers[part], (-weights[edge], edge))

        def expand_frontier(part, edge):
            for vertex in network.endpoints[edge]:
                vertex = int(vertex)
                if vertex in expanded_vertices[part]:
                    continue
                expanded_vertices[part].add(vertex)
                for adjacent in incident_edges[vertex]:
                    add_frontier(part, adjacent)

        def frontier_edge(part):
            frontier = frontiers[part]
            while frontier and owners[frontier[0][1]] >= 0:
                heapq.heappop(frontier)
            return None if not frontier else frontier[0][1]

        for part, edge in enumerate(seeds):
            owners[edge] = part
            loads[part] += weights[edge]
            counts[part] += 1
        for part, edge in enumerate(seeds):
            expand_frontier(part, edge)

        # A part appears exactly once, either in the active heap or in the
        # exhausted heap. Only the selected part changes load or gains frontier
        # edges, so its key can be updated after popping, without stale keys or
        # a scan over all parts. Other parts may lose their remaining frontier
        # to competing regions; discard those owned edges lazily when popped.
        active_parts = []
        exhausted_parts = []
        for part in range(num_parts):
            heap = active_parts if frontiers[part] else exhausted_parts
            heap.append((loads[part], counts[part], part))
        heapq.heapify(active_parts)
        heapq.heapify(exhausted_parts)

        remaining = network.num_edges - num_seeds
        while remaining:
            while active_parts:
                key = heapq.heappop(active_parts)
                part = key[2]
                if frontier_edge(part) is not None:
                    _, edge = heapq.heappop(frontiers[part])
                    break
                heapq.heappush(exhausted_parts, key)
            else:
                # No region can reach an unowned edge. All parts are now in
                # the exhausted heap. Attach a fresh disconnected component
                # to its minimum, which is also the global minimum.
                _, _, part = heapq.heappop(exhausted_parts)
                while owners[unowned[0][1]] >= 0:
                    heapq.heappop(unowned)
                _, edge = heapq.heappop(unowned)
            owners[edge] = part
            loads[part] += weights[edge]
            counts[part] += 1
            remaining -= 1
            expand_frontier(part, edge)
            heap = active_parts if frontiers[part] else exhausted_parts
            heapq.heappush(heap, (loads[part], counts[part], part))
        return cls(network, owners, num_parts=num_parts)

    def __repr__(self):
        return (
            f"NetworkPartition(num_parts={self.num_parts}, "
            f"separator_vertices={self.separator_vertices.size})"
        )

    def __reduce__(self):
        return (_rebuild_network_partition, (self.network, self.edge_owners, self.num_parts))


def _rebuild_network_partition(network, owners, num_parts):
    return NetworkPartition(network, owners, num_parts=num_parts)


class NetworkVertexExchangePlan:
    """Immutable owner/ghost exchange plan for one Network partition.

    The first data axis selected by ``axis`` must contain the local resident
    vertices in ``partition.resident_vertices[comm.rank]`` order.  All other
    axes are transported as an arbitrary per-vertex data block.  Communication
    uses typed owner/ghost messages and never constructs a global vertex
    vector.
    """

    def __init__(self, partition, comm):
        if not isinstance(partition, NetworkPartition):
            raise TypeError(
                "NetworkVertexExchangePlan requires a NetworkPartition."
            )
        if comm.size != partition.num_parts:
            raise ValueError(
                "Network vertex exchange communicator size must match the "
                "partition part count."
            )

        rank = comm.rank
        resident = partition.resident_vertices[rank]
        owned = partition.owned_vertices[rank]
        ghost = partition.ghost_vertices[rank]
        owned_positions = np.searchsorted(resident, owned)
        ghost_positions = np.searchsorted(resident, ghost)

        # Each rank advertises only its own ghosts to their owners. Constructing
        # a local exchange plan must not force every other partition's metadata.
        ghost_owners = partition.vertex_equation_owners[ghost]
        request_order = np.argsort(ghost_owners, kind='stable')
        request_counts = np.bincount(ghost_owners, minlength=comm.size).astype(np.int64)
        incoming_counts = np.empty_like(request_counts)
        comm.Alltoall(request_counts, incoming_counts)
        request_offsets = np.r_[0, np.cumsum(request_counts[:-1])]
        incoming_offsets = np.r_[0, np.cumsum(incoming_counts[:-1])]
        requested = np.empty(int(incoming_counts.sum()), dtype=np.int64)
        comm.Alltoallv(
            [ghost[request_order], request_counts, request_offsets, MPI.INT64_T],
            [requested, incoming_counts, incoming_offsets, MPI.INT64_T],
        )

        forward_send_positions = []
        forward_recv_positions = []
        forward_send_vertices = []
        forward_recv_vertices = []
        for peer in range(comm.size):
            begin, count = incoming_offsets[peer], incoming_counts[peer]
            send_vertices = requested[begin:begin+count]
            recv_vertices = ghost[
                partition.vertex_equation_owners[ghost] == peer
            ]
            forward_send_vertices.append(
                _readonly_array(send_vertices, np.int64, ndim=1)
            )
            forward_recv_vertices.append(
                _readonly_array(recv_vertices, np.int64, ndim=1)
            )
            forward_send_positions.append(
                _readonly_array(
                    np.searchsorted(resident, send_vertices),
                    np.int64,
                    ndim=1,
                )
            )
            forward_recv_positions.append(
                _readonly_array(
                    np.searchsorted(resident, recv_vertices),
                    np.int64,
                    ndim=1,
                )
            )

        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "comm", comm)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "owned", owned)
        object.__setattr__(self, "ghost", ghost)
        object.__setattr__(self, "resident", resident)
        object.__setattr__(
            self,
            "owned_positions",
            _readonly_array(owned_positions, np.int64, ndim=1),
        )
        object.__setattr__(
            self,
            "ghost_positions",
            _readonly_array(ghost_positions, np.int64, ndim=1),
        )
        object.__setattr__(
            self,
            "forward_send_vertices",
            tuple(forward_send_vertices),
        )
        object.__setattr__(
            self,
            "forward_recv_vertices",
            tuple(forward_recv_vertices),
        )
        object.__setattr__(
            self,
            "forward_send_positions",
            tuple(forward_send_positions),
        )
        object.__setattr__(
            self,
            "forward_recv_positions",
            tuple(forward_recv_positions),
        )
        # Additive reduction is exactly the reverse communication pattern.
        object.__setattr__(
            self,
            "reduce_send_positions",
            tuple(forward_recv_positions),
        )
        object.__setattr__(
            self,
            "reduce_recv_positions",
            tuple(forward_send_positions),
        )
        # Requests already give the forward counts and vertex offsets. Reuse
        # them for reverse reduction, and pack only nonempty peer segments.
        vertex_layout = tuple(_readonly_array(a, np.int64, ndim=1) for a in (
            incoming_counts, request_counts, incoming_offsets, request_offsets))
        object.__setattr__(self, "_vertex_layout", vertex_layout)
        object.__setattr__(self, "_send_vertices", int(incoming_counts.sum()))
        object.__setattr__(self, "_recv_vertices", int(request_counts.sum()))
        for name, positions, counts, offsets in (
            ("_send_segments", self.forward_send_positions, incoming_counts, incoming_offsets),
            ("_recv_segments", self.forward_recv_positions, request_counts, request_offsets),
        ):
            segments = tuple((positions[peer], int(offsets[peer]),
                              int(offsets[peer] + counts[peer]))
                             for peer in np.flatnonzero(counts))
            object.__setattr__(self, name, segments)
        # Cache only immutable counts/offsets, never mutable message buffers.
        # Bound retained layouts when applications use many payload sizes.
        object.__setattr__(self, "_payload_layouts", {1: vertex_layout})
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("NetworkVertexExchangePlan objects are immutable.")
        object.__setattr__(self, name, value)

    @staticmethod
    def _mpi_dtype(dtype):
        dtype = np.dtype(dtype)
        if not dtype.isnative:
            raise TypeError(
                "Network vertex exchange requires native-endian data."
            )
        mpi_dtype = MPI._typedict.get(dtype.char)
        if mpi_dtype is None:
            raise TypeError(
                f"Network vertex exchange does not support dtype {dtype}."
            )
        return mpi_dtype

    def _payload_layout(self, block_size):
        """MPI element counts depend on payload size, not dtype or axis shape."""
        layout = self._payload_layouts.get(block_size)
        if layout is None:
            layout = tuple(_readonly_array(a * block_size, np.int64, ndim=1)
                           for a in self._vertex_layout)
            if len(self._payload_layouts) >= 8:
                self._payload_layouts.pop(next(iter(self._payload_layouts)))
            self._payload_layouts[block_size] = layout
        return layout

    def _exchange(self, data, axis, *, add):
        data = np.asarray(data)
        if not isinstance(axis, (int, np.integer)):
            raise TypeError("Network vertex exchange axis must be an integer.")
        axis = int(axis)
        if axis < 0:
            axis += data.ndim
        if axis < 0 or axis >= data.ndim:
            raise ValueError(
                f"Network vertex exchange axis {axis} is outside data with "
                f"{data.ndim} dimensions."
            )
        view = np.moveaxis(data, axis, 0)
        if view.shape[0] != self.resident.size:
            raise ValueError(
                "Network vertex exchange axis must contain exactly the local "
                f"resident vertices ({self.resident.size}, got {view.shape[0]})."
            )

        block_shape = view.shape[1:]
        signature = (data.dtype.str, block_shape)
        changed = np.asarray(signature != getattr(self, '_validated_payload', None), dtype=np.int32)
        self.comm.Allreduce(MPI.IN_PLACE, changed, op=MPI.MAX)
        if changed:
            _validate_collective_signature(self.comm, signature,
                                           "Network vertex exchange payload validation")
            object.__setattr__(self, '_validated_payload', signature)
        block_size = int(np.prod(block_shape, dtype=np.int64))
        send_counts, recv_counts, send_displacements, recv_displacements = (
            self._payload_layout(block_size))
        send_segments, recv_segments = self._send_segments, self._recv_segments
        send_vertices, recv_vertices = self._send_vertices, self._recv_vertices
        if add:
            send_counts, recv_counts = recv_counts, send_counts
            send_displacements, recv_displacements = recv_displacements, send_displacements
            send_segments, recv_segments = recv_segments, send_segments
            send_vertices, recv_vertices = recv_vertices, send_vertices

        buffer_key = (data.dtype.str, block_size)
        cache = getattr(self, '_exchange_buffers', None)
        if cache is None:
            cache = {}
            object.__setattr__(self, '_exchange_buffers', cache)
        buffers = cache.get(add)
        if buffers is None or buffers[0] != buffer_key:
            buffers = (buffer_key,
                       np.empty(send_vertices * block_size, dtype=data.dtype),
                       np.empty(recv_vertices * block_size, dtype=data.dtype))
            cache[add] = buffers
        _, send_buffer, recv_buffer = buffers
        if block_size:
            for positions, begin, end in send_segments:
                send_buffer[begin * block_size:end * block_size] = (
                    np.ascontiguousarray(view[positions]).reshape(-1))

        mpi_dtype = self._mpi_dtype(data.dtype)
        # Typed collective transport keeps data on the owner/ghost routes
        # without consuming tags from a caller's communicator namespace.
        self.comm.Alltoallv(
            [send_buffer, send_counts, send_displacements, mpi_dtype],
            [recv_buffer, recv_counts, recv_displacements, mpi_dtype],
        )
        if block_size:
            for positions, begin, end in recv_segments:
                incoming = recv_buffer[begin * block_size:end * block_size].reshape(
                    (positions.size,) + block_shape)
                if add:
                    view[positions] += incoming
                else:
                    view[positions] = incoming
        return data

    def forward(self, data, *, axis=0):
        """Copy authoritative owner values into all resident ghost entries."""
        return self._exchange(
            data,
            axis,
            add=False,
        )

    def reduce_add(self, data, *, axis=0, refresh=True):
        """Add ghost contributions into owners and optionally refresh ghosts."""
        self._exchange(
            data,
            axis,
            add=True,
        )
        if refresh:
            self.forward(data, axis=axis)
        return data

    def __repr__(self):
        return (
            f"NetworkVertexExchangePlan(rank={self.rank}, "
            f"owned={self.owned.size}, ghost={self.ghost.size})"
        )


def _resolve_partition(network, partition, num_parts, weights):
    if partition is None:
        return NetworkPartition.connected(network, num_parts, weights)
    if isinstance(partition, str):
        if partition == "connected":
            return NetworkPartition.connected(network, num_parts, weights)
        if partition == "balanced_edges":
            return NetworkPartition.balanced(network, num_parts, weights)
        raise ValueError(
            "network_partition must be 'connected', 'balanced_edges', or "
            "an explicit edge-owner specification."
        )
    protocol = getattr(partition, "__dedalus_network_partition__", None)
    if protocol is not None:
        partition = protocol(network, num_parts)
    elif callable(partition):
        partition = partition(network, num_parts)
    if isinstance(partition, NetworkPartition):
        if partition.network is not network and (
            partition.network.vertex_ids != network.vertex_ids
            or partition.network.edge_ids != network.edge_ids
            or not np.array_equal(partition.network.endpoints, network.endpoints)
        ):
            raise ValueError("Network partition belongs to a different network.")
        if partition.num_parts > num_parts:
            raise ValueError("Network partition contains unavailable MPI ranks.")
        return NetworkPartition(
            network,
            partition.edge_owners,
            num_parts=num_parts,
        )
    if isinstance(partition, Mapping):
        try:
            partition = [partition[edge_id] for edge_id in network.edge_ids]
        except KeyError as error:
            raise ValueError(
                f"Network partition is missing edge {error.args[0]!r}."
            ) from error
    return NetworkPartition(network, partition, num_parts=num_parts)


def _normalize_edge_indices(edge_indices, num_edges):
    """Return a canonical sorted tuple of unique edge-table indices."""
    if edge_indices is None:
        return range(num_edges)
    try:
        values = tuple(edge_indices)
    except TypeError as error:
        raise TypeError("NetworkBasis edge_indices must be an iterable.") from error
    if not values:
        raise ValueError("NetworkBasis edge_indices must be nonempty.")
    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                "NetworkBasis edge_indices must not contain booleans."
            )
        if isinstance(value, (int, np.integer)):
            value = int(value)
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or not float(value).is_integer():
                raise ValueError(
                    "NetworkBasis edge_indices must contain finite "
                    "integer-valued entries."
                )
            value = int(value)
        else:
            raise TypeError(
                "NetworkBasis edge_indices must contain numeric canonical "
                "indices."
            )
        if value < 0 or value >= num_edges:
            raise ValueError(
                "NetworkBasis edge_indices contain an index outside the "
                "Network edge table."
            )
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("NetworkBasis edge_indices must be unique.")
    normalized.sort()
    if len(normalized) == num_edges and all(i == v for i, v in enumerate(normalized)):
        return range(num_edges)
    return tuple(normalized)


def _rebuild_network_basis(
    network,
    size,
    interval_basis,
    dealias,
    library,
    edge_indices,
):
    """Reconstruct a network basis bundle during unpickling."""
    return NetworkBasis(
        network,
        size,
        interval_basis=interval_basis,
        dealias=dealias,
        library=library,
        edge_indices=edge_indices,
    )


class _NetworkCardinalMap(CardinalMap):
    """Topology map assembled only on requested rows/columns.

    The public matrix remains available for serial use and explicit inspection.
    Distributed operators request local submatrices and never construct a
    global CSR merely to select a small part of it.
    """

    _network_topology_local = True
    matrix_dtype = np.dtype(np.float64)

    def __init__(self, source, target, network, edge_indices, kind, name,
                 transpose_of=None):
        self.source, self.target = source, target
        self.network, self.edge_indices = network, edge_indices
        self.kind, self.name = kind, name
        self.transpose_of = transpose_of

    def __repr__(self):
        return f"CardinalMap(name={self.name!r}, shape={(self.target.size, self.source.size)})"

    @CachedAttribute
    def matrix(self):
        matrix = self.submatrix(np.arange(self.target.size), np.arange(self.source.size))
        for array in (matrix.data, matrix.indices, matrix.indptr):
            array.flags.writeable = False
        return matrix

    def submatrix(self, rows, columns):
        rows, columns = np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)
        if self.transpose_of is not None:
            return self.transpose_of.submatrix(columns, rows).T.tocsr()
        positions = np.arange(rows.size)
        if self.kind == 'inverse_lengths':
            cols = rows
            values = 1 / self.network.lengths[self.edge_indices[rows]]
        elif self.kind == 'restriction':
            cols = self.network.endpoints[self.edge_indices[rows // 2], rows % 2]
            values = np.ones(rows.size)
        else:
            parity = rows % 2
            keep = np.ones(rows.size, dtype=bool)
            if self.kind.startswith('left'):
                keep = parity == 0
            elif self.kind.startswith('right'):
                keep = parity == 1
            positions = positions[keep]
            cols = rows[keep] // 2
            values = np.ones(cols.size)
            if self.kind.endswith('neumann'):
                values = (2*parity[keep]-1) / self.network.lengths[self.edge_indices[cols]]
        matrix = sparse.csr_matrix((values, (positions, cols)),
                                   shape=(rows.size, self.source.size))
        # This sparse column selector allocates by local nnz, never global width.
        unique, inverse = np.unique(columns, return_inverse=True)
        selected = _csr_local_columns(matrix, unique)
        return selected if np.array_equal(columns, unique) else selected[:, inverse]

    @CachedAttribute
    def T(self):
        if self.transpose_of is not None:
            return self.transpose_of
        return _NetworkCardinalMap(self.target, self.source, self.network,
                                   self.edge_indices, self.kind, self.name+'.T', self)

    @property
    def H(self):
        return self.T


class NetworkBasis(metaclass=CachedClass):
    r"""Bundle of edge, vertex, port, and interval bases for a network.

    Parameters
    ----------
    network : Network or graph-like
        Metric network defining topology, orientation, and edge lengths.
    size : int
        Spectral resolution on every edge.
    interval_basis : Basis class, optional
        Basis factory for the normalized interval :math:`s \in [0, 1]`
        (default: :class:`~dedalus.core.basis.Chebyshev`).
    dealias : float or 1-tuple of float, optional
        Interval dealiasing scale (default: 1).
    library : str, optional
        Transform library forwarded to ``interval_basis``.
    edge_indices : iterable of int, optional
        Canonical Network edge-table indices represented by this bucket.
        Omitting it selects every edge.

    Attributes
    ----------
    edges : CardinalBasis
        One discrete element per network edge, weighted by physical length.
    vertices : CardinalBasis
        One discrete element per network vertex.
    ports : CardinalBasis
        Two discrete elements per edge in canonical left/right order.
    interval : Basis
        Spectral basis on the normalized reference interval.
    restriction, R : CardinalMap
        Vertex-to-port restriction map.

    Notes
    -----
    Fields with ``bases=basis`` use ``(basis.edges, basis.interval)``.  A
    single bucket has one interval family and resolution; multiple disjoint
    buckets can share ``basis.vertices`` through :class:`NetworkBasisSet`.
    """

    @classmethod
    def _preprocess_cache_args(
        cls,
        network,
        size,
        interval_basis,
        dealias,
        library,
        edge_indices,
    ):
        """Normalize constructor options before caching the basis bundle."""
        if not isinstance(network, Network):
            network = Network(network)
        size = int(size)
        if size <= 0:
            raise ValueError("NetworkBasis size must be positive.")
        if network.num_edges == 0:
            raise ValueError("NetworkBasis requires at least one edge.")
        edge_indices = _normalize_edge_indices(
            edge_indices,
            network.num_edges,
        )

        if np.isscalar(dealias):
            dealias = (dealias,)
        else:
            dealias = tuple(dealias)
        if len(dealias) != 1:
            raise ValueError("NetworkBasis dealias must have length 1.")
        try:
            dealias = (float(dealias[0]),)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("NetworkBasis dealias must be numeric.") from error
        if not np.isfinite(dealias[0]) or dealias[0] <= 0:
            raise ValueError("NetworkBasis dealias must be positive and finite.")

        # Let the interval factory resolve its own canonical options.  The
        # built-in interval bases are CachedClasses, so this probe is reused by
        # __init__ and makes None equivalent to an explicitly selected default
        # library for Chebyshev, Legendre, and other supported factories.
        interval = interval_basis(
            network.position_coord,
            size=size,
            bounds=(0, 1),
            dealias=dealias,
            library=library,
        )
        dealias = tuple(interval.dealias)
        library = getattr(interval, "library", library)
        return (
            network,
            size,
            interval_basis,
            dealias,
            library,
            edge_indices,
        )

    def __init__(
        self,
        network,
        size,
        interval_basis=Chebyshev,
        dealias=1,
        library=None,
        edge_indices=None,
    ):
        full_bucket = isinstance(edge_indices, range) and edge_indices == range(network.num_edges)
        if full_bucket:
            edge_indices_array = CanonicalIndices(network.num_edges)
            port_indices_array = CanonicalIndices(2*network.num_edges)
        else:
            if getattr(network, '_local_input', False):
                raise NotImplementedError('Local graph input currently requires the full edge basis')
            edge_indices_array = _readonly_array(edge_indices, np.int64, ndim=1)
            port_indices_array = _readonly_array(
                np.column_stack((2*edge_indices_array, 2*edge_indices_array+1)).ravel(), np.int64)
        num_bucket_edges = edge_indices_array.size
        num_bucket_ports = port_indices_array.size
        lengths = network.lengths if full_bucket else network.lengths[edge_indices_array]
        edges = CardinalBasis(
            network.edge_coord,
            size=num_bucket_edges,
            weights=lengths,
        )
        vertices = network._vertices_basis
        ports = CardinalBasis(network.edge_coord, size=num_bucket_ports)
        interval = interval_basis(
            network.position_coord,
            size=size,
            bounds=(0, 1),
            dealias=dealias,
            library=library,
        )
        self.network = network
        self.size = size
        self.interval_basis = interval_basis
        self.interval_dealias = dealias
        self.library = library
        self.edge_indices = edge_indices_array
        self.port_indices = port_indices_array
        self.edges = edges
        self.vertices = vertices
        self.ports = ports
        self.interval = interval
        self.coordsys = network
        self.coords = network
        self.dim = 2
        self.shape = (num_bucket_edges, size)
        self.dealias = (1, interval.dealias[0])
        self.volume = edges.volume * interval.volume

        for role_basis, role, elements in (
            (edges, "edges", edge_indices_array),
            (ports, "ports", port_indices_array),
        ):
            role_basis._network = network
            role_basis._network_basis = self
            role_basis._network_role = role
            role_basis._network_elements = elements
        # The interval basis may itself be cached and shared by several
        # buckets, so it receives only direct Network metadata.
        interval._network = network
        interval._network_role = "position"

        def topology_map(source, target, kind, name):
            return _NetworkCardinalMap(source, target, network,
                                       edge_indices_array, kind, name)

        self.restriction = topology_map(vertices, ports, 'restriction', 'R')
        self.R = self.restriction
        self._left_trace = topology_map(edges, ports, 'left', 'dtrace_left')
        self._right_trace = topology_map(edges, ports, 'right', 'dtrace_right')
        self.edge_to_ports = topology_map(edges, ports, 'both', 'edge_to_ports')
        self._left_neumann = topology_map(edges, ports, 'left_neumann', 'ntrace_left')
        self._right_neumann = topology_map(edges, ports, 'right_neumann', 'ntrace_right')
        self._inverse_lengths = topology_map(edges, edges, 'inverse_lengths', 'inverse_edge_lengths')

    def __dedalus_bases__(self):
        """Return the ordinary bases used internally by a Field domain."""
        return (self.edges, self.interval)

    def local_grids(self, dist, scales):
        """Return canonical edge indices and the normalized reference grid."""
        logical_edge_grid = dist.local_grid(self.edges, scale=scales[0])
        edge_grid = self.edge_indices[logical_edge_grid.astype(np.int64)]
        reference_grid = dist.local_grid(self.interval, scale=scales[1])
        return edge_grid, reference_grid

    def global_grids(self, dist, scales):
        """Return global edge indices and the normalized reference grid."""
        logical_edge_grid = self.edges.global_grid(dist, scale=scales[0])
        edge_grid = self.edge_indices[logical_edge_grid.astype(np.int64)]
        reference_grid = self.interval.global_grid(dist, scale=scales[1])
        return edge_grid, reference_grid

    def local_physical_grid(self, dist, scale=1):
        """Return local physical arclength from each oriented edge tail."""
        edge_grid, reference_grid = self.local_grids(
            dist,
            scales=(1, scale),
        )
        edge_indices = edge_grid.astype(np.int64)
        return self.network.lengths[edge_indices] * reference_grid

    def global_physical_grid(self, dist, scale=1):
        """Return global physical arclength from each oriented edge tail."""
        edge_grid, reference_grid = self.global_grids(
            dist,
            scales=(1, scale),
        )
        edge_indices = edge_grid.astype(np.int64)
        return self.network.lengths[edge_indices] * reference_grid

    def physical_grid(self, dist, scale=1):
        """Compatibility alias for :meth:`local_physical_grid`."""
        return self.local_physical_grid(dist, scale=scale)

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self.__dedalus_bases__())

    def __getitem__(self, index):
        return self.__dedalus_bases__()[index]

    def __repr__(self):
        selected = self.edge_indices.size
        if selected == self.network.num_edges:
            return f"NetworkBasis(num_edges={selected}, size={self.size})"
        return (
            f"NetworkBasis(num_edges={selected}/{self.network.num_edges}, "
            f"size={self.size})"
        )

    def __reduce__(self):
        return (
            _rebuild_network_basis,
            (
                self.network,
                self.size,
                self.interval_basis,
                self.interval_dealias,
                self.library,
                tuple(self.edge_indices),
            ),
        )

    def derivative(self, operand):
        r"""Differentiate with respect to physical edge arclength.

        This applies :math:`L_e^{-1}\partial_s` on edge ``e``, where
        :math:`s \in [0, 1]` is the reference coordinate.

        Parameters
        ----------
        operand : Operand
            Dedalus edge field or deferred expression.

        Returns
        -------
        Operand
            Deferred physical derivative.
        """
        derivative = operators.Differentiate(operand, self.network.position_coord)
        if not isinstance(derivative, Operand):
            return derivative
        return self._inverse_lengths @ derivative

    dx = derivative

    def dtrace(self, operand):
        """Return endpoint values in canonical port order.

        Parameters
        ----------
        operand : Operand
            Dedalus field or expression on ``edges``, optionally with the
            interval basis.

        Returns
        -------
        Operand
            Port expression ordered ``(edge 0 left, edge 0 right, ...)``.
        """
        if operand.domain.get_basis(self.network.position_coord) is None:
            return self.edge_to_ports @ operand
        left = operators.Interpolate(
            operand,
            self.network.position_coord,
            "left",
        )
        right = operators.Interpolate(
            operand,
            self.network.position_coord,
            "right",
        )
        return (self._left_trace @ left) + (self._right_trace @ right)

    def ntrace(self, operand):
        r"""Return outward physical derivatives in canonical port order.

        The result is :math:`-\partial_x u` at each oriented edge tail and
        :math:`+\partial_x u` at each head.

        Parameters
        ----------
        operand : Operand
            Dedalus edge field or deferred expression.

        Returns
        -------
        Operand
            Outward physical derivative on ``ports``.
        """
        derivative = operators.Differentiate(
            operand,
            self.network.position_coord,
        )
        if not isinstance(derivative, Operand):
            return derivative
        left = operators.Interpolate(
            derivative,
            self.network.position_coord,
            "left",
        )
        right = operators.Interpolate(
            derivative,
            self.network.position_coord,
            "right",
        )
        return (self._left_neumann @ left) + (self._right_neumann @ right)


class NetworkBasisSet(Mapping):
    """Immutable, insertion-ordered exact cover of Network edge buckets.

    Parameters
    ----------
    network : Network or graph-like
        The common metric Network.
    buckets : mapping
        Insertion-ordered mapping from bucket names to either existing
        :class:`NetworkBasis` objects or keyword mappings accepted by
        ``NetworkBasis(network, **options)``.  Every configuration mapping
        must include ``edge_indices``.

    Notes
    -----
    A BasisSet is a collection, not a ragged Field basis.  Select an individual
    bucket for each edge Field and bind all bucket fields to ``basis_set.vertices``.
    """

    _dedalus_network_basis_set = True

    def __init__(self, network, buckets):
        network = Network(network)
        if not isinstance(buckets, Mapping):
            raise TypeError("NetworkBasisSet buckets must be a mapping.")
        if not buckets:
            raise ValueError("NetworkBasisSet requires at least one bucket.")

        items = []
        coverage = np.zeros(network.num_edges, dtype=np.int64)
        for name, specification in buckets.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "NetworkBasisSet bucket names must be nonempty strings."
                )
            if isinstance(specification, NetworkBasis):
                basis = specification
                if basis.network is not network:
                    raise ValueError(
                        f"NetworkBasisSet bucket {name!r} belongs to a "
                        "different Network."
                    )
            elif isinstance(specification, Mapping):
                options = dict(specification)
                if "network" in options:
                    raise TypeError(
                        "NetworkBasisSet bucket configurations must not "
                        "repeat the Network argument."
                    )
                if "edge_indices" not in options:
                    raise ValueError(
                        f"NetworkBasisSet bucket {name!r} must specify "
                        "edge_indices."
                    )
                basis = NetworkBasis(network, **options)
            else:
                raise TypeError(
                    "NetworkBasisSet bucket values must be NetworkBasis "
                    "objects or option mappings."
                )
            coverage[basis.edge_indices] += 1
            items.append((name, basis))

        overlapping = np.flatnonzero(coverage > 1)
        missing = np.flatnonzero(coverage == 0)
        if overlapping.size or missing.size:
            details = []
            if overlapping.size:
                details.append(
                    "overlapping canonical edges "
                    f"{overlapping.tolist()}"
                )
            if missing.size:
                details.append(
                    f"missing canonical edges {missing.tolist()}"
                )
            raise ValueError(
                "NetworkBasisSet buckets must form an exact edge cover; "
                + "; ".join(details)
                + "."
            )

        mapping = dict(items)
        edge_costs = np.empty(network.num_edges, dtype=np.float64)
        for basis in mapping.values():
            edge_costs[basis.edge_indices] = basis.size**3
            if basis.vertices is not network._vertices_basis:
                raise RuntimeError(
                    "NetworkBasis buckets did not reuse the shared vertex basis."
                )
        edge_costs.flags.writeable = False

        object.__setattr__(self, "network", network)
        object.__setattr__(self, "_mapping", MappingProxyType(mapping))
        object.__setattr__(self, "buckets", self._mapping)
        object.__setattr__(self, "_items", tuple(items))
        object.__setattr__(self, "vertices", network._vertices_basis)
        object.__setattr__(self, "edge_costs", edge_costs)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("NetworkBasisSet objects are immutable.")
        object.__setattr__(self, name, value)

    def __getitem__(self, name):
        return self._mapping[name]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def __repr__(self):
        return (
            f"NetworkBasisSet(num_buckets={len(self)}, "
            f"num_edges={self.network.num_edges})"
        )

    def __reduce__(self):
        return (type(self), (self.network, dict(self._items)))


def _network_element_labels(operand, subproblem):
    """Label flattened field coefficients by their owning network edge."""
    shape = subproblem.field_shape(operand)
    tensor_rank = len(operand.tensorsig)
    for basis in operand.domain.bases:
        role = getattr(basis, "_network_role", None)
        if role not in ("edges", "ports", "vertices"):
            continue
        if role == "vertices":
            return -np.ones(np.prod(shape, dtype=int), dtype=int)
        axis = operand.dist.get_basis_axis(basis)
        axis_slice = subproblem.coeff_slices(operand.domain)[axis]
        local_elements = operand.dist.coeff_layout.local_elements(
            operand.domain,
            scales=1,
        )[axis]
        elements = local_elements[axis_slice]
        elements = take_elements(basis._network_elements, elements)
        if role == "ports":
            elements = elements // 2
        label_shape = [1] * len(shape)
        label_shape[tensor_rank + axis] = elements.size
        labels = elements.reshape(label_shape)
        return np.broadcast_to(labels, shape).ravel()
    return -np.ones(np.prod(shape, dtype=int), dtype=int)


def _network_vertex_labels(operand, subproblem):
    """Label flattened vertex-field coefficients by network vertex."""
    shape = subproblem.field_shape(operand)
    tensor_rank = len(operand.tensorsig)
    for basis in operand.domain.bases:
        if getattr(basis, "_network_role", None) != "vertices":
            continue
        axis = operand.dist.get_basis_axis(basis)
        axis_slice = subproblem.coeff_slices(operand.domain)[axis]
        local_elements = operand.dist.coeff_layout.local_elements(
            operand.domain,
            scales=1,
        )[axis]
        vertices = take_elements(basis._network_elements, local_elements[axis_slice])
        label_shape = [1] * len(shape)
        label_shape[tensor_rank + axis] = vertices.size
        labels = vertices.reshape(label_shape)
        return np.broadcast_to(labels, shape).ravel()
    return -np.ones(np.prod(shape, dtype=int), dtype=int)


def _is_global_scalar(operand):
    """Return whether an operand is a basis-free scalar degree of freedom."""
    return not operand.tensorsig and not operand.domain.bases


def _is_network_interface_operand(operand):
    """Return whether an operand belongs in the condensed interface block."""
    return _has_network_role(operand, "vertices") or _is_global_scalar(operand)


def _network_interface_labels(operands, subproblem, num_vertices):
    """Label vertex and global-scalar interface degrees of freedom.

    Physical vertices retain their canonical labels.  Basis-free scalar
    degrees of freedom receive synthetic labels after the physical vertices;
    the subgraph solver consequently keeps them on the global separator.
    """
    labels = []
    next_global_label = num_vertices
    for operand in operands:
        operand_labels = _network_vertex_labels(operand, subproblem)
        if _is_global_scalar(operand):
            size = operand_labels.size
            operand_labels = np.arange(
                next_global_label,
                next_global_label + size,
                dtype=int,
            )
            next_global_label += size
        labels.append(operand_labels)
    return np.concatenate(labels)


def _network_interface_keys(labels):
    """Pair every interface label with its deterministic occurrence index.

    Several independent vertex fields can contribute one degree of freedom at
    the same physical vertex.  Labels alone therefore do not identify a
    compressed interface entry.  The occurrence is counted in expanded field
    order, before Dedalus's one-to-one preconditioner permutation is applied.
    Non-interface entries retain the sentinel key ``(-1, -1)``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("Network interface labels must be one-dimensional.")
    keys = np.full((labels.size, 2), -1, dtype=np.int64)
    occurrences = {}
    for index, label in enumerate(labels):
        label = int(label)
        if label < 0:
            continue
        occurrence = occurrences.get(label, 0)
        occurrences[label] = occurrence + 1
        keys[index] = (label, occurrence)
    return keys


def _network_interface_owners(labels, partition, *, condition, root=0):
    """Map physical vertices and synthetic global labels to MPI owners."""
    labels = np.asarray(labels, dtype=int)
    owners = np.full(labels.shape, root if condition else -1, dtype=int)
    physical = (labels >= 0) & (labels < partition.network.num_vertices)
    source = (
        partition.vertex_equation_owners
        if condition
        else partition.interior_vertex_owners
    )
    owners[physical] = source[labels[physical]]
    return owners


def _interface_occurrence_error(sorted_keys, counts):
    """Validate lexicographically sorted keys in one grouped pass."""
    if not sorted_keys.shape[0]:
        return None
    labels = sorted_keys[:, 0]
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    lengths = np.diff(np.append(starts, labels.size))
    expected_occurrences = np.arange(labels.size) - np.repeat(starts, lengths)
    bad_occurrences = np.logical_or.reduceat(
        sorted_keys[:, 1] != expected_occurrences, starts,
    )
    bad_groups = np.flatnonzero(
        bad_occurrences | (lengths != counts[labels[starts]])
    )
    if bad_groups.size:
        group = bad_groups[0]
        start = starts[group]
        label = int(labels[start])
        local = sorted_keys[start : start + lengths[group], 1]
        expected = np.arange(counts[label], dtype=np.int64)
        return (
            f"label {label} has occurrences "
            f"{local.tolist()}, expected {expected.tolist()}"
        )
    return None


class _SparseLabelCounts:
    """Counts indexed by canonical labels without a global-width array."""

    def __init__(self, labels, values):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.int64)

    def __getitem__(self, labels):
        labels = np.asarray(labels, dtype=np.int64)
        positions = np.searchsorted(self.labels, labels)
        # All consumers query labels included in the distributed reduction.
        if np.any(positions >= self.labels.size) or np.any(self.labels[positions] != labels):
            raise KeyError("Count requested for an unregistered interface label")
        return self.values[positions]

    def get(self, labels):
        labels = np.asarray(labels, dtype=np.int64)
        positions = np.searchsorted(self.labels, labels)
        found = positions < self.labels.size
        found[found] = self.labels[positions[found]] == labels[found]
        result = np.zeros(labels.shape, dtype=np.int64)
        result[found] = self.values[positions[found]]
        return result


def _matching_interface_counts(row, column, comm):
    """Compare global counts on the union of locally relevant labels."""
    labels = np.union1d(row.labels, column.labels)
    counts = _distributed_label_max(comm, labels,
                                    np.column_stack((row.get(labels), column.get(labels))))
    rows, columns = counts.T
    return np.array_equal(rows, columns), _SparseLabelCounts(labels, rows)


def _distributed_label_max(comm, labels, values, *, operation='max'):
    """Reduce locally requested labels through one sparse rendezvous.

    ``values`` may have one or several columns. A tuple of operations selects
    sum/max per column, allowing related checks to share routing and messages.
    Labels and nonnegative count values are validated by their callers.
    """
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    scalar = values.ndim == 1
    columns = values[:, None] if scalar else values
    if columns.ndim != 2 or columns.shape[0] != labels.size or not columns.shape[1]:
        raise ValueError('Sparse label values must have one row per label')
    nvalues = columns.shape[1]
    operations = (operation,)*nvalues if isinstance(operation, str) else tuple(operation)
    if len(operations) != nvalues or any(op not in ('sum', 'max') for op in operations):
        raise ValueError('Require one sum/max operation per value column')
    if comm.size >= 128:
        # Coalesce tiny cross-node requests before the rendezvous. Random
        # vertex numbering otherwise sends a few words to almost every rank.
        node = comm.Split_type(MPI.COMM_TYPE_SHARED)
        try:
            result = _grouped_label_reduce(comm, node, labels, columns, operations)
        finally:
            node.Free()
    else:
        result = _flat_label_reduce(comm, labels, columns, operations)
    return result[:, 0] if scalar else result


def _grouped_label_reduce(comm, group, labels, columns, operations):
    """Gather requests within groups; exchange only between group leaders."""
    counts = np.asarray(group.allgather(labels.size), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    nvalues = columns.shape[1]
    requests = np.column_stack((labels, columns))
    gathered = np.empty((int(counts.sum()), nvalues+1), dtype=np.int64) if group.rank == 0 else None
    group.Gatherv(requests, [gathered, counts*(nvalues+1), offsets*(nvalues+1), MPI.INT64_T]
                  if group.rank == 0 else None, root=0)
    leaders = comm.Split(0 if group.rank == 0 else MPI.UNDEFINED, key=comm.rank)
    try:
        replies = (_flat_label_reduce(leaders, gathered[:, 0], gathered[:, 1:], operations)
                   if group.rank == 0 else None)
        result = np.empty_like(columns)
        group.Scatterv([replies, counts*nvalues, offsets*nvalues, MPI.INT64_T]
                       if group.rank == 0 else None, result, root=0)
    finally:
        if leaders != MPI.COMM_NULL:
            leaders.Free()
    return result


def _flat_label_reduce(comm, labels, columns, operations):
    """Sparse rendezvous on a communicator of ranks or group leaders."""
    nvalues = columns.shape[1]
    extent = int(comm.allreduce(int(labels.max(initial=-1)), op=MPI.MAX)) + 1
    width = max(1, (extent + comm.size - 1) // comm.size)
    peers = np.minimum(labels // width, comm.size - 1)
    order = np.argsort(peers, kind='stable')
    send_counts = np.bincount(peers, minlength=comm.size).astype(np.int64)
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    send_offsets = np.r_[0, np.cumsum(send_counts[:-1])]
    recv_offsets = np.r_[0, np.cumsum(recv_counts[:-1])]
    requests = np.column_stack((labels[order], columns[order]))
    record_size = nvalues + 1
    received = np.empty((int(recv_counts.sum()), record_size), dtype=np.int64)
    comm.Alltoallv(
        [requests, record_size*send_counts, record_size*send_offsets, MPI.INT64_T],
        [received, record_size*recv_counts, record_size*recv_offsets, MPI.INT64_T],
    )
    unique, inverse = np.unique(received[:, 0], return_inverse=True)
    reduced = np.zeros((unique.size, nvalues), dtype=np.int64)
    for column, op in enumerate(operations):
        reducer = np.maximum if op == 'max' else np.add
        reducer.at(reduced[:, column], inverse, received[:, column+1])
    replies = reduced[inverse]
    returned = np.empty((labels.size, nvalues), dtype=np.int64)
    comm.Alltoallv(
        [replies, nvalues*recv_counts, nvalues*recv_offsets, MPI.INT64_T],
        [returned, nvalues*send_counts, nvalues*send_offsets, MPI.INT64_T],
    )
    result = np.empty_like(returned)
    result[order] = returned
    return result


class _ResidentInterfaceDofs:
    """Canonical owner/ghost transport for compressed interface entries.

    Physical entries are packed as ``(resident vertex, occurrence, rhs)`` and
    exchanged through :class:`NetworkVertexExchangePlan`.  Synthetic labels
    representing basis-free global scalars are present on every rank and use
    ``root`` as their deterministic owner.
    """

    def __init__(self, keys, partition, comm, *, root=0, name="interface", exchange=None):
        self.partition = partition
        self.comm = comm
        self.root = int(root)
        self.name = str(name)
        self.exchange = (NetworkVertexExchangePlan(partition, comm)
                         if exchange is None else exchange)
        self.num_vertices = partition.network.num_vertices

        keys = np.asarray(keys, dtype=np.int64)
        validation_error = None
        try:
            if (self.exchange.partition is not partition
                    or MPI.Comm.Compare(self.exchange.comm, comm) != MPI.IDENT):
                raise ValueError('Interface exchange must use the same partition and communicator')
            if keys.ndim != 2 or keys.shape[1] != 2:
                raise ValueError(
                    "Resident interface keys must have shape (n, 2)."
                )
            if np.any(keys < 0):
                raise ValueError(
                    "Resident interface entries require nonnegative keys."
                )
            # Reuse this ordering for both duplicate and occurrence checks.
            order = np.lexsort((keys[:, 1], keys[:, 0]))
            sorted_keys = keys[order]
            if np.any(np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)):
                raise ValueError("Resident interface keys must be unique locally.")
        except Exception as error:
            validation_error = f"{type(error).__name__}: {error}"
        _raise_collective_error(
            comm,
            validation_error,
            f"Resident Network {self.name} key validation",
        )
        self.keys = keys
        self.labels = keys[:, 0]
        self.occurrences = keys[:, 1]

        local_max_label = self.labels.max(initial=-1)
        global_max_label = comm.allreduce(local_max_label, op=MPI.MAX)
        self.num_labels = max(self.num_vertices, int(global_max_label) + 1)
        local_labels, local_counts = np.unique(self.labels, return_counts=True)
        # Only basis-free scalar labels are replicated. Physical vertex labels
        # are checked by sparse request/reply, including differing peer counts.
        synthetic_labels = np.unique(np.concatenate(comm.allgather(
            local_labels[local_labels >= self.num_vertices]
        )))
        query_labels = np.union1d(local_labels, synthetic_labels)
        query_counts = np.zeros(query_labels.size, dtype=np.int64)
        query_counts[np.searchsorted(query_labels, local_labels)] = local_counts
        counts = _distributed_label_max(comm, query_labels, query_counts)
        self.counts = _SparseLabelCounts(query_labels, counts)

        occurrence_error = _interface_occurrence_error(sorted_keys, self.counts)
        _raise_collective_error(
            comm,
            occurrence_error,
            f"Resident Network {self.name} occurrence validation",
        )

        physical = self.labels < self.num_vertices
        physical_positions = np.flatnonzero(physical)
        physical_labels = self.labels[physical_positions]
        resident = self.exchange.resident
        resident_positions = np.searchsorted(resident, physical_labels)
        resident_error = None
        if physical_labels.size and (
            np.any(resident_positions >= resident.size)
            or not np.array_equal(
                resident[resident_positions],
                physical_labels,
            )
        ):
            resident_error = (
                "a local physical interface key is outside this rank's "
                "resident vertex set"
            )
        _raise_collective_error(
            comm,
            resident_error,
            f"Resident Network {self.name} storage validation",
        )
        self.physical_positions = physical_positions
        self.physical_resident_positions = resident_positions
        self.physical_occurrences = self.occurrences[physical_positions]
        self.physical_slots = int(comm.allreduce(
            counts[query_labels < self.num_vertices].max(initial=0), op=MPI.MAX,
        ))

        synthetic_positions = np.flatnonzero(~physical)
        synthetic_keys = keys[synthetic_positions]
        if synthetic_keys.size:
            synthetic_order = np.lexsort(
                (synthetic_keys[:, 1], synthetic_keys[:, 0])
            )
            synthetic_positions = synthetic_positions[synthetic_order]
            synthetic_keys = synthetic_keys[synthetic_order]
        expected_synthetic = []
        for label in synthetic_labels:
            expected_synthetic.extend(
                (label, occurrence)
                for occurrence in range(int(self.counts[label]))
            )
        expected_synthetic = np.asarray(
            expected_synthetic,
            dtype=np.int64,
        ).reshape((-1, 2))
        synthetic_error = None
        if not np.array_equal(synthetic_keys, expected_synthetic):
            synthetic_error = (
                "basis-free global scalar interface keys are not present "
                "identically on every rank"
            )
        _raise_collective_error(
            comm,
            synthetic_error,
            f"Resident Network {self.name} scalar validation",
        )
        self.synthetic_positions = synthetic_positions
        self.synthetic_keys = synthetic_keys

        owners = np.full(keys.shape[0], self.root, dtype=np.int64)
        owners[physical] = partition.vertex_equation_owners[
            self.labels[physical]
        ]
        self.owners = owners
        self.owned = owners == comm.rank

    def _view(self, values):
        values = np.asarray(values)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.keys.shape[0]:
            raise ValueError(
                f"Resident Network {self.name} data must have shape "
                f"({self.keys.shape[0]}, nrhs)."
            )
        return values, squeeze

    def _physical_buffer(self, values):
        buffer = np.zeros(
            (
                self.exchange.resident.size,
                self.physical_slots,
                values.shape[1],
            ),
            dtype=values.dtype,
        )
        if self.physical_positions.size:
            buffer[
                self.physical_resident_positions,
                self.physical_occurrences,
            ] = values[self.physical_positions]
        # Vertex fields and packed interface slots use the same payload rank.
        # This avoids alternating equal-width signatures on their shared plan.
        return buffer.reshape(self.exchange.resident.size,
                              self.physical_slots * values.shape[1])

    def _unpack_physical(self, buffer, values):
        buffer = buffer.reshape(self.exchange.resident.size,
                                self.physical_slots, values.shape[1])
        if self.physical_positions.size:
            values[self.physical_positions] = buffer[
                self.physical_resident_positions,
                self.physical_occurrences,
            ]

    def forward(self, values):
        """Refresh local ghost and replicated-scalar entries in place."""
        values, _ = self._view(values)
        physical = self._physical_buffer(values)
        self.exchange.forward(physical, axis=0)
        self._unpack_physical(physical, values)
        if self.synthetic_positions.size:
            synthetic = np.ascontiguousarray(values[self.synthetic_positions])
            self.comm.Bcast(synthetic, root=self.root)
            values[self.synthetic_positions] = synthetic
        return values

    def reduce_add(self, values, *, refresh=True):
        """Sum duplicate local contributions and optionally refresh ghosts."""
        values, _ = self._view(values)
        physical = self._physical_buffer(values)
        self.exchange.reduce_add(physical, axis=0, refresh=refresh)
        self._unpack_physical(physical, values)
        if self.synthetic_positions.size:
            synthetic = np.ascontiguousarray(values[self.synthetic_positions])
            assembled = np.empty_like(synthetic)
            self.comm.Allreduce(synthetic, assembled, op=MPI.SUM)
            values[self.synthetic_positions] = assembled
        return values



def assemble_distributed_network_matrix(
    replicated,
    additive,
    solver,
    subproblem,
):
    """Assign replicated vertex rows once and retain rank-local contributions.

    Symbolic matrix assembly separates every linear path into a pure replicated
    vertex contribution ``D`` and a rank-additive contribution ``E_r``.  This
    routine forms ``P_r D + E_r``, where ``P_r`` owns each vertex-equation row
    on exactly one rank.  The resulting ordinary sparse matrices can be freely
    combined later by implicit timesteppers without losing that ownership.
    """
    comm = solver.dist.network_comm
    partition = solver.dist.network_partition
    equations = solver.problem.equations
    variables = solver.problem.LHS_variables
    replicated = sparse.csr_matrix(replicated)
    additive = sparse.csr_matrix(additive)
    row_vertices = _network_interface_labels(
        [equation["eqn"] for equation in equations],
        subproblem,
        partition.network.num_vertices,
    )
    column_vertices = _network_interface_labels(
        variables,
        subproblem,
        partition.network.num_vertices,
    )

    validation_error = None
    if replicated.shape != additive.shape:
        validation_error = (
            "Replicated and additive matrices have different shapes."
        )
    elif replicated.shape != (row_vertices.size, column_vertices.size):
        validation_error = (
            "Network ownership labels do not match the matrix shape."
        )
    else:
        entries = replicated.tocoo()
        invalid = (
            ((row_vertices[entries.row] < 0) | (column_vertices[entries.col] < 0))
            & (np.abs(entries.data) > solver.entry_cutoff)
        )
        if np.any(invalid):
            validation_error = (
                "A replicated matrix path reaches an edge or port degree of "
                "freedom; its Network ownership provenance is invalid."
            )
    _raise_collective_error(
        comm,
        validation_error,
        "Distributed Network matrix-provenance validation",
    )

    interface_rows = np.flatnonzero(row_vertices >= 0)
    interface_columns = np.flatnonzero(column_vertices >= 0)
    if solver.dist.network_vertex_storage != "owned_ghost":
        # Replicated vertex storage constructs identical interface blocks on
        # every rank.  Resident storage deliberately has different local
        # shapes, so equality is instead enforced through canonical DOF keys
        # when its condensation layout is built.
        replicated_interface = replicated[
            interface_rows
        ][:, interface_columns].tocsr()
        _validate_collective_signature(
            comm,
            _sparse_signature(replicated_interface),
            "Distributed Network replicated vertex matrix validation",
        )

    owned_rows = np.zeros(row_vertices.size, dtype=bool)
    row_owners = _network_interface_owners(
        row_vertices[interface_rows],
        partition,
        condition=True,
        root=0,
    )
    owned_rows[interface_rows] = row_owners == comm.rank
    owned_replicated = sparse.diags(
        owned_rows.astype(replicated.dtype)
    ) @ replicated
    return (owned_replicated + additive).tocsr()


class _DistributedNetworkMatrixLayout:
    """Rank-local edge and replicated-interface indices for one subproblem."""

    def __init__(self, solver, subproblem):
        problem = solver.problem
        variables = problem.LHS_variables
        equations = problem.equations
        interface_fields = [
            variable
            for variable in variables
            if _is_network_interface_operand(variable)
        ]
        if not interface_fields:
            raise ValueError(
                "Distributed Network problems require an explicit vertex field."
            )
        full_column_mask = np.concatenate(
            [
                np.full(
                    subproblem.field_size(variable),
                    variable in interface_fields,
                    dtype=bool,
                )
                for variable in variables
            ]
        )
        full_row_mask = np.concatenate(
            [
                np.full(
                    subproblem.field_size(equation["eqn"]),
                    _is_network_interface_operand(equation["eqn"]),
                    dtype=bool,
                )
                for equation in equations
            ]
        )
        self.interface_columns = np.flatnonzero(
            _select_preconditioned(
                subproblem.pre_right_pinv,
                full_column_mask,
                "right preconditioner",
            )
        )
        self.interface_rows = np.flatnonzero(
            _select_preconditioned(
                subproblem.pre_left,
                full_row_mask,
                "left preconditioner",
            )
        )
        self.edge_rows = np.setdiff1d(
            np.arange(subproblem.pre_left.shape[0]),
            self.interface_rows,
            assume_unique=True,
        )
        self.edge_columns = np.setdiff1d(
            np.arange(subproblem.pre_right_pinv.shape[0]),
            self.interface_columns,
            assume_unique=True,
        )
        full_column_vertices = _network_interface_labels(
            variables,
            subproblem,
            problem.dist.network.num_vertices,
        )
        full_row_vertices = _network_interface_labels(
            [equation["eqn"] for equation in equations],
            subproblem,
            problem.dist.network.num_vertices,
        )
        minimal_column_vertices = _select_preconditioned(
            subproblem.pre_right_pinv,
            full_column_vertices,
            "right preconditioner",
        )
        minimal_row_vertices = _select_preconditioned(
            subproblem.pre_left,
            full_row_vertices,
            "left preconditioner",
        )
        self.interface_column_vertices = minimal_column_vertices[
            self.interface_columns
        ]
        self.interface_row_vertices = minimal_row_vertices[self.interface_rows]
        if (
            np.any(self.interface_row_vertices < 0)
            or np.any(self.interface_column_vertices < 0)
        ):
            raise ValueError(
                "Distributed Network interface equations and unknowns must "
                "be labelled by their vertices or global scalars."
            )

        self.resident = (
            problem.dist.network_vertex_storage == "owned_ghost"
        )
        if self.resident:
            full_column_keys = _network_interface_keys(
                full_column_vertices
            )
            full_row_keys = _network_interface_keys(full_row_vertices)
            minimal_column_keys = _select_preconditioned(
                subproblem.pre_right_pinv,
                full_column_keys,
                "right preconditioner",
            )
            minimal_row_keys = _select_preconditioned(
                subproblem.pre_left,
                full_row_keys,
                "left preconditioner",
            )
            self.interface_column_keys = minimal_column_keys[
                self.interface_columns
            ]
            self.interface_row_keys = minimal_row_keys[self.interface_rows]
            partition = problem.dist.network_partition
            comm = problem.dist.network_comm
            self.column_dofs = _ResidentInterfaceDofs(
                self.interface_column_keys,
                partition,
                comm,
                name="matrix-column interface",
                exchange=problem.dist.network_vertex_exchange_plan,
            )
            self.row_dofs = _ResidentInterfaceDofs(
                self.interface_row_keys,
                partition,
                comm,
                name="matrix-row interface",
                exchange=problem.dist.network_vertex_exchange_plan,
            )
            count_error = None
            counts_match, _ = _matching_interface_counts(
                self.row_dofs.counts, self.column_dofs.counts, comm)
            if not counts_match:
                count_error = (
                    "each physical vertex or global scalar must have the "
                    "same number of interface equations and unknowns"
                )
            _raise_collective_error(
                comm,
                count_error,
                "Resident Network matrix interface-count validation",
            )
            self.interface_column_owned = self.column_dofs.owned
            self.interface_row_owned = self.row_dofs.owned
        else:
            self.interface_column_keys = None
            self.interface_row_keys = None
            self.column_dofs = None
            self.row_dofs = None
            rank_is_root = problem.dist.network_comm.rank == 0
            self.interface_column_owned = np.full(
                self.interface_columns.size,
                rank_is_root,
                dtype=bool,
            )
            self.interface_row_owned = np.full(
                self.interface_rows.size,
                rank_is_root,
                dtype=bool,
            )


def _distributed_network_matrix_layout(solver, subproblem):
    layout = getattr(subproblem, "_distributed_network_matrix_layout", None)
    if layout is None:
        layout = _DistributedNetworkMatrixLayout(solver, subproblem)
        subproblem._distributed_network_matrix_layout = layout
    return layout


def apply_distributed_network_subproblem_matrix(
    matrix,
    array,
    solver,
    subproblem,
    *,
    axis=0,
    out=None,
):
    """Apply a local matrix and assemble its replicated vertex-row output."""
    comm = solver.dist.network_comm
    input_error = None
    try:
        if axis != 0:
            raise NotImplementedError(
                "Distributed Network subproblem matrices currently act on "
                "axis zero."
            )
        layout = _distributed_network_matrix_layout(solver, subproblem)
        local_matrix = sparse.csr_matrix(matrix)
        values = np.asarray(array)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != local_matrix.shape[1]:
            raise ValueError(
                "Distributed Network matrix input does not match the local "
                "matrix."
            )
        result_shape = (local_matrix.shape[0], values.shape[1])
        output_shape = (local_matrix.shape[0],) if squeeze else result_shape
        if out is not None:
            out_array = np.asarray(out)
            empty_placeholder = (
                local_matrix.shape[0] == 0 and out_array.size == 0
            )
            if out_array.shape != output_shape and not empty_placeholder:
                raise ValueError(
                    "Distributed Network matrix output does not match the "
                    f"local matrix: got {out_array.shape}, expected "
                    f"{output_shape}."
                )
    except Exception as error:
        input_error = f"{type(error).__name__}: {error}"
    validation = getattr(solver, "_network_matrix_metadata", None)
    if validation is None:
        validation = solver._network_matrix_metadata = CollectiveMetadata()
    signature = None
    if input_error is None:
        signature = (values.shape[1], values.dtype.str, bool(layout.resident))
        if not layout.resident:
            signature += (layout.interface_rows.size, layout.interface_columns.size,
                          _array_signature(layout.interface_row_vertices))
    validation.check(comm, signature,
                     "Distributed Network matrix-application validation",
                     error=input_error)

    interface_input = np.ascontiguousarray(values[layout.interface_columns])
    if layout.resident:
        layout.column_dofs.forward(interface_input)
    else:
        comm.Bcast(interface_input, root=0)
    # Local edge values remain resident. Refresh only the interface part of a
    # private input, then apply the original assembled sparse operator once.
    # No numerical matrix blocks are cached, so changed entries are observed.
    synchronized = values.copy()
    synchronized[layout.interface_columns] = interface_input
    result = np.asarray(local_matrix @ synchronized)
    interface_output = np.ascontiguousarray(result[layout.interface_rows])
    if layout.resident:
        layout.row_dofs.reduce_add(interface_output)
    else:
        comm.Allreduce(MPI.IN_PLACE, interface_output, op=MPI.SUM)
    result[layout.interface_rows] = interface_output
    if squeeze:
        result = result[:, 0]
    if out is None:
        return result
    np.copyto(out, result)
    return out


def _network_from_operands(operands):
    networks = []
    for operand in operands:
        for basis in operand.domain.bases:
            network_basis = getattr(basis, "_network_basis", None)
            basis_network = getattr(basis, "_network", None)
            if basis_network is None and network_basis is not None:
                basis_network = network_basis.network
            if basis_network is not None and all(
                basis_network is not network for network in networks
            ):
                networks.append(basis_network)
    if len(networks) != 1:
        raise ValueError(
            "DtN condensation requires exactly one Network across the problem."
        )
    return networks[0]


def _domain_has_network_role(domain, role):
    return any(
        getattr(basis, "_network_role", None) == role
        for basis in domain.bases
    )


def _select_preconditioned(transform, values, name):
    """Apply a Dedalus selection/permutation without averaging metadata."""
    transform = sparse.csr_matrix(transform)
    entries_per_row = np.diff(transform.indptr)
    if np.any(entries_per_row != 1) or not np.allclose(
        np.abs(transform.data),
        1,
    ):
        raise ValueError(
            f"DtN condensation requires {name} to select and permute "
            "degrees of freedom one-to-one."
        )
    return np.asarray(values)[transform.indices]


def _rhs_matrix(values, rows, columns):
    """Normalize matsolver output to a two-dimensional RHS matrix."""
    values = np.asarray(values)
    if values.ndim == 1 and columns == 1 and values.size == rows:
        values = values[:, None]
    if values.shape != (rows, columns):
        raise ValueError(
            "Matrix solver returned shape "
            f"{values.shape}; expected {(rows, columns)}."
        )
    return values


def _factorization_pivot_ratio(factorization):
    """Return a cheap singularity indicator for exposed sparse LU factors."""
    lu = getattr(factorization, "LU", None)
    upper = getattr(lu, "U", None)
    if upper is None:
        return None
    pivots = np.abs(upper.diagonal())
    if not pivots.size:
        return 1.0
    largest = pivots.max(initial=0)
    if largest == 0:
        return 0.0
    return float(pivots.min(initial=largest) / largest)


def _validate_independent_edge_blocks(
    matrix,
    row_labels,
    column_labels,
    entry_cutoff,
):
    """Reject direct couplings between distinct non-interface edge blocks."""
    row_labels = np.asarray(row_labels, dtype=int)
    column_labels = np.asarray(column_labels, dtype=int)
    if matrix.shape != (row_labels.size, column_labels.size):
        raise ValueError("Edge labels do not match the matrix dimensions.")
    coo = matrix.tocoo()
    row_edges = row_labels[coo.row]
    column_edges = column_labels[coo.col]
    edge_entry = (row_edges >= 0) & (column_edges >= 0)
    cross_edge = edge_entry & (row_edges != column_edges)
    if np.any(cross_edge & (np.abs(coo.data) > entry_cutoff)):
        raise ValueError(
            "The non-interface matrix contains direct coupling between edges."
        )


def _group_label_indices(labels):
    """Group positions by label, preserving the original order within groups."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    if not order.size:
        return sorted_labels, order, np.zeros(1, dtype=int)
    starts = np.concatenate((
        [0], np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1,
    ))
    return sorted_labels[starts], order, np.append(starts, labels.size)


def _csr_local_columns(matrix, columns):
    """Select sorted unique columns using only row-local sparse workspace.

    CSR fancy column indexing allocates a lookup spanning the original matrix
    width. Repeating that for tiny edge blocks makes extraction quadratic in
    graph size. Binary search in the selected columns instead costs workspace
    proportional to the selected rows' nnz. Preserve duplicates and explicit
    zeros, and discard entries outside the selected columns as slicing does.
    """
    matrix = sparse.csr_matrix(matrix)
    columns = np.asarray(columns)
    positions = np.searchsorted(columns, matrix.indices)
    keep = positions < columns.size
    keep[keep] = columns[positions[keep]] == matrix.indices[keep]
    shape = (matrix.shape[0], columns.size)
    if np.all(keep):
        return sparse.csr_matrix(
            (matrix.data, positions, matrix.indptr), shape=shape,
        )
    offsets = np.empty(matrix.nnz + 1, dtype=matrix.indptr.dtype)
    offsets[0] = 0
    np.cumsum(keep, out=offsets[1:])
    return sparse.csr_matrix(
        (matrix.data[keep], positions[keep], offsets[matrix.indptr]),
        shape=shape,
    )


class _EdgeBlockFactorization:
    """Factor independent owned-edge blocks and return packed local data."""

    def __init__(
        self,
        matrix,
        row_labels,
        column_labels,
        matsolver,
        solver,
        owned_edges=None,
    ):
        row_labels = np.asarray(row_labels, dtype=int)
        column_labels = np.asarray(column_labels, dtype=int)
        if np.any(row_labels < 0) or np.any(column_labels < 0):
            raise ValueError(
                "All non-interface equations and unknowns must belong to an edge."
            )
        row_edges, row_order, row_offsets = _group_label_indices(row_labels)
        column_edges, column_order, column_offsets = _group_label_indices(
            column_labels
        )
        if not np.array_equal(row_edges, column_edges):
            raise ValueError("Edge equation and unknown blocks do not match.")

        _validate_independent_edge_blocks(
            matrix,
            row_labels,
            column_labels,
            solver.entry_cutoff,
        )

        self.shape = matrix.shape
        self.dtype = matrix.dtype
        self.blocks = {}
        restricted_ownership = owned_edges is not None
        if owned_edges is None:
            owned_edges = row_edges
        owned_edges = set(np.asarray(owned_edges, dtype=int))
        # Exact, bounded keys; never reuse across operator assemblies.
        factor_cache = {}
        for group, edge in enumerate(row_edges):
            if edge not in owned_edges:
                continue
            rows = row_order[row_offsets[group] : row_offsets[group + 1]]
            columns = column_order[
                column_offsets[group] : column_offsets[group + 1]
            ]
            if restricted_ownership:
                # Do not retain the global permutations through views when
                # compatibility-mode ranks own only a subset of the blocks.
                rows = rows.copy()
                columns = columns.copy()
            if rows.size != columns.size:
                raise ValueError(
                    f"Edge {edge} has {rows.size} equations and "
                    f"{columns.size} unknowns."
                )
            block = _csr_local_columns(matrix[rows], columns)
            key = (block.shape, block.dtype.str, block.indptr.tobytes(),
                   block.indices.tobytes(), block.data.tobytes())
            factorization = factor_cache.get(key)
            if factorization is None:
                factorization = matsolver(block, solver)
                if len(factor_cache) >= 128:
                    factor_cache.pop(next(iter(factor_cache)))
                factor_cache[key] = factorization
            pivot_ratio = _factorization_pivot_ratio(factorization)
            real_dtype = np.empty((), dtype=block.dtype).real.dtype
            pivot_tolerance = 100 * np.finfo(real_dtype).eps
            if pivot_ratio is not None and pivot_ratio <= pivot_tolerance:
                raise np.linalg.LinAlgError(
                    f"Edge {edge} Dirichlet block is numerically singular "
                    f"(LU pivot ratio {pivot_ratio:.3e}).  A global problem "
                    "can remain regular at an edge DtN pole; use the direct "
                    "solver or a non-Dirichlet condensation formulation."
                )
            self.blocks[int(edge)] = (
                rows,
                columns,
                factorization,
            )

        local_columns = [
            columns
            for rows, columns, factorization in self.blocks.values()
        ]
        if local_columns:
            self.local_columns = np.sort(np.concatenate(local_columns))
        else:
            self.local_columns = np.empty(0, dtype=int)
        self.block_positions = {
            edge: np.searchsorted(self.local_columns, columns)
            for edge, (rows, columns, factorization) in self.blocks.items()
        }

        # Group ONLY identical factor objects already shared by the exact
        # operator cache. This never approximates or merges different blocks.
        groups = {}
        for edge, (rows, columns, factorization) in self.blocks.items():
            group = groups.setdefault(id(factorization), (factorization, [], []))
            group[1].append(rows)
            group[2].append(self.block_positions[edge])
        self.solve_groups = tuple((factor, np.asarray(rows, dtype=np.int64),
                                   np.asarray(positions, dtype=np.int64))
                                  for factor, rows, positions in groups.values())

    def solve(self, vector):
        """Solve owned blocks and pack results in ``local_columns`` order."""
        vector = np.asarray(vector)
        squeeze = vector.ndim == 1
        if squeeze:
            vector = vector[:, None]
        output = np.empty(
            (self.local_columns.size, vector.shape[1]),
            dtype=np.result_type(vector.dtype, self.dtype),
        )
        nrhs = vector.shape[1]
        for factorization, rows, positions in self.solve_groups:
            block_size = rows.shape[1]
            if not nrhs:
                continue
            # Bound each dense workspace while batching independent RHSs.
            batch_size = max(1, (32*1024**2)//max(1, block_size*nrhs*output.dtype.itemsize))
            for begin in range(0, rows.shape[0], batch_size):
                end = min(rows.shape[0], begin+batch_size)
                rhs = vector[rows[begin:end]].transpose(1, 0, 2).reshape(block_size, -1)
                solved = _rhs_matrix(factorization.solve(rhs), block_size, (end-begin)*nrhs)
                output[positions[begin:end]] = solved.reshape(block_size, end-begin, nrhs).transpose(1, 0, 2)
        if squeeze:
            return output[:, 0]
        return output

    def response(self, matrix):
        """Apply owned inverse blocks and return a packed sparse response."""
        matrix = sparse.csr_matrix(matrix)
        data = []
        rows_out = []
        columns_out = []
        for edge, (rows, columns, factorization) in self.blocks.items():
            block_rhs = matrix[rows]
            active_columns = np.unique(block_rhs.indices)
            if not active_columns.size:
                continue
            block_response = _rhs_matrix(
                factorization.solve(
                    _csr_local_columns(block_rhs, active_columns).toarray()
                ),
                columns.size,
                active_columns.size,
            )
            block_response = sparse.coo_matrix(block_response)
            if block_response.nnz:
                data.append(block_response.data)
                rows_out.append(
                    self.block_positions[edge][block_response.row]
                )
                columns_out.append(active_columns[block_response.col])
        if not data:
            return sparse.csr_matrix(
                (self.local_columns.size, matrix.shape[1]),
                dtype=matrix.dtype,
            )
        return sparse.csr_matrix(
            (
                np.concatenate(data),
                (np.concatenate(rows_out), np.concatenate(columns_out)),
            ),
            shape=(self.local_columns.size, matrix.shape[1]),
        )


def _allgatherv_packed_rows(values, row_counts, comm):
    """All-gather packed dense row blocks without global zero buffers."""
    values = np.ascontiguousarray(values)
    row_counts = np.asarray(row_counts, dtype=np.int64)
    layout_error = None
    if values.ndim != 2:
        layout_error = "Packed row values must be a two-dimensional array."
    elif row_counts.shape != (comm.size,):
        layout_error = "Packed row counts must contain one entry per rank."
    elif values.shape[0] != row_counts[comm.rank]:
        layout_error = (
            "Local packed row count does not match the collective layout."
        )
    _raise_collective_error(
        comm,
        layout_error,
        "Packed row-layout validation",
    )
    _validate_collective_signature(
        comm,
        (
            values.shape[1],
            values.dtype.str,
            tuple(row_counts.tolist()),
        ),
        "Packed row-buffer validation",
    )
    element_counts = row_counts * values.shape[1]
    displacements = np.zeros(comm.size, dtype=np.int64)
    if comm.size > 1:
        displacements[1:] = np.cumsum(element_counts[:-1])
    gathered = np.empty(
        (row_counts.sum(), values.shape[1]),
        dtype=values.dtype,
    )
    comm.Allgatherv(
        values.ravel(),
        [
            gathered.ravel(),
            (element_counts.tolist(), displacements.tolist()),
        ],
    )
    return gathered


def _gather_sparse_sum(matrix, comm, root=0):
    """Gather sparse contributions and sum duplicate entries on one rank."""
    matrix = sparse.coo_matrix(matrix)
    payload = (matrix.data, matrix.row, matrix.col)
    payloads = comm.gather(payload, root=root)
    if comm.rank != root:
        return None
    data = [part[0] for part in payloads if part[0].size]
    rows = [part[1] for part in payloads if part[0].size]
    columns = [part[2] for part in payloads if part[0].size]
    if not data:
        return sparse.csr_matrix(matrix.shape, dtype=matrix.dtype)
    return sparse.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=matrix.shape,
        dtype=matrix.dtype,
    ).tocsr()


def _raise_collective_error(comm, error, stage):
    """Raise the same error on every rank before the next collective."""
    failed = np.asarray(error is not None, dtype=np.int32)
    comm.Allreduce(MPI.IN_PLACE, failed, op=MPI.MAX)
    if not failed:
        return
    errors = comm.allgather(error)
    for rank, rank_error in enumerate(errors):
        if rank_error is not None:
            raise RuntimeError(
                f"{stage} failed on network rank {rank}: {rank_error}"
            )


def _array_signature(values):
    """Return a compact deterministic signature for a numerical array."""
    values = np.ascontiguousarray(values)
    digest = hashlib.blake2b(values.tobytes(), digest_size=16).digest()
    return (values.shape, values.dtype.str, digest)


def _sparse_signature(matrix):
    """Return a canonical signature for a sparse matrix."""
    matrix = sparse.csr_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return (
        matrix.shape,
        matrix.dtype.str,
        _array_signature(matrix.indptr),
        _array_signature(matrix.indices),
        _array_signature(matrix.data),
    )


def _validate_collective_signature(comm, signature, stage, root=0):
    """Require replicated structural metadata to agree before collectives."""
    if comm.size == 1:
        return
    signatures = comm.allgather(signature)
    reference = signatures[root]
    for rank, rank_signature in enumerate(signatures):
        if rank_signature != reference:
            raise RuntimeError(
                f"{stage} failed on network rank {rank}: structural metadata "
                f"does not match authoritative rank {root}."
            )


def _sparse_factor_response(factorization, matrix, operator=None,
                            workspace_bytes=32 * 1024**2):
    """Apply a factor with bounded dense RHSs, packing disconnected components.

    Components share RHS slots because a block-diagonal inverse cannot mix
    their rows. Map every resulting entry back to its original global column;
    retain all actual fill, including arbitrarily small nonzeros. The bound is
    per dense array (input and output), with a minimum of one full column.
    """
    matrix = sparse.csr_matrix(matrix)
    if not matrix.nnz:
        return sparse.csr_matrix(matrix.shape, dtype=matrix.dtype)
    n = matrix.shape[0]
    if operator is None:
        component_count, labels = 1, np.zeros(n, dtype=int)
    else:
        component_count, labels = sparse.csgraph.connected_components(
            operator.astype(bool), directed=False, return_labels=True)
    rhs = matrix.tocoo()
    components = labels[rhs.row]
    order = np.lexsort((rhs.col, components))
    sorted_components, sorted_columns = components[order], rhs.col[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = ((sorted_components[1:] != sorted_components[:-1]) |
                 (sorted_columns[1:] != sorted_columns[:-1]))
    columns = sorted_columns[first]
    counts = np.bincount(sorted_components[first], minlength=component_count)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    pair_index = np.empty(order.size, dtype=np.int64)
    pair_index[order] = np.cumsum(first) - 1
    slots = pair_index - offsets[components]
    width = int(counts.max())
    packed = sparse.csr_matrix((rhs.data, (rhs.row, slots)), shape=(n, width))
    chunk_width = max(1, workspace_bytes // max(1, n * matrix.dtype.itemsize))
    data, rows, cols = [], [], []
    for start in range(0, width, chunk_width):
        stop = min(width, start + chunk_width)
        response = sparse.coo_matrix(_rhs_matrix(
            factorization.solve(packed[:, start:stop].toarray()), n, stop-start))
        local_slots = response.col + start
        row_components = labels[response.row]
        valid = local_slots < counts[row_components]
        data.append(response.data[valid])
        rows.append(response.row[valid])
        cols.append(columns[offsets[row_components[valid]] + local_slots[valid]])
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=matrix.shape)


class _SubgraphDtNFactorization:
    """Eliminate rank-interior vertices onto the partition separator."""

    def __init__(
        self,
        direct_matrix,
        local_edge_schur,
        row_vertices,
        column_vertices,
        partition,
        matsolver,
        solver,
        comm,
        root=0,
    ):
        self.comm = comm
        self.root = root
        self.partition = partition
        rank = comm.rank
        row_vertices = np.asarray(row_vertices, dtype=int)
        column_vertices = np.asarray(column_vertices, dtype=int)
        label_error = None
        if np.any(row_vertices < 0) or np.any(column_vertices < 0):
            label_error = (
                "Subgraph DtN requires labelled vertex or global-scalar "
                "interface DOFs."
            )
        _raise_collective_error(
            comm,
            label_error,
            "Subgraph label validation",
        )
        num_interface_labels = max(
            partition.network.num_vertices,
            row_vertices.max(initial=-1) + 1,
            column_vertices.max(initial=-1) + 1,
        )
        row_counts = np.bincount(
            row_vertices,
            minlength=num_interface_labels,
        )
        column_counts = np.bincount(
            column_vertices,
            minlength=num_interface_labels,
        )
        count_error = None
        if not np.array_equal(row_counts, column_counts):
            count_error = (
                "Each network vertex or global scalar must have the same "
                "number of interface equations and unknowns for subgraph "
                "condensation."
            )
        _raise_collective_error(
            comm,
            count_error,
            "Subgraph interface-count validation",
        )

        direct_matrix = sparse.csr_matrix(direct_matrix)
        local_edge_schur = sparse.csr_matrix(local_edge_schur)
        num_vertices = partition.network.num_vertices
        local_promoted = np.zeros(num_vertices, dtype=np.int8)
        for contribution in (direct_matrix, local_edge_schur):
            entries = contribution.tocoo()
            global_rows = row_vertices[entries.row] >= num_vertices
            physical_columns = (
                (column_vertices[entries.col] >= 0)
                & (column_vertices[entries.col] < num_vertices)
            )
            significant = np.abs(entries.data) > solver.entry_cutoff
            touched = column_vertices[
                entries.col[global_rows & physical_columns & significant]
            ]
            local_promoted[touched] = 1
        promoted = np.empty_like(local_promoted)
        comm.Allreduce(local_promoted, promoted, op=MPI.MAX)
        promoted_vertices = np.flatnonzero(promoted)

        row_vertex_owners = _network_interface_owners(
            row_vertices,
            partition,
            condition=False,
            root=root,
        )
        column_vertex_owners = _network_interface_owners(
            column_vertices,
            partition,
            condition=False,
            root=root,
        )
        row_condition_owners = _network_interface_owners(
            row_vertices,
            partition,
            condition=True,
            root=root,
        )
        column_condition_owners = _network_interface_owners(
            column_vertices,
            partition,
            condition=True,
            root=root,
        )

        # A global scalar equation cannot belong to one rank's local vertex
        # elimination.  Promote every physical vertex it touches into the
        # shared separator, including vertices that would otherwise be wholly
        # interior to a partition.  The matching vertex equation follows the
        # same label, preserving square interior and separator blocks.
        if promoted_vertices.size:
            row_vertex_owners[
                np.isin(row_vertices, promoted_vertices)
            ] = -1
            column_vertex_owners[
                np.isin(column_vertices, promoted_vertices)
            ] = -1

        self.interior_rows = np.flatnonzero(row_vertex_owners == rank)
        self.interior_columns = np.flatnonzero(column_vertex_owners == rank)
        self.separator_rows = np.flatnonzero(row_vertex_owners < 0)
        self.separator_columns = np.flatnonzero(column_vertex_owners < 0)
        shape_error = None
        if self.interior_rows.size != self.interior_columns.size:
            shape_error = (
                f"Subgraph {rank} has different numbers of interior equations "
                "and unknowns."
            )
        elif self.separator_rows.size != self.separator_columns.size:
            shape_error = (
                "The network separator has different numbers of equations "
                "and unknowns."
            )
        _raise_collective_error(
            comm,
            shape_error,
            "Subgraph shape validation",
        )

        # Allocate every direct vertex equation to exactly one rank.  Standard
        # pointwise Robin data are diagonal here.  More general direct vertex
        # coupling is accepted when it does not reach another rank's interior.
        direct_coo = direct_matrix.tocoo()
        direct_row_owners = row_condition_owners[direct_coo.row]
        direct_column_owners = column_vertex_owners[direct_coo.col]
        crosses_remote_interior = (
            (direct_column_owners >= 0)
            & (direct_row_owners != direct_column_owners)
            & (np.abs(direct_coo.data) > solver.entry_cutoff)
        )
        direct_coupling_error = None
        if np.any(crosses_remote_interior):
            direct_coupling_error = (
                "Direct vertex coupling crosses partition interiors; promote "
                "those vertices to the separator or choose another partition."
            )
        _raise_collective_error(
            comm,
            direct_coupling_error,
            "Subgraph direct-coupling validation",
        )
        owned_direct_rows = row_condition_owners == rank
        local_direct = sparse.diags(
            owned_direct_rows.astype(direct_matrix.dtype)
        ) @ direct_matrix
        local_matrix = (local_direct + local_edge_schur).tocsr()

        # Rank-local interface contributions may touch this rank's interiors
        # and the shared separator, but never another rank's interior.
        local_coo = local_matrix.tocoo()
        remote_row = (
            (row_vertex_owners[local_coo.row] >= 0)
            & (row_vertex_owners[local_coo.row] != rank)
        )
        remote_column = (
            (column_vertex_owners[local_coo.col] >= 0)
            & (column_vertex_owners[local_coo.col] != rank)
        )
        invalid_local = (
            (remote_row | remote_column)
            & (np.abs(local_coo.data) > solver.entry_cutoff)
        )
        local_structure_error = None
        if np.any(invalid_local):
            local_structure_error = (
                "Rank-local interface contributions cross another "
                "partition's interior."
            )
        _raise_collective_error(
            comm,
            local_structure_error,
            "Subgraph structure validation",
        )

        I_rows = self.interior_rows
        I_columns = self.interior_columns
        G_rows = self.separator_rows
        G_columns = self.separator_columns
        self.interior_matrix = local_matrix[I_rows][:, I_columns].tocsr()
        self.interior_to_separator = local_matrix[I_rows][:, G_columns].tocsr()
        self.separator_to_interior = local_matrix[G_rows][:, I_columns].tocsr()
        local_separator = local_matrix[G_rows][:, G_columns].tocsr()
        factor_error = None
        if I_rows.size:
            try:
                self.interior_solver = matsolver(self.interior_matrix, solver)
                self.interior_response = _sparse_factor_response(
                    self.interior_solver,
                    self.interior_to_separator,
                    self.interior_matrix,
                )
                local_separator = (
                    local_separator
                    - self.separator_to_interior @ self.interior_response
                ).tocsr()
            except Exception as error:
                self.interior_solver = None
                factor_error = f"{type(error).__name__}: {error}"
        else:
            self.interior_solver = None
            self.interior_response = sparse.csr_matrix(
                (I_columns.size, G_columns.size),
                dtype=local_matrix.dtype,
            )
        _raise_collective_error(
            comm,
            factor_error,
            "Subgraph interior factorization",
        )

        self.row_condition_owners = row_condition_owners
        self.separator_row_owners = row_condition_owners[G_rows]
        self.separator_column_owners = column_condition_owners[G_columns]
        self.separator_owned_rows = self.separator_row_owners == rank
        self.local_schur_matrix = local_separator

        # The separator is assembled and solved collectively.  Small systems
        # use a replicated factorization, while larger systems can remain as
        # additive rank-local sparse contributions throughout restarted GMRES.
        # No communicator rank has a distinguished factorization or solve role.
        if G_rows.size:
            self.separator_solver = CollectiveSparseSolver(
                local_separator,
                comm,
                owned_rows=self.separator_owned_rows,
                method=getattr(solver, "network_separator_method", "auto"),
                direct_max_size=getattr(
                    solver,
                    "network_separator_direct_max_size",
                    256,
                ),
                restart=getattr(solver, "network_separator_restart", 40),
                maxiter=getattr(solver, "network_separator_maxiter", None),
                rtol=getattr(solver, "network_separator_rtol", 1e-10),
                atol=getattr(solver, "network_separator_atol", 0.0),
                preconditioner=getattr(
                    solver,
                    "network_separator_preconditioner",
                    "jacobi",
                ),
            )
        else:
            self.separator_solver = None

        # Direct solves expose the replicated matrix already assembled by the
        # collective solver.  GMRES deliberately never assembles that matrix,
        # so ``schur_matrix`` is each rank's additive local contribution.
        self.schur_matrix_is_local = bool(
            self.separator_solver is not None
            and self.separator_solver.method == "gmres"
        )
        if self.schur_matrix_is_local:
            self.schur_matrix = local_separator
        elif self.separator_solver is None:
            self.schur_matrix = local_separator
        else:
            self.schur_matrix = self.separator_solver.global_matrix

    def solve(self, interface_rhs, local_edge_correction):
        """Solve the separator system and reconstruct all vertex unknowns."""
        rank = self.comm.rank
        local_rhs = np.array(local_edge_correction, copy=True)
        owned_rows = self.row_condition_owners == rank
        local_rhs[owned_rows] += interface_rhs[owned_rows]

        interior_error = None
        if self.interior_rows.size:
            try:
                free_interior = _rhs_matrix(
                    self.interior_solver.solve(local_rhs[self.interior_rows]),
                    self.interior_columns.size,
                    interface_rhs.shape[1],
                )
            except Exception as error:
                free_interior = None
                interior_error = f"{type(error).__name__}: {error}"
        else:
            free_interior = np.zeros(
                (0, interface_rhs.shape[1]),
                dtype=interface_rhs.dtype,
            )
        _raise_collective_error(
            self.comm,
            interior_error,
            "Subgraph interior solve",
        )
        local_separator_rhs = (
            local_rhs[self.separator_rows]
            - self.separator_to_interior @ free_interior
        )
        separator_error = None
        if self.separator_rows.size:
            try:
                separator_solution = _rhs_matrix(
                    self.separator_solver.solve(local_separator_rhs),
                    self.separator_columns.size,
                    interface_rhs.shape[1],
                )
            except Exception as error:
                separator_solution = None
                separator_error = f"{type(error).__name__}: {error}"
        else:
            separator_solution = np.zeros(
                (0, interface_rhs.shape[1]),
                dtype=interface_rhs.dtype,
            )
        _raise_collective_error(
            self.comm,
            separator_error,
            "Collective separator solve",
        )

        interior_solution = (
            free_interior
            - self.interior_response @ separator_solution
        )
        local_solution = np.zeros(
            (interface_rhs.shape[0], interface_rhs.shape[1]),
            dtype=interface_rhs.dtype,
        )
        local_solution[self.interior_columns] = interior_solution
        owned_separator_columns = self.separator_column_owners == rank
        local_solution[
            self.separator_columns[owned_separator_columns]
        ] = separator_solution[owned_separator_columns]
        solution = np.empty_like(local_solution)
        self.comm.Allreduce(local_solution, solution, op=MPI.SUM)
        return solution


class _ResidentSubgraphDtNFactorization:
    """Eliminate resident rank interiors onto a compact shared separator."""

    def __init__(
        self,
        local_matrix,
        row_dofs,
        column_dofs,
        partition,
        matsolver,
        solver,
        comm,
        root=0,
        coupled=None,
    ):
        self.comm = comm
        self.root = int(root)
        self.partition = partition
        self.row_dofs = row_dofs
        self.column_dofs = column_dofs
        rank = comm.rank
        num_vertices = partition.network.num_vertices
        row_keys = row_dofs.keys
        column_keys = column_dofs.keys
        row_labels = row_dofs.labels
        column_labels = column_dofs.labels
        local_matrix = sparse.csr_matrix(local_matrix)

        count_error = None
        counts_match, row_counts = _matching_interface_counts(
            row_dofs.counts, column_dofs.counts, comm)
        if local_matrix.shape != (row_keys.shape[0], column_keys.shape[0]):
            count_error = (
                f"local interface matrix {local_matrix.shape} does not match "
                f"resident keys {(row_keys.shape[0], column_keys.shape[0])}"
            )
        elif not counts_match:
            count_error = (
                "each physical vertex or global scalar must have the same "
                "number of interface equations and unknowns"
            )
        _raise_collective_error(
            comm,
            count_error,
            "Resident subgraph interface-count validation",
        )
        self.interface_counts = row_counts

        # A basis-free global equation or unknown cannot enter a rank-local
        # interior factorization.  Promote every physical vertex coupled to a
        # synthetic label, in either matrix direction, onto the separator.
        entries = local_matrix.tocoo()
        significant = np.abs(entries.data) > solver.entry_cutoff
        entry_row_labels = row_labels[entries.row]
        entry_column_labels = column_labels[entries.col]
        global_rows = entry_row_labels >= num_vertices
        global_columns = entry_column_labels >= num_vertices
        physical_rows = entry_row_labels < num_vertices
        physical_columns = entry_column_labels < num_vertices
        touched_columns = entry_column_labels[
            significant & global_rows & physical_columns
        ]
        touched_rows = entry_row_labels[
            significant & physical_rows & global_columns
        ]
        local_promoted = np.union1d(touched_columns, touched_rows)
        self.promoted_vertices = np.unique(np.concatenate(comm.allgather(local_promoted)))

        row_interior_owners = _network_interface_owners(
            row_labels,
            partition,
            condition=False,
            root=root,
        )
        column_interior_owners = _network_interface_owners(
            column_labels,
            partition,
            condition=False,
            root=root,
        )
        if self.promoted_vertices.size:
            row_interior_owners[
                np.isin(row_labels, self.promoted_vertices)
            ] = -1
            column_interior_owners[
                np.isin(column_labels, self.promoted_vertices)
            ] = -1

        self.interior_rows = np.flatnonzero(row_interior_owners == rank)
        self.interior_columns = np.flatnonzero(
            column_interior_owners == rank
        )
        self.separator_rows = np.flatnonzero(row_interior_owners < 0)
        self.separator_columns = np.flatnonzero(
            column_interior_owners < 0
        )
        shape_error = None
        if self.interior_rows.size != self.interior_columns.size:
            shape_error = (
                f"resident subgraph {rank} has different numbers of interior "
                "equations and unknowns"
            )
        _raise_collective_error(
            comm,
            shape_error,
            "Resident subgraph shape validation",
        )

        # Every significant local entry may reach this rank's interiors and
        # any separator key, but never another rank's interior.
        remote_rows = (
            (row_interior_owners[entries.row] >= 0)
            & (row_interior_owners[entries.row] != rank)
        )
        remote_columns = (
            (column_interior_owners[entries.col] >= 0)
            & (column_interior_owners[entries.col] != rank)
        )
        structure_error = None
        if np.any(significant & (remote_rows | remote_columns)):
            structure_error = (
                "a rank-local resident interface contribution reaches "
                "another partition's interior"
            )
        _raise_collective_error(
            comm,
            structure_error,
            "Resident subgraph structure validation",
        )

        I_rows = self.interior_rows
        I_columns = self.interior_columns
        G_rows = self.separator_rows
        G_columns = self.separator_columns
        # With coupled input, factor edge coefficients/tau and internal
        # vertices together. The structural vertex matrix above determines
        # separator ownership and conservative global-constraint promotions.
        self.coupled = coupled
        self.solution_size = column_keys.shape[0]
        if coupled is not None:
            full, edge_rows, edge_columns, vertex_rows, vertex_columns = coupled
            I_rows = np.concatenate((edge_rows, vertex_rows[I_rows]))
            I_columns = np.concatenate((edge_columns, vertex_columns[I_columns]))
            G_rows = vertex_rows[G_rows]
            G_columns = vertex_columns[G_columns]
            local_matrix = full
            self.solution_size = full.shape[1]
            self.full_interior_rows = I_rows
            self.full_interior_columns = I_columns
            self.full_separator_rows = G_rows
            self.full_separator_columns = G_columns
        self.interior_matrix = local_matrix[I_rows][:, I_columns].tocsr()
        self.interior_to_separator = local_matrix[I_rows][
            :, G_columns
        ].tocsr()
        self.separator_to_interior = local_matrix[G_rows][
            :, I_columns
        ].tocsr()
        local_separator = local_matrix[G_rows][:, G_columns].tocsr()

        factor_error = None
        if I_rows.size:
            try:
                self.interior_solver = matsolver(self.interior_matrix, solver)
                self.interior_response = _sparse_factor_response(
                    self.interior_solver,
                    self.interior_to_separator,
                    self.interior_matrix,
                )
                local_separator = (
                    local_separator
                    - self.separator_to_interior @ self.interior_response
                ).tocsr()
            except Exception as error:
                self.interior_solver = None
                self.interior_response = None
                factor_error = f"{type(error).__name__}: {error}"
        else:
            self.interior_solver = None
            self.interior_response = sparse.csr_matrix(
                (I_columns.size, G_columns.size),
                dtype=local_matrix.dtype,
            )
        _raise_collective_error(
            comm,
            factor_error,
            "Resident subgraph interior factorization",
        )

        # Only actual local separator keys enter the collective interface.
        # Do not enumerate every physical vertex to discover this small set.
        G_rows, G_columns = self.separator_rows, self.separator_columns
        gathered_keys = comm.allgather(row_keys[G_rows])
        self.separator_keys = np.unique(np.concatenate(gathered_keys), axis=0)
        separator_index = {
            tuple(key): index
            for index, key in enumerate(self.separator_keys)
        }
        mapping_error = None
        try:
            self.separator_row_positions = np.asarray(
                [separator_index[tuple(key)] for key in row_keys[G_rows]],
                dtype=np.int64,
            )
            self.separator_column_positions = np.asarray(
                [separator_index[tuple(key)] for key in column_keys[G_columns]],
                dtype=np.int64,
            )
        except KeyError as error:
            self.separator_row_positions = None
            self.separator_column_positions = None
            mapping_error = f"separator key {error.args[0]!r} is missing"
        _raise_collective_error(
            comm,
            mapping_error,
            "Resident subgraph separator-key validation",
        )

        separator_size = self.separator_keys.shape[0]
        local_coo = local_separator.tocoo()
        embedded_separator = sparse.csr_matrix(
            (
                local_coo.data,
                (
                    self.separator_row_positions[local_coo.row],
                    self.separator_column_positions[local_coo.col],
                ),
            ),
            shape=(separator_size, separator_size),
            dtype=local_separator.dtype,
        )
        embedded_separator.sum_duplicates()
        self.local_schur_matrix = embedded_separator

        separator_labels_array = self.separator_keys[:, 0]
        separator_owners = np.full(separator_size, root, dtype=np.int64)
        physical_separator_keys = separator_labels_array < num_vertices
        separator_owners[physical_separator_keys] = (
            partition.vertex_equation_owners[
                separator_labels_array[physical_separator_keys]
            ]
        )
        self.separator_row_owners = separator_owners
        self.separator_column_owners = separator_owners.copy()
        self.separator_owned_rows = separator_owners == rank

        if separator_size:
            self.separator_solver = CollectiveSparseSolver(
                embedded_separator,
                comm,
                owned_rows=self.separator_owned_rows,
                method=getattr(solver, "network_separator_method", "auto"),
                direct_max_size=getattr(
                    solver,
                    "network_separator_direct_max_size",
                    256,
                ),
                restart=getattr(solver, "network_separator_restart", 40),
                maxiter=getattr(solver, "network_separator_maxiter", None),
                rtol=getattr(solver, "network_separator_rtol", 1e-10),
                atol=getattr(solver, "network_separator_atol", 0.0),
                preconditioner=getattr(
                    solver,
                    "network_separator_preconditioner",
                    "jacobi",
                ),
            )
        else:
            self.separator_solver = None

        self.resident_separator = bool(self.separator_solver is not None
                                       and self.separator_solver._owned_krylov is not None)
        if self.resident_separator:
            self.separator_solver.configure_resident(self.separator_row_positions,
                                                       self.separator_column_positions)

        self.schur_matrix_is_local = bool(
            self.separator_solver is not None
            and self.separator_solver.method == "gmres"
        )
        if self.schur_matrix_is_local or self.separator_solver is None:
            self.schur_matrix = embedded_separator
        else:
            self.schur_matrix = self.separator_solver.global_matrix

    def solve(self, interface_rhs, local_edge_correction, edge_rhs=None):
        """Solve compact separator data and reconstruct resident unknowns."""
        interface_rhs = np.asarray(interface_rhs)
        local_edge_correction = np.asarray(local_edge_correction)
        columns = interface_rhs.shape[1]
        dtype = np.result_type(
            interface_rhs.dtype,
            local_edge_correction.dtype,
        )
        local_rhs = np.asarray(
            local_edge_correction,
            dtype=dtype,
        ).copy()
        local_rhs[self.row_dofs.owned] += interface_rhs[
            self.row_dofs.owned
        ]

        I_rows, I_columns = self.interior_rows, self.interior_columns
        G_rows, G_columns = self.separator_rows, self.separator_columns
        if self.coupled is not None:
            full, edge_rows, edge_columns, vertex_rows, vertex_columns = self.coupled
            rhs = np.zeros((full.shape[0], columns), dtype=dtype)
            rhs[vertex_rows] = local_rhs
            rhs[edge_rows] = edge_rhs
            local_rhs = rhs
            I_rows, I_columns = self.full_interior_rows, self.full_interior_columns
            G_rows, G_columns = self.full_separator_rows, self.full_separator_columns
        interior_error = None
        if I_rows.size:
            try:
                free_interior = _rhs_matrix(
                    self.interior_solver.solve(
                        local_rhs[I_rows]
                    ),
                    I_columns.size,
                    columns,
                )
            except Exception as error:
                free_interior = None
                interior_error = f"{type(error).__name__}: {error}"
        else:
            free_interior = np.zeros((0, columns), dtype=dtype)
        _raise_collective_error(
            self.comm,
            interior_error,
            "Resident subgraph interior solve",
        )

        local_separator_rhs = (
            local_rhs[G_rows]
            - self.separator_to_interior @ free_interior
        )
        separator_size = self.separator_keys.shape[0]
        separator_error = None
        local_separator_solution = None
        try:
            if self.resident_separator:
                local_separator_solution = self.separator_solver.solve_resident(local_separator_rhs)
            elif separator_size:
                embedded_rhs = np.zeros((separator_size, columns), dtype=np.result_type(dtype, local_separator_rhs.dtype))
                embedded_rhs[self.separator_row_positions] += local_separator_rhs
                full_solution = _rhs_matrix(self.separator_solver.solve(embedded_rhs), separator_size, columns)
                local_separator_solution = full_solution[self.separator_column_positions]
            else:
                local_separator_solution = np.zeros((self.separator_columns.size, columns), dtype=dtype)
        except Exception as error:
            separator_error = f"{type(error).__name__}: {error}"
        _raise_collective_error(self.comm, separator_error, "Resident collective separator solve")

        interior_solution = (
            free_interior
            - self.interior_response @ local_separator_solution
        )
        solution = np.zeros(
            (self.solution_size, columns),
            dtype=np.result_type(
                dtype,
                interior_solution.dtype,
                local_separator_solution.dtype,
            ),
        )
        solution[I_columns] = interior_solution
        solution[G_columns] = local_separator_solution
        return solution


class DistributedNetworkDtNFactorization:
    """DtN factorization for fields already stored by whole-edge ownership."""

    def __init__(
        self,
        matrix,
        solver,
        subproblem,
        matsolver,
        partition=None,
        comm=None,
    ):
        if comm is None:
            comm = MPI.COMM_SELF
        if not solver.dist.distributed_network:
            raise ValueError(
                "DistributedNetworkDtNFactorization requires distributed "
                "Network field layouts."
            )
        self.comm = comm
        self.root = 0
        self.network = solver.dist.network
        self.partition = solver.dist.network_partition
        self.owned_edges = self.partition.edges[comm.rank]
        problem = solver.problem
        variables = problem.LHS_variables
        equations = problem.equations
        matrix = sparse.csr_matrix(matrix)

        interface_fields = [
            variable
            for variable in variables
            if _is_network_interface_operand(variable)
        ]
        if not interface_fields:
            raise ValueError(
                "DtN condensation requires an explicit NetworkBasis vertex field."
            )

        full_column_mask = np.concatenate(
            [
                np.full(
                    subproblem.field_size(variable),
                    variable in interface_fields,
                    dtype=bool,
                )
                for variable in variables
            ]
        )
        full_row_mask = np.concatenate(
            [
                np.full(
                    subproblem.field_size(equation["eqn"]),
                    _is_network_interface_operand(equation["eqn"]),
                    dtype=bool,
                )
                for equation in equations
            ]
        )
        interface_columns = np.flatnonzero(
            _select_preconditioned(
                subproblem.pre_right_pinv,
                full_column_mask,
                "right preconditioner",
            )
        )
        interface_rows = np.flatnonzero(
            _select_preconditioned(
                subproblem.pre_left,
                full_row_mask,
                "left preconditioner",
            )
        )
        all_rows = np.arange(matrix.shape[0])
        all_columns = np.arange(matrix.shape[1])
        edge_rows = np.setdiff1d(all_rows, interface_rows, assume_unique=True)
        edge_columns = np.setdiff1d(
            all_columns,
            interface_columns,
            assume_unique=True,
        )
        shape_error = None
        if matrix.shape[0] != matrix.shape[1]:
            shape_error = f"Local Network matrix is not square: {matrix.shape}."
        elif interface_rows.size != interface_columns.size:
            shape_error = (
                "DtN condensation found different local numbers of vertex "
                "equations and interface unknowns."
            )
        elif edge_rows.size != edge_columns.size:
            shape_error = (
                "The local non-interface block is not square and cannot be "
                "condensed."
            )
        _raise_collective_error(
            comm,
            shape_error,
            "Distributed Network block-layout validation",
        )

        full_column_labels = np.concatenate(
            [
                _network_element_labels(variable, subproblem)
                for variable in variables
            ]
        )
        full_row_labels = np.concatenate(
            [
                _network_element_labels(equation["eqn"], subproblem)
                for equation in equations
            ]
        )
        full_column_vertices = _network_interface_labels(
            variables,
            subproblem,
            self.network.num_vertices,
        )
        full_row_vertices = _network_interface_labels(
            [equation["eqn"] for equation in equations],
            subproblem,
            self.network.num_vertices,
        )
        minimal_column_labels = _select_preconditioned(
            subproblem.pre_right_pinv,
            full_column_labels,
            "right preconditioner",
        )
        minimal_row_labels = _select_preconditioned(
            subproblem.pre_left,
            full_row_labels,
            "left preconditioner",
        )
        minimal_column_vertices = _select_preconditioned(
            subproblem.pre_right_pinv,
            full_column_vertices,
            "right preconditioner",
        )
        minimal_row_vertices = _select_preconditioned(
            subproblem.pre_left,
            full_row_vertices,
            "left preconditioner",
        )
        edge_column_labels = minimal_column_labels[edge_columns]
        edge_row_labels = minimal_row_labels[edge_rows]
        interface_column_vertices = minimal_column_vertices[interface_columns]
        interface_row_vertices = minimal_row_vertices[interface_rows]
        self.resident = (
            solver.dist.network_vertex_storage == "owned_ghost"
        )

        label_error = None
        if np.any(edge_row_labels < 0) or np.any(edge_column_labels < 0):
            label_error = (
                "All local non-interface equations and unknowns must belong "
                "to an owned edge."
            )
        elif not np.array_equal(
            np.unique(edge_row_labels),
            self.owned_edges,
        ):
            label_error = "Local edge equations do not match field ownership."
        elif not np.array_equal(
            np.unique(edge_column_labels),
            self.owned_edges,
        ):
            label_error = "Local edge unknowns do not match field ownership."
        _raise_collective_error(
            comm,
            label_error,
            "Distributed Network edge-label validation",
        )
        if self.resident:
            layout_error = None
            try:
                layout = _distributed_network_matrix_layout(
                    solver,
                    subproblem,
                )
                if not np.array_equal(
                    layout.interface_rows,
                    interface_rows,
                ) or not np.array_equal(
                    layout.interface_columns,
                    interface_columns,
                ):
                    raise ValueError(
                        "matrix and DtN interface selections do not match"
                    )
                if not np.array_equal(
                    layout.interface_row_vertices,
                    interface_row_vertices,
                ) or not np.array_equal(
                    layout.interface_column_vertices,
                    interface_column_vertices,
                ):
                    raise ValueError(
                        "matrix and DtN interface labels do not match"
                    )
            except Exception as error:
                layout = None
                layout_error = f"{type(error).__name__}: {error}"
            _raise_collective_error(
                comm,
                layout_error,
                "Resident Network interface-layout validation",
            )
            _validate_collective_signature(
                comm,
                matrix.dtype.str,
                "Resident Network matrix-dtype validation",
                root=self.root,
            )
            self.interface_row_dofs = layout.row_dofs
            self.interface_column_dofs = layout.column_dofs
            self.interface_row_owned = layout.interface_row_owned
            self.interface_column_owned = layout.interface_column_owned
        else:
            _validate_collective_signature(
                comm,
                (
                    matrix.dtype.str,
                    _array_signature(interface_row_vertices),
                    _array_signature(interface_column_vertices),
                ),
                "Distributed Network interface-layout validation",
                root=self.root,
            )
            self.interface_row_dofs = None
            self.interface_column_dofs = None
            rank_is_root = comm.rank == self.root
            self.interface_row_owned = np.full(
                interface_rows.size,
                rank_is_root,
                dtype=bool,
            )
            self.interface_column_owned = np.full(
                interface_columns.size,
                rank_is_root,
                dtype=bool,
            )

        self.interface_rows = interface_rows
        self.interface_columns = interface_columns
        self.edge_rows = edge_rows
        self.edge_columns = edge_columns
        local_matrix = matrix[edge_rows][:, edge_columns].tocsr()
        local_edge_to_interface = matrix[
            edge_rows
        ][:, interface_columns].tocsr()
        self.local_interface_to_edge = matrix[
            interface_rows
        ][:, edge_columns].tocsr()
        local_interface_matrix = matrix[
            interface_rows
        ][:, interface_columns].tocsr()

        self.coupled = self.resident and (
            solver.network_interior_elimination == "coupled"
        )
        if self.coupled:
            # Symbolic fill through an edge conservatively identifies vertex
            # couplings without requiring invertible individual edge blocks.
            # It can retain extra separator vertices if numerical terms cancel.
            _, edge_row_positions = np.unique(edge_row_labels, return_inverse=True)
            _, edge_column_positions = np.unique(edge_column_labels, return_inverse=True)
            num_edges = self.owned_edges.size
            left = self.local_interface_to_edge.tocoo()
            right = local_edge_to_interface.tocoo()
            left_pattern = sparse.csr_matrix(
                (np.ones(left.nnz), (left.row, edge_column_positions[left.col])),
                shape=(interface_rows.size, num_edges),
            )
            right_pattern = sparse.csr_matrix(
                (np.ones(right.nnz), (edge_row_positions[right.row], right.col)),
                shape=(num_edges, interface_columns.size),
            )
            structure = abs(local_interface_matrix) + left_pattern @ right_pattern
            self.subgraph_solver = _ResidentSubgraphDtNFactorization(
                structure, self.interface_row_dofs, self.interface_column_dofs,
                self.partition, matsolver, solver, comm, root=self.root,
                coupled=(matrix, edge_rows, edge_columns, interface_rows, interface_columns),
            )
            self.schur_matrix = self.subgraph_solver.schur_matrix
            self.interface_solver = self.subgraph_solver.separator_solver
            self.hierarchical = True
            self.edge_response_rows_local = edge_columns.size
            self.edge_column_counts = np.asarray(
                comm.allgather(edge_columns.size), dtype=np.int64
            )
            return

        edge_factor_error = None
        try:
            self.edge_solver = _EdgeBlockFactorization(
                local_matrix,
                edge_row_labels,
                edge_column_labels,
                matsolver,
                solver,
            )
            self.edge_response = self.edge_solver.response(
                local_edge_to_interface
            )
        except Exception as error:
            self.edge_solver = None
            self.edge_response = None
            edge_factor_error = f"{type(error).__name__}: {error}"
        _raise_collective_error(
            comm,
            edge_factor_error,
            "Distributed Network edge factorization",
        )
        packed_error = None
        if not np.array_equal(
            self.edge_solver.local_columns,
            np.arange(edge_columns.size),
        ):
            packed_error = "The local edge solver does not cover every edge column."
        _raise_collective_error(
            comm,
            packed_error,
            "Distributed Network packed edge-layout validation",
        )

        self.edge_column_counts = np.asarray(
            comm.allgather(edge_columns.size),
            dtype=np.int64,
        )
        self.edge_response_rows_local = edge_columns.size
        local_edge_schur = (
            -self.local_interface_to_edge @ self.edge_response
        ).tocsr()
        local_interface_schur = (
            local_interface_matrix + local_edge_schur
        ).tocsr()
        self.hierarchical = True
        if self.resident:
            self.subgraph_solver = _ResidentSubgraphDtNFactorization(
                local_interface_schur,
                self.interface_row_dofs,
                self.interface_column_dofs,
                self.partition,
                matsolver,
                solver,
                comm,
                root=self.root,
            )
        else:
            self.subgraph_solver = _SubgraphDtNFactorization(
                sparse.csr_matrix(
                    local_interface_matrix.shape,
                    dtype=local_interface_matrix.dtype,
                ),
                local_interface_schur,
                interface_row_vertices,
                interface_column_vertices,
                self.partition,
                matsolver,
                solver,
                comm,
                root=self.root,
            )
        self.schur_matrix = self.subgraph_solver.schur_matrix
        self.interface_solver = self.subgraph_solver.separator_solver

    def synchronize_interface_columns(self, values):
        """Refresh ghost interface unknowns after an external update."""
        values = np.asarray(values)
        if self.resident:
            return self.interface_column_dofs.forward(values)
        self.comm.Bcast(values, root=self.root)
        return values

    def synchronize_interface_rows(self, values):
        """Refresh ghost interface equation data after an external update."""
        values = np.asarray(values)
        if self.resident:
            return self.interface_row_dofs.forward(values)
        self.comm.Bcast(values, root=self.root)
        return values

    def solve(self, vector):
        """Solve a rank-local system and retain rank-local edge output."""
        vector = np.asarray(vector)
        original_shape = vector.shape
        squeeze = vector.ndim == 1
        if squeeze:
            vector = vector[:, None]
        rhs_error = None
        if vector.ndim != 2 or vector.shape[0] != (
            self.edge_rows.size + self.interface_rows.size
        ):
            rhs_error = (
                "Rank-local right-hand side does not match its Network matrix."
            )
        validation = getattr(self, '_rhs_metadata', None)
        if validation is None:
            validation = self._rhs_metadata = CollectiveMetadata()
        signature = None if rhs_error else (
            vector.dtype.str, vector.shape[1], vector.shape[0] == 0,
        )

        def validate_rhs(signatures):
            if any(item[0] != signatures[self.root][0] for item in signatures):
                raise RuntimeError('Resident Network right-hand-side dtype mismatch')
            columns = max(item[1] for item in signatures)
            for dtype, count, empty in signatures:
                if self.resident and count == 0 and not empty:
                    raise RuntimeError('A nonempty local matrix has no right-hand-side columns')
                if count != columns and not (self.resident and count == 0 and empty):
                    raise RuntimeError('Distributed Network right-hand-side columns disagree')
            return columns

        columns = validation.check(
            self.comm, signature, 'Distributed Network right-hand-side validation',
            error=rhs_error, validate=validate_rhs,
        )
        empty_placeholder = (self.resident and vector.shape == (0, 0) and columns > 0)
        if empty_placeholder:
            vector = np.empty((0, columns), dtype=vector.dtype)

        edge_rhs = vector[self.edge_rows]
        interface_rhs = np.ascontiguousarray(vector[self.interface_rows])
        if not self.resident:
            self.comm.Bcast(interface_rhs, root=self.root)
        if self.coupled:
            solution = self.subgraph_solver.solve(
                interface_rhs, np.zeros_like(interface_rhs), edge_rhs=edge_rhs
            )
            if empty_placeholder:
                return np.empty(original_shape, dtype=solution.dtype)
            return solution[:, 0] if squeeze else solution
        edge_solve_error = None
        try:
            free_edge_solution = self.edge_solver.solve(edge_rhs)
        except Exception as error:
            free_edge_solution = None
            edge_solve_error = f"{type(error).__name__}: {error}"
        _raise_collective_error(
            self.comm,
            edge_solve_error,
            "Distributed Network edge solve",
        )
        local_rhs_correction = (
            -self.local_interface_to_edge @ free_edge_solution
        )
        interface_solution = self.subgraph_solver.solve(
            interface_rhs,
            local_rhs_correction,
        )
        local_edge_solution = (
            free_edge_solution
            - self.edge_response @ interface_solution
        )
        solution = np.empty(
            vector.shape,
            dtype=np.result_type(
                vector.dtype,
                local_edge_solution.dtype,
                interface_solution.dtype,
            ),
        )
        solution[self.edge_columns] = local_edge_solution
        solution[self.interface_columns] = interface_solution
        if empty_placeholder:
            return np.empty(original_shape, dtype=solution.dtype)
        if squeeze:
            return solution[:, 0]
        return solution


class NetworkDtNFactorization:
    """Schur factorization onto explicit network vertex variables.

    The assembled edge equations and port-continuity equations form the local
    block.  Eliminating that block leaves the exact discrete Dirichlet-to-
    Neumann system on vertex fields.  The public equations are unchanged.
    """

    def __init__(
        self,
        matrix,
        solver,
        subproblem,
        matsolver,
        partition=None,
        comm=None,
    ):
        problem = solver.problem
        variables = problem.LHS_variables
        equations = problem.equations
        if comm is None:
            comm = MPI.COMM_SELF
        self.comm = comm
        self.root = 0
        network_error = None
        try:
            self.network = _network_from_operands(variables)
        except Exception as error:
            self.network = None
            network_error = f"{type(error).__name__}: {error}"
        _raise_collective_error(
            comm,
            network_error,
            "Network metadata construction",
        )
        network_signature = (
            self.network.num_vertices,
            self.network.num_edges,
            _array_signature(self.network.endpoints),
            _array_signature(self.network.lengths),
        )
        _validate_collective_signature(
            comm,
            network_signature,
            "Network topology validation",
            root=self.root,
        )
        matrix = sparse.csr_matrix(matrix)
        matrix_header = (matrix.shape, matrix.dtype.str)
        _validate_collective_signature(
            comm,
            matrix_header,
            "Network matrix-layout validation",
            root=self.root,
        )

        interface_fields = [
            variable
            for variable in variables
            if _is_network_interface_operand(variable)
        ]
        if not interface_fields:
            raise ValueError(
                "DtN condensation requires an explicit NetworkBasis vertex field."
            )

        full_column_mask = np.concatenate(
            [
                np.full(
                    subproblem.field_size(variable),
                    variable in interface_fields,
                    dtype=bool,
                )
                for variable in variables
            ]
        )
        full_row_mask = np.concatenate(
            [
                np.full(
                    subproblem.field_size(equation["eqn"]),
                    _is_network_interface_operand(equation["eqn"]),
                    dtype=bool,
                )
                for equation in equations
            ]
        )

        interface_columns = np.flatnonzero(
            _select_preconditioned(
                subproblem.pre_right_pinv,
                full_column_mask,
                "right preconditioner",
            )
        )
        interface_rows = np.flatnonzero(
            _select_preconditioned(
                subproblem.pre_left,
                full_row_mask,
                "left preconditioner",
            )
        )
        if interface_rows.size != interface_columns.size:
            raise ValueError(
                "DtN condensation found different numbers of vertex equations "
                f"({interface_rows.size}) and interface unknowns "
                f"({interface_columns.size})."
            )

        all_rows = np.arange(matrix.shape[0])
        all_columns = np.arange(matrix.shape[1])
        edge_rows = np.setdiff1d(all_rows, interface_rows, assume_unique=True)
        edge_columns = np.setdiff1d(
            all_columns,
            interface_columns,
            assume_unique=True,
        )
        if edge_rows.size != edge_columns.size:
            raise ValueError(
                "The non-interface block is not square and cannot be condensed."
            )

        self.interface_rows = interface_rows
        self.interface_columns = interface_columns
        self.edge_rows = edge_rows
        self.edge_columns = edge_columns

        full_column_labels = np.concatenate(
            [
                _network_element_labels(variable, subproblem)
                for variable in variables
            ]
        )
        full_row_labels = np.concatenate(
            [
                _network_element_labels(equation["eqn"], subproblem)
                for equation in equations
            ]
        )
        full_column_vertices = _network_interface_labels(
            variables,
            subproblem,
            self.network.num_vertices,
        )
        full_row_vertices = _network_interface_labels(
            [equation["eqn"] for equation in equations],
            subproblem,
            self.network.num_vertices,
        )
        minimal_column_labels = _select_preconditioned(
            subproblem.pre_right_pinv,
            full_column_labels,
            "right preconditioner",
        )
        minimal_row_labels = _select_preconditioned(
            subproblem.pre_left,
            full_row_labels,
            "left preconditioner",
        )
        edge_column_labels = minimal_column_labels[edge_columns]
        edge_row_labels = minimal_row_labels[edge_rows]
        minimal_column_vertices = _select_preconditioned(
            subproblem.pre_right_pinv,
            full_column_vertices,
            "right preconditioner",
        )
        minimal_row_vertices = _select_preconditioned(
            subproblem.pre_left,
            full_row_vertices,
            "left preconditioner",
        )
        interface_column_vertices = minimal_column_vertices[interface_columns]
        interface_row_vertices = minimal_row_vertices[interface_rows]

        metadata_signature = (
            _array_signature(interface_rows),
            _array_signature(interface_columns),
            _array_signature(edge_row_labels),
            _array_signature(edge_column_labels),
            _array_signature(interface_row_vertices),
            _array_signature(interface_column_vertices),
        )
        _validate_collective_signature(
            comm,
            metadata_signature,
            "Network interface-layout validation",
            root=self.root,
        )

        if np.any(edge_row_labels < 0) or np.any(edge_column_labels < 0):
            raise ValueError(
                "All non-interface equations and unknowns must belong to an edge."
            )
        if comm.size == 1:
            _validate_independent_edge_blocks(
                matrix,
                minimal_row_labels,
                minimal_column_labels,
                solver.entry_cutoff,
            )
        else:
            edge_structure_error = None
            if comm.rank == self.root:
                try:
                    _validate_independent_edge_blocks(
                        matrix,
                        minimal_row_labels,
                        minimal_column_labels,
                        solver.entry_cutoff,
                    )
                except Exception as error:
                    edge_structure_error = (
                        f"{type(error).__name__}: {error}"
                    )
            _raise_collective_error(
                comm,
                edge_structure_error,
                "Edge structure validation",
            )
        edge_sizes = np.bincount(
            edge_row_labels,
            minlength=self.network.num_edges,
        ).astype(np.float64)
        # Sparse direct factorization cost is problem-dependent.  Cubic block
        # size is a conservative default that strongly avoids oversized edges;
        # callers can provide any graph partition explicitly.
        edge_weights = edge_sizes**3
        if comm.rank == self.root:
            try:
                resolved_partition = _resolve_partition(
                    self.network,
                    partition,
                    comm.size,
                    edge_weights,
                )
                partition_payload = (
                    np.asarray(resolved_partition.edge_owners),
                    None,
                )
            except Exception as error:
                partition_payload = (
                    None,
                    f"{type(error).__name__}: {error}",
                )
        else:
            partition_payload = None
        edge_owners, partition_error = comm.bcast(
            partition_payload,
            root=self.root,
        )
        if partition_error is not None:
            raise ValueError(
                f"Failed to construct the network partition: {partition_error}"
            )
        self.partition = NetworkPartition(
            self.network,
            edge_owners,
            num_parts=comm.size,
        )
        self.owned_edges = self.partition.edges[comm.rank]

        edge_label_error = None
        if (
            np.any(edge_row_labels >= self.network.num_edges)
            or np.any(edge_column_labels >= self.network.num_edges)
        ):
            edge_label_error = "Non-interface edge labels exceed the network."
        _raise_collective_error(
            comm,
            edge_label_error,
            "Edge-label validation",
        )
        edge_row_owners = self.partition.edge_owners[edge_row_labels]
        edge_column_owners = self.partition.edge_owners[edge_column_labels]
        self.local_edge_rows = np.flatnonzero(edge_row_owners == comm.rank)
        self.local_edge_columns = np.flatnonzero(
            edge_column_owners == comm.rank
        )
        if comm.size == 1:
            interface_matrix = matrix[
                interface_rows
            ][:, interface_columns].tocsr()
            local_matrix = matrix[
                edge_rows[self.local_edge_rows]
            ][:, edge_columns[self.local_edge_columns]].tocsr()
            local_edge_to_interface = matrix[
                edge_rows[self.local_edge_rows]
            ][:, interface_columns].tocsr()
            self.local_interface_to_edge = matrix[
                interface_rows
            ][:, edge_columns[self.local_edge_columns]].tocsr()
        else:
            matrix_distribution_error = None
            if comm.rank == self.root:
                try:
                    interface_matrix = matrix[
                        interface_rows
                    ][:, interface_columns].tocsr()
                    matrix_payloads = []
                    for rank in range(comm.size):
                        rank_edge_rows = np.flatnonzero(
                            edge_row_owners == rank
                        )
                        rank_edge_columns = np.flatnonzero(
                            edge_column_owners == rank
                        )
                        matrix_payloads.append(
                            (
                                matrix[
                                    edge_rows[rank_edge_rows]
                                ][
                                    :,
                                    edge_columns[rank_edge_columns],
                                ].tocsr(),
                                matrix[
                                    edge_rows[rank_edge_rows]
                                ][:, interface_columns].tocsr(),
                                matrix[
                                    interface_rows
                                ][
                                    :,
                                    edge_columns[rank_edge_columns],
                                ].tocsr(),
                            )
                        )
                except Exception as error:
                    interface_matrix = None
                    matrix_payloads = None
                    matrix_distribution_error = (
                        f"{type(error).__name__}: {error}"
                    )
            else:
                interface_matrix = None
                matrix_payloads = None
            _raise_collective_error(
                comm,
                matrix_distribution_error,
                "Network matrix distribution",
            )
            interface_matrix = comm.bcast(
                interface_matrix,
                root=self.root,
            )
            (
                local_matrix,
                local_edge_to_interface,
                self.local_interface_to_edge,
            ) = comm.scatter(matrix_payloads, root=self.root)

        edge_factor_error = None
        try:
            self.edge_solver = _EdgeBlockFactorization(
                local_matrix,
                edge_row_labels[self.local_edge_rows],
                edge_column_labels[self.local_edge_columns],
                matsolver,
                solver,
            )
            self.edge_response = self.edge_solver.response(
                local_edge_to_interface
            )
        except Exception as error:
            self.edge_solver = None
            self.edge_response = None
            edge_factor_error = f"{type(error).__name__}: {error}"
        _raise_collective_error(
            comm,
            edge_factor_error,
            "Edge factorization",
        )

        # Each response now contains only this rank's edge unknowns.  Record
        # the packed collective order once so reconstructed fields can be
        # gathered without allocating a global zero contribution per rank.
        self.edge_column_counts = np.bincount(
            edge_column_owners,
            minlength=comm.size,
        ).astype(np.int64)
        self.packed_edge_columns = np.argsort(
            edge_column_owners,
            kind="stable",
        )
        edge_layout_error = None
        if not np.array_equal(
            self.edge_solver.local_columns,
            np.arange(self.local_edge_columns.size),
        ):
            edge_layout_error = (
                "The local edge solver does not cover every owned column."
            )
        _raise_collective_error(
            comm,
            edge_layout_error,
            "Packed edge-layout validation",
        )
        local_edge_schur = (
            -self.local_interface_to_edge @ self.edge_response
        ).tocsr()
        self.hierarchical = comm.size > 1
        if self.hierarchical:
            self.subgraph_solver = _SubgraphDtNFactorization(
                interface_matrix,
                local_edge_schur,
                interface_row_vertices,
                interface_column_vertices,
                self.partition,
                matsolver,
                solver,
                comm,
                root=self.root,
            )
            self.schur_matrix = self.subgraph_solver.schur_matrix
            self.interface_solver = self.subgraph_solver.separator_solver
        else:
            self.subgraph_solver = None
            edge_schur = _gather_sparse_sum(
                local_edge_schur,
                comm,
                root=self.root,
            )
            self.schur_matrix = (
                interface_matrix + edge_schur
            ).tocsr()
            self.interface_solver = matsolver(self.schur_matrix, solver)

    def solve(self, vector):
        """Solve through the condensed vertex system and reconstruct edges."""
        vector = np.asarray(vector)
        squeeze = vector.ndim == 1
        if squeeze:
            vector = vector[:, None]
        rhs_header = (vector.shape, vector.dtype.str)
        _validate_collective_signature(
            self.comm,
            rhs_header,
            "Network right-hand-side layout validation",
            root=self.root,
        )
        # The symbolic problem and user fields are replicated in this initial
        # backend.  Root is authoritative so rank-local initialization cannot
        # silently produce inconsistent right-hand sides.
        vector = np.ascontiguousarray(vector)
        self.comm.Bcast(vector, root=self.root)

        edge_rhs = vector[
            self.edge_rows[self.local_edge_rows]
        ]
        interface_rhs = vector[self.interface_rows]
        edge_solve_error = None
        try:
            free_edge_solution = self.edge_solver.solve(edge_rhs)
        except Exception as error:
            free_edge_solution = None
            edge_solve_error = f"{type(error).__name__}: {error}"
        _raise_collective_error(
            self.comm,
            edge_solve_error,
            "Edge solve",
        )
        local_rhs_correction = (
            -self.local_interface_to_edge @ free_edge_solution
        )
        if self.hierarchical:
            interface_solution = self.subgraph_solver.solve(
                interface_rhs,
                local_rhs_correction,
            )
        else:
            rhs_correction = np.empty_like(local_rhs_correction)
            self.comm.Reduce(
                local_rhs_correction,
                rhs_correction,
                op=MPI.SUM,
                root=self.root,
            )
            condensed_rhs = interface_rhs + rhs_correction
            interface_solution = _rhs_matrix(
                self.interface_solver.solve(condensed_rhs),
                self.interface_columns.size,
                vector.shape[1],
            )
        local_edge_solution = (
            free_edge_solution
            - self.edge_response @ interface_solution
        )
        packed_edge_solution = _allgatherv_packed_rows(
            local_edge_solution,
            self.edge_column_counts,
            self.comm,
        )

        solution = np.empty(
            vector.shape,
            dtype=np.result_type(
                vector.dtype,
                packed_edge_solution.dtype,
                interface_solution.dtype,
            ),
        )
        solution[
            self.edge_columns[self.packed_edge_columns]
        ] = packed_edge_solution
        solution[self.interface_columns] = interface_solution
        if squeeze:
            return solution[:, 0]
        return solution


def _has_network_role(operand, role):
    return any(
        getattr(basis, "_network_role", None) == role
        for basis in operand.domain.bases
    )


def _network_basis(operand):
    if not isinstance(operand, Operand):
        raise TypeError("Network traces require a Dedalus field or operator.")
    for basis in operand.domain.bases:
        network_basis = getattr(basis, "_network_basis", None)
        if (
            network_basis is not None
            and getattr(basis, "_network_role", None) == "edges"
        ):
            return network_basis
    raise ValueError("Operand does not use the edge basis of a NetworkBasis.")


def dirichlet_trace(operand):
    """Evaluate an edge field at both endpoints in canonical port order."""
    return _network_basis(operand).dtrace(operand)


def neumann_trace(operand):
    """Evaluate outward physical derivatives in canonical port order."""
    return _network_basis(operand).ntrace(operand)
