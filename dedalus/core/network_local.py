"""Collective metric-graph construction from preassigned local edge shards."""
from types import MappingProxyType

from mpi4py import MPI
import numpy as np

from .basis import CardinalBasis
from .coords import Coordinate
from .network_storage import CanonicalIndices, LocalGraphTable


class _LocalGroups:
    def __init__(self, size, rank, values):
        self.size, self.rank = size, rank
        self.values = np.array(values, dtype=np.int64, copy=True)
        self.values.flags.writeable = False

    def __len__(self):
        return self.size

    def __getitem__(self, rank):
        if rank != self.rank:
            raise IndexError('A local graph stores only this rank\'s partition groups')
        return self.values


def from_local(cls, *, comm, num_vertices, num_edges, edge_indices, endpoints,
               lengths, coordinate_names):
    from .network import (NetworkPartition, _CanonicalIDs, _readonly_array,
                          _raise_collective_error, _validate_collective_signature,
                          _distributed_label_max)
    error = None
    try:
        if int(num_vertices) != num_vertices or int(num_edges) != num_edges:
            raise ValueError('Global graph counts must be integers')
        num_vertices, num_edges = int(num_vertices), int(num_edges)
        if num_vertices < 1 or num_edges < 1:
            raise ValueError('Require positive global graph counts')
        edge_indices = np.asarray(edge_indices)
        endpoints = np.asarray(endpoints)
        if edge_indices.ndim != 1 or (edge_indices.size and edge_indices.dtype.kind not in 'iu'):
            raise ValueError('Canonical edge indices must be integers')
        edge_indices = np.asarray(edge_indices, dtype=np.int64)
        if endpoints.size and endpoints.dtype.kind not in 'iu':
            raise ValueError('Canonical endpoint indices must be integers')
        endpoints = np.asarray(endpoints, dtype=np.int64).reshape((-1, 2))
        lengths = np.asarray(lengths, dtype=np.float64)
        if endpoints.shape != (edge_indices.size, 2) or lengths.shape != (edge_indices.size,):
            raise ValueError('Local edge arrays have inconsistent shapes')
        if (np.any(edge_indices < 0) or np.any(edge_indices >= num_edges)
                or np.any(edge_indices[1:] <= edge_indices[:-1])):
            raise ValueError('Local edge indices must be sorted, unique and in range')
        if np.any(endpoints < 0) or np.any(endpoints >= num_vertices):
            raise ValueError('Endpoint index out of range')
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError('Edge lengths must be finite and positive')
        names = tuple(map(str, coordinate_names))
        if len(names) != 2 or names[0] == names[1]:
            raise ValueError('Require two distinct coordinate names')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    _raise_collective_error(comm, error, 'Local Network input validation')
    _validate_collective_signature(comm, (num_vertices, num_edges, names), 'Local Network global header')
    occurrences = _distributed_label_max(comm, edge_indices, np.ones(edge_indices.size, dtype=np.int64), operation='sum')
    total_edges = comm.allreduce(edge_indices.size, op=MPI.SUM)
    error = None if total_edges == num_edges and np.all(occurrences == 1) else 'Edges must cover the global index range exactly once'
    _raise_collective_error(comm, error, 'Local Network edge coverage')

    vertices, local_degree = np.unique(endpoints, return_counts=True)
    metadata = _distributed_label_max(
        comm, vertices, np.column_stack((local_degree,
                                       np.full(vertices.size, comm.rank),
                                       np.full(vertices.size, comm.size-1-comm.rank))),
        operation=('sum', 'max', 'max'))
    degree, maximum, reversed_minimum = metadata.T
    minimum = comm.size - 1 - reversed_minimum
    owned = vertices[minimum == comm.rank]
    covered = comm.allreduce(owned.size, op=MPI.SUM)
    _raise_collective_error(comm, None if covered == num_vertices else
                            'Local Network input does not yet support isolated vertices', 'Local Network vertex coverage')
    local_separator = (minimum != maximum)
    owned_separator = local_separator & (minimum == comm.rank)
    separator_pairs = np.concatenate(comm.allgather(np.column_stack((vertices[owned_separator], minimum[owned_separator]))))
    separator_pairs = separator_pairs[np.argsort(separator_pairs[:, 0])]
    separator = separator_pairs[:, 0]
    vertex_labels = np.union1d(vertices, separator)
    equation_owners = np.empty(vertex_labels.size, dtype=np.int64)
    equation_owners[np.searchsorted(vertex_labels, vertices)] = minimum
    equation_owners[np.searchsorted(vertex_labels, separator)] = separator_pairs[:, 1]
    interior_owners = equation_owners.copy()
    interior_owners[np.searchsorted(vertex_labels, separator)] = -1
    total_length = comm.allreduce(float(lengths.sum()), op=MPI.SUM)

    obj = object.__new__(cls)
    obj.names = names
    obj.coords = (Coordinate(names[0]), Coordinate(names[1]))
    obj.edge_coord, obj.position_coord = obj.coords
    obj.dim = 2
    obj.curvilinear = False
    obj.default_nonconst_groups = (1, 1)
    obj.num_vertices, obj.num_edges = num_vertices, num_edges
    obj.num_ports = 2*num_edges
    obj.vertex_ids, obj.edge_ids = _CanonicalIDs(num_vertices), _CanonicalIDs(num_edges)
    obj.endpoints = LocalGraphTable(num_edges, edge_indices, endpoints)
    obj.lengths = LocalGraphTable(num_edges, edge_indices, lengths, global_sum=total_length)
    obj._degree = LocalGraphTable(num_vertices, vertices, degree)
    obj.total_length = total_length
    obj.vertex_groups = MappingProxyType({})
    obj.source_type = None
    obj._local_input = True
    obj._input_comm = comm
    basis = CardinalBasis(obj.edge_coord, size=num_vertices)
    basis._network, basis._network_role = obj, 'vertices'
    basis._network_elements = CanonicalIndices(num_vertices)
    obj._vertices_basis = basis

    partition = object.__new__(NetworkPartition)
    partition.network, partition.num_parts = obj, comm.size
    partition.edge_owners = LocalGraphTable(num_edges, edge_indices, np.full(edge_indices.size, comm.rank, dtype=np.int64))
    partition.vertex_equation_owners = LocalGraphTable(num_vertices, vertex_labels, equation_owners)
    partition.interior_vertex_owners = LocalGraphTable(num_vertices, vertex_labels, interior_owners)
    partition.separator_vertices = _readonly_array(separator, np.int64)
    groups = dict(edges=edge_indices, owned_vertices=owned, incident_vertices=vertices,
                  resident_vertices=vertices, ghost_vertices=vertices[minimum != comm.rank],
                  interior_vertices=vertices[(minimum == maximum) & (minimum == comm.rank)])
    for name, values in groups.items():
        setattr(partition, name, _LocalGroups(comm.size, comm.rank, values))
    obj._local_partition = partition
    obj._frozen = True
    return obj


def from_partition(cls, *, graph, partition, comm, root):
    """Scatter preassigned input from one rank; never broadcast the full graph."""
    from .network import (NetworkPartition, _network_id_signature, _CanonicalIDs,
                          _raise_collective_error, _validate_collective_signature)
    error = None
    try:
        if not isinstance(root, (int, np.integer)) or not 0 <= root < comm.size:
            raise ValueError('Input root must be an integer communicator rank')
        root = int(root)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    _raise_collective_error(comm, error, 'Partitioned Network input root')
    _validate_collective_signature(comm, root, 'Partitioned Network input root')
    header = shards = None
    error = None
    if comm.rank == root:
        try:
            net = cls(graph)
            if getattr(net, '_local_input', False):
                raise ValueError('Root input must describe the full graph')
            for ids, size in [(net.vertex_ids, net.num_vertices),
                              (net.edge_ids, net.num_edges)]:
                if _network_id_signature(ids) != _network_id_signature(_CanonicalIDs(size)):
                    raise ValueError('Local graph input currently requires canonical integer IDs')
            if isinstance(partition, NetworkPartition):
                if partition.network is not net or partition.num_parts != comm.size:
                    raise ValueError('Partition must belong to the supplied graph and communicator size')
                part = partition
            else:
                if partition is None or isinstance(partition, str) or callable(partition):
                    raise ValueError('Supply preassigned edge owners or a NetworkPartition')
                part = NetworkPartition(net, partition, num_parts=comm.size)
            header = (net.num_vertices, net.num_edges, net.names)
            shards = []
            for rank in range(comm.size):
                edges = np.asarray(part.edges[rank], dtype=np.int64)
                shards.append(dict(edge_indices=edges, endpoints=net.endpoints[edges],
                                   lengths=net.lengths[edges]))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    _raise_collective_error(comm, error, 'Partitioned Network root input')
    num_vertices, num_edges, names = comm.bcast(header, root=root)
    shard = comm.scatter(shards, root=root)
    # Release preparation arrays before collective graph validation. A caller
    # retaining its original input still owns that memory on the root rank.
    del shards
    return from_local(cls, comm=comm, num_vertices=num_vertices, num_edges=num_edges,
                      coordinate_names=names, **shard)
