"""
Distributor, Layout, Transform, and Transpose class definitions.
"""

import logging
from mpi4py import MPI
from .network_storage import take_elements, CanonicalIndices

import numpy as np
import itertools
from contextlib import contextmanager
from collections import OrderedDict
from math import prod
import numbers
from weakref import WeakSet

from .coords import CoordinateSystem, DirectProduct
from ..tools.array import reshape_vector
from ..tools.cache import CachedMethod, CachedAttribute
from ..tools.config import config
from ..tools.general import OrderedSet
from ..tools.collective_metadata import CollectiveMetadata

logger = logging.getLogger(__name__.split('.')[-1])
GROUP_TRANSFORMS = config['transforms'].getboolean('GROUP_TRANSFORMS')
TRANSPOSE_LIBRARY = config['parallelism'].get('TRANSPOSE_LIBRARY')
GROUP_TRANSPOSES = config['parallelism'].getboolean('GROUP_TRANSPOSES')
SYNC_TRANSPOSES = config['parallelism'].getboolean('SYNC_TRANSPOSES')

if TRANSPOSE_LIBRARY.upper() == 'FFTW':
    from .transposes import FFTWTranspose as TransposePlanner
elif TRANSPOSE_LIBRARY.upper() == 'MPI':
    from .transposes import AlltoallvTranspose as TransposePlanner
from .transposes import RowDistributor, ColDistributor

# Public interface
__all__ = ['Distributor']


class Distributor:
    """
    Directs parallelized distribution and transformation of fields.

    Parameters
    ----------
    coordsystems : CoordinateSystem or sequence of CoordinateSystem
        Coordinate system or systems defining the field dimensions.  A
        :class:`~dedalus.core.network.Network` may be supplied directly.
    comm : MPI communicator, optional
        MPI communicator (default: comm world)
    mesh : tuple of ints, optional
        Process mesh for parallelization (default: 1-D mesh of available processes)
    dtype : dtype-like, optional
        Scalar data type for distributed fields.
    network_partition : optional
        Whole-edge ownership for a Network coordinate system.  Accepts
        ``"connected"``, ``"balanced_edges"``, an edge-ID mapping, an owner
        array, a callable, or a
        :class:`~dedalus.core.network.NetworkPartition`.  When provided, edge
        and port fields are stored only on their owning MPI rank.  Omitting it
        selects replicated compatibility storage.
    network_vertex_storage : {"replicated", "owned_ghost", "auto"}, optional
        Storage for vertex-basis fields when ``network_partition`` is present.
        ``"owned_ghost"`` stores authoritative vertices and locally incident
        endpoint ghosts.  ``"auto"`` selects owned/ghost storage when a
        partition is supplied and replicated storage otherwise.  The default
        remains ``"replicated"`` for compatibility.

    Attributes
    ----------
    comm_cart : MPI communicator
        Cartesian MPI communicator over mesh
    coords : array of ints
        Coordinates in cartesian communicator
    layouts : list of layout objects
        Available layouts

    Notes
    -----
    Computations are parallelized by splitting D-dimensional data fields over
    an R-dimensional mesh of MPI processes, where R < D.  In coefficient space,
    we take the first R dimensions of the data to be distributed over the mesh,
    leaving the last (D-R) dimensions local.  To transform such a data cube to
    grid space, we loop backwards over the D dimensions, performing each
    transform if the corresponding dimension is local, and performing an MPI
    transpose with the next dimension otherwise.  This effectively bubbles the
    first local dimension up from the (D-R)-th to the first dimension,
    transforming to grid space along the way.  In grid space, then, the first
    dimensional is local, followed by R dimensions distributed over the mesh,
    and the last (D-R-1) dimensions local.

    The distributor object for a given dimension constructs layout objects
    describing each of the (D+R+1) layouts (sets of transform/distribution
    states) and the paths between them (D transforms and R transposes).
    """

    def __init__(
        self,
        coordsystems,
        comm=None,
        mesh=None,
        dtype=None,
        network_partition=None,
        network_vertex_storage=None,
    ):
        if comm is None:
            comm = MPI.COMM_WORLD
        # Network PDEs use one of two explicit storage modes.  The compatibility
        # mode keeps complete symbolic fields on every process while retaining
        # the user communicator for distributed DtN factorizations.  Supplying
        # a partition opts into whole-edge field storage on that communicator.
        replicated_network = getattr(
            coordsystems,
            "_dedalus_replicated_distributor",
            False,
        )
        local_input = bool(getattr(coordsystems, '_local_input', False))
        if network_vertex_storage is None:
            network_vertex_storage = 'auto' if local_input else 'replicated'
        if replicated_network:
            local_modes = comm.allgather(local_input)
            if any(mode != local_modes[0] for mode in local_modes):
                raise RuntimeError('All ranks must use the same Network input mode')
        if local_input:
            if MPI.Comm.Compare(comm, coordsystems._input_comm) not in (MPI.IDENT, MPI.CONGRUENT):
                raise ValueError('Local Network communicator must preserve its input rank order')
            if network_partition is not None and network_partition is not coordsystems._local_partition:
                raise ValueError('Local Network input already supplies its partition')
            network_partition = coordsystems._local_partition
            if network_vertex_storage == 'replicated':
                raise ValueError("Local Network input requires network_vertex_storage='owned_ghost' or 'auto'")
        if replicated_network and comm.size > 1:
            storage_signatures = comm.allgather(
                (
                    isinstance(network_vertex_storage, str),
                    network_vertex_storage
                    if isinstance(network_vertex_storage, str)
                    else repr(network_vertex_storage),
                )
            )
            if any(
                signature != storage_signatures[0]
                for signature in storage_signatures[1:]
            ):
                raise RuntimeError(
                    "Network Distributor construction requires the same "
                    "network_vertex_storage option on every MPI rank."
                )
        valid_vertex_storage = {"replicated", "owned_ghost", "auto"}
        if (
            not isinstance(network_vertex_storage, str)
            or network_vertex_storage not in valid_vertex_storage
        ):
            raise ValueError(
                "network_vertex_storage must be 'replicated', "
                "'owned_ghost', or 'auto'."
            )
        if not replicated_network and network_vertex_storage != "replicated":
            raise ValueError(
                "network_vertex_storage is only valid for a Network "
                "distributor."
            )
        requested_network_vertex_storage = network_vertex_storage
        # Auto selects the scalable vertex layout only when whole-edge
        # ownership has also been requested.  Keep the explicit default
        # replicated so existing distributed scripts retain their field
        # shapes unless they opt into this policy.
        if network_vertex_storage == "auto":
            network_vertex_storage = (
                "owned_ghost"
                if network_partition is not None
                else "replicated"
            )
        if not replicated_network and network_partition is not None:
            raise ValueError(
                "network_partition is only valid for a Network distributor."
            )
        if replicated_network and comm.size > 1:
            partition_modes = comm.allgather(network_partition is not None)
            if any(mode != partition_modes[0] for mode in partition_modes[1:]):
                raise RuntimeError(
                    "Network Distributor construction must either provide "
                    "network_partition on every MPI rank or omit it on every "
                    "rank."
                )
        distributed_network = (
            replicated_network and network_partition is not None
        )
        if network_vertex_storage == "owned_ghost" and not distributed_network:
            raise ValueError(
                "network_vertex_storage='owned_ghost' requires a "
                "network_partition."
            )
        network = coordsystems if replicated_network else None
        if replicated_network:
            if mesh is not None:
                raise ValueError(
                    "Network distributors do not accept a Dedalus process mesh."
                )
            network_comm = comm
            if comm.size > 1 and not local_input:
                from .network import (
                    _array_signature,
                    _network_id_signature,
                    _network_vertex_groups_signature,
                    _validate_collective_signature,
                )

                network_signature = (
                    network.num_vertices,
                    network.num_edges,
                    tuple(network.names),
                    _array_signature(network.endpoints),
                    _array_signature(network.lengths),
                    _network_id_signature(network.vertex_ids),
                    _network_id_signature(network.edge_ids),
                    _network_vertex_groups_signature(network.vertex_groups),
                )
                validation_stage = (
                    "Distributed Network topology validation"
                    if distributed_network
                    else "Network topology validation"
                )
                _validate_collective_signature(
                    comm,
                    network_signature,
                    validation_stage,
                )

            if local_input:
                resolved_network_partition = network._local_partition
                layout_comm = comm
            elif distributed_network:
                from .network import NetworkPartition, _resolve_partition

                if comm.rank == 0:
                    try:
                        resolved_partition = _resolve_partition(
                            network,
                            network_partition,
                            comm.size,
                            np.ones(network.num_edges),
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
                    root=0,
                )
                if partition_error is not None:
                    raise ValueError(
                        "Failed to construct the distributed Network layout: "
                        f"{partition_error}"
                    )
                # Root already validated and built this exact owner table.
                # Other ranks build their local copy after the broadcast.
                resolved_network_partition = (
                    resolved_partition if comm.rank == 0 else NetworkPartition(
                        network,
                        edge_owners,
                        num_parts=comm.size,
                    )
                )
                layout_comm = comm
            else:
                resolved_network_partition = None
                layout_comm = MPI.COMM_SELF
        else:
            network_comm = None
            resolved_network_partition = None
            layout_comm = comm
        # A Network is a public coordinate container, while its replicated
        # symbolic layouts still operate on the two scalar coordinates.
        if replicated_network:
            coordsystems = tuple(coordsystems.coords)
        # Accept single coordsys in place of tuple/list
        elif not isinstance(coordsystems, (tuple, list)):
            coordsystems = (coordsystems,)
        # Note if only a single coordsys for simplicity
        if len(coordsystems) == 1:
            self.single_coordsys = coordsystems[0]
        else:
            self.single_coordsys = False
        # Get coords
        self.coords = sum((cs.coords for cs in coordsystems), ())
        self.coordsystems = coordsystems
        # Defaults
        if mesh is None:
            mesh = np.array([layout_comm.size], dtype=int)
        else:
            if isinstance(mesh, list) or isinstance(mesh, tuple):
                mesh = np.array(mesh, dtype=int)
        # Trim trailing ones
        mesh = 1 + np.trim_zeros(mesh - 1, trim='b')
        self.dim = dim = len(self.coords)
#        self.dim = dim = sum(coordsys.dim for coordsys in coordsystems)
        self.comm = comm
        self.network_comm = network_comm
        self.network = network
        self.network_partition = resolved_network_partition
        self.distributed_network = distributed_network
        self.requested_network_vertex_storage = (
            requested_network_vertex_storage
        )
        self.network_vertex_storage = network_vertex_storage
        if network_vertex_storage == "owned_ghost":
            from .network import NetworkVertexExchangePlan

            self.network_vertex_exchange_plan = NetworkVertexExchangePlan(
                resolved_network_partition,
                layout_comm,
            )
        else:
            self.network_vertex_exchange_plan = None
        self.mesh = mesh = np.array(mesh)
        # Check mesh compatibility
        logger.debug('Mesh: %s' %str(mesh))
        if mesh.size >= dim:
            raise ValueError("Mesh (%s) must have lower dimension than distributor (%i)" %(mesh, dim))
        if prod(mesh) != layout_comm.size:
            raise ValueError("Wrong number of processes (%i) for specified mesh (%s)" %(layout_comm.size, mesh))
        self.dtype = dtype
        self.layout_comm = layout_comm
        # Replicated Network inputs are synchronized once at the outer edge of
        # each expression evaluation.  The depth counter keeps recursive
        # Future evaluation from broadcasting derived intermediate fields.
        self._network_evaluation_depth = 0
        self._network_replica_metadata = CollectiveMetadata()
        self._network_vertex_metadata = CollectiveMetadata()
        # Create cartesian communicator, ignoring axes with m=1
        reduced_mesh = [m for m in mesh if m > 1]
        self.comm_cart = layout_comm.Create_cart(reduced_mesh)
        self.comm_coords = np.array(self.comm_cart.coords, dtype=int)
        # Build layout objects
        if distributed_network:
            self._build_network_layouts()
        else:
            self._build_layouts()
        # Keep set of weak field references
        self.fields = WeakSet()

    @CachedAttribute
    def cs_by_axis(self):
        cs_dict = {}
        for cs in self.coordsystems:
            for subaxis in range(cs.dim):
                axis = self.get_axis(cs)
                cs_dict[axis+subaxis] = cs
        return cs_dict

    def get_coordsystem(self, axis):
        return self.cs_by_axis[axis]

    def _build_layouts(self, dry_run=False):
        """Construct layout objects."""
        D = self.dim
        R = np.sum(self.mesh > 1)
        # First layout: full coefficient space
        local = np.array([True] * D)
        local[:self.mesh.size][self.mesh > 1] = False
        grid_space = [False] * D
        layout_0 = Layout(self, local, grid_space)
        layout_0.index = 0
        # Layout and path lists
        self.layouts = [layout_0]
        self.paths = []
        self.transforms = []
        # Subsequent layouts
        for i in range(1, R+D+1):
            # Iterate backwards over bases to last coefficient space basis
            for d in reversed(range(D)):
                if not grid_space[d]:
                    # Transform if local
                    if local[d]:
                        grid_space[d] = True
                        layout_i = Layout(self, local, grid_space)
                        if not dry_run:
                            path_i = Transform(self.layouts[-1], layout_i, d)
                            self.transforms.insert(0, path_i)
                        break
                    # Otherwise transpose
                    else:
                        local[d] = True
                        local[d+1] = False
                        layout_i = Layout(self, local, grid_space)
                        if not dry_run:
                            path_i = Transpose(self.layouts[-1], layout_i, d, self.comm_cart)
                        break
            layout_i.index = i
            self.layouts.append(layout_i)
            if not dry_run:
                self.paths.append(path_i)
        # Directly reference coefficient and grid space layouts
        self.coeff_layout = self.layouts[0]
        self.grid_layout = self.layouts[-1]
        # Allow string references to coefficient and grid space layouts
        self.layout_references = {'c': self.coeff_layout,
                                  'g': self.grid_layout}

    def _build_network_layouts(self):
        """Build whole-edge layouts without distributed transposes."""
        if self.dim != 2:
            raise ValueError("Distributed Network layouts require two axes.")
        edge_axis_local = self.layout_comm.size == 1
        local = np.array([edge_axis_local, True], dtype=bool)

        coeff_layout = Layout(self, local, (False, False))
        coeff_layout.index = 0
        interval_layout = Layout(self, local, (False, True))
        interval_layout.index = 1
        grid_layout = Layout(self, local, (True, True))
        grid_layout.index = 2

        interval_transform = Transform(coeff_layout, interval_layout, 1)
        cardinal_transform = Transform(interval_layout, grid_layout, 0)
        self.layouts = [coeff_layout, interval_layout, grid_layout]
        self.paths = [interval_transform, cardinal_transform]
        self.transforms = [cardinal_transform, interval_transform]
        self.coeff_layout = coeff_layout
        self.grid_layout = grid_layout
        self.layout_references = {
            'c': self.coeff_layout,
            'g': self.grid_layout,
        }

    def network_local_elements(
        self,
        layout,
        domain,
        scales,
        rank=None,
        broadcast=False,
    ):
        """Return logical elements under whole-edge Network ownership."""
        if not self.distributed_network:
            raise ValueError("Distributor does not use distributed Network layouts.")
        if rank is None:
            rank = self.layout_comm.rank
        if rank < 0 or rank >= self.layout_comm.size:
            raise ValueError("Network layout rank is outside the communicator.")
        global_shape = layout.global_shape(domain, scales)
        elements = []
        for axis, (axis_size, basis) in enumerate(
            zip(global_shape, domain.full_bases)
        ):
            if axis != 0:
                elements.append(np.arange(axis_size))
                continue
            if basis is None:
                # Domains with no Network cardinal basis are replicated on
                # the Network axis.  This includes full-network scalars and
                # position-only intermediates produced by reducing over
                # edges.  The latter must remain present on every rank so a
                # subsequent interval reduction retains every rank's local
                # matrix contribution.
                elements.append(np.arange(axis_size))
                continue
            network_basis = getattr(basis, "_network_basis", None)
            basis_network = getattr(basis, "_network", None)
            if basis_network is None and network_basis is not None:
                basis_network = network_basis.network
            if (
                basis_network is None
                or basis_network is not self.network
            ):
                raise ValueError(
                    "Distributed Network fields require bases belonging to "
                    "the Distributor's Network."
                )
            axis_elements = self._network_axis_elements(
                basis, rank, self.network_partition,
                self.network_vertex_storage, axis_size,
            )
            # Preserve the public result's independent, writable-array behavior.
            elements.append(axis_elements.copy())
        return tuple(elements)

    @CachedMethod
    def _network_axis_elements(self, basis, rank, partition, storage, axis_size):
        """Cache immutable cardinal selections independently of interval scales."""
        role = getattr(basis, "_network_role", None)
        if isinstance(basis._network_elements, CanonicalIndices):
            if role == 'edges':
                axis_elements = partition.edges[rank].copy()
            elif role == 'ports':
                edges = partition.edges[rank]
                axis_elements = np.column_stack((2*edges, 2*edges+1)).ravel()
            elif role == 'vertices' and storage == 'owned_ghost':
                axis_elements = partition.resident_vertices[rank].copy()
            else:
                axis_elements = np.arange(axis_size)
            axis_elements.setflags(write=False)
            return axis_elements
        if role == "edges":
            canonical_edges = np.asarray(
                basis._network_elements,
                dtype=int,
            )
            axis_elements = np.flatnonzero(
                partition.edge_owners[canonical_edges]
                == rank
            )
        elif role == "ports":
            canonical_ports = np.asarray(
                basis._network_elements,
                dtype=int,
            )
            port_owners = partition.edge_owners[
                (canonical_ports // 2)
            ]
            axis_elements = np.flatnonzero(port_owners == rank)
        elif role == "vertices":
            if storage == "owned_ghost":
                canonical_vertices = np.asarray(
                    basis._network_elements,
                    dtype=int,
                )
                resident_vertices = (
                    partition.resident_vertices[rank]
                )
                axis_elements = np.flatnonzero(
                    np.isin(
                        canonical_vertices,
                        resident_vertices,
                        assume_unique=True,
                    )
                )
            else:
                axis_elements = np.arange(axis_size)
        else:
            raise ValueError(
                "Distributed Network fields require an edge, port, or "
                "vertex basis on the network axis."
            )
        axis_elements = np.asarray(axis_elements, dtype=int)
        axis_elements.setflags(write=False)
        return axis_elements

    def network_axis_role(self, domain):
        """Return the storage role of a distributed Network field."""
        if not self.distributed_network:
            return None
        basis = domain.full_bases[0]
        if basis is None:
            return None
        network_basis = getattr(basis, "_network_basis", None)
        basis_network = getattr(basis, "_network", None)
        if basis_network is None and network_basis is not None:
            basis_network = network_basis.network
        if basis_network is None or basis_network is not self.network:
            raise ValueError(
                "Distributed Network fields require bases belonging to the "
                "Distributor's Network."
            )
        return getattr(basis, "_network_role", None)

    def network_axis_is_replicated(self, domain):
        """Whether a Network domain is complete on every Network rank."""
        if not self.distributed_network:
            return False
        basis = domain.full_bases[0]
        if basis is None:
            return True
        return (
            self.network_axis_role(domain) == "vertices"
            and self.network_vertex_storage == "replicated"
        )

    def network_axis_uses_owned_ghost(self, domain):
        """Whether a domain uses resident owned/ghost vertex storage."""
        return (
            self.distributed_network
            and self.network_vertex_storage == "owned_ghost"
            and self.network_axis_role(domain) == "vertices"
        )

    def network_vertex_canonical_elements(self, domain, rank=None):
        """Return canonical vertices represented by a local vertex domain."""
        if not self.network_axis_uses_owned_ghost(domain):
            raise ValueError(
                "Domain does not use owned/ghost Network vertex storage."
            )
        if rank is None:
            rank = self.layout_comm.rank
        basis = domain.full_bases[0]
        logical_elements = self.coeff_layout.local_elements(
            domain,
            scales=1,
            rank=rank,
        )[0]
        return take_elements(basis._network_elements, logical_elements)

    def synchronize_network_vertex_ghosts(self, field):
        """Refresh the ghost values of one resident vertex field in place."""
        if not self.network_axis_uses_owned_ghost(field.domain):
            return field
        canonical = self.network_vertex_canonical_elements(field.domain)
        if not np.array_equal(
            canonical,
            self.network_vertex_exchange_plan.resident,
        ):
            raise NotImplementedError(
                "Owned/ghost exchange currently requires the canonical full "
                "Network vertex basis."
            )
        data_axis = len(field.tensorsig)
        self.network_vertex_exchange_plan.forward(
            field.data,
            axis=data_axis,
        )
        return field

    def _network_replica_signature(self, field):
        """Return rank-independent metadata for one replicated input field."""
        basis_signature = []
        for basis in field.domain.full_bases:
            if basis is None:
                basis_signature.append(None)
            else:
                basis_signature.append(
                    (
                        type(basis).__module__,
                        type(basis).__qualname__,
                        getattr(basis, "_network_role", None),
                        tuple(basis.shape),
                    )
                )
        return (
            None if field.name is None else str(field.name),
            tuple(coordsys.dim for coordsys in field.tensorsig),
            np.dtype(field.dtype).str,
            tuple(basis_signature),
            tuple(self.coeff_layout.global_shape(field.domain, scales=1)),
        )

    def synchronize_network_replicas(self, fields):
        """Broadcast replicated Network input fields from authoritative rank 0.

        ``fields`` must have deterministic expression-tree order on every
        rank.  A single collective signature check runs before any field-wise
        broadcasts so divergent task trees fail collectively instead of
        entering mismatched broadcasts.  Rank zero's layout and scales are
        propagated with its data, allowing a previous idle-rank evaluation to
        have left replicas in different layouts.
        """
        if not self.distributed_network or self.layout_comm.size == 1:
            return

        replicas = OrderedSet()
        for field in fields:
            role = None
            if field.dist is self:
                role = self.network_axis_role(field.domain)
            # Anonymous, position-only fields are Dedalus's evaluation-local
            # polynomial/additive intermediates (for example the two Jacobi
            # lift polynomials).  They are constructed independently on every
            # rank and must not be overwritten as authoritative user inputs.
            # A user-supplied position-only input can opt into synchronization
            # simply by giving the field a semantic name.
            position_only_input = (
                role is None
                and bool(field.domain.bases)
                and field.name is not None
            )
            if (
                field.dist is self
                and self.network_axis_is_replicated(field.domain)
                and (
                    role == "vertices"
                    or not field.domain.bases
                    or position_only_input
                )
            ):
                replicas.add(field)
        signatures = tuple(
            self._network_replica_signature(field) for field in replicas
        )
        self._network_replica_metadata.check(
            self.layout_comm, signatures,
            "Distributed Network evaluation received inconsistent replicated input fields",
        )
        if len(signatures) != len(set(signatures)):
            raise ValueError(
                "A distributed Network expression must give unique names "
                "to otherwise indistinguishable replicated fields."
            )

        if not replicas:
            return
        root = 0
        metadata = None
        if self.layout_comm.rank == root:
            metadata = [(field.layout.index, tuple(field.scales)) for field in replicas]
        metadata = self.layout_comm.bcast(metadata, root=root)
        groups = {}
        for field, (layout_index, scales) in zip(replicas, metadata):
            if self.layout_comm.rank != root:
                field.preset_scales(scales)
                field.preset_layout(self.layouts[layout_index])
            groups.setdefault(np.dtype(field.data.dtype).str, []).append(field)
        for group in groups.values():
            if len(group) == 1:
                self.layout_comm.Bcast(group[0].data, root=root)
                continue
            sizes = [field.data.size for field in group]
            buffer = np.empty(sum(sizes), dtype=group[0].data.dtype)
            offset = 0
            for field, size in zip(group, sizes):
                if self.layout_comm.rank == root:
                    buffer[offset:offset+size] = field.data.ravel()
                offset += size
            self.layout_comm.Bcast(buffer, root=root)
            if self.layout_comm.rank != root:
                offset = 0
                for field, size in zip(group, sizes):
                    field.data[...] = buffer[offset:offset+size].reshape(field.data.shape)
                    offset += size

    @contextmanager
    def network_evaluation(self, fields):
        """Synchronize Network inputs once around recursive evaluation."""
        outermost = self._network_evaluation_depth == 0
        self._network_evaluation_depth += 1
        try:
            if outermost:
                self.synchronize_network_replicas(fields)
            yield
        finally:
            self._network_evaluation_depth -= 1

    def get_layout_object(self, input):
        """Dereference layout identifiers."""
        if isinstance(input, Layout):
            return input
        else:
            return self.layout_references[input]

    def buffer_size(self, domain, scales, dtype):
        """Compute necessary buffer size (bytes) for all layouts."""
        return max(layout.buffer_size(domain, scales, dtype) for layout in self.layouts)

    def remedy_scales(self, scales):
        """Remedy different scale inputs."""
        if scales is None:
            scales = 1
        if isinstance(scales, numbers.Number):
            scales = (scales,) * self.dim
        scales = tuple(scales)
        if 0 in scales:
            raise ValueError("Scales must be nonzero.")
        return scales

    def get_transform_object(self, axis):
        return self.transforms[axis]

    def get_axis(self, coord):
        if isinstance(coord, CoordinateSystem):
            coord = coord.coords[0]
        return self.coords.index(coord)

    def get_basis_axis(self, basis):
        return self.get_axis(basis.coordsys.coords[0])

    def first_axis(self, basis):
        return self.get_basis_axis(basis)

    def last_axis(self, basis):
        return self.first_axis(basis) + basis.dim - 1

    def Field(self, *args, **kw):
        """Alternate constructor for fields."""
        from .field import Field
        return Field(self, *args, **kw)

    def ScalarField(self, *args, **kw):
        """Alternate constructor for scalar fields."""
        from .field import ScalarField
        return ScalarField(self, *args, **kw)

    def VectorField(self, *args, **kw):
        """Alternate constructor for vector fields."""
        from .field import VectorField
        return VectorField(self, *args, **kw)

    def TensorField(self, *args, **kw):
        """Alternate constructor for tensor fields."""
        from .field import TensorField
        return TensorField(self, *args, **kw)

    def IdentityTensor(self, coordsys_in, coordsys_out=None, bases=None, dtype=None):
        """Identity tensor field."""
        if coordsys_out is None:
            coordsys_out = coordsys_in
        from .field import TensorField
        I = TensorField(self, (coordsys_out, coordsys_in), bases=bases, dtype=dtype)
        if coordsys_in is coordsys_out:
            for i in range(coordsys_in.dim):
                I['g'][i, i] = 1
        elif isinstance(coordsys_in, DirectProduct) and (coordsys_out in coordsys_in.coordsystems):
            i0 = coordsys_in.subaxis_by_cs[coordsys_out]
            for i in range(coordsys_out.dim):
                I['g'][i, i0+i] = 1
        elif isinstance(coordsys_out, DirectProduct) and (coordsys_in in coordsys_out.coordsystems):
            i0 = coordsys_out.subaxis_by_cs[coordsys_in]
            for i in range(coordsys_in.dim):
                I['g'][i0+i, i] = 1
        else:
            raise ValueError("Unsupported coordinate systems.")
        return I

    def local_grid(self, basis, scale=None):
        # TODO: remove from bases and do it all here?
        if scale is None:
            scale = 1
        if basis.dim == 1:
            return basis.local_grid(self, scale=scale)
        else:
            raise ValueError("Use `local_grids` for multidimensional bases.")

    # def global_grids(self, *bases, scales=None):
    #     """Global grids."""
    #     grids = []
    #     scales = self.remedy_scales(scales)
    #     for basis in bases:
    #         basis_axis = self.get_basis_axis(basis)
    #         basis_scales = scales[basis_axis:basis_axis+basis.dim]
    #         global_grids = basis.global_grids(scales=basis_scales)
    #         for subaxis in range(basis.dim):
    #             axis = basis_axis + subaxis
    #             grids.append(reshape_vector(global_grids[subaxis], dim=self.dim, axis=axis))
    #     return tuple(grids)

    # def local_grids(self, *bases, scales=None):
    #     """Local grid."""
    #     grids = []
    #     scales = self.remedy_scales(scales)
    #     for basis in bases:
    #         basis_axis = self.get_basis_axis(basis)
    #         basis_scales = scales[basis_axis:basis_axis+basis.dim]
    #         local_elements = self.grid_layout.local_elements(basis.domain(self), scales=scales)
    #         global_grids = basis.global_grids(scales=basis_scales)
    #         for subaxis in range(basis.dim):
    #             axis = basis_axis + subaxis
    #             local_grid = global_grids[subaxis][local_elements[axis]]
    #             grids.append(reshape_vector(local_grid, dim=self.dim, axis=axis))
    #     return tuple(grids)

    def local_grids(self, *bases, scales=None):
        scales = self.remedy_scales(scales)
        grids = []
        for basis in bases:
            basis_scales = scales[self.first_axis(basis):self.last_axis(basis)+1]
            grids.extend(basis.local_grids(self, scales=basis_scales))
        return grids

    def local_modes(self, basis):
        # TODO: remove from bases and do it all here?
        return basis.local_modes(self)

    @CachedAttribute
    def default_nonconst_groups(self):
        return sum((cs.default_nonconst_groups for cs in self.coordsystems), ())


class Layout:
    """
    Object describing the data distribution for a given transform and
    distribution state.

    Attributes
    ----------
    local : array of bools
        Axis locality flags (True/False for local/distributed)
    grid_space : array of bools
        Axis grid-space flags (True/False for grid/coeff space)

    """

    def __init__(self, dist, local, grid_space):
        self.dist = dist
        # Freeze local and grid_space lists into boolean arrays
        self.local = np.array(local)
        self.grid_space = np.array(grid_space)
        # Extend mesh and coordinates to distributor dimension
        self.ext_mesh = np.ones(dist.dim, dtype=int)
        reduced_mesh = [m for m in dist.mesh if m > 1]
        self.ext_mesh[~self.local] = reduced_mesh
        self.ext_coords = np.zeros(dist.dim, dtype=int)
        self.ext_coords[~self.local] = dist.comm_coords

    def global_shape(self, domain, scales):
        """Global data shape."""
        scales = self.dist.remedy_scales(scales)
        #global_shape = np.array(domain.coeff_shape).copy()
        #global_shape[self.grid_space] = np.array(domain.grid_shape(scales))[self.grid_space]
        global_shape = domain.global_shape(self, scales)
        return tuple(global_shape)

    def axis_is_local(self, domain, axis):
        """Whether an axis is complete for a particular field domain."""
        if (
            self.dist.distributed_network
            and axis == 0
            and self.dist.network_axis_is_replicated(domain)
        ):
            return True
        return bool(self.local[axis])

    def chunk_shape(self, domain):
        """Chunk shape."""
        #scales = self.dist.remedy_scales(scales)
        #chunk_shape = np.array(domain.coeff_group_shape).copy()
        #chunk_shape[self.grid_space] = 1
        chunk_shape = domain.chunk_shape(self)
        return tuple(chunk_shape)

    def group_shape(self, domain):
        """Chunk shape."""
        return tuple(domain.group_shape(self))

    def local_chunks(self, domain, scales, rank=None, broadcast=False):
        """Local chunk indices by axis."""
        if self.dist.distributed_network:
            elements = self.dist.network_local_elements(
                self,
                domain,
                scales,
                rank=rank,
                broadcast=broadcast,
            )
            chunk_shape = self.chunk_shape(domain)
            return tuple(
                np.unique(axis_elements // chunk_size)
                for axis_elements, chunk_size in zip(elements, chunk_shape)
            )
        global_shape = self.global_shape(domain, scales)
        chunk_shape = self.chunk_shape(domain)
        chunk_nums = -(-np.array(global_shape) // np.array(chunk_shape))  # ceil
        local_chunks = []
        # Get coordinates
        if rank is None:
            ext_coords = self.ext_coords
        else:
            ext_coords = np.zeros(self.dist.dim, dtype=int)
            ext_coords[~self.local] = self.dist.comm_cart.Get_coords(rank)
        # Get chunks axis by axis
        for axis, basis in enumerate(domain.full_bases):
            if self.local[axis]:
                # All chunks for local dimensions
                local_chunks.append(np.arange(chunk_nums[axis]))
            else:
                # Block distribution otherwise
                mesh = self.ext_mesh[axis]
                if broadcast and (basis is None):
                    coord = 0
                else:
                    coord = ext_coords[axis]
                block = -(-chunk_nums[axis] // mesh)
                start = min(chunk_nums[axis], block*coord)
                end = min(chunk_nums[axis], block*(coord+1))
                local_chunks.append(np.arange(start, end))
        return tuple(local_chunks)

    def global_elements(self, domain, scales):
        """Global element indices by axis."""
        global_shape = self.global_shape(domain, scales)
        indices = [np.arange(n) for n in global_shape]
        return tuple(indices)

    def local_elements(self, domain, scales, rank=None, broadcast=False):
        """Local element indices by axis."""
        if self.dist.distributed_network:
            return self.dist.network_local_elements(
                self,
                domain,
                scales,
                rank=rank,
                broadcast=broadcast,
            )
        chunk_shape = self.chunk_shape(domain)
        local_chunks = self.local_chunks(domain, scales, rank=rank, broadcast=broadcast)
        indices = []
        for chunk_size, chunks in zip(chunk_shape, local_chunks):
            ax_indices = chunk_size*np.repeat(chunks, chunk_size) + np.tile(np.arange(chunk_size), len(chunks))
            indices.append(ax_indices)
        return tuple(indices)

    @CachedMethod
    def valid_elements(self, tensorsig, domain, scales, rank=None, broadcast=False):
        """Make dense array of mode inclusion."""
        # Make dense array of local elements
        elements = self.local_elements(domain, scales, rank=rank, broadcast=broadcast)
        elements = np.array(np.meshgrid(*elements, indexing='ij'))
        # Check validity basis-by-basis
        grid_space = self.grid_space
        vshape = tuple(cs.dim for cs in tensorsig) + elements[0].shape
        valid = np.ones(shape=vshape, dtype=bool)
        for basis in domain.bases:
            basis_axes = slice(self.dist.first_axis(basis), self.dist.last_axis(basis)+1)
            valid &= basis.valid_elements(tensorsig, grid_space[basis_axes], elements[basis_axes])
        return valid

    def _group_arrays(self, elements, domain):
        # Convert to groups basis-by-basis
        grid_space = self.grid_space
        groups = np.zeros_like(elements)
        groups = np.ma.masked_array(groups)
        for basis in domain.bases:
            basis_axes = slice(self.dist.first_axis(basis), self.dist.last_axis(basis)+1)
            groups[basis_axes] = basis.elements_to_groups(grid_space[basis_axes], elements[basis_axes])
        return groups

    @CachedMethod
    def local_group_arrays(self, domain, scales, rank=None, broadcast=False):
        """Dense array of local groups (first axis)."""
        # Make dense array of local elements
        elements = self.local_elements(domain, scales, rank=rank, broadcast=broadcast)
        elements = np.array(np.meshgrid(*elements, indexing='ij'))
        return self._group_arrays(elements, domain)

    @CachedMethod
    def global_group_arrays(self, domain, scales):
        """Dense array of local groups (first axis)."""
        # Make dense array of local elements
        elements = self.global_elements(domain, scales)
        elements = np.array(np.meshgrid(*elements, indexing='ij'))
        return self._group_arrays(elements, domain)

    @CachedMethod
    def local_groupsets(self, group_coupling, domain, scales, rank=None, broadcast=False):
        local_groupsets = self.local_group_arrays(domain, scales, rank=rank, broadcast=broadcast).astype(object)
        # Replace non-enumerated axes with None
        for axis in range(local_groupsets.shape[0]):
            if group_coupling[axis]:
                local_groupsets[axis] = None
        # Flatten local groupsets
        local_groupsets = local_groupsets.reshape((local_groupsets.shape[0], -1))
        # Drop masked groups
        local_groupsets = np.ma.compress_cols(local_groupsets)
        # Return unique groupsets
        local_groupsets = tuple(map(tuple, local_groupsets.T))
        return OrderedSet(local_groupsets)

    @CachedMethod
    def local_groupset_slices(self, groupset, domain, scales, rank=None, broadcast=False):
        groups = self.local_group_arrays(domain, scales, rank=rank, broadcast=broadcast)
        dim = groups.shape[0]
        group_shape = self.group_shape(domain)
        # find all elements which match group
        selections = np.ones(groups[0].shape, dtype=int)
        for i, subgroup in enumerate(groupset):
            if subgroup is not None:
                selections *= (subgroup == groups[i])
                # Note: seems to exclude masked elements for ==, unlike other comparisons
        # determine which axes to loop over, which to find bounds for
        slices = []
        for i, subgroup in enumerate(groupset):
            if subgroup is None:
                subslices = [slice(None)]
            else:
                # loop over axis i but taking into account group_shape
                subslices = [slice(j, j+group_shape[i]) for j in range(0, groups.shape[i+1], group_shape[i])]
            slices.append(subslices)
        group_slices = []
        for s in itertools.product(*slices):
            sliced_selections = selections[tuple(s)]
            if np.any(sliced_selections): # some elements match group
                # assume selected groups are cartesian product, find left and right bounds
                lefts = list(map(np.min, np.where(sliced_selections)))
                rights = list(map(np.max, np.where(sliced_selections)))
                # build multidimensional group slice
                group_slice = []
                for i in range(dim):
                    if s[i] != slice(None):
                        group_slice.append(s[i])
                    else:
                        group_slice.append(slice(lefts[i], rights[i]+1))
                group_slices.append(tuple(group_slice))
        return group_slices

    def slices(self, domain, scales):
        """Local element slices by axis."""
        local_elements = self.local_elements(domain, scales)
        slices = []
        for LE in local_elements:
            if LE.size:
                if (
                    self.dist.distributed_network
                    and not np.array_equal(
                        LE,
                        np.arange(LE[0], LE[0] + LE.size),
                    )
                ):
                    raise NotImplementedError(
                        "A noncontiguous distributed Network layout cannot be "
                        "represented by slices; use local_elements instead."
                    )
                slices.append(slice(LE.min(), LE.max()+1))
            else:
                slices.append(slice(0, 0))
        return tuple(slices)

    @CachedMethod
    def local_shape(self, domain, scales, rank = None):
        """Local data shape."""
        local_elements = self.local_elements(domain, scales, rank = rank)
        shape = tuple(LE.size for LE in local_elements)
        return shape

    def buffer_size(self, bases, scales, dtype):
        """Local buffer size (bytes)."""
        local_shape = self.local_shape(bases, scales)
        return prod(local_shape) * np.dtype(dtype).itemsize

    # def local_group_index(self, group, domain, scales):
    #     """Index of a group within local groups."""
    #     index = []
    #     for grp, local_grps in zip(group, self.local_groups(domain, scales)):
    #         if grp is None:
    #             index.append(None)
    #         else:
    #             index.append(local_grps.index(grp))
    #     return index

        # if distribution == 'block':
        #     index[~local] = (group - start)[~local]
        # elif distribution == 'cyclic':
        #     index[~local] = np.mod(group, mesh)[~local]

    # def group(self, local_index):
    #     pass
    #     # if distribution == 'block':
    #     #     group = start + index
    #     # elif distribution == 'cyclic':
    #     #     group = mesh *

    # @CachedMethod
    # def groups(self, domain, scales):
    #     """Comptue group sizes."""
    #     groups = []
    #     for axis, space in enumerate(domain.spaces):
    #         if space is None:
    #             groups.append(1)
    #         elif self.grid_space[axis]:
    #             groups.append(1)
    #         else:
    #             groups.append(space.group_size)
    #     return np.array(groups, dtype=int)

    # @CachedMethod
    # def blocks(self, domain, scales):
    #     """Compute block sizes for data distribution."""
    #     global_shape = self.global_shape(domain, scales)
    #     groups = self.groups(domain, scales)
    #     return groups * np.ceil(global_shape / groups / self.ext_mesh).astype(int)

    # @CachedMethod
    # def start(self, domain, scales):
    #     """Compute starting coordinates for local data."""
    #     blocks = self.blocks(domain, scales)
    #     start = self.ext_coords * blocks
    #     start[domain.constant] = 0
    #     return start

    # @CachedMethod
    # def local_shape(self, domain, scales):
    #     """Compute local data shape."""
    #     global_shape = self.global_shape(domain, scales)
    #     blocks = self.blocks(domain, scales)
    #     start = self.start(domain, scales)
    #     local_shape = np.minimum(blocks, global_shape-start)
    #     local_shape = np.maximum(0, local_shape)
    #     return local_shape

    # @CachedMethod
    # def slices(self, domain, scales):
    #     """Compute slices for selecting local portion of global data."""
    #     start = self.start(domain, scales)
    #     local_shape = self.local_shape(domain, scales)
    #     return tuple(slice(s, s+l) for (s, l) in zip(start, local_shape))



class Transform:
    """
    Directs spectral transforms between two layouts.

    TODO:
        - Implement grouped transforms
    """

    def __init__(self, layout0, layout1, axis):
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis

    def increment(self, fields):
        """Backward transform a list of fields."""
        if len(fields) == 1:
            self.increment_single(*fields)
        elif GROUP_TRANSFORMS:
            self.increment_group(fields)
        else:
            for field in fields:
                self.increment_single(field)

    def decrement(self, fields):
        """Forward transform a list of fields."""
        if len(fields) == 1:
            self.decrement_single(*fields)
        elif GROUP_TRANSFORMS:
            self.decrement_group(fields)
        else:
            for field in fields:
                self.decrement_single(field)

    def increment_single(self, field):
        """Backward transform a field."""
        axis = self.axis
        basis = field.domain.full_bases[axis]
        # Reference views from both layouts
        cdata = field.data
        field.preset_layout(self.layout1)
        gdata = field.data
        # Transform non-constant bases with local data
        if (basis is not None) and prod(cdata.shape):
            basis.backward_transform(field, axis, cdata, gdata)
            #basis.backward_transform(cdata, gdata, axis, field.scales[axis], field.tensorsig)
            #plan = basis.transform_plan(cdata.shape, self.axis, field.scales[self.axis], field.dtype)
            #plan.backward(cdata, gdata)

    def decrement_single(self, field):
        """Forward transform a field."""
        axis = self.axis
        basis = field.domain.full_bases[axis]
        # Reference views from both layouts
        gdata = field.data
        field.preset_layout(self.layout0)
        cdata = field.data
        # Transform non-constant bases with local data
        if (basis is not None) and prod(gdata.shape):
            basis.forward_transform(field, axis, gdata, cdata)
            #basis.forward_transform(gdata, cdata, axis, field.scales[axis], field.tensorsig)
            #plan = basis.transform_plan(cdata.shape, self.axis, field.scales[self.axis], field.dtype)
            #plan.forward(gdata, cdata)

    def increment_group(self, fields):
        """Backward transform multiple fields simultaneously."""
        #logger.warning("Group transforms not implemented.")
        for field in fields:
            self.increment_single(field)

    def decrement_group(self, fields):
        """Forward transform multiple fields simultaneously."""
        #logger.warning("Group transforms not implemented.")
        for field in fields:
            self.decrement_single(field)

    # @CachedMethod
    # def group_data(self, nfields, scales):
    #     local_shape0 = self.layout0.local_shape(scales)
    #     local_shape1 = self.layout1.local_shape(scales)
    #     group_shape0 = [nfields] + list(local_shape0)
    #     group_shape1 = [nfields] + list(local_shape1)
    #     group_cdata = fftw.create_array(group_shape0, self.layout0.dtype)
    #     group_gdata = fftw.create_array(group_shape1, self.layout1.dtype)
    #     return group_cdata, group_gdata

    # def increment_group(self, fields):
    #     fields = list(fields)
    #     scales = fields[0].meta[:]['scale']
    #     cdata, gdata = self.group_data(len(fields), scales)
    #     for i, field in enumerate(fields):
    #         np.copyto(cdata[i], field.data)
    #     self.basis.backward(cdata, gdata, self.axis+1, fields[0].meta[self.axis])
    #     for i, field in enumerate(fields):
    #         field.layout = self.layout1
    #         np.copyto(field.data, gdata[i])

    # def decrement_group(self, fields):
    #     fields = list(fields)
    #     scales = fields[0].meta[:]['scale']
    #     cdata, gdata = self.group_data(len(fields), scales)
    #     for i, field in enumerate(fields):
    #         np.copyto(gdata[i], field.data)
    #     self.basis.forward(gdata, cdata, self.axis+1, fields[0].meta[self.axis])
    #     for i, field in enumerate(fields):
    #         field.layout = self.layout0
    #         np.copyto(field.data, cdata[i])


class Transpose:
    """
    Directs distributed transposes between two layouts.

    TODO:
        - Implement grouped transposes
        - Transpose all components simultaneously
    """

    def __init__(self, layout0, layout1, axis, comm_cart):
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        self.comm_cart = comm_cart
        # Create subgrid communicator along the moving mesh axis
        remain_dims = [0] * comm_cart.dim
        # No comm cart across axes where mesh = 1
        mesh = layout0.dist.mesh
        comm_cart_axis = axis - np.sum(mesh[:axis]==1)
        remain_dims[comm_cart_axis] = 1
        self.comm_sub = comm_cart.Sub(remain_dims)

    @CachedMethod
    def _sub_shape(self, domain, scales):
        """Build global shape of data assigned to sub-communicator."""
        local_shape = self.layout0.local_shape(domain, scales)
        global_shape = self.layout0.global_shape(domain, scales)
        # Global shape along transposing axes, local shape along others
        sub_shape = np.array(local_shape)
        sub_shape[self.axis] = global_shape[self.axis]
        sub_shape[self.axis+1] = global_shape[self.axis+1]
        return tuple(sub_shape)

    @CachedMethod
    def _plan(self, ncomp, sub_shape, chunk_shape, dtype):
        axis = self.axis
        if prod(sub_shape) == 0:
            return None  # no data
        elif (sub_shape[axis] == chunk_shape[axis]) and (sub_shape[axis+1] == chunk_shape[axis+1]):
            return None  # no change
        else:
            # Add axis for components
            full_sub_shape = (ncomp,) + sub_shape
            full_chunk_shape = (ncomp,) + chunk_shape
            return TransposePlanner(full_sub_shape, full_chunk_shape, dtype, axis+1, self.comm_sub)

    def _single_plan(self, field):
        """Build single transpose plan."""
        ncomp = prod([cs.dim for cs in field.tensorsig])
        sub_shape = self._sub_shape(field.domain, field.scales)
        chunk_shape = field.domain.chunk_shape(self.layout0)
        return self._plan(ncomp, sub_shape, chunk_shape, field.dtype)

    def _group_plans(self, fields):
        """Build group transpose plan."""
        # Segment fields by sub_shapes and chunk_shapes
        field_groups = OrderedDict()
        for field in fields:
            sub_shape = self._sub_shape(field.domain, field.scales)
            chunk_shape = field.domain.chunk_shape(self.layout0)
            if (sub_shape, chunk_shape) in field_groups:
                field_groups[(sub_shape, chunk_shape)].append(field)
            else:
                field_groups[(sub_shape, chunk_shape)] = [field]
        # Plan for each field group
        plans = []
        for (sub_shape, chunk_shape), fields in field_groups.items():
            ncomp = 0
            for field in fields:
                ncomp += prod([cs.dim for cs in field.tensorsig])
            plan = self._plan(ncomp, sub_shape, chunk_shape, field.dtype) # Assumes last field's dtype is good for everybody
            plans.append((fields, plan))
        return plans

    def increment(self, fields):
        """Backward transpose a list of fields."""
        if SYNC_TRANSPOSES:
            self.comm_sub.Barrier()
        if len(fields) == 1:
            self.increment_single(*fields)
        elif GROUP_TRANSPOSES:
            self.increment_group(fields)
        else:
            for field in fields:
                self.increment_single(field)

    def decrement(self, fields):
        """Forward transpose a list of fields."""
        if SYNC_TRANSPOSES:
            self.comm_sub.Barrier()
        if len(fields) == 1:
            self.decrement_single(*fields)
        elif GROUP_TRANSPOSES:
            self.decrement_group(fields)
        else:
            for field in fields:
                self.decrement_single(field)

    def plan_localize_columns(self, plan, data0, data1):
        plan.localize_columns(data0, data1)

    def increment_single(self, field):
        """Backward transpose a field."""
        plan = self._single_plan(field)
        if plan:
            # Reference views from both layouts
            data0 = field.data
            field.preset_layout(self.layout1)
            data1 = field.data
            # Transpose between data views
            self.plan_localize_columns(plan, data0, data1)
        else:
            # No communication: just update field layout
            field.preset_layout(self.layout1)

    def plan_localize_rows(self, plan, data0, data1):
        plan.localize_rows(data1, data0)

    def decrement_single(self, field):
        """Forward transpose a field."""
        plan = self._single_plan(field)
        if plan:
            # Reference views from both layouts
            data1 = field.data
            field.preset_layout(self.layout0)
            data0 = field.data
            # Transpose between data views
            self.plan_localize_rows(plan, data0, data1)
        else:
            # No communication: just update field layout
            field.preset_layout(self.layout0)

    def increment_group(self, fields):
        """Backward transpose multiple fields simultaneously."""
        plans = self._group_plans(fields)
        for fields, plan in plans:
            if plan:
                if len(fields) == 1:
                    field = fields[0]
                    # Reference views from both layouts
                    data0 = field.data
                    field.preset_layout(self.layout1)
                    data1 = field.data
                    # Transpose between data views
                    self.plan_localize_columns(plan, data0, data1)
                else:
                    # Gather data across fields
                    data0 = []
                    data1 = []
                    for field in fields:
                        rank = len(field.tensorsig)
                        # Reference views from both layouts
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data0.append(field.data.reshape(flat_comp_shape))
                        field.preset_layout(self.layout1)
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data1.append(field.data.reshape(flat_comp_shape))
                    if data0:
                        data0 = np.concatenate(data0)
                    else:
                        data0 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    if data1:
                        data1 = np.concatenate(data1)
                    else:
                        data1 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    # Transpose between data views
                    self.plan_localize_columns(plan, data0, data1)
                    # Split up transposed data
                    i = 0
                    for field in fields:
                        ncomp = prod([cs.dim for cs in field.tensorsig])
                        data = data1[i:i+ncomp]
                        field.data[:] = data.reshape(field.data.shape)
                        i += ncomp
            else:
                # No communication: just update field layouts
                for field in fields:
                    field.preset_layout(self.layout1)

    def decrement_group(self, fields):
        """Forward transpose multiple fields simultaneously."""
        plans = self._group_plans(fields)
        for fields, plan in plans:
            if plan:
                if len(fields) == 1:
                    field = fields[0]
                    # Reference views from both layouts
                    data1 = field.data
                    field.preset_layout(self.layout0)
                    data0 = field.data
                    # Transpose between data views
                    self.plan_localize_rows(plan, data0, data1)
                else:
                    # Gather data across fields
                    data0 = []
                    data1 = []
                    for field in fields:
                        rank = len(field.tensorsig)
                        # Reference views from both layouts
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data1.append(field.data.reshape(flat_comp_shape))
                        field.preset_layout(self.layout0)
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data0.append(field.data.reshape(flat_comp_shape))
                    if data0:
                        data0 = np.concatenate(data0)
                    else:
                        data0 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    if data1:
                        data1 = np.concatenate(data1)
                    else:
                        data1 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    # Transpose between data views
                    self.plan_localize_rows(plan, data0, data1)
                    # Split up transposed data
                    i = 0
                    for field in fields:
                        ncomp = prod([cs.dim for cs in field.tensorsig])
                        data = data0[i:i+ncomp]
                        field.data[:] = data.reshape(field.data.shape)
                        i += ncomp
            else:
                # No communication: just update field layouts
                for field in fields:
                    field.preset_layout(self.layout1)

    # def increment_group(self, *fields):
    #     """Transpose group from layout0 to layout1."""
    #     scales = unify(field.scales for field in fields)
    #     plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
    #     if plan:
    #         # Copy fields to group buffer
    #         for i, field in enumerate(fields):
    #             np.copyto(buffer0[i], field.data)
    #         # Transpose between group buffer views
    #         plan.localize_columns(buffer0, buffer1)
    #         # Copy from group buffer to fields in new layout
    #         for i, field in enumerate(fields):
    #             field.preset_layout(self.layout1)
    #             np.copyto(field.data, buffer1[i])
    #     else:
    #         # No data: just update field layouts
    #         for field in fields:
    #             field.preset_layout(self.layout1)

    # def decrement_group(self, *fields):
    #     """Transpose group from layout1 to layout0."""
    #     scales = unify(field.scales for field in fields)
    #     plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
    #     if plan:
    #         # Copy fields to group buffer
    #         for i, field in enumerate(fields):
    #             np.copyto(buffer1[i], field.data)
    #         # Transpose between group buffer views
    #         plan.localize_rows(buffer1, buffer0)
    #         # Copy from group buffer to fields in new layout
    #         for i, field in enumerate(fields):
    #             field.preset_layout(self.layout0)
    #             np.copyto(field.data, buffer0[i])
    #     else:
    #         # No data: just update field layouts
    #         for field in fields:
    #             field.preset_layout(self.layout0)
