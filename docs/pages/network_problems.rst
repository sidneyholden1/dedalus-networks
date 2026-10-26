PDEs on metric networks
***********************

Dedalus represents a metric network as oriented one-dimensional edges joined
at vertices.  The orientation fixes the order of endpoint data but does not
change an orientation-invariant PDE.  Each edge is discretized on the
reference interval :math:`s \in [0,1]`; its physical length is included by the
network derivative and integration measure.

The :doc:`complete network Poisson example <examples/lbvp_network_poisson>`
constructs a graph, declares fields and vertex equations, solves with the
Dirichlet-to-Neumann backend, and writes the solution using the ordinary
Dedalus analysis interface.

Public API
==========

The package-level ``dedalus.public`` namespace exports
:class:`~dedalus.core.network.Network`,
:class:`~dedalus.core.network.NetworkPartition`,
:class:`~dedalus.core.network.NetworkBasis`, and
:class:`~dedalus.core.network.NetworkBasisSet` as ``d3.Network``,
``d3.NetworkPartition``, ``d3.NetworkBasis``, and ``d3.NetworkBasisSet``.
The endpoint operators
:func:`dtrace <dedalus.core.operators.network_dirichlet_trace>` and
:func:`ntrace <dedalus.core.operators.network_neumann_trace>` are likewise
available as ``d3.dtrace`` and ``d3.ntrace``.

Constructing a network
======================

``Network`` accepts package-neutral arrays:

.. code-block:: python

    network = d3.Network(
        vertex_ids=("left", "junction", "right"),
        edge_ids=("short", "long"),
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition="connected",
        network_vertex_storage="auto",
    )
    basis = d3.NetworkBasis(network, size=32)
    edge_index, s = dist.local_grids(basis)
    x = basis.local_physical_grid(dist)

``endpoints[e] = (v0, v1)`` orients edge ``e`` from :math:`s=0` to
:math:`s=1`.  Vertex and edge IDs may be any unique hashable objects.  Loops
and parallel edges are retained.  The default coordinate names are ``("e",
"s")``; the second coordinate is always the normalized reference position.

A NetworkX graph or ``metricgraph.MetricGraph`` can instead be passed directly
as ``d3.Network(graph)``.  Other graph packages can implement
``__dedalus_network__()`` and return a mapping containing ``vertex_ids``,
``edge_ids``, ``endpoints``, and ``lengths``.  Dedalus snapshots this data, so
the numerical problem is independent of the originating graph package.

``Network`` contains topology and metric geometry, not PDE-specific material
data.  Store edge or vertex coefficients as ordinary Cardinal fields so they
can participate in Dedalus expressions and be changed independently of the
graph.  For example, edgewise conductivity and vertexwise Robin data can be
declared with:

.. code-block:: python

    kappa = dist.Field(name="kappa", bases=basis.edges)
    alpha = dist.Field(name="alpha", bases=basis.vertices)
    kappa.set_global_data(np.asarray(edge_kappa)[:, None])
    alpha.set_global_data(np.asarray(vertex_alpha)[:, None])

Use ``bases=basis`` instead when a coefficient varies with position along an
edge.  Physical edge lengths remain on ``Network`` because they define the
derivative scaling and integration measure.

Supplying ``network_partition`` to the distributor opts into whole-edge
distributed storage.  On one MPI rank this has the same field shapes as the
serial case.  On multiple ranks, every edge and its complete interval
discretization belong to one rank; ``edge_index`` and ``x`` therefore contain
only the locally owned edges, while the separable ``s`` grid is common to all
edges.  Omitting ``network_partition`` selects the replicated compatibility
mode described below.

Row ``x[i, :]`` is physical arclength from the tail of the edge at numeric
position ``i``; ``edge_index[i, 0]`` contains that ordinal.  In contrast,
``dist.local_grids(basis)`` follows the ordinary Dedalus convention and
returns the separable coordinate grids.  Its second result is therefore the
same array as ``dist.local_grid(basis.interval)``.  Global physical arclength
is available from ``basis.global_physical_grid(dist)``.  The earlier
``basis.physical_grid(dist)`` spelling remains a compatibility alias for
``basis.local_physical_grid(dist)``.

Preassigned local graph input
----------------------------

For large distributed graphs, ``Network.from_local`` avoids giving every rank
all of the graph's endpoints, lengths and edge owners. This optional collective
constructor takes the edges already assigned to the calling rank. For example,
a contiguous chain with ``q`` edges per rank can be constructed without a global
endpoint array:

.. code-block:: python

    from mpi4py import MPI
    import numpy as np
    import dedalus.public as d3

    comm = MPI.COMM_WORLD
    q = 2048
    edges = np.arange(q*comm.rank, q*(comm.rank+1), dtype=np.int64)
    network = d3.Network.from_local(
        comm=comm,
        num_vertices=q*comm.size+1,
        num_edges=q*comm.size,
        edge_indices=edges,
        endpoints=np.column_stack((edges, edges+1)),
        lengths=np.ones(q),
    )
    dist = d3.Distributor(network, comm=comm, dtype=np.float64,
                          network_vertex_storage="owned_ghost")
    basis = d3.NetworkBasis(network, size=32)

The constructor does not partition the graph. Each rank's sorted, unique
``edge_indices`` establish its ownership; together they must cover the global
integer range exactly once. Endpoint labels use the global vertex integer
range. The constructor checks coverage, lengths and matching global headers,
then establishes vertex degrees, owners and ghosts collectively. All ranks,
including ranks with no edges, must call it on the same communicator. The
associated distributor must use that communicator with the same rank order.
No separate ``network_partition`` should be supplied.

The graph has global logical dimensions, but its endpoint and length tables
contain only owned edges, and its degree table contains resident vertices.
Index these tables with local canonical labels, such as
``network.lengths[edges]``. Whole-table NumPy conversion and requests for absent
entries raise errors instead of silently allocating replicated data. Build
coefficient fields in local edge order and vertex fields in
``dist.network_partition.resident_vertices[comm.rank]`` order. Fill owned values
and use the existing vertex exchange interface when ghost updates are needed.
Default IDs are immutable integer sequences; code requiring a literal tuple can
explicitly call ``tuple(network.edge_ids)`` at the cost of global materialization.

This path currently supports canonical integer labels, graphs without isolated
vertices, and a single full ``NetworkBasis``. Loops, parallel edges, heterogeneous
positive lengths, idle ranks, and real-valued linear solves have regression
coverage. Small steady chain/grid/irregular cases and an evolving chain have
also been compared with the replicated input path. Explicit partial edge-basis
buckets, arbitrary external IDs and the graph metadata used by ordinary
analysis-file output/restart are not supported here. The existing ``Network``
constructor remains the supported path for those workflows; the optional
constructor is not yet a replacement for all scientific examples.

Local graph tables scale with owned edges and resident vertices plus the
separator, whose labels remain replicated. Separator vectors and rank-sized
MPI exchange metadata also remain. Thus this interface removes whole-graph
replication, but does not promise strictly constant memory at unlimited rank
counts or on graphs whose separators grow with the entire graph.

In both input modes, built-in topology maps now construct requested local
blocks lazily, and partition transport setup exchanges ghost requests rather
than building every other rank's groups. Constructing a
``NetworkVertexExchangePlan`` is collective and must occur in matching order
on all ranks. Full topology matrices remain available through the replicated
input API when explicitly requested.

Network fields and traces
=========================

``NetworkBasis`` combines three discrete Cardinal bases with an edge-interval
basis, which is Chebyshev by default:

* ``basis.edges`` indexes one value or one interval field per edge;
* ``basis.vertices`` indexes vertex unknowns;
* ``basis.ports`` indexes the two oriented endpoints of every edge.

The canonical port order is ``(edge 0 left, edge 0 right, edge 1 left, ...)``.
For an edge field ``u``, ``dtrace(u)`` returns its endpoint values in this
order.  ``ntrace(u)`` returns outward physical derivatives: ``-du/dx`` at a
left endpoint and ``+du/dx`` at a right endpoint.  Use
``dx = basis.derivative`` for physical arclength derivatives; it applies the
factor ``1 / network.lengths[e]`` to the reference derivative.  Generic
``d3.Differentiate(u, network.position_coord)`` intentionally returns
:math:`du/ds`, preserving the usual Dedalus meaning of differentiation with
respect to a coordinate.

Network PDE problem variables are currently scalar.  The two Network axes
``(edge, s)`` describe graph-index storage and position along an edge; they do
not form a two-component physical vector space.  Represent a coupled system
with one scalar ``Field`` per component.  ``VectorField`` and ``TensorField``
with a Network tensor signature remain available as data containers for
analysis and restart, but cannot yet be used as LBVP, NLBVP, IVP, or EVP
unknowns.

When a first-order system evolves an oriented edge flux ``q`` independently,
its outward endpoint values are the signed Dirichlet trace.  The signs are
available without another special operator:

.. code-block:: python

    port_sign = dist.Field(name="port_sign", bases=basis.ports)
    port_sign.set_global_data(network.port_signs[:, None])
    outward_q = port_sign * dtrace(q)
    problem.add_equation("R.T @ outward_q = source")

``network.port_signs`` is ``(-1, +1)`` for each oriented edge in canonical
port order.  This form is useful for coupled pressure--flux, wave, and
transmission-line systems; ``ntrace(u)`` remains the differentiated Neumann
trace of a primal field.

Heterogeneous edge discretizations
==================================

One ``NetworkBasis`` represents a dense rectangular edge-by-interval field:
all edges selected by that basis use the same interval family and resolution.
Use ``NetworkBasisSet`` when disjoint groups of edges need different families
or resolutions.  Its named buckets must form an exact cover of the canonical
edge table; overlapping or missing edge indices are rejected.  For example, a
four-edge network can mix Chebyshev and Legendre discretizations as follows:

.. code-block:: python

    from mpi4py import MPI
    import numpy as np
    from scipy import sparse

    bases = d3.NetworkBasisSet(
        network,
        {
            "coarse": {
                "edge_indices": (0, 2),
                "size": 12,
            },
            "fine": {
                "edge_indices": (1, 3),
                "size": 18,
                "interval_basis": d3.Legendre,
            },
        },
    )

    comm = MPI.COMM_WORLD
    partition = d3.NetworkPartition.connected(
        network,
        comm.size,
        weights=bases.edge_costs,
    )
    dist = d3.Distributor(
        network,
        comm=comm,
        dtype=np.float64,
        network_partition=partition,
        network_vertex_storage="auto",
    )
    coarse = bases["coarse"]
    fine = bases["fine"]

    u_coarse = dist.Field(name="u_coarse", bases=coarse)
    u_fine = dist.Field(name="u_fine", bases=fine)
    tau_c1 = dist.Field(name="tau_c1", bases=coarse.edges)
    tau_c2 = dist.Field(name="tau_c2", bases=coarse.edges)
    tau_f1 = dist.Field(name="tau_f1", bases=fine.edges)
    tau_f2 = dist.Field(name="tau_f2", bases=fine.edges)
    U = dist.Field(name="U", bases=bases.vertices)

    dt = d3.TimeDerivative
    lift_c_basis = coarse.interval.derivative_basis(2)
    lift_f_basis = fine.interval.derivative_basis(2)
    lift_c = lambda A, n: d3.Lift(A, lift_c_basis, n)
    lift_f = lambda A, n: d3.Lift(A, lift_f_basis, n)

    # Dirichlet leaves and Kirchhoff interior vertices on a four-edge chain.
    D = d3.CardinalMap(
        bases.vertices,
        bases.vertices,
        sparse.diags((1.0, 0, 0, 0, 1)),
        name="D",
    )
    K = d3.CardinalMap(
        bases.vertices,
        bases.vertices,
        sparse.diags((0.0, 1, 1, 1, 0)),
        name="K",
    )

    problem = d3.IVP(
        [u_coarse, u_fine, U, tau_c1, tau_c2, tau_f1, tau_f2],
        network_interfaces=[(u_coarse, U), (u_fine, U)],
        namespace=locals(),
    )
    problem.add_equation(
        "dt(u_coarse) - coarse.dx(coarse.dx(u_coarse)) "
        "+ lift_c(tau_c1,-1) + lift_c(tau_c2,-2) = 0"
    )
    problem.add_equation(
        "dt(u_fine) - fine.dx(fine.dx(u_fine)) "
        "+ lift_f(tau_f1,-1) + lift_f(tau_f2,-2) = 0"
    )
    problem.add_equation("continuity(u_coarse) = 0")
    problem.add_equation("continuity(u_fine) = 0")
    problem.add_equation(
        "D @ U + K @ (coarse.R.T @ coarse.ntrace(u_coarse) "
        "+ fine.R.T @ fine.ntrace(u_fine)) = 0"
    )

Each edge field remains a dense field on one bucket; ``NetworkBasisSet`` is a
collection and cannot itself be passed as ``bases`` to create a ragged or
padded field.  The buckets share the exact same full-vertex basis, so one
explicit ``U`` joins all traces.  Each bucket has its own PDE, tau fields, and
continuity equation, while the single vertex equation sums the outward fluxes
after mapping them to that shared vertex basis.  The explicit associations in
``network_interfaces`` are required here: creating separate automatic
interfaces would incorrectly give the two pieces independent vertex values.

``bases.edge_costs`` assigns the estimate ``size**3`` to every canonical edge
in its bucket.  Passing these costs to a partitioner accounts for unequal
dense spectral-factorization work.  ``basis.local_grids(dist)`` returns the
canonical edge indices for that bucket, which can be used to initialize or
interpret its field without reference to bucket-local numbering.  Arithmetic
within a bucket has the ordinary dense-field semantics; combine different
buckets only after mapping them to a common space such as ``bases.vertices``.
The :repo:`heterogeneous heat example <examples/ivp_network_heterogeneous_heat/heat.py>`
supplies a complete time-dependent problem and analytic check.

Physical measure
================

The edge Cardinal basis carries the physical quadrature weights ``lengths``.
Ordinary Dedalus integration, averaging, domain volume, and field norms
therefore use

.. math::

    d\mu(e,s) = L_e\,ds.

For example, ``d3.Integrate(u)`` and ``d3.Integrate(u, network)`` both compute
the full physical network integral, while ``d3.Average(u)`` divides by
``network.total_length``.  Integrating only over ``network.position_coord``
intentionally returns a per-edge reference integral in ``s``; the edge-length
Jacobian enters when the edge coordinate is also integrated.

Vertex coupling with equations
==============================

The sparse restriction map ``R = basis.restriction`` copies vertex values to
their incident ports.  Its transpose sums port data at each vertex.  An
explicit vertex field ``U`` therefore gives continuity and generalized Robin
or Kirchhoff conditions using ordinary Dedalus equations:

.. code-block:: python

    u = dist.Field(name="u", bases=basis)
    U = dist.Field(name="U", bases=basis.vertices)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)

    R = basis.restriction
    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    problem = d3.LBVP([u, U, tau_1, tau_2], namespace=locals())
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = f"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation(
        "alpha*U + beta*(R.T @ ntrace(u)) = g"
    )

The continuity equation supplies one equation per port while introducing one
``U`` unknown per vertex.  At a degree-:math:`d` vertex this leaves
:math:`d-1` independent equality conditions, exactly matching pairwise
continuity.  The final equation supplies the remaining vertex condition.
Pointwise ``alpha`` and ``beta`` fields can select Dirichlet, Kirchhoff, or
Robin behavior independently at every vertex.

For an edgewise diffusion field ``kappa``, write the edge operator as
``-dx(kappa*dx(u))`` and its outward port flux as
``dtrace(kappa)*ntrace(u)``.  ``kappa`` may be edge-constant or vary along each
edge; ``dtrace(kappa)`` returns its endpoint values.

Global constraints and pure-Neumann gauges
------------------------------------------

A basis-free scalar tau closes a full-network integral constraint in the same
way that a scalar pressure tau closes a gauge equation on an ordinary Dedalus
domain.  For example, a mean-zero pure-Neumann Poisson problem adds
``tau_gauge`` to the edge equation and one global equation:

.. code-block:: python

    tau_gauge = dist.Field(name="tau_gauge")

    problem = d3.LBVP(
        [u, U, tau_1, tau_2, tau_gauge],
        namespace=locals(),
    )
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) "
        "+ tau_gauge = f"
    )
    problem.add_equation("dtrace(u) - R @ U = 0")
    problem.add_equation("R.T @ ntrace(u) = g")
    problem.add_equation("integ(u) = 0")

Here ``tau_gauge`` has ``bases=None``: it is one global unknown, unlike the two
edgewise tau fields used by the interval lifts.  The DtN backend retains the
scalar and integral equation in its condensed interface system.  In a
whole-edge MPI layout the scalar is replicated with rank zero authoritative,
while every rank contributes its owned-edge part of ``integ(u)``.  The same
syntax is supported in implicit IVP stages and in NLBVP Newton iterations.
Use one independent global constraint and conjugate scalar tau for each
independent null direction; a connected pure-Neumann scalar problem has one.
``ave(u) = 0`` is the equivalent mean-zero form and can be used anywhere the
integral form is accepted.

Full-network reductions use the interval-then-edge order by default.  An
explicit edge-first expression such as
``integ(integ(u, network.edge_coord), network.position_coord)`` is also
supported: its position-only intermediate is replicated so contributions from
every edge owner survive the second reduction.  A global row that directly,
or after edge condensation, touches a vertex promotes that vertex to the
shared separator.  This preserves exact coupling to vertices that the graph
partition would otherwise classify as rank-interior, at the cost of enlarging
the separator for genuinely global constraints.

Problem-managed vertex interfaces
---------------------------------

For static vertex conditions, the vertex field can be managed by the problem
rather than declared explicitly.  Register each continuous edge field with
``network_interfaces`` and use the problem-local ``continuity`` and
``vertex_value`` helpers in ordinary equations:

.. code-block:: python

    problem = d3.LBVP(
        [u, tau_1, tau_2],
        network_interfaces=[u],
        namespace=locals(),
    )
    problem.add_equation(
        "-dx(dx(u)) + lift(tau_1,-1) + lift(tau_2,-2) = f"
    )
    problem.add_equation("continuity(u) = 0")
    problem.add_equation(
        "alpha*vertex_value(u) + beta*(R.T @ ntrace(u)) = g"
    )

This is exact shorthand for the explicit equations above.  The problem creates
a genuine vertex field internally and lowers the helpers to
``dtrace(u) - R @ U`` and ``U`` before matrix assembly.  Consequently the
direct and DtN solvers see the same algebra in either form.  Registration is
deliberately opt-in: for example, an independently evolved edge-flux field need
not be continuous.  Automatically managed interfaces require uniquely named
edge fields; the hidden checkpoint name is derived from that semantic name, so
reordering ``network_interfaces`` cannot silently rebind restart data.

The managed field remains accessible when initial data, analysis output, or a
dynamic vertex equation requires it:

.. code-block:: python

    u_vertex = problem.vertex_field(u)
    u_vertex["g"] = initial_vertex_data

    dt = d3.TimeDerivative
    problem.add_equation(
        "mass*dt(vertex_value(u)) + damping*vertex_value(u) "
        "+ R.T @ ntrace(u) = source"
    )

For coupled systems, list every independently continuous field, such as
``network_interfaces=[u, v]``.  Each receives its own vertex field, and
``vertex_value(u)`` and ``vertex_value(v)`` can be coupled in the same vertex
equations.  The fully explicit form remains available for unusual interface
physics.  An existing vertex field can also be associated with the helpers via
``network_interfaces=[(u, U)]`` when both fields already appear in the problem
variable list.

Dirichlet-to-Neumann solver
===========================

For linear and nonlinear boundary-value problems and implicit
initial-value-problem stages, build with ``network_solver="dtn"``.  The
equations are unchanged.  Dedalus identifies the explicit vertex field,
factors every independent edge block, and forms the exact discrete Schur
complement on vertex unknowns.  Repeated linear solves and fixed-timestep
stages reuse these factorizations until the left-hand-side matrices are
rebuilt.  Nonlinear boundary-value problems rebuild and condense the exact
Frechet Jacobian at every Newton iteration.

Vertex fields can also carry dynamic conditions in an IVP, producing a
coupled edge-PDE/vertex-ODE system rather than a static boundary condition.
For example, with scalar diffusivity ``kappa``:

.. code-block:: python

    dt = d3.TimeDerivative
    problem.add_equation(
        "mass*dt(U) + damping*U + kappa*(R.T @ ntrace(u)) = source"
    )

Whole-edge distributed storage
==============================

For the scalable MPI path, bind the partition when constructing the
distributor and do not repeat it when building the solver:

.. code-block:: python

    dist = d3.Distributor(
        network,
        dtype=np.float64,
        network_partition="connected",
        network_vertex_storage="auto",
    )
    basis = d3.NetworkBasis(network, size=32)
    # ... fields and equations as above ...
    solver = problem.build_solver(network_solver="dtn")

Edge and port fields then store only the entries owned by the local rank.  An
edge interval is indivisible: its transforms, tau variables, and numerical
matrix remain together with its edge.  With
``network_vertex_storage="owned_ghost"``, a vertex field stores the vertices
whose equations belong to that rank together with the endpoint ghosts needed
by its owned edges.  ``"auto"`` selects this layout when a partition is
present.  The explicit default, ``"replicated"``, retains the earlier full
vertex vector on every rank for compatibility.

Basis-free scalars and position-only fields with no Network Cardinal basis
remain replicated.  In replicated mode, rank zero is authoritative for
vertex fields, basis-free scalars, and named position-only input profiles at
an outer expression or evaluator boundary.  In owned/ghost mode, every vertex
has one canonical owner and its current value is forwarded only to incident
ranks.  Internal additive reduction intermediates are not overwritten by
either synchronization.  Prefer
``field.set_global_data(global_array)`` for partition-independent
initialization after selecting the intended layout and scales, and call it
with the same global-shape array on every rank.  For example, call
``field.change_layout("g")`` first when ``global_array`` contains grid values.
An edge or port field selects its locally owned entries, an owned/ghost vertex
field selects its resident entries and refreshes ghosts, and a replicated
field uses the rank-zero copy at evaluation.  Use
``field.allgather_data(layout)`` when analysis genuinely needs the canonical
global vertex array.  Different non-root values in a replicated input must not
be used to encode rank-local contributions.

Runtime Cardinal conversions preserve these ownership rules, while gathers,
output, and norms count every edge or vertex exactly once.  With owned/ghost
storage, persistent numerical data scale with the locally owned edges and
resident vertices; the complete symbolic graph and Network metadata still
exist on every rank.  The restriction maps exchange only endpoint halo data
between local ports and resident vertices.  Selected-vertex interpolation is
also collective-safe: the canonical owner supplies the value returned on all
ranks.

Vertex equations may freely combine direct terms and edge- or port-mediated
terms.  For example, all three contributions in
``A @ U + R.T @ ntrace(u) + R.T @ (w * (R @ U))`` have their natural meaning:
the direct vertex matrix ``A`` is counted once, while the locally owned port
contributions are summed across ranks.  Matrix assembly tracks this ownership
through the symbolic operator tree, so the same rule applies after implicit
IVP matrix combinations and timestep-dependent rebuilds.

A direct vertex matrix in owned/ghost mode must be topology-local: every
source needed by an owned equation row must lie in that rank's resident
endpoint halo.  Pointwise conditions, incidence-based maps, and coupling among
separator vertices satisfy this naturally.  A genuinely nonlocal custom map
must use replicated vertex storage, enlarge the halo in a future custom
partition implementation, or be expressed through a global reduction.

Likewise, an implicit ``CardinalMap`` directly between edge or port entries
must preserve edge ownership.  A map entry that reaches another rank is
rejected collectively rather than silently dropped.  Express inter-subgraph
coupling through vertex interface fields, or choose a partition that keeps the
coupled entries together.  Cross-owner maps used only in explicit/runtime
expressions retain their usual gathered-source semantics.

The parallel reduction is exact for the assembled spectral discretization and
proceeds hierarchically:

#. With distributed ``owned_ghost`` storage, each rank factors one sparse
   interior block containing its interval-spectral coefficients, tau variables,
   and internal vertex values. Vertices whose incident edges have the same
   owner are internal unless retained by a global constraint.
#. Vertices incident to edges owned by different ranks, plus any vertices
   promoted by a global constraint, form a compact separator.  The reduced
   separator is solved collectively.  Small separators are assembled and
   directly factored on every rank; larger separators remain additive
   rank-local sparse matrices throughout a root-free restarted GMRES solve.
#. The compact separator solution is available on every rank.  Each rank
   reconstructs its internal vertex and edge values together; it never
   materializes a persistent full vertex vector.

Thus the final edge field is not gathered as part of a solve.  Implicit IVP
matrix products likewise apply local edge blocks and reduce only their vertex
interface contributions.  The result is algebraically the same Schur
complement solve as eliminating all edge and internal-vertex unknowns from one
global matrix; the partition changes data placement and work, not the
discrete equations.

The default is ``network_interior_elimination="coupled"``. Pass
``network_interior_elimination="two_stage"`` to ``problem.build_solver`` to
first eliminate individual edge blocks and then internal vertices. This
fallback can be useful for regular networks with identical edge operators.
The option applies to distributed ``owned_ghost`` DtN solves; serial and
replicated-storage paths retain their existing elimination algorithms.
Both choices use the same separator-solver policy below. Symbolic coupling
in the coupled path conservatively retains vertices touched by global
constraints; exact numerical cancellation can make this separator larger
than in the two-stage path.

Separator solver options and diagnostics
----------------------------------------

The separator policy is configured with ordinary solver-construction
keywords.  The defaults are shown here for a boundary-value problem; an IVP
passes its timestepper in the usual way as well as these keywords:

.. code-block:: python

    solver = problem.build_solver(
        network_solver="dtn",
        network_separator_method="auto",
        network_interior_elimination="coupled",
        network_separator_direct_max_size=256,
        network_separator_restart=40,
        network_separator_maxiter=None,
        network_separator_rtol=1e-10,
        network_separator_atol=0.0,
        network_separator_preconditioner="jacobi",
    )

``network_separator_method`` may be ``"auto"``, ``"replicated_direct"``, or
``"gmres"``.  In ``"auto"`` mode, a separator whose dimension is no larger
than ``network_separator_direct_max_size`` is densely assembled and factored
on every rank.  Larger separators use collective GMRES without assembling a
global matrix on a distinguished rank.  Setting the threshold to zero selects
GMRES for every nonempty separator; the two explicit method names override
the threshold.

``network_separator_restart`` is the maximum Krylov dimension between GMRES
restarts.  ``network_separator_maxiter`` limits total iterations; ``None``
uses a size-dependent default.  ``network_separator_rtol`` and
``network_separator_atol`` test the explicitly recomputed residual according
to

.. math::

    \lVert b-Ax \rVert \leq
    \max\left(\mathtt{atol},\mathtt{rtol}\lVert b\rVert\right).

``network_separator_preconditioner`` selects the collective right
preconditioner: ``"jacobi"`` assembles the diagonal across ranks and
``"none"`` applies no preconditioner.  The GMRES-specific settings do not
alter a replicated direct solve.  All separator options are validated for
type, range, and agreement across the participating ranks before collective
work begins.

For a nonempty distributed separator, the DtN factorization's
``interface_solver.method`` records the resolved method and
``interface_solver.last_solve_info`` contains one immutable record per
right-hand side from the most recent solve.  Each record reports ``method``,
``converged``, ``iterations``, ``restart_cycles``, ``residual_norm``,
``relative_residual``, and a textual ``reason``.  A successful direct solve
has zero Krylov iterations and still reports its measured residual.  GMRES
returns only after the explicit residual satisfies the tolerance; failure to
converge raises collectively instead of returning an unconverged state.  An
empty separator has no ``interface_solver``.

The factorization's matrix diagnostic also depends on the method.  With
replicated direct factorization, ``schur_matrix`` is the globally assembled
separator matrix on every rank.  With GMRES, ``schur_matrix_is_local`` is true
and ``schur_matrix`` is only that rank's additive contribution; summing those
contributions would form the global matrix that the iterative solve avoids
assembling.

The built-in ``"connected"`` partition is a deterministic,
connectivity-aware region-growth heuristic.  It balances indivisible edges
while traversing vertex incidence without explicitly constructing the line
graph.  It remains a lightweight starting point rather than a replacement for
a large-scale graph partitioner.  Use ``"balanced_edges"`` for topology-blind
edge balancing, or pass ``NetworkPartition(network, edge_owners)``, an
edge-ID-to-rank mapping, a callable, or an object implementing
``__dedalus_network_partition__`` to supply an external partition.  The
separator size and edge-work imbalance should both be considered when
comparing partitions.

Scaling benchmark
=================

``benchmarks/network_dtn_scaling.py`` provides a reproducible MPI benchmark
for the whole-edge backend.  It constructs a deterministic graph and a
manufactured piecewise-linear solution, then reports separate maximum-rank
wall times for problem setup, matrix assembly, hierarchical DtN
factorization, the first solve, and repeated solves with cached factors.
The benchmark uses ``--vertex-storage owned_ghost`` by default; pass
``--vertex-storage replicated`` for a direct compatibility comparison.
Correctness is checked with relative manufactured-solution error and a
coefficient-space normwise backward error for the assembled system.  The
individual grid-space equation residuals remain in the record as diagnostics.
A rounded, partition-independent checksum is also recorded as a reproducibility
fingerprint; it is deliberately not used as a tolerance test because values
arbitrarily close to a rounding boundary can change the hash.

For strong scaling, keep the graph and edge resolution fixed while changing
the number of MPI ranks.  For example, run the following command at each
desired rank count:

.. code-block:: bash

    OMP_NUM_THREADS=1 PYTHONPATH=. mpiexec -n 4 python3 \
        benchmarks/network_dtn_scaling.py \
        --scaling strong --graph binary_tree --edges 4096 \
        --resolution 32 --partition connected --repeats 5 \
        --output strong.jsonl

For weak scaling, ``--edges-per-rank`` fixes the target local work and the
benchmark constructs the corresponding global graph:

.. code-block:: bash

    OMP_NUM_THREADS=1 PYTHONPATH=. mpiexec -n 4 python3 \
        benchmarks/network_dtn_scaling.py \
        --scaling weak --graph chain --edges-per-rank 512 \
        --resolution 32 --partition connected --repeats 5 \
        --output weak.jsonl

The graph families ``chain``, ``binary_tree``, ``star``, and ``cycle`` expose
different separator structures.  Comparing ``connected`` with
``balanced_edges`` also separates the effects of load balance and graph
locality.  Each invocation prints one JSON object on rank zero; ``--output``
appends that same record to a JSON Lines file.  Records include source commit
and dirty state, a benchmark-script hash, topology and edge-owner hashes,
separator and local-matrix sizes, solver settings, rank timing spreads, BLAS,
software and MPI versions, thread and scheduler binding settings, and the
command line.  Use the same graph, resolution, dtype, partition strategy, and
machine placement within a strong scaling series.  Set ``OMP_NUM_THREADS=1``
unless deliberately benchmarking a hybrid configuration.  Run multiple
independent invocations when estimating assembly and factorization
variability: ``--repeats`` samples only solves that reuse the cached factors.

Replicated compatibility mode
=============================

Omitting ``network_partition`` from ``Distributor`` retains complete edge and
port fields on every MPI rank.  The DtN solver can still distribute edge and
subgraph factorizations (and accepts the partition at solver construction),
but rank zero assembles the global numerical matrix and the solved edge state
is replicated.  This mode is useful for compatibility and small regression
problems; its per-rank field memory scales with the complete network.  Inputs
used for rank-local diagnostics must be initialized identically on every rank.

Output and restart
==================

Whole-edge distributed Network fields support the gather-mode HDF5 file
handler.  This is the default for a distributed Network, so the usual analysis
interface works without an additional parallel-output setting:

.. code-block:: python

    snapshots = solver.evaluator.add_file_handler(
        "snapshots",
        sim_dt=0.1,
    )
    snapshots.add_task(u, name="u")

Passing ``parallel="gather"`` explicitly is equivalent.  Each write gathers
fields by their global edge or port indices and writes the complete dataset on
rank zero.  The ``"virtual"`` and ``"mpio"`` handlers are not supported
because they assume contiguous local slices, whereas a graph partition may
own noncontiguous edges.  MPI file handlers also remain unavailable in the
replicated compatibility mode.

Every gathered Network file contains a partition-independent ``/network``
group.  It records the canonical endpoint table, physical edge lengths,
coordinate names, edge orientation, port ordering, outward-normal signs, and
the map from the stored reference coordinate :math:`s` to physical arclength
:math:`x_e = L_e s`.  Portable string, numeric, and tuple vertex and edge IDs
are stored as UTF-8 JSON.  Arbitrary Python-object IDs are intentionally not
serialized; canonical integer indices still make such files complete for
numerical reconstruction.  Each task records whether its two Network axes
represent edges, vertices, ports, position, or a constant direction.

For small custom diagnostics, all distributed ranks can instead gather a
field explicitly before rank zero writes it:

.. code-block:: python

    global_u = u.gather_data(root=0, layout="g")
    if dist.comm.rank == 0:
        np.save("u.npy", global_u)

``gather_data`` is collective and allocates the complete requested field only
on the root.  ``allgather_data`` is also available when every rank genuinely
needs a global diagnostic array.  Neither operation is used by the DtN solve
itself.

Gathered grid- or coefficient-space tasks can be loaded directly with the
ordinary restart interface:

.. code-block:: python

    write, dt = solver.load_state("snapshots/snapshots_s1.h5")

Network construction already requires topology, portable semantic IDs,
coordinate names, and vertex groups to agree across ranks.  Restart loading is
also collective for a Network using more than one MPI rank: metadata and task
validation failures are propagated to every rank before field data are read.

Distributed ranks read only the canonical edge or port indices that they own;
vertex fields are restored as the replicated interface vector.  The restart
partition need not match the partition used to write the file, so a run can be
continued with a different MPI size or a different edge assignment.  Before
loading data, Dedalus validates the stored topology, lengths, coordinate and
port conventions, task roles, and complete cardinal-axis sizes against the
active Network, so a same-shaped file from a different graph is rejected.

The DtN reduction requires an invertible edge problem for prescribed endpoint
values.  At a Dirichlet resonance of a Helmholtz or Schrödinger edge operator,
the local DtN map has a pole even when the fully coupled Robin problem is
regular.  The solver diagnoses exposed singular LU factors; use the direct
serial solver at such a parameter or, in future, a Robin-to-Robin or
nullspace-aware condensation.

Targeted distributed eigenvalue problems
=========================================

Whole-edge and replicated-compatibility MPI layouts support targeted right
eigenmodes through the ordinary sparse-EVP call:

.. code-block:: python

    solver = problem.build_solver()
    solver.solve_sparse(
        solver.subproblems[0],
        N=6,
        target=0.8,
        ncv=48,
        tol=1e-8,
    )

For the Dedalus pencil :math:`Lx=\lambda(-M)x`, the collective backend
factorizes the single shifted matrix :math:`L+\sigma M` at the requested
target :math:`\sigma` and applies

.. math::

    T = (L+\sigma M)^{-1}(-M)

inside a dependency-free MPI Arnoldi iteration.  The DtN map is therefore
formed only at the fixed shift; no eigenvalue-dependent nonlinear separator
problem is introduced.  Inner products count every owned edge degree of
freedom once and count each vertex-interface degree of freedom only on its
canonical owner (or the authoritative root in replicated storage).
``set_state`` scatters the resulting rank-local eigenvector and refreshes
endpoint ghosts through the normal field interface.

This first collective implementation is intentionally targeted and
unrestarted.  ``ncv`` is both the Arnoldi-space dimension and the iteration
limit, so increasing it trades memory for convergence.  Every returned mode
is checked with a collective relative residual of the original generalized
pencil.  Only ``which="LM"`` in shift-invert space and right eigenvectors are
currently supported; distributed dense and left-eigenvector solves are
rejected explicitly.  A real problem can set a real mode as state, but use
``dtype=np.complex128`` when the desired mode or eigenvalue may be complex.
The shifted global pencil and each edge Dirichlet block must be nonsingular at
the target.  Moving the target away from an eigenvalue or an edge Dirichlet
resonance resolves the usual shift-invert or DtN pole, respectively.

Current scope
=============

The optimized DtN path currently targets linear and nonlinear boundary-value
problems, implicit stages of initial-value problems, and the targeted
distributed right-eigenmode path described above.  Changing an implicit
timestep or taking a Newton iteration rebuilds the corresponding condensed
matrix.  Direct serial boundary-value solves remain available by passing
``network_solver=None`` in serial compatibility mode.  Serial Network
eigenvalue problems retain the direct dense and sparse paths.  Distributed
Network EVPs currently provide targeted, unrestarted right modes rather than
dense, left-mode, or general-selection eigensolvers.  The separator solve is
collective and root-free when GMRES is selected; only its compact Krylov
vectors are replicated.  Gather-based MPI analysis output and
partition-independent restart are supported for both replicated and
owned/ghost vertex layouts.

The current equation backend supports scalar fields and systems of coupled
scalar fields, including first-order pressure--flux formulations.  Geometric
``VectorField`` and ``TensorField`` calculus on a graph is not yet defined;
represent physical components as named scalar edge fields.  Every individual
``NetworkBasis`` uses one interval family and resolution over its selected
edge subset.  ``NetworkBasisSet`` supports mixed families and resolutions by
combining an exact cover of dense bucket fields through a shared vertex basis;
it does not create a single ragged field.

Default local input for partitioned simulations
----------------------------------------------

``Network.from_partition(graph, owners, comm=comm, root=0)`` is the convenience
entry point for an already partitioned graph. It returns local graph storage.
Only the root rank supplies ``graph`` and ``owners``; other ranks pass ``None``.
All ranks participate. ``owners`` may be a precomputed owner array or a
``NetworkPartition`` for that graph and communicator size. No partition search
is performed by this constructor. Root input errors are reported collectively.

.. code-block:: python

    # graph and owners are supplied on rank zero; None on the other ranks.
    network = d3.Network.from_partition(graph, owners, comm=comm)
    dist = d3.Distributor(network, comm=comm, dtype=np.float64)
    basis = d3.NetworkBasis(network, size=32)

For ``from_partition`` and ``from_local`` input, omitting
``network_vertex_storage`` selects owned/ghost storage automatically. Explicit
incompatible settings still raise an error. The full ``Network(...)`` data
constructor and its existing distributor behavior remain available as the
compatibility path; existing caller-owned graph objects are never mutated.
The local-input restrictions described above still apply. In particular,
ordinary scientific output/restart has not been migrated by changing defaults.

The convenience constructor temporarily stores the complete input and outgoing
pieces on root. Callers retaining that original graph retain its memory too.
Use preassigned per-rank input files for large runs to avoid that preparation
bottleneck. Neither input route broadcasts the complete graph to every rank.

The ``paired_full_sim.py`` benchmark now defaults to local input. With a saved
``*.owners.npy`` fixture, it reads the adjacent ``*.shards/manifest.json``;
``--local-input`` can supply another manifest explicitly. Missing prepared
shards are an error, not a silent fallback. ``partition_probe.py`` prepares
shards by default; ``--no-local-shards`` opts out for fixture-only diagnostics.
Without saved files, example graph generation and partition selection happen
on root and are included in simulation time. Published supplied-input timing
continues to use prepared files, excluding offline partition construction.
Use ``--graph-input full`` only to select the explicit full-input comparison.


Spatial partition preparation
-----------------------------

For embedded networks, ``NetworkPartition.spatial(network, parts, positions,
weights=estimated_edge_work)`` constructs connected regions using recursive
geometric bisection and fragment repair. ``positions`` has one row per vertex
in the full graph's vertex order. Weights must be positive; omitted weights
balance edge counts. Coefficient counts are a useful proxy when spectral orders
vary; lengths are appropriate when work scales with physical length.

This optional offline heuristic favors compact interfaces, but does not promise
an optimal cut or a fixed imbalance bound. Audit connectedness, actual work and
shared vertices. It does not override prescribed ownership or change defaults.
Prepare on the full graph, then use ``Network.from_partition`` or saved local
shards for the distributed simulation.
