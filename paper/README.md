# Paper examples and figure reproduction

These are executable examples for *A scalable computational framework for
PDEs on physical networks*. Each entry
point includes its required scientific helper modules here. The library is the
`dedalus/` source in this checkout.

The prepared companion release contains the solver, example scripts, reproduction
inputs/settings, figure data (bundled or linked), plotting code and instructions.
Here, `paper/` means code accompanying the paper. The manuscript LaTeX sources,
manuscript bibliography/build files and complete paper PDF are excluded from
this release. The manuscript will be archived separately and linked by citation.

The default runs build and solve or advance each example, then check its
numerical results. The companion archive bundles the saved figure data,
plotting scripts and exact production settings. See
[REPRODUCING.md](REPRODUCING.md) to redraw all five figures from saved results,
and [release-checks.json](release-checks.json) for the validation record.

| Paper material | Start here | Check performed by the default run |
| --- | --- | --- |
| Methods: beam code listings | [methods/beam.py](methods/beam.py) | Three-edge static beam against an exact polynomial |
| Figure 1: steady rods | [rods/steady.py](rods/steady.py) | 24 rods against the analytic hyperbolic profiles |
| Figure 1: transient rods | [rods/transient.py](rods/transient.py) | Three RK222 steps; continuity and Kirchhoff residuals |
| Figure 2: scaling equations | [scaling/benchmark.py](scaling/benchmark.py) | Two coupled fields, heterogeneous resolutions, manufactured solution |
| Figure 3: optical resonances | [photonic/eigenmodes.py](photonic/eigenmodes.py) | Five-edge open loop; pencil, physical equations and endpoint laws |
| Figure 4: mineral replacement | [mineral/evolve.py](mineral/evolve.py) | 44 pores, three geometry updates; water, solute and inventory balances |
| Figure 5: periodic KdV | [kdv/evolve.py](kdv/evolve.py) | 18 edges, three steps; crossing constraints, mass and L2 balances |

The beam belongs to the methods listings. It is not an additional application
figure. A general getting-started/tutorial collection is outside this release.

## Run the small cases

Use an environment built for this checkout, with its compiled Dedalus extension,
NumPy, SciPy, mpi4py and the normal Dedalus dependencies. From the repository
root, run these commands with one process:

```sh
python paper/methods/beam.py
python paper/rods/steady.py
python paper/rods/transient.py
python paper/scaling/benchmark.py
python paper/photonic/eigenmodes.py
python paper/photonic/eigenmodes.py --interval-check
python paper/mineral/evolve.py
python paper/kdv/evolve.py
```

Each command prints a `RESULT` record and raises an error if its numerical
check fails. These short checks write no simulation files. The recorded
execution results and source hashes are in [smoke-checks.json](smoke-checks.json).
They were obtained with one rank and one numerical-library thread in the
existing Rusty project environment. Automated Linux/macOS source builds and
test results are recorded in
[GitHub Actions](https://github.com/sidneyholden1/dedalus-networks/actions).

## What the interfaces look like

The native steady heat equation is directly expressed in
[steady.py](rods/steady.py):

```python
u = dist.Field(name='u', bases=basis, regularity='continuous')
problem = d3.LBVP([u], namespace=locals())
problem.add_equation('-div(grad(u)) + u = 0')
problem.add_vertex_constraint('endpoint(u, on=hot) = 1')
```

Continuity matches incident values at each vertex. The vertex constraint fixes
the hot corner and generates its unknown reaction; the other vertices retain
the Kirchhoff balance from the distributional equation. Edge taus are generated
internally. The beam uses the same field/problem interface:

```python
problem.add_equation('H.T(EI*H(u)) = load')
problem.add_vertex_constraint('endpoint(u, deriv=0, on=clamped) = 0')
problem.add_vertex_constraint('endpoint(u, deriv=1, on=clamped) = 0')
problem.add_vertex_constraint(
    'endpoint(u, deriv=1, on=selected_junctions, sum=True) = 0')
```

The last line is this example's chosen joint law. Continuous displacement alone
does not impose shared slopes or rigid-joint physics.

Transient rods use the native IVP in [transient.py](rods/transient.py):

```python
v = dist.Field(name='v'+name, bases=basis, regularity='continuous')
# After setting up each resolution group and its coefficients:
problem = d3.IVP(variables, time=t, network_interfaces=interfaces, namespace=locals())
for v, K, C, a, w in coefficients:
    problem.add_equation(
        'C*dt(v) - div(K*grad(v)) + a*v = -C*(8*t*exp(-4*t*t))*w')
problem.add_vertex_constraint('endpoint(hot_field, on=[0]) = 0')
```

The physical temperature is `u = v + (1-exp(-4*t*t))*w`, where `w` is the
steady heated-corner profile. This keeps the original time-dependent heating
and its homogeneous correction boundary. Resolution groups retain the paper's
different numbers of modes and share one physical vertex field. Dedalus
generates their taus, continuity rows and combined Kirchhoff balance internally.
The script declares the fields, coefficients and heated vertex; it no longer
constructs endpoint maps or accumulates vertex flux equations.

The [scaling workload](scaling/benchmark.py) writes its two coupled fields,
`u` and `v`, explicitly. One loop visits the resolution groups; each group adds
this pair of equations after setting up its coefficients:

```python
for name, basis in bases.items():
    u, v = u_fields[name], v_fields[name]
    # Set this group's coefficients and known profiles as shown in the script.

    problem.add_equation(
        'C_u*dt(u) - div(K_u*grad(u)) + a_u*u + coupling*(u-v) = '
        'exp(-t)*f_u + nonlinear*(u**3-exp(-3*t)*h_u3) '
        '+ coupling*exp(-t)*(h_u-h_v)')
    problem.add_equation(
        'C_v*dt(v) - div(K_v*grad(v)) + a_v*v + coupling*(v-u) = '
        'exp(-t)*f_v + nonlinear*(v**3-exp(-3*t)*h_v3) '
        '+ coupling*exp(-t)*(h_v-h_u)')

    boundary_vertices = np.intersect1d(
        np.flatnonzero(leaf), network.endpoints[basis.edge_indices])
    if boundary_vertices.size:
        problem.add_vertex_constraint('endpoint(u, on=boundary_vertices) = 0')
        problem.add_vertex_constraint('endpoint(v, on=boundary_vertices) = 0')
```

`h_u` and `h_v` are the known spatial profiles; `h_u3` and `h_v3` store their
cubes. The explicit coefficient and forcing calculations in the script preserve
the original manufactured solutions `exp(-t)*h_u` and `exp(-t)*h_v`. The default
has seven resolution groups, so the loop adds seven pairs of equations.
Each balance is associated with its unique time-differentiated unknown,
so ordinary equations need no `test=` option. Mixed time derivatives and
cross-fluxes require an explicit association; no equation-order or field-name
guess is used. The default is a short correctness check. Publishing this workload makes the
benchmark equations inspectable; the strong/weak scaling figures can be replotted
from retained measurements without running another scaling campaign.

Optics uses the exact fixed-carrier representation in
[photonic_carrier.py](photonic/photonic_carrier.py):

```python
problem = d3.EVP([A, B, U], delta, namespace=locals())
problem.add_equation('E@grad(A) - 1j*delta*n*A = 0')
problem.add_equation('E@grad(B) + 1j*delta*n*B = 0')
problem.add_equation('phaseA*endpoint(A) + phaseB*endpoint(B) - endpoint(U) = 0')
problem.add_equation(
    'endpoint(signed_index*(phaseA*endpoint(A) - phaseB*endpoint(B)), sum=True) - nout*U = 0')
```

The eigenvalue is `k = carrier + delta`. `E` is the intrinsic unit tangent,
and `U` is the physical vertex value. Continuity applies to the reconstructed
field; the envelopes remain piecewise. `endpoint(U)` supplies the vertex value
at every incident endpoint and `sum=True` adds contributions at each vertex.
The phase factors and signed refractive indices express the original optical
matching and outgoing-wave laws. Taus and lifts are generated internally.
The default checks a small branching graph. `--interval-check` additionally
compares an eigenvalue with the analytic dielectric-interval resonance.
`--graph` accepts an optical graph JSON; the diagnostic gather is limited to
small graphs. This entry point does not perform the full 454-mode search or
the large-graph production output workflow.

Mineral replacement keeps its pressure, incoming-flux mixing and finite-grain
updates in [evolve.py](mineral/evolve.py). The actual Dedalus equations are in
[mineral_dedalus.py](mineral/mineral_dedalus.py). Pressure now uses:

```python
p = dist.Field(name='p', bases=basis, regularity='continuous')
pressure_defect = dist.Field(name='pressure_defect', bases=basis.dirac)
pressure_defect['g'] = 0
problem = d3.LBVP([p], namespace=locals())
problem.add_equation('-div(k*grad(p)) = pressure_defect')
problem.add_vertex_constraint('endpoint(p, on=pressure_vertices) = boundary')
```

The vertex source is zero for the physical solve. The existing compensated
pressure refinement sets it to the measured Kirchhoff defect for correction
solves, with homogeneous reservoir values.

Transport uses the exact uniform-pore integrating factor. Dedalus solves for
constant edge amplitudes `B`, `C` and physical junction concentrations `VB`, `VC`:

```python
problem = d3.LBVP([B, C, VB, VC], namespace=locals())
problem.add_equation('E@grad(B) = 0')
problem.add_equation('E@grad(C) = 0')
problem.add_equation('endpoint(B, side=upstream) - endpoint(VB, side=upstream) = 0')
problem.add_equation('endpoint(C, side=upstream) - endpoint(VC, side=upstream) = 0')
problem.add_equation('alpha*VB + beta*(VB-endpoint(incoming*TB*endpoint(B), sum=True)) = prescribed')
problem.add_equation('alpha*VC + beta*(VC-endpoint(incoming*(TC*endpoint(C)+TH*endpoint(B)), sum=True)) = 0')
```

`upstream` selects the incoming end of each pore from its flow direction.
`incoming` contains normalized incoming water-flux weights; `alpha` selects
prescribed vertices and `beta` selects mixing vertices. Concentrations remain
piecewise because junction mixing equates outgoing values to a flux-weighted
mean of incoming values. The coefficients include the evolving pore geometry.

`TB`, `TC` and `TH` are endpoint values of the known reaction propagator. The
physical concentration profiles and integrated reaction rates are reconstructed
explicitly below the solve in the same source file. This representation is exact
for the uniform coefficients within each pore at a given geometry step. The
coefficients can differ between pores and change as the geometry evolves.

The example has one transport formulation and no `fitted` switch. Its taus and
lifts are internal. Geometry updates, exhaustion and closure retain their
existing rules. The direct concentration formulation and optional custom mixing
maps are retained in [verification code](../dedalus/tests/mineral_transport_verification.py).
The random graph and uniform-equation-21 initialization are declared
reconstructions; this tiny graph is not the paper's saved network.

KdV uses the native calculus for the ordinary nonlinear model in
[evolve.py](kdv/evolve.py):

```python
u = dist.Field(name='u', bases=basis, regularity='continuous')
w = E @ d3.grad(E @ d3.grad(u))
problem = d3.IVP([u], time=t, namespace=locals())
# Construct tau corrections from the adjoints of the junction conditions.
problem.add_equation('dt(u) + div(E*w) = -6*u*(E@grad(u))', tau='adjoint')
problem.add_vertex_constraint('endpoint(horizontal*u, deriv=1, sum=True) = 0')
problem.add_vertex_constraint('endpoint(vertical*u, deriv=1, sum=True) = 0')
```

`E` is the intrinsic unit tangent in the assigned edge orientation; `horizontal`
and `vertical` select the two edge directions. Thus `w` is edge curvature and
`div(E*w)` supplies its signed vertex balance. The two explicit constraints pair
the straight-through slopes. Continuity is declared on `u`; it is not imposed
on curvature. On this balanced oriented grid, the vertex flux of `3*E*u**2`
cancels for continuous `u`, leaving the regular nonlinear term shown above.

`tau='adjoint'` constructs the tau correction shapes from the junction
conditions using the physical L2 inner product. Their amplitudes are solved
for internally. This example uses adjoint tau corrections because the default
high-mode prescription fails its stability checks; both choices use spectral
polynomials. The previous `tau='galerkin'` spelling remains an alias.

The numerical method used for the paper is unchanged. Dedalus generates the
independent continuity rows and physical-L2 adjoints of all six crossing
constraints internally. The explicit nonlinear product is
projected in physical L2 before conversion to derivative bases. RKGFY and the
[scaled linear solves](kdv/linear_solvers.py) are retained. The
[unscaled constraint diagnostics](kdv/conditions.py) independently check the laws.

The [translation audit](../dedalus/tests/test_network_native_ordinary_kdv.py)
compares the matrices, arbitrary retained-mode nonlinear forcing, three time
steps and instantaneous rate against [explicit_reference.py](kdv/explicit_reference.py),
the retained original assembly. This adjoint tau path currently supports
serial execution, one continuous scalar, a regular-degree graph and static
homogeneous vertex constraints. Results and exact source hashes are recorded in
[kdv-native-checks.json](kdv-native-checks.json).

## Native interface checks

[native-interface-checks.json](native-interface-checks.json) records the latest
small-case translations and shared-library regressions. The comparisons retain
the previous explicit implementations as audit references: transient rods and
scaling compare physical coefficients after stepping; optics compares the
carrier eigenproblem matrices and modes; mineral compares pressure, transport
and geometry updates. Additional checks cover the direct optical formulation,
direct mineral concentrations, pressure defect sources, cached pressure solves,
heterogeneous vertex balances and selected endpoints. The direct optical helper
[photonic_dedalus.py](photonic/photonic_dedalus.py) also uses native `grad/div`
and automatic taus; the carrier formulation remains the paper entry point.

## Paper settings and provenance

[paper-settings.json](paper-settings.json) distinguishes the recorded paper
settings from the small runs. It is a review reference, not a complete production
case manifest. Full retained settings and figure inputs are indexed in
[REPRODUCING.md](REPRODUCING.md). [source-map.json](source-map.json) records the copied or extracted
scientific source files and the deliberate interface edits. All application
imports resolve to modules here or to installed dependencies; no script imports
from private research directories or cluster controllers.

The retained paper outputs include their original source and run provenance.
