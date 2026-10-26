# Dedalus on Networks

A research fork of [Dedalus](https://github.com/DedalusProject/dedalus) for
solving partial differential equations on networks. The extension represents
edges with spectral bases and couples them through vertex values and flux
conditions. It distributes subgraphs across MPI processes and uses
Dirichlet-to-Neumann solvers.

This is an experimental extension, maintained separately from upstream Dedalus.
The Python package remains `dedalus`; install this fork in its own environment.

## Start here

1. Follow [installation and development instructions](NETWORK_DEVELOPMENT.md).
2. Run the [three-edge Poisson example](examples/lbvp_network_poisson/poisson.py).
   Its built-in analytic check gives the central vertex value `6/13`.
3. Read the [network API and solver guide](docs/pages/network_problems.rst).
4. Try the [heterogeneous heat example](examples/ivp_network_heterogeneous_heat/heat.py)
   for time evolution on edges with different lengths and resolutions.

The implementation includes linear and nonlinear boundary-value problems,
initial-value problems, selected eigenvalue problems, dynamic vertex storage,
constraints, and restart support. See the guide for the precise supported cases.
Current network fields are scalar; the distributed eigenvalue path is restricted
to selected right eigenmodes, and network file output uses gathering.

## Development and review

Network tests live in `dedalus/tests/test_network*.py` and
`dedalus/tests_parallel/test_network*.py`. The development guide gives serial
and MPI commands. The documented scaling harness is included under `benchmarks/`.

This public branch consolidates the network implementation onto upstream commit
`2a2ad15108f2cd2d24d0b103f8167962562b5cbc`. Library source, tests, examples, and
the scaling harness match development snapshot `e6050873e952678d0d879764e12a4bd510835fc8`.
Research archives, cluster administration, and simulation outputs are maintained
separately. This initial publication is for use and architectural feedback;
it does not claim upstream acceptance or a new performance validation.

## Attribution and license

Dedalus is developed by the Dedalus collaboration, whose original project
information follows below. The existing [license](LICENSE.txt) and
[citation metadata](CITATION.cff) are retained.

---

<!-- Title -->
<h1 align="center">
  Dedalus Project
</h1>

<!-- Information badges -->
<p align="center">
  <a href="https://www.repostatus.org/#active">
    <img alt="Repo status" src="https://www.repostatus.org/badges/latest/active.svg" />
  </a>
  <a href="http://dedalus-project.readthedocs.org">
    <img alt="Read the Docs" src="https://img.shields.io/readthedocs/dedalus-project">
  </a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dedalus">
  <a href="https://pypi.org/project/dedalus/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/dedalus">
  </a>
  <a href="https://github.com/conda-forge/dedalus-feedstock">
  <img alt="Conda Version" src="https://img.shields.io/conda/vn/conda-forge/dedalus">
  </a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html">
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/dedalus">
  </a>
</p>

Dedalus is a flexible framework for solving partial differential equations using modern spectral methods.
The code is open-source and developed by a team of researchers studying astrophysical, geophysical, and biological fluid dynamics.

Dedalus is written primarily in Python and features an easy-to-use interface with symbolic vectorial equation specification.
For example, to simulate incompressible hydrodynamics in a ball, you can symbolically enter the equations, including [gauge conditions](https://dedalus-project.readthedocs.io/en/latest/pages/gauge_conditions.html) and [boundary conditions enforced with the tau method](https://dedalus-project.readthedocs.io/en/latest/pages/tau_method.html), as:

```python
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u)")
problem.add_equation("u(r=1) = 0")
problem.add_equation("integ(p) = 0")
```

Our numerical algorithms produce sparse and spectrally accurate discretizations of PDEs on simple domains, including Cartesian domains of any dimension, disks, annuli, spheres, spherical shells, and balls:

<table style="background-color:#FFFFFF;">
  <tr>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_1d_kdv_burgers.html">
        <figure>
          <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/kdv_burgers.png">
          <figcaption>KdV-Burgers equation (1D IVP)</figcaption>
        </figure>
      </a>
    </th>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_rayleigh_benard.html">
        <figure>
          <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/rayleigh_benard.png">
          <figcaption>Rayleigh-Benard convection (2D IVP)</figcaption>
        </figure>
      </a>
    </th>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_shear_flow.html">
        <figure>
            <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/shear_flow.png">
            <figcaption>Periodic shear flow (2D IVP)</figcaption>
        </figure>
      </a>
    </th>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/lbvp_2d_poisson.html">
        <figure>
            <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/poisson.png">
            <figcaption>Poisson equation (2D LBVP)</figcaption>
        </figure>
      </a>
    </th>
  </tr>
  <tr>
  </tr>
  <tr>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_disk_libration.html">
        <figure>
          <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/libration.png">
          <figcaption>Librational instability (disk IVP)</figcaption>
        </figure>
      </a>
    </th>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_sphere_shallow_water.html">
        <figure>
          <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/shallow_water.png">
          <figcaption>Spherical shallow water (sphere IVP)</figcaption>
        </figure>
      </a>
    </th>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_shell_convection.html">
        <figure>
            <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/shell_convection.png">
            <figcaption>Spherical shell convection (shell IVP)</figcaption>
        </figure>
      </a>
    </th>
    <th width="25%">
      <a href="https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_ball_internally_heated_convection.html">
        <figure>
            <img src="https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/pages/examples/images/internally_heated_convection.png">
            <figcaption>Internally heated convection (ball IVP)</figcaption>
        </figure>
      </a>
    </th>
  </tr>
</table>

The resulting systems are efficiently solved using compiled libraries and are automatically parallelized using MPI.
See the [documentation](http://dedalus-project.readthedocs.org) for tutorials and additional examples.

## Links

* Project homepage: <http://dedalus-project.org>
* Code repository: <https://github.com/DedalusProject/dedalus>
* Documentation: <http://dedalus-project.readthedocs.org>
* Mailing list: <https://groups.google.com/forum/#!forum/dedalus-users>

## Developers

* [Keaton Burns (@kburns)](https://github.com/kburns)
* [Geoff Vasil (@geoffvasil)](https://github.com/geoffvasil)
* [Jeff Oishi (@jsoishi)](https://github.com/jsoishi)
* [Daniel Lecoanet (@lecoanet)](https://github.com/lecoanet/)
* [Ben Brown (@bpbrown)](https://github.com/bpbrown)
