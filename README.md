# Dedalus on Networks

A research fork of [Dedalus](https://github.com/DedalusProject/dedalus) for
solving partial differential equations on physical networks. Edges use spectral
bases and are coupled through vertex values and flux conditions.

The network-calculus interface provides graph differential operators and
automatic tau assembly. The solver supports linear and nonlinear boundary-value
problems, initial-value problems and selected eigenvalue problems, with MPI
distribution and Dirichlet-to-Neumann solvers.

This repository accompanies *A scalable computational framework for PDEs on
physical networks* (2026, in preparation), by Sidney Holden, Keaton J. Burns,
Daniel Lecoanet, Jeffrey S. Oishi, Benjamin P. Brown, and Geoffrey M. Vasil.
It includes example scripts and data for reproducing the paper's figures.

[Installation](INSTALL.md) · [Paper examples](paper/README.md) ·
[Reproduce the figures](paper/REPRODUCING.md) ·
[Download the release](https://github.com/sidneyholden1/dedalus-networks/releases/tag/paper-companion-20261106.1)

The Python package remains `dedalus`; install this fork in a separate environment.

## Examples

<table align="center">
  <tr>
    <td align="center" valign="top">
      <h3>Three-edge Poisson</h3>
      <a href="examples/lbvp_network_poisson"><img src="docs/pages/examples/images/network_poisson_readme.png" width="332" alt="Three-edge Poisson solution with a linear colour bar labelled u" /></a>
      <p>A boundary-value problem at one junction.</p>
      <a href="examples/lbvp_network_poisson">Run the example</a>
    </td>
    <td align="center" valign="top">
      <h3>Heat diffusion</h3>
      <a href="examples/ivp_network_grid_heat"><img src="docs/pages/examples/images/network_disc_heat.gif" width="332" alt="Animated heat diffusion on a triangulated disc network, with a logarithmic colour bar labelled u" /></a>
      <p>A hot spot spreads through a network.</p>
      <a href="examples/ivp_network_grid_heat">Run the example</a>
    </td>
  </tr>
</table>

### KdV on a periodic lattice

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/pages/examples/images/network_kdv_readme_dark.png" />
    <img src="docs/pages/examples/images/network_kdv_readme_light.png" width="790" alt="KdV snapshots at t = 0, 6, 20 and 200, including the initial pulse. The sign key distinguishes u greater than zero and u less than zero; drawn width is proportional to the magnitude of u." />
  </picture>
</p>

A localized pulse propagates and disperses across a periodic network.

[Example script and equations](paper/kdv/README.md) ·
[Full figure](paper/data/kdv/published.pdf)

---

[Network API guide](docs/pages/network_problems.rst) ·
[Development and tests](NETWORK_DEVELOPMENT.md) · [Citation](CITATION.cff)

Built on [Dedalus](https://github.com/DedalusProject/dedalus), developed by the
Dedalus collaboration. This research fork is maintained independently.

[Upstream documentation](https://dedalus-project.readthedocs.io) ·
[Dedalus citation](CITATION_UPSTREAM.cff) · [License](LICENSE.txt)
