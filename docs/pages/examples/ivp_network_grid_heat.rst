Heat diffusion on a disc network (network IVP)
*********************************************

A central hot spot spreads along a random triangulation of a disc with
100 vertices and 277 edges. The equation on each edge is
:math:`\partial_t u = \partial_x^2 u`, where :math:`x` is physical arclength.
Temperature is continuous at every junction and the sum of outward fluxes
is zero. The network is insulated, so total heat is conserved.

The default geometry uses seed 42, 20 boundary vertices and 80 interior
vertices. Each edge has 16 Chebyshev modes; SBDF2 time stepping and the
Dirichlet-to-Neumann solver evolve the solution to time 0.4. The saved
solution contains 121 states.

Run
---

From the repository root, in the environment containing this fork:

.. code-block:: console

    python examples/ivp_network_grid_heat/heat.py --output output/disc_heat

This writes ``network_disc_heat.gif``, initial/spreading/final PNGs,
``snapshots/snapshots_s1.h5`` with ``u`` and ``U``, geometry and parameters
in ``run.json``, numerical checks in ``checks.json``, and rendering metadata
in ``animation_receipt.json``. Checks cover heat conservation, energy decay,
temperature bounds, continuity and Kirchhoff flux balance.

To redraw existing saved data without rerunning the solver:

.. code-block:: console

    python examples/ivp_network_grid_heat/plot_heat.py output/disc_heat

Use a fresh output directory to retain earlier results; rerunning overwrites
the outputs. The example README describes the optional square geometry.

Output
------

The animation uses the same logarithmic inferno scale, 0.001 to 1, throughout.
Values at or below 0.001 share the lowest colour. The colour bar is on the
right, matches the network's height, and has :math:`u` above it. Each frame
uses one saved simulation state, with no interpolation in time.

.. image:: images/network_disc_heat.gif
    :width: 100 %
    :align: center
    :alt: Heat diffusion along a triangulated disc network, with a colour bar.

Solver
------

.. literalinclude:: ../../../examples/ivp_network_grid_heat/heat.py

Renderer
--------

.. literalinclude:: ../../../examples/ivp_network_grid_heat/plot_heat.py
