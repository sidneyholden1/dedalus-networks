Poisson equation (network LBVP)
********************************

The three-edge star has junction :math:`v_0` at the centre and leaves
:math:`v_1` on the left, :math:`v_2` above, and :math:`v_3` below. The directed
edges are :math:`v_1\to v_0`, :math:`v_0\to v_2`, and :math:`v_0\to v_3`,
with lengths 1, 1.5, and 2. The boundary values are :math:`u(v_1)=1` and
:math:`u(v_2)=u(v_3)=0`; continuity and Kirchhoff flux balance give
:math:`u(v_0)=6/13`.

The script, plot labels, coefficient arrays, and saved HDF5 metadata all use
vertex order ``("v_0", "v_1", "v_2", "v_3")``. The plot shows the coloured
edges and mathematical vertex labels, with no vertex markers or title.
The colour bar sits on the right, spans the network's height, and has
:math:`u` above it.

This example uses the whole-edge distributed storage path.  Under MPI, each
rank solves only its owned edge intervals and stores only its owned/incident
vertices.  The usual Dedalus analysis handler writes ``u`` and ``U`` to a
gathered HDF5 file, and one additional gather is performed to make the small
rank-zero plot.

Run
---

From the repository root, in the environment containing this fork:

.. code-block:: console

    mkdir -p output/poisson
    (cd output/poisson && python ../../examples/lbvp_network_poisson/poisson.py)

The output directory contains ``network_poisson.png`` and
``network_poisson/network_poisson_s1.h5``. Use a fresh directory to retain
earlier results. Prefix the Python command with ``mpiexec -n 4`` for MPI.

Script
------

.. literalinclude:: ../../../examples/lbvp_network_poisson/poisson.py

Output
------

.. image:: images/network_poisson.png
    :width: 100 %
    :align: center
    :alt: Three-edge Poisson solution, with v_0 at the central junction.
