Poisson equation (network LBVP)
********************************

This example uses the whole-edge distributed storage path.  Under MPI, each
rank solves only its owned edge intervals and stores only its owned/incident
vertices.  The usual Dedalus analysis handler writes ``u`` and ``U`` to a
gathered HDF5 file, and one additional gather is performed to make the small
rank-zero plot.

Script
------

.. literalinclude:: ../../../examples/lbvp_network_poisson/poisson.py

Output
------

.. image:: images/network_poisson.png
    :width: 100 %
    :align: center
