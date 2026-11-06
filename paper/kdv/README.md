# Periodic KdV

`evolve.py` expresses the ordinary KdV equation using continuous `u`, `grad`,
`div`, the crossing constraints and `tau='adjoint'`. This option constructs
correction shapes from the physical-L2 adjoints of the junction constraints.
The original discretization is preserved; only the assembly is internal.

`plot.py --output <new-directory>` redraws the four saved states at t=0,6,20,200.
The archive includes the 512-edge periodic 16-by-16 graph, length-8 edges,
original N=64 fields, configuration and qualification receipts. Production
used RKGFY, dt=0.000125, dealiasing 1.5 and the recorded polynomial initial
pulse. `paper/data/kdv/production.json` retains all model settings and the
provenance of the saved stages. The plotter preserves sign colours, width
scaling and all physical edges without temporal interpolation.

`render_fields.py` constructs the four data panels from those fields. `plot.py`
then arranges them in the paper's two-by-two layout, with the shared sign and
orientation key. The layout preserves the panel drawing commands and image
samples. Its TeX, Poppler and qpdf requirements are listed in `INSTALL.md` at
the repository root.

The native adjoint assembly has serial checks against the original method,
covering matrices, nonlinear forcing, short time evolution and instantaneous
rates. The saved production fields include their original source and MPI run
provenance.
