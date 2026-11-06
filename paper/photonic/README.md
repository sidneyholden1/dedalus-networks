# Optical resonances

`eigenmodes.py` presents the native fixed-carrier envelope equations. Its small
open-loop default and optional `--interval-check` verify the eigenproblem and
physical boundary laws. Figure 3 displays a separate, retained 100,485-edge
interior resonance with 32 envelope coefficients per edge. The exact carrier,
target, Krylov settings and rank counts are in `paper/data/photonic/fine-status.json`;
the coarse run and refinement comparison are included alongside it.

`plot.py --output <new-directory>` redraws the approved figure using the exact
saved pointwise-intensity raster objects and the 2,049-point resolved trace.
The trace is checked against the bundled travelling-wave amplitudes. Maps and
trace share the same pointwise maximum; they are not edge-mean intensity plots.

The full physical mode and geometry are in `selected-fields.npz`. To regenerate
the pointwise rasters from those fields, the original renderer and its C++
helper are included. This optional step needs a C++17 compiler and `TMPDIR`
pointing to writable temporary storage:

```sh
python paper/photonic/plot_photonic_quantile_pair.py \
  --selection paper/data/photonic --boxes paper/data/photonic/boxes.json \
  --output figures/photonic-rasters --scales log --palette vivid \
  --color-floor 1e-6 --overview-stroke-width 3.06 \
  --hotspot-stroke-width 32 --zoom-fraction .5 --edge-profile
```

This evaluates the retained mode, not a new eigenproblem. The exact original
renderer is also preserved in `render-source.zip`; the unpacked helper only
changes its import path to resolve within this release. Selection, physical
checks, refinement and source identities accompany the data.
