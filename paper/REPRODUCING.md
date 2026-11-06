# Reproducing the figures

The companion archive includes `paper/data/`. Start by verifying the saved
inputs, then redraw the figures from the repository root:

```sh
python paper/verify_data.py
python paper/rods/plot.py --output figures/rods
python paper/scaling/plot.py --output figures/scaling
python paper/photonic/plot.py --output figures/photonic
python paper/mineral/plot.py --output figures/mineral
python paper/kdv/plot.py --output figures/kdv
```

Each output directory must be new. These commands read saved results and write
PNG/PDF figures; they do not run simulations or request cluster resources.
`--data` selects another input directory (for mineral, the `plot-data.npz` file).
The exact manuscript figure exports are also included as
`paper/data/<application>/published.pdf`. The manuscript figure snapshot is
identified by commit and file hashes in [data-manifest.json](data-manifest.json).
PDF metadata and font rendering can differ across machines; numerical input
checksums and plotting definitions are the reproducibility reference.

The plotting commands need NumPy, SciPy, Matplotlib and Pillow, plus TeX Live
with `latex`, `dvipng`, `pdflatex`, Computer Modern/Latin Modern, `amsmath`,
`type1cm`, `standalone` and TikZ. The optical redraw uses Poppler's `pdfimages`;
the KdV layout uses `pdftoppm`, `pdftotext`, `qpdf` and `fix-qdf`. It first
renders the saved fields, then composes the four panels in the paper layout.
Tested Python versions are
recorded in [environment.json](environment.json); the compact plotting-only
requirements are in [requirements-plotting.txt](requirements-plotting.txt).
The Dedalus solver and MPI are not imported by the five plotting commands.

| Figure | Saved inputs | Production settings and provenance |
| --- | --- | --- |
| 1: rods | Chebyshev coefficients, error curves and 1/2/4-process partitions | [Rod notes](rods/README.md); original case tables in `data/rods/` |
| 2: scaling | All 180 selected measurements, their full raw records and 129 retained partition arrays | `data/scaling/production.json`; [scaling notes](scaling/README.md) |
| 3: optical mode | Full 100,485-edge physical mode, geometry, exact map rasters and 2,049-point trace | `data/photonic/{coarse,fine}-status.json`, selection and refinement receipts; [optical notes](photonic/README.md) |
| 4: mineral replacement | Four diameter fields and the complete plotted permeability history | `data/mineral/production.json` and replay audit; [mineral notes](mineral/README.md) |
| 5: KdV | Exact saved fields at t=0,6,20,200, geometry and qualification receipts | `data/kdv/production.json`; [KdV notes](kdv/README.md) |

Historical receipts retain their original run paths as provenance. Plotters
resolve every required input inside the supplied data directory; those old
paths are not prerequisites. `data/scaling/production.json` relocates retained
partition paths relative to `paper/data/`, while preserving all numerical settings.

The short solver examples in [README.md](README.md) exercise the network
interface. Their defaults intentionally differ from production settings, which
are included with the saved results and original source identities. The scaling
workload is available to inspect and run; its figures can also be redrawn
directly from the retained measurements.
