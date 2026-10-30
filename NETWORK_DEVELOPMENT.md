# Installing and developing the network fork

## Get the source

```sh
git clone https://github.com/sidneyholden1/dedalus-networks.git
cd dedalus-networks
conda env create -f environment-network-pdes.yml
conda activate dedalus-network-pdes
```

The environment supplies the compiled dependencies without installing another
copy of Dedalus. Build this fork in editable mode:

```sh
export MPI_PREFIX="$CONDA_PREFIX"
export FFTW_PREFIX="$CONDA_PREFIX"
export OMP_NUM_THREADS=1
CC=mpicc CXX=mpicxx python -m pip install -e . --no-deps --no-build-isolation
```

Rebuild after changing a Cython source file or the MPI stack. The environment
uses MPICH; use that environment's `mpiexec` when running parallel examples.

## Poisson example

Keep generated files in an output directory. From the repository root:

```sh
mkdir -p output/poisson
(cd output/poisson && python ../../examples/lbvp_network_poisson/poisson.py)
```

This solves a three-edge star with junction `v_0` and leaves `v_1` (left),
`v_2` (upper), and `v_3` (lower). The directed edges `v_1` to `v_0`,
`v_0` to `v_2`, and `v_0` to `v_3` have lengths 1, 1.5, and 2.
The boundary values are `u(v_1)=1` and `u(v_2)=u(v_3)=0`; the analytic
check gives `u(v_0)=6/13`. Vertex arrays and HDF5 metadata use the order
`("v_0", "v_1", "v_2", "v_3")`. The script writes HDF5 solution data and
`network_poisson.png`, with mathematical vertex labels and no vertex markers.
Its linear colour bar spans 0 to 1, sits on the right and has `u` above it.
For a small MPI run, use a separate output directory:

```sh
mkdir -p output/poisson-mpi
(cd output/poisson-mpi && mpiexec -n 4 python ../../examples/lbvp_network_poisson/poisson.py)
```

## Disc heat animation

From the repository root:

```sh
python examples/ivp_network_grid_heat/heat.py --output output/disc_heat
```

This solves heat diffusion along a 100-node random triangulation of a disc,
with seed 42. The default run saves 121 states up to time 0.4 and produces:

- `network_disc_heat.gif`: the animation shown in the README, with a colour bar
  on the right and `u` above it.
- `heat_initial.png`, `heat_spreading.png`, `heat_final.png`: static snapshots.
- `snapshots/snapshots_s1.h5`: saved edge temperatures `u` and vertex values `U`.
- `run.json`, `checks.json`, `animation_receipt.json`: geometry and parameters,
  heat/junction checks, and frame times and colour mapping.

The colour scale stays logarithmic from 0.001 to 1; lower values share the
lowest colour. The renderer evaluates the saved spectral fields along each
rod and does not interpolate in time. To redraw existing data:

```sh
python examples/ivp_network_grid_heat/plot_heat.py output/disc_heat
```

Use `--no-render` with the solver to save data and checks only. Its
`--geometry square` option selects a 5-by-5 network; see the
[example README](examples/ivp_network_grid_heat) for the corresponding settings.
Use fresh output directories to retain earlier results.

## Heterogeneous heat example

Run the time-dependent example from its own output directory:

```sh
mkdir -p output/heat
(cd output/heat && python ../../examples/ivp_network_heterogeneous_heat/heat.py)
```

It writes:

- `network_heat.png`: final temperature profiles and temperature decay at an
  interior vertex, compared with the analytic solution.
- `network_heat/network_heat_s1.h5`: time snapshots of `u_coarse`, `u_fine` and
  `U`, with grid and network metadata for each discretization bucket.

The script checks the numerical solution against the analytic heat mode and
prints the output paths. For MPI, use a separate directory:

```sh
mkdir -p output/heat-mpi
(cd output/heat-mpi && mpiexec -n 4 python ../../examples/ivp_network_heterogeneous_heat/heat.py)
```

Both runs produce complete output files on rank zero. Rerunning in the same
directory overwrites these outputs. The full API and solver limitations are in
`docs/pages/network_problems.rst`.

## Tests

From the repository root, with the environment activated:

```sh
python -m pytest -q dedalus/tests/test_network*.py
mpiexec -n 4 python -m pytest -q dedalus/tests_parallel/test_network*.py
```

For the full upstream regression command, see [CONTRIBUTING.md](CONTRIBUTING.md).
Run larger MPI examples and benchmarks through the scheduler on a cluster.

## Initial publication checks (26 October 2026)

Five focused network normalization, invalid-length, and restriction-map checks
passed in the existing development environment. The numerical portion of the
three-edge Poisson example passed its analytic vertex check in serial. These
are lightweight smoke checks; the full test suite, MPI suite, plotting/output,
and a clean Conda installation were not rerun for this publication.

## Example output checks (29 October 2026)

The Poisson example passes its analytic vertex check and produces its PNG and
HDF5 output in serial. The disc heat example passes checks for heat conservation,
energy decay, temperature bounds and junction conditions; all 121 animation
frames decode. The heterogeneous heat output and analytic checks pass in serial
and on four MPI ranks. These are example checks in the existing development
environment; the full suite and a fresh Conda installation were not rerun.
