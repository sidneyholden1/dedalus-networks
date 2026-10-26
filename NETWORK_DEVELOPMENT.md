# Installing and developing the network fork

## Get the source

```sh
git clone https://github.com/sidneyholden1/dedalus-network.git
cd dedalus-network
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

## First example

Keep generated files in an output directory. From the repository root:

```sh
mkdir -p output/poisson
(cd output/poisson && python ../../examples/lbvp_network_poisson/poisson.py)
```

This solves a three-edge star with unequal edge lengths, checks the analytic
vertex values (central value `6/13`), and writes HDF5 output and a plot.
For a small MPI run, use a separate output directory:

```sh
mkdir -p output/poisson-mpi
(cd output/poisson-mpi && mpiexec -n 4 python ../../examples/lbvp_network_poisson/poisson.py)
```

The time-dependent example is `examples/ivp_network_heterogeneous_heat/heat.py`.
The full API and solver limitations are in `docs/pages/network_problems.rst`.

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
