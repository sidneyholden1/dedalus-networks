# Installing the paper companion

For figure redraws alone, create a Python environment and install:

```sh
python -m pip install -r paper/requirements-plotting.txt
```

Also install TeX Live (`latex`, `dvipng`, Computer Modern/Latin Modern,
`amsmath`, `type1cm`, `standalone`, TikZ and `pdflatex`), Poppler (`pdfimages`,
`pdftoppm`, `pdftotext`) and qpdf (`qpdf`, `fix-qdf`) using your system's package
manager. The optional optical raster regeneration needs a C++17 compiler.
The plotting commands do not import Dedalus or MPI. See
[paper/REPRODUCING.md](paper/REPRODUCING.md) for commands and input checksums.

The solver examples require this release's Dedalus source, a C compiler,
MPI with development headers, FFTW3 with MPI support, and the Python build
requirements in `pyproject.toml`. MPI and mpi4py must use compatible libraries.
Start from an environment with those native libraries installed. Set `MPI_PATH`
and `FFTW_PATH` if they are not in standard compiler search locations, then:

```sh
python -m pip install -c paper/constraints-tested.txt 'cython>=3.0.5' numpy mpi4py setuptools wheel
python -m pip install --no-build-isolation -c paper/constraints-tested.txt -e .
```

Use an environment dedicated to this release: its package is still named
`dedalus`. `setup.py` declares the full Python runtime dependencies and builds
four Cython extensions. Do not install an unmodified Dedalus package in place
of the bundled network source. Run the examples from the extracted release
root so `paper/` and the figure inputs are available.

For the bounded serial checks, use one numerical-library thread:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python paper/methods/beam.py
```

The tested Python, NumPy/SciPy, MPI and plotting versions are recorded in
[paper/environment.json](paper/environment.json). Verification reused the
existing compatible compiled extensions and dependencies with the extracted
release source; it did not rebuild or install into that active environment.
A clean build on a second machine remains untested. Native KdV adjoint assembly
is currently qualified only in serial. Large production runs and performance
experiments need appropriately provisioned compute resources; no supplied
plotting or short-check command submits jobs.

The paper integration tests use the companion source directory. When testing
an installed library outside its checkout, set `DEDALUS_PAPER_DIR` to the
absolute path of the checkout's `paper/` directory. These integration tests
are skipped when the optional companion sources are absent; the library tests
remain available in the installed package.
