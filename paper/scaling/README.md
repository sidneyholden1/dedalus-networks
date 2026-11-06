# Parallel scaling

`benchmark.py` exposes the two coupled evolution equations, with one pair per
resolution group. Its default run is a small correctness check. `plot.py
--output <new-directory>` redraws all three scaling panels from 180 retained
measurements, without MPI or new benchmark runs.

The production workload uses two fields, coupling 0.2, cubic coefficient 0.1,
RK222, dt=0.001, ten warmup steps and 1,000 measured steps. Full settings for
every selected configuration are in `paper/data/scaling/production.json`,
including graph/partition seeds, topology size, tolerances, solver options and
rank counts. Raw records retain hardware/timing and scientific checks; exact
owner arrays are in `paper/data/scaling/partitions/`. `inputs.py` supplies the
network generators. Use the recorded seed 20261026 for production topologies;
the short example uses its original review seed 20261010.

Strong-scaling curves use each topology's own P=1 baseline within a node and
P=96 baseline across nodes. Weak scaling shows recurring timestep seconds.
The selected repeat is fixed in the saved results. The curves show those
measurements without uncertainty bars; the associated source and hardware are
recorded alongside the timings.
