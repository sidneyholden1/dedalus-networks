# Rod heat conduction

`steady.py` and `transient.py` show the continuous-field network interface.
Run `plot.py --output <new-directory>` to redraw Figure 1 from saved data.

The figure uses 16 vertices, 24 rods, total length 24 and mean resolution 32.
The transient state is at t=1 with RK222 and dt=1/128; its imposed temperature
is 1-exp(-4t²). Geometry seed 7304, individual rod lengths, coefficient arrays,
material values and ownership arrays are retained in the JSON reports.
The small negative undershoot in the stored transient solution is displayed.

`paper/data/rods/cases-*.json` retains the original steady, transient, parallel
and time-convergence case settings. The accompanying `*-verification.json`
files contain the plotted convergence data and checks. Their original output
paths are provenance, not inputs required by the plotter. `inputs.py` contains
the geometry and steady-profile construction used by the examples.
