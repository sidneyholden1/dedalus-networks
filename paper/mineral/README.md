# Mineral replacement

`evolve.py` shows pressure, exact uniform-pore transport and finite-inventory
updates on a small network. `plot.py --output <new-directory>` redraws the
accepted 29,996-pore Gamma=0.3 example and its entire plotted permeability curve.
The full saved geometry, four diameter fields and 12,497 history records are
bundled in `paper/data/mineral/plot-data.npz`.

`production.json` records the actual geometry seed, chemistry, controller,
midpoint integrator, stopping rules, tolerances and source identity. In
particular, the recorded internal `chemistry.da` is 0.627201506132748 and the
imposed total flow is 64.84413939429024. These belong to the historical
normalization used for the paper's Da=0.5 example; replacing the recorded value
with 0.5 would change that run. The short example is not a production preset.

The first two displayed states come from verified replays with the accepted
source; the later two states and curve come from the retained original history.
The replay audit, hashes and exact frame times are included. No interpolation
is used. This is one qualitative Dedalus demonstration on a reconstructed
network, not the authors' exact network realization or an exhaustive recreation
of their parameter regimes. The current short example does not implement the
historical campaign's complete midpoint/restart controller.
