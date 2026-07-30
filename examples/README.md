# CoolingTNS Examples

Runnable demonstrations of the dispatch-based cooling interface. Run each from
the repository root, e.g. `julia --project=. examples/ed_kspace_demo.jl`.

## Shared helpers

- `ed_dm_example_utils.jl` — typed parameters for the ED density-matrix Ising
  k-space example (`ed_dm_ising_example`), the matching `Cooling.jl` command
  (`ed_dm_driver_command`), and the canonical k-space plot wrapper
  (`plot_ed_dm_kspace_results`). Included by the scripts below rather than run
  directly.

## Scripts

- `ed_cooling_example.jl` — ED backend with `DensityMatrix` and
  `MonteCarloWavefunction`, the equivalent `Cooling.jl` command line, and a
  backend comparison sweep.
- `unified_interface_demo.jl` — one problem driven through every
  backend × simulation method × evolution method combination to show
  backend-agnostic dispatch.
- `ed_kspace_demo.jl` — k-space measurements under periodic/antiperiodic
  boundary conditions, including the Bogoliubov dispersion and the distinction
  between raw Fourier occupations and mode occupations.
- `test_ed_kspace.jl` — smoke example for ED k-space observables on the
  integrable transverse-field Ising chain.
- `run_dm_and_plot.jl` — runs the ED density-matrix Ising k-space simulation via
  `ed_dm_example_utils.jl` and emits the canonical `n_k` and mode-energy plots.
- `run_short_dm.jl` — shells out to `Cooling.jl` for a shorter ED density-matrix
  run; useful as a quick end-to-end check of the CLI driver.

## Test coverage

- `test/test_ed_dm_kspace_examples.jl` — `ed_dm_example_utils.jl` parameters,
  filenames, and driver flags; also asserts `run_dm_and_plot.jl` carries no
  obsolete filenames, plot scripts, dispersion, or energy labels.
- `test/test_ed_kspace_smoke_example.jl` — includes and runs `test_ed_kspace.jl`.
- `test/test_ed_kspace_demo_text.jl` — checks `ed_kspace_demo.jl` text against
  the canonical dispersion and occupation conventions in `CLAUDE.md`.

`ed_cooling_example.jl`, `unified_interface_demo.jl`, and `run_short_dm.jl` are
not covered by the test suite; they execute full simulations.
