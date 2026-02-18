"""
    plot_time_randomization_resonances_tn_dm_trotter.jl

Driver script (TN deterministic) generating a publication-style figure showing how
randomizing cycle times suppresses accidental resonances/heating.

This uses:
- `TNBackend()`
- `DensityMatrix() + TrotterEvolution()` (MPO + Trotter)

Compared to the ED demo, this is much more expensive. The defaults here are set
for a *small* TN run to validate the workflow; for publication runs, increase
`N`, `steps`, and tighten `cutoff`/increase `Dmax` as needed.

Usage:
    julia --project=. scripts/plotting/plot_time_randomization_resonances_tn_dm_trotter.jl

Output (default):
    scripts/plotting/Figs/randomized_times_remove_accidental_heating_tn_dm_trotter.pdf
"""

include(joinpath(@__DIR__, "randomized_time_resonance_figure.jl"))

using CoolingTNS

backend = TNBackend()

# Interacting model used in the TN paper draft
ham_params = NiIsingParameters(10, 1.0, -1.05, 0.5)

# Deterministic TN evolution
sim_params = UnifiedSimulationParameters(
    DensityMatrix(),
    TrotterEvolution();
    Dmax=40,
    cutoff=1e-6,
    tau=0.2,
    pe=0.0,
)

# Keep scan modest by default
plot_time_randomization_resonances(
    backend=backend,
    ham_params=ham_params,
    sim_params=sim_params,
    steps=30,
    window_size=5,
    t_values=collect(range(0.2, 4.0; length=9)),
    seed_list=collect(1:5),
    output_name="randomized_times_remove_accidental_heating_tn_dm_trotter.pdf",
)
