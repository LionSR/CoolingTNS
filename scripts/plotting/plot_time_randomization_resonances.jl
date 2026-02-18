""" 
    plot_time_randomization_resonances.jl

Driver script (ED demo) generating a publication-style figure showing how
randomizing cycle times suppresses accidental resonances/heating.

This uses the shared implementation in `randomized_time_resonance_figure.jl`.

Usage:
    julia --project=. scripts/plotting/plot_time_randomization_resonances.jl

Output (default):
    scripts/plotting/Figs/randomized_times_remove_accidental_heating_ed.pdf
"""

include(joinpath(@__DIR__, "randomized_time_resonance_figure.jl"))

using CoolingTNS

backend = EDBackend()
ham_params = NiIsingParameters(4, 1.0, -1.05, 0.5)
sim_params = UnifiedSimulationParameters(DensityMatrix(), ContinuousEvolution(); pe=0.0)

plot_time_randomization_resonances(
    backend=backend,
    ham_params=ham_params,
    sim_params=sim_params,
    output_name="randomized_times_remove_accidental_heating_ed.pdf",
)
