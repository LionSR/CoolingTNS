"""
    plot_ground_state_cooling_ising_tn_mcwf_trotter_multi_long_from_data.jl

Plot (publication-style) the long-run TN multi-frequency cooling curve for the
integrable Ising model from an HDF5 cache file produced by
`run_ground_state_cooling_ising_tn_mcwf_trotter_multi_long.jl`.

Usage:
  julia --project=. scripts/plotting/plot_ground_state_cooling_ising_tn_mcwf_trotter_multi_long_from_data.jl
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

using CoolingTNS
using Printf

infile = joinpath(
    @__DIR__,
    "Data",
    "ground_state_cooling_ising_tn_mcwf_trotter_multi_ZZ_N20_g0.2_te3.2_steps120_R15_seed1.h5",
)

data = read_h5_data(infile)
data === nothing && error("Missing data file: $infile")

E_list = Float64.(data[CoolingTNS.RESULT_ENERGY])
rel_list = Float64.(data["rel_list"])
E0 = Float64(_maybe_scalar(data["E0"]))
N = Int(_maybe_scalar(data["N"]))
steps = Int(_maybe_scalar(data["steps"]))

E_over_N = E_list ./ N
E0_over_N = E0 / N

plt = get_pyplot()
plt.rcParams.update(Dict(
    "font.size" => 9,
    "axes.labelsize" => 9,
    "axes.titlesize" => 9,
    "legend.fontsize" => 8,
    "xtick.labelsize" => 8,
    "ytick.labelsize" => 8,
    "lines.linewidth" => 1.8,
    "pdf.fonttype" => 42,
    "ps.fonttype" => 42,
))

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
steps_axis = collect(0:steps)

ax = axs[0]
ax.plot(steps_axis, E_over_N, color="C1", label="multi-Δ")
ax.axhline(E0_over_N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")
ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density $E/N$")
ax.set_title("TN MCWF+Trotter (Ising, N=20)")
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

ax = axs[1]
ax.plot(steps_axis, rel_list, color="C1", label="multi-Δ")
ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
ax.set_xlabel("cooling step")
ax.set_ylabel(L"relative energy $e$")
ax.set_title(L"$e = |(E-E_0)/E_0|$")
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

fig.tight_layout()

save_figure(fig, @__DIR__, "ground_state_cooling_ising_tn_mcwf_trotter_multi_ZZ_N20_g0.2_te3.2_steps120_R15.pdf")
