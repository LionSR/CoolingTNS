"""
    plot_mode_energy_consistency_ed.jl

Generate a compact figure checking consistency between:

- direct energy expectation value E = \\langle H_S \\rangle
- reconstructed energy from mode measurements (integrable Ising, PBC)

This is intended for seminar slides (one-figure consistency check).

Usage:
  julia --project=. --startup-file=no scripts/plotting/plot_mode_energy_consistency_ed.jl

Output:
  scripts/plotting/Figs/mode_energy_consistency_ed.pdf
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

using CoolingTNS
using Random
using Statistics
using Printf

Random.seed!(1)

# ----------------------------------------------------------------------------
# Config (integrable Ising with PBC so that mode observables are available)
# ----------------------------------------------------------------------------

N = 6
J = 1.0
h = 0.5
bc = :periodic

ham_params = IsingParameters(N, J, h, bc)

coupling = "XX"
g = 0.3
te = 2.0
steps = 60

sim_params = UnifiedSimulationParameters(DensityMatrix(), ContinuousEvolution(); pe=0.0)

# Choose an initial state in a definite Px sector (|+x⟩^{\otimes N} gives Px=+1)
init_type = "theta"
init_theta = 0.0

# ----------------------------------------------------------------------------
# Run cooling with mode measurements
# ----------------------------------------------------------------------------

cp = BasicCouplingParameters(coupling, g, steps, te, nothing)
prob = setup_problem(EDBackend(), ham_params, cp, sim_params)
state0 = setup_initial_state(prob, sim_params, init_type, init_theta)

# Silence step-by-step printing (keeps runtime output short)
res = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        run_cooling(prob, state0, cp, sim_params, ham_params; measure_modes=true)
    end
end

E_direct = Float64.(res[RESULT_ENERGY])
mode_hk = Float64.(res[RESULT_MODE_HK])
k_indices = res[RESULT_MODE_K_INDICES]
E_modes = ising_energy_from_mode_hk(k_indices, mode_hk, ham_params)

ΔE_abs = abs.(E_direct .- E_modes)
ΔE_rel = similar(ΔE_abs)
for i in eachindex(ΔE_abs)
    ΔE_rel[i] = abs(E_direct[i]) > 0 ? ΔE_abs[i] / abs(E_direct[i]) : ΔE_abs[i]
end

@printf("Mode-energy reconstruction consistency (ED, Ising PBC): N=%d, steps=%d\n", N, steps)
@printf("  max |E_direct - E_modes| = %.3e\n", maximum(ΔE_abs))
@printf("  max rel. error           = %.3e\n", maximum(ΔE_rel))

# ----------------------------------------------------------------------------
# Plot (seminar style)
# ----------------------------------------------------------------------------

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
steps_ax = collect(0:steps)

ax = axs[0]
ax.plot(steps_ax, E_direct ./ N, color="C0", label=L"direct $E/N$")
ax.plot(steps_ax, E_modes ./ N, color="C1", linestyle="--", label=L"modes $E_{\mathrm{modes}}/N$")
ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density")
ax.set_title("Energy from modes vs direct")
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

ax = axs[1]
ax.semilogy(steps_ax, ΔE_abs ./ N .+ 1e-18, color="C3")
ax.set_xlabel("cooling step")
ax.set_ylabel(L"$|E-E_{\mathrm{modes}}|/N$")
ax.set_title("Reconstruction error")
ax.grid(true, alpha=0.25, which="both")

fig.suptitle(@sprintf("ED Ising (PBC): N=%d, %s, g=%.2f, t_e=%.1f", N, coupling, g, te), y=1.02)
fig.tight_layout()

save_figure(fig, @__DIR__, "mode_energy_consistency_ed.pdf")
