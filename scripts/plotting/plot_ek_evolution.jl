"""
Plot evolution of Bogoliubov mode energy contributions during cooling.

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plot_ek_evolution.jl <filename.h5>
"""

if !isdefined(@__MODULE__, :get_pyplot)
    Base.include(@__MODULE__, joinpath(@__DIR__, "PlotUtils.jl"))
end

if !isdefined(@__MODULE__, :_COOLINGTNS_PLOT_EK_EVOLUTION_INCLUDED)
const _COOLINGTNS_PLOT_EK_EVOLUTION_INCLUDED = true

const _MODE_ENERGY_DATA_KEYS = (
    CoolingTNS.RESULT_MODE_HK,
    CoolingTNS.RESULT_MODE_K_INDICES,
    CoolingTNS.RESULT_MODE_ENERGIES,
)

_has_mode_energy_data(data::AbstractDict) = all(key -> haskey(data, key), _MODE_ENERGY_DATA_KEYS)

function _warn_missing_mode_energy_data(filename, data)
    if haskey(data, CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION) &&
       haskey(data, CoolingTNS.RESULT_K_VALUES)
        @warn "File $filename contains raw Fourier occupations tilde_n_k, but not Bogoliubov mode data. " *
              "Not plotting epsilon_k*tilde_n_k as an energy; rerun with measure_modes=true."
    else
        @warn "No Bogoliubov mode energy data found in file $filename"
    end
end

function _mode_matrix_steps_by_modes(values::AbstractMatrix, n_modes::Int, name::AbstractString;
                                     n_steps=nothing)
    if size(values, 1) == size(values, 2) == n_modes
        @warn "$name is square; assuming canonical steps-by-modes layout; no transposition is applied"
    end
    if n_steps !== nothing && size(values, 1) != n_steps
        throw(DimensionMismatch(
            "$name has $(size(values, 1)) rows, but expected $n_steps cooling steps",
        ))
    end
    size(values, 2) == n_modes || throw(DimensionMismatch(
        "$name has $(size(values, 2)) columns, but expected $n_modes modes",
    ))
    return Float64.(values)
end

function _mode_energy_coefficients(k_indices, N::Int, J::Real, h::Real)
    θ = CoolingTNS.theta_from_Jh(J, h)
    Λ = CoolingTNS.energy_scale(J, h)
    return [(Λ / 2) * CoolingTNS.coeff_k(Float64(k), θ, N) for k in k_indices]
end

function _checked_mode_energy_coefficients(εk_values, k_indices, N::Int, J::Real, h::Real)
    length(εk_values) == length(k_indices) || throw(DimensionMismatch(
        "$(CoolingTNS.RESULT_MODE_ENERGIES) length $(length(εk_values)) does not match " *
        "$(CoolingTNS.RESULT_MODE_K_INDICES) length $(length(k_indices))",
    ))

    coeffs = _mode_energy_coefficients(k_indices, N, J, h)
    computed_εk = abs.(2 .* coeffs)
    if !isapprox(Float64.(εk_values), computed_εk; rtol=1e-8, atol=1e-10)
        @warn "Stored mode energies differ from coefficients reconstructed from N, J, h"
    end
    return coeffs
end

function _mode_energy_contributions(mode_hk::AbstractMatrix, coeffs::AbstractVector;
                                    n_steps=nothing)
    hk = _mode_matrix_steps_by_modes(
        mode_hk, length(coeffs), CoolingTNS.RESULT_MODE_HK; n_steps=n_steps)
    return hk .* reshape(coeffs, 1, :)
end

function _mode_energy_contributions(mode_hk::AbstractMatrix, k_indices, N::Int, J::Real, h::Real;
                                    n_steps=nothing)
    coeffs = _mode_energy_coefficients(k_indices, N, J, h)
    return _mode_energy_contributions(mode_hk, coeffs; n_steps=n_steps)
end

_mode_phase_over_pi(k_indices, N::Int) = [2 * Float64(k) / N for k in k_indices]

"""
    plot_ek_evolution(filename; steps_to_plot=nothing, save_fig=true)

Plot the evolution of the mode energy contributions
``E_k = (Λ/2) coeff_k <h_k>`` in code units during cooling.

This uses the Bogoliubov mode observable ``h_k`` from `RESULT_MODE_HK`, in the
notation of `Notes/NotesED/MapToSpin.tex`. The Fourier occupation
``<tilde a_k^dag tilde a_k>`` stored in `RESULT_MOMENTUM_DISTRIBUTION` is not a
mode energy and is deliberately not used here.  If the file contains
`RESULT_MODE_MEASUREMENT_CYCLES`, `steps_to_plot` selects among the measured
rows and the plotted labels use the corresponding physical cooling cycles.
"""
function plot_ek_evolution(filename; steps_to_plot=nothing, save_fig=true)
    plt = get_pyplot()

    data = read_h5_data(filename)
    data === nothing && return

    if !_has_mode_energy_data(data)
        _warn_missing_mode_energy_data(filename, data)
        return
    end

    mode_hk = data[CoolingTNS.RESULT_MODE_HK]
    k_indices = data[CoolingTNS.RESULT_MODE_K_INDICES]
    εk_values = Float64.(data[CoolingTNS.RESULT_MODE_ENERGIES])
    N = Int(_maybe_scalar(data["N"]))
    J = Float64(_maybe_scalar(get(data, "J", 1.0)))
    h = Float64(_maybe_scalar(get(data, "h", 1.0)))
    bc = Symbol(get(data, "bc", "open"))

    # `RESULT_MODE_ENERGIES` stores positive excitation energies. The plot uses
    # signed `coeff_k` instead, because special modes carry the sign through w_k.
    coeffs = _checked_mode_energy_coefficients(εk_values, k_indices, N, J, h)
    n_steps_expected = haskey(data, CoolingTNS.RESULT_ENERGY) ?
                       length(data[CoolingTNS.RESULT_ENERGY]) : nothing
    energy_contrib_full = _mode_energy_contributions(mode_hk, coeffs; n_steps=n_steps_expected)
    measured = _mode_measurement_cycle_rows(data, size(energy_contrib_full, 1))
    energy_contrib = energy_contrib_full[measured.rows, :]
    x_values = _mode_phase_over_pi(k_indices, N)
    total_steps = size(energy_contrib, 1)
    step_indices = select_evolution_steps(total_steps; steps_to_plot=steps_to_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_evolution_colors(plt, length(step_indices))

    for (idx, step_idx) in enumerate(step_indices)
        if step_idx <= total_steps
            E_k = energy_contrib[step_idx, :]
            cycle = measured.cycles[step_idx]
            label = cycle == 0 ? "Initial" : "Cycle $cycle"
            ax.plot(x_values, E_k, "o-",
                   color=colors[idx], linewidth=2, markersize=6, label=label)
        end
    end

    ax.plot(x_values, -coeffs, "k--", linewidth=2.5, label=L"\langle h_k\rangle=-1")
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_xlabel(L"\phi_k/\pi", fontsize=14)
    ax.set_ylabel(L"E_k = \frac{\Lambda}{2}\mathrm{coeff}_k\langle h_k\rangle", fontsize=14)
    ax.set_title("Bogoliubov Mode Energy Contributions\n(N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best", fontsize=12)
    ax.set_xlim(minimum(x_values), maximum(x_values))
    plt.tight_layout()

    if save_fig
        base_filename = extract_filename_base(filename)
        save_figure(fig, dirname(filename), "mode_energy_evolution_$(base_filename).pdf")
    end

    return fig
end

end
