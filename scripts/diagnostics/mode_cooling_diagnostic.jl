#!/usr/bin/env julia
"""
    mode_cooling_diagnostic.jl

Standalone diagnostic script that:
1. Sets up a small Ising chain (N=6, PBC) with reasonable parameters
2. Runs ED DensityMatrix cooling for ~50 steps with mode measurement
3. Prints a summary table of mode energies at each step
4. Verifies energy consistency: (Λ/2) Σ_k coeff_k · ⟨h_k⟩ ≈ ⟨H⟩
5. Optionally generates a matplotlib plot

Usage:
    julia --project=. --startup-file=no scripts/diagnostics/mode_cooling_diagnostic.jl [--plot]
"""

using CoolingTNS
using Printf

# ============================================================================
# Parameters
# ============================================================================

const N = 6           # System size
const J = 1.0         # Ising coupling
const h = 0.5         # Transverse field
const BC = :periodic   # Boundary condition
const COUPLING = "XX"  # System-bath coupling
const G_COUPLING = 0.3 # Coupling strength
const TE = 2.0         # Evolution time per step
const STEPS = 50       # Number of cooling steps
const INIT_TYPE = "theta"
const INIT_THETA = π / 4

# ============================================================================
# Setup
# ============================================================================

function run_diagnostic(; do_plot::Bool=false)
    println("=" ^ 80)
    println("  Mode-Resolved Cooling Diagnostic")
    println("=" ^ 80)
    println()

    # Model parameters
    ham_params = CoolingTNS.IsingParameters(N, J, h, BC)
    coupling_params = CoolingTNS.BasicCouplingParameters(COUPLING, G_COUPLING, STEPS, TE, nothing)
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution()
    )

    # Print parameter summary
    θ = CoolingTNS.theta_from_Jh(J, h)
    Λ = CoolingTNS.energy_scale(J, h)
    println("Model: Ising, N=$N, J=$J, h=$h, BC=$BC")
    println("θ = $(round(θ, digits=4)) rad = $(round(rad2deg(θ), digits=2))°")
    println("Λ = 2√(J²+h²) = $(round(Λ, digits=4))")
    println("Coupling: $COUPLING, g=$G_COUPLING, te=$TE, steps=$STEPS")
    println()

    # Setup problem
    println("Setting up problem...")
    problem = CoolingTNS.setup_problem(CoolingTNS.EDBackend(), ham_params, coupling_params, sim_params)
    Δ = problem.extra.coupling_params.delta
    @printf("Ground state energy: %.6f (E/N = %.6f)\n", problem.e₀, problem.e₀ / N)
    @printf("System gap (Δ): %.6f\n", Δ)
    println()

    # Ground state parity and modes
    px = CoolingTNS.measure_state_parity(problem.ϕ₀, N)
    gF = CoolingTNS.fermionic_bc(BC, round(Int, px))
    ks = CoolingTNS.allowed_k_indices(N, gF)
    println("Ground state parity ⟨Px⟩ = $(round(px, digits=6)), gF = $gF")
    println("Allowed k-indices: $ks")
    println()

    # Print mode dispersion
    println("─" ^ 60)
    println("  Mode Dispersion")
    println("─" ^ 60)
    @printf("  %-8s  %-12s  %-12s  %-12s  %-8s\n", "k", "φ_k", "ε_k(code)", "ε_k(notes)", "|ε_k - Δ|")
    println("  " * "─" ^ 56)
    εk_all = [Λ * CoolingTNS.mode_energy(Float64(ki), θ, N) for ki in ks]
    res_disp_idx = argmin(abs.(εk_all .- abs(Δ)))
    for (idx, k) in enumerate(ks)
        εk_notes = CoolingTNS.mode_energy(Float64(k), θ, N)
        εk_code = Λ * εk_notes
        φk = 2π * Float64(k) / N
        k_str = k isa Rational ? "$(numerator(k))/$(denominator(k))" : "$k"
        @printf("  %-8s  %12.6f  %12.6f  %12.6f  %8.4f%s\n",
                k_str, φk, εk_code, εk_notes, abs(εk_code - abs(Δ)),
                idx == res_disp_idx ? "  ← resonant" : "")
    end
    println()

    # Setup initial state and run cooling
    println("Setting up initial state ($INIT_TYPE, θ=$(round(INIT_THETA, digits=4)))...")
    state0 = CoolingTNS.setup_initial_state(problem, sim_params, INIT_TYPE, INIT_THETA)
    px_init = CoolingTNS.measure_state_parity(state0.state, N)
    @printf("Initial state parity ⟨Px⟩ = %.6f\n", px_init)
    println()

    println("Running cooling with mode measurement...")
    println("─" ^ 80)
    results = redirect_stdout(devnull) do
        CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params; measure_modes=true)
    end
    println("Cooling completed.")
    println()

    # ========================================================================
    # Print mode cooling summary
    # ========================================================================

    mode_hk = results["mode_hk"]
    k_indices = results["mode_k_indices"]
    εk_values = results["mode_ek_values"]
    E_list = results["E_list"]
    overlap_list = results["GS_overlap_list"]
    n_steps_total = size(mode_hk, 1)
    n_modes = size(mode_hk, 2)

    # Find resonant mode
    res_idx = argmin(abs.(εk_values .- abs(Δ)))

    println("=" ^ 80)
    println("  Cooling Summary Table")
    println("=" ^ 80)

    # Header
    k_headers = [k isa Rational ? "k=$(numerator(k))/$(denominator(k))" : "k=$k" for k in k_indices]
    @printf("  %-6s  %-10s  %-8s", "Step", "E/N", "Overlap")
    for kh in k_headers
        @printf("  %-10s", kh)
    end
    println()
    println("  " * "─" ^ (28 + 12 * n_modes))

    # Print selected steps
    print_steps = unique(vcat(
        [1, 2, 3],
        [div(n_steps_total, 4), div(n_steps_total, 2), div(3 * n_steps_total, 4)],
        [n_steps_total]
    ))
    sort!(print_steps)

    for step in print_steps
        if step < 1 || step > n_steps_total
            continue
        end
        @printf("  %-6d  %10.6f  %8.5f", step - 1, E_list[step] / N, overlap_list[step])
        for i in 1:n_modes
            marker = i == res_idx ? "*" : " "
            @printf("  %9.5f%s", mode_hk[step, i], marker)
        end
        println()
    end
    println()
    println("  * = resonant mode (closest to Δ = $(round(abs(Δ), digits=4)))")
    println()

    # ========================================================================
    # Energy consistency check
    # ========================================================================

    println("=" ^ 80)
    println("  Energy Consistency Check: (Λ/2) Σ_k coeff_k · ⟨h_k⟩  vs  ⟨H⟩")
    println("=" ^ 80)
    println()
    println("  Note: The decomposition H = (Λ/2) Σ_k coeff_k · h_k is exact only within")
    println("  a single fermion parity sector. For mixed states (density matrices) that span")
    println("  both sectors, a discrepancy is expected. The check becomes more accurate as")
    println("  the state is cooled toward the ground state (definite parity).")
    println()

    @printf("  %-6s  %-14s  %-14s  %-14s  %-8s\n", "Step", "⟨H⟩ direct", "⟨H⟩ modes", "Δ(abs)", "Rel err")
    println("  " * "─" ^ 60)

    for step in print_steps
        if step < 1 || step > n_steps_total
            continue
        end
        E_direct = E_list[step]
        E_modes = (Λ / 2) * sum(
            CoolingTNS.coeff_k(k, θ, N) * mode_hk[step, i]
            for (i, k) in enumerate(k_indices)
        )
        abs_err = abs(E_direct - E_modes)
        rel_err = abs(E_direct) > 1e-10 ? abs_err / abs(E_direct) : abs_err
        @printf("  %-6d  %14.8f  %14.8f  %14.2e  %8.2e\n",
                step - 1, E_direct, E_modes, abs_err, rel_err)
    end
    println()

    # ========================================================================
    # Cooling effectiveness per mode
    # ========================================================================

    println("=" ^ 80)
    println("  Mode Cooling Effectiveness")
    println("=" ^ 80)
    @printf("  %-10s  %-10s  %-12s  %-12s  %-10s  %-10s\n",
            "k", "ε_k", "⟨h_k⟩_init", "⟨h_k⟩_final", "Δ⟨h_k⟩", "|ε_k - Δ|")
    println("  " * "─" ^ 66)

    for i in 1:n_modes
        k = k_indices[i]
        k_str = k isa Rational ? "$(numerator(k))/$(denominator(k))" : "$k"
        hk_init = mode_hk[1, i]
        hk_final = mode_hk[end, i]
        marker = i == res_idx ? "  ← resonant" : ""
        @printf("  %-10s  %10.4f  %12.6f  %12.6f  %10.6f  %10.4f%s\n",
                k_str, εk_values[i], hk_init, hk_final, hk_final - hk_init,
                abs(εk_values[i] - abs(Δ)), marker)
    end
    println()

    # ========================================================================
    # Optional plotting
    # ========================================================================

    if do_plot
        println("Generating plot...")
        try
            include(joinpath(@__DIR__, "..", "plotting", "plot_mode_cooling.jl"))
            savepath = joinpath(@__DIR__, "..", "..", "Figs", "mode_cooling_diagnostic_N$(N).pdf")
            mkpath(dirname(savepath))
            fig = plot_mode_cooling_from_data(mode_hk, k_indices, εk_values;
                                              delta=Δ, savepath=savepath,
                                              title="Mode cooling: N=$N, J=$J, h=$h, $COUPLING coupling")
            println("Plot saved to $savepath")
        catch e
            if isa(e, LoadError) || isa(e, MethodError)
                @warn "Plotting not available (matplotlib/PythonCall not installed?): $e"
            else
                @warn "Plotting failed: $e"
                for (exc, bt) in Base.catch_stack()
                    showerror(stderr, exc, bt)
                    println(stderr)
                end
            end
            println("Skipping plot generation. Run with PythonCall + matplotlib for plots.")
        end
    end

    println("=" ^ 80)
    println("  Diagnostic complete.")
    println("=" ^ 80)

    return results
end

# ============================================================================
# Main
# ============================================================================

do_plot = "--plot" in ARGS || "-p" in ARGS
run_diagnostic(; do_plot=do_plot)
