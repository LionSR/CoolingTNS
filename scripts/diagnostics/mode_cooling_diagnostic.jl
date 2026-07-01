#!/usr/bin/env julia
"""
    mode_cooling_diagnostic.jl

Standalone diagnostic script that:
1. Sets up a small periodic or antiperiodic Ising chain.
2. Runs ED DensityMatrix cooling with mode measurement.
3. Prints a summary table of Bogoliubov occupations n_k^Bog at each step
4. Verifies energy consistency: (Λ/2) Σ_k coeff_k · ⟨h_k⟩ ≈ ⟨H⟩
5. Optionally generates a matplotlib plot

Usage:
    julia --project=. --startup-file=no scripts/diagnostics/mode_cooling_diagnostic.jl [options]

Useful short exact control:
    julia --project=. --startup-file=no scripts/diagnostics/mode_cooling_diagnostic.jl \
        --N 4 --steps 3 --h -1.05 --te 1.0 --coupling XX
"""

using CoolingTNS
using Printf

# ============================================================================
# Parameters and CLI parsing
# ============================================================================

const DEFAULT_MODE_COOLING_DIAGNOSTIC_CONFIG = (
    N=6,
    J=1.0,
    h=0.5,
    bc=:periodic,
    coupling="XX",
    g=0.3,
    te=2.0,
    steps=50,
    init_type="theta",
    init_angle=π / 4,
    theta_code=CoolingTNS.theta_code_from_initial_product_angle(π / 4),
    do_plot=false,
)

function mode_cooling_diagnostic_usage(io=stdout)
    println(io, "usage: julia --project=. --startup-file=no scripts/diagnostics/mode_cooling_diagnostic.jl [options]")
    println(io)
    println(io, "Options:")
    println(io, "  --N INT              even system size, default 6")
    println(io, "  --steps INT          cooling cycles, default 50")
    println(io, "  --J FLOAT            Ising coupling, default 1.0")
    println(io, "  --h FLOAT            transverse field, default 0.5")
    println(io, "  --bc periodic|antiperiodic")
    println(io, "  --coupling AB        system-bath Pauli label in {X,Y,Z}^2, default XX")
    println(io, "  --g FLOAT            system-bath coupling strength, default 0.3")
    println(io, "  --te FLOAT           evolution time per cycle, default 2.0")
    println(io, "  --init-angle FLOAT   per-site angle alpha for cos(alpha)|0>+sin(alpha)|1>")
    println(io, "  --theta-code FLOAT   code theta parameter; mutually exclusive with --init-angle")
    println(io, "  --plot, -p           also write a mode-occupation plot")
    println(io, "  --help, -h           print this message")
    return nothing
end

function _parse_mode_bc(value::AbstractString)
    label = Symbol(strip(lowercase(value)))
    label in (:periodic, :antiperiodic) && return label
    throw(ArgumentError("--bc must be periodic or antiperiodic, got $(repr(value))"))
end

function _mode_arg_value(args, index::Integer, flag::AbstractString)
    index < length(args) || throw(ArgumentError("$flag requires a value"))
    return args[index + 1]
end

function validate_mode_cooling_diagnostic_config(config)
    config.N >= 2 && iseven(config.N) ||
        throw(ArgumentError("--N must be an even integer at least 2"))
    config.steps >= 1 || throw(ArgumentError("--steps must be at least 1"))
    all(isfinite, (config.J, config.h, config.g, config.te, config.theta_code)) ||
        throw(ArgumentError("--J, --h, --g, --te, and theta_code must be finite"))
    config.te >= 0 || throw(ArgumentError("--te must be non-negative"))
    config.bc in (:periodic, :antiperiodic) ||
        throw(ArgumentError("--bc must be periodic or antiperiodic"))
    CoolingTNS.parse_coupling(config.coupling)
    config.init_type == "theta" ||
        throw(ArgumentError("mode cooling diagnostic currently uses init_type=\"theta\""))
    return config
end

function parse_mode_cooling_diagnostic_args(args=ARGS; io=stdout)
    config = Dict{Symbol,Any}(pairs(DEFAULT_MODE_COOLING_DIAGNOSTIC_CONFIG))
    init_angle_seen = false
    theta_code_seen = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            mode_cooling_diagnostic_usage(io)
            return nothing
        elseif arg in ("--plot", "-p")
            config[:do_plot] = true
            i += 1
        elseif arg == "--N"
            config[:N] = parse(Int, _mode_arg_value(args, i, arg))
            i += 2
        elseif arg == "--steps"
            config[:steps] = parse(Int, _mode_arg_value(args, i, arg))
            i += 2
        elseif arg in ("--J", "--h", "--g", "--te")
            key = Symbol(arg[3:end])
            config[key] = parse(Float64, _mode_arg_value(args, i, arg))
            i += 2
        elseif arg == "--bc"
            config[:bc] = _parse_mode_bc(_mode_arg_value(args, i, arg))
            i += 2
        elseif arg == "--coupling"
            config[:coupling] = _mode_arg_value(args, i, arg)
            i += 2
        elseif arg == "--init-angle"
            init_angle_seen = true
            theta_code_seen && throw(ArgumentError(
                "--init-angle and --theta-code are mutually exclusive"
            ))
            angle = parse(Float64, _mode_arg_value(args, i, arg))
            config[:init_angle] = angle
            config[:theta_code] = CoolingTNS.theta_code_from_initial_product_angle(angle)
            i += 2
        elseif arg == "--theta-code"
            theta_code_seen = true
            init_angle_seen && throw(ArgumentError(
                "--init-angle and --theta-code are mutually exclusive"
            ))
            theta_code = parse(Float64, _mode_arg_value(args, i, arg))
            config[:theta_code] = theta_code
            config[:init_angle] = CoolingTNS.initial_product_angle(theta_code)
            i += 2
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end

    return validate_mode_cooling_diagnostic_config((; config...))
end

_mode_index_label(k) = k isa Rational ? "$(numerator(k))/$(denominator(k))" : "$(k)"

function _mode_occupation_from_diagnostic_results(results)
    return CoolingTNS.mode_occupation_from_results(results; require_hk=true)
end

# ============================================================================
# Setup
# ============================================================================

function run_diagnostic(config=DEFAULT_MODE_COOLING_DIAGNOSTIC_CONFIG; do_plot=nothing)
    config = validate_mode_cooling_diagnostic_config(config)
    N = config.N
    J = config.J
    h = config.h
    bc = config.bc
    coupling = config.coupling
    g_coupling = config.g
    te = config.te
    steps = config.steps
    init_type = config.init_type
    init_product_angle = config.init_angle
    init_theta_code = config.theta_code
    do_plot = do_plot === nothing ? config.do_plot : do_plot

    println("=" ^ 80)
    println("  Mode-Resolved Cooling Diagnostic")
    println("=" ^ 80)
    println()

    # Model parameters
    ham_params = CoolingTNS.IsingParameters(N, J, h, bc)
    coupling_params = CoolingTNS.BasicCouplingParameters(
        coupling, g_coupling, steps, te, nothing
    )
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution()
    )

    # Print parameter summary
    θ = CoolingTNS.theta_from_Jh(J, h)
    Λ = CoolingTNS.energy_scale(J, h)
    println("Model: Ising, N=$N, J=$J, h=$h, BC=$bc")
    println("θ = $(round(θ, digits=4)) rad = $(round(rad2deg(θ), digits=2))°")
    println("Λ = 2√(J²+h²) = $(round(Λ, digits=4))")
    println("Coupling: $coupling, g=$g_coupling, te=$te, steps=$steps")
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
    parity = CoolingTNS._reference_parity_sector(px)
    gF = CoolingTNS.fermionic_bc(bc, parity)
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
    res_disp_indices = Set(CoolingTNS.nearest_bath_resonance_indices(εk_all, Δ))
    for (idx, k) in enumerate(ks)
        εk_notes = CoolingTNS.mode_energy(Float64(k), θ, N)
        εk_code = Λ * εk_notes
        φk = 2π * Float64(k) / N
        k_str = _mode_index_label(k)
        @printf("  %-8s  %12.6f  %12.6f  %12.6f  %8.4f%s\n",
                k_str, φk, εk_code, εk_notes, abs(εk_code - abs(Δ)),
                idx in res_disp_indices ? "  ← resonant" : "")
    end
    println()

    # Setup initial state and run cooling
    println(
        "Setting up initial state ($init_type, alpha=$(round(init_product_angle, digits=4)), " *
        "theta_code=$(round(init_theta_code, digits=4)))..."
    )
    state0 = CoolingTNS.setup_initial_state(problem, sim_params, init_type, init_theta_code)
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

    mode_hk = results[CoolingTNS.RESULT_MODE_HK]
    mode_nk = _mode_occupation_from_diagnostic_results(results)
    k_indices = results[CoolingTNS.RESULT_MODE_K_INDICES]
    εk_values = results[CoolingTNS.RESULT_MODE_ENERGIES]
    E_list = results[CoolingTNS.RESULT_ENERGY]
    overlap_list = results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP]
    n_steps_total = size(mode_hk, 1)
    n_modes = size(mode_hk, 2)

    # Find every mode closest to the bath detuning.
    res_indices = Set(CoolingTNS.nearest_bath_resonance_indices(εk_values, Δ))

    println("=" ^ 80)
    println("  Cooling Summary Table: Bogoliubov occupations n_k^Bog")
    println("=" ^ 80)
    println("  Table entries are quasiparticle occupations n_k^Bog; resonant modes are marked by *.")
    println()

    # Header
    k_headers = ["k=$(_mode_index_label(k))" for k in k_indices]
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
            marker = i in res_indices ? "*" : " "
            @printf("  %9.5f%s", mode_nk[step, i], marker)
        end
        println()
    end
    println()
    println("  * = resonant modes (closest to Δ = $(round(abs(Δ), digits=4)))")
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
        E_modes = CoolingTNS.ising_energy_from_mode_hk(
            k_indices, view(mode_hk, step, :), ham_params
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
    println("  Mode Cooling Effectiveness: Bogoliubov occupations n_k^Bog")
    println("=" ^ 80)
    @printf("  %-10s  %-10s  %-12s  %-12s  %-10s  %-10s\n",
            "k", "ε_k", "n_Bog(init)", "n_Bog(final)", "Δn_Bog", "|ε_k - Δ|")
    println("  " * "─" ^ 66)

    for i in 1:n_modes
        k = k_indices[i]
        k_str = _mode_index_label(k)
        nk_init = mode_nk[1, i]
        nk_final = mode_nk[end, i]
        marker = i in res_indices ? "  ← resonant" : ""
        @printf("  %-10s  %10.4f  %12.6f  %12.6f  %10.6f  %10.4f%s\n",
                k_str, εk_values[i], nk_init, nk_final, nk_final - nk_init,
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
            fig = plot_mode_occupation_from_data(mode_nk, k_indices, εk_values;
                                                 delta=Δ, savepath=savepath,
                                                 title="Mode occupation cooling: N=$N, J=$J, h=$h, $coupling coupling")
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

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_mode_cooling_diagnostic_args(ARGS)
    config === nothing && exit(0)
    run_diagnostic(config)
end
