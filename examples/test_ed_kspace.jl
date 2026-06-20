#!/usr/bin/env julia

"""
Smoke example for ED k-space measurements with periodic and antiperiodic
boundary conditions.

The k-space observables are defined only for the integrable transverse-field
Ising chain.  This file therefore uses `IsingParameters`, not the
nonintegrable Ising model.
"""

using CoolingTNS
using Printf

function _print_momentum_row(k_values, values; label, symbol)
    println(label)
    for (φ, value) in zip(k_values, values)
        @printf("  φ/π = %+8.5f  (φ = %+9.6f):  %s = %.6f\n", φ / π, φ, symbol, value)
    end
end

function run_ed_kspace_case(; bc::Symbol, sim_method, init_state::String, steps::Int=4)
    N = 6
    J = 1.0
    h = 2.0
    backend = CoolingTNS.EDBackend()
    ham_params = CoolingTNS.IsingParameters(N, J, h, bc)
    coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.3, steps, 2.0, nothing)
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        sim_method,
        CoolingTNS.ContinuousEvolution();
        pe=0.0,
        n_trajectories=1,
    )

    problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    state = CoolingTNS.setup_initial_state(problem, sim_params, init_state, 0.0)
    results = CoolingTNS.run_cooling(
        problem,
        state,
        coupling_params,
        sim_params,
        ham_params;
        measure_modes=true,
    )

    haskey(results, CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION) || error("missing momentum distribution")
    haskey(results, CoolingTNS.RESULT_K_VALUES) || error("missing k-grid")

    k_values = results[CoolingTNS.RESULT_K_VALUES]
    momentum_dist = results[CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION]

    println("\n$(typeof(sim_method)) with $bc spin boundary conditions")
    println("Number of momentum points: $(length(k_values))")
    _print_momentum_row(k_values, momentum_dist[1, :];
        label="Initial raw Fourier occupations:", symbol="tilde n_k")
    _print_momentum_row(k_values, momentum_dist[end, :];
        label="Final raw Fourier occupations:", symbol="tilde n_k")

    if haskey(results, CoolingTNS.RESULT_MODE_NK)
        mode_nk = results[CoolingTNS.RESULT_MODE_NK]
        mode_k_indices = results[CoolingTNS.RESULT_MODE_K_INDICES]
        mode_momenta = [2π * Float64(k) / N for k in mode_k_indices]
        _print_momentum_row(mode_momenta, mode_nk[end, :];
            label="Final Bogoliubov occupations:", symbol="n_k^Bog")
    end

    return results
end

function test_ed_kspace()
    println("Testing ED k-space measurements for the integrable Ising chain.")

    for bc in (:periodic, :antiperiodic)
        run_ed_kspace_case(
            bc=bc,
            sim_method=CoolingTNS.MonteCarloWavefunction(),
            init_state="product",
        )
        run_ed_kspace_case(
            bc=bc,
            sim_method=CoolingTNS.DensityMatrix(),
            init_state="identity",
        )
    end

    println("\nAll ED k-space smoke checks completed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_ed_kspace()
end
