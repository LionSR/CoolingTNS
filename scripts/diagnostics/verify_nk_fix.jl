#!/usr/bin/env julia
# Verify that measure_momentum_distribution_ed_clean now:
# 1. Auto-detects parity and uses correct k-grid
# 2. Rotates to notes basis before computing n_k
# 3. Matches BdG predictions to machine precision
using CoolingTNS, LinearAlgebra

function verify_nk_fix()
    println("="^70)
    println("VERIFICATION: measure_momentum_distribution_ed_clean fix")
    println("="^70)

    all_pass = true

    for (N, θ_val) in [(4, 0.3), (6, 0.4), (8, 0.7)]
        J = cos(θ_val)/2; h = sin(θ_val)/2
        ham_params = CoolingTNS.IsingParameters(N, J, h, :periodic)
        H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
        E0, ψ0, gap = CoolingTNS.ground_state_ed(H_sys)

        px = CoolingTNS.measure_state_parity(ψ0, N)
        parity = round(Int, px)
        gF = CoolingTNS.fermionic_bc(:periodic, parity)
        ks = CoolingTNS.allowed_k_indices(N, gF)

        # Test auto-detection
        k_vals, nk = CoolingTNS.measure_momentum_distribution_ed_clean(ψ0, ham_params)

        # BdG predictions
        nk_pred = [sin(CoolingTNS.bogoliubov_angle(Float64(k), θ_val, N))^2 for k in ks]

        max_err = maximum(abs.(nk .- nk_pred))
        pass = max_err < 1e-10

        status = pass ? "✅" : "❌"
        println("\nN=$N, θ=$(round(θ_val,digits=2)), gF=$gF: max_err=$(round(max_err,sigdigits=3)) $status")

        if !pass
            all_pass = false
            for (i, k) in enumerate(ks)
                println("  k=$k: measured=$(round(nk[i],digits=6)), predicted=$(round(nk_pred[i],digits=6))")
            end
        end
    end

    println("\n" * "="^70)
    if all_pass
        println("ALL TESTS PASSED ✅")
    else
        println("SOME TESTS FAILED ❌")
    end
end

verify_nk_fix()
