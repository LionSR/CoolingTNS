#!/usr/bin/env julia
# Verify the σ_z sign convention in MapToSpin.tex vs code
using CoolingTNS, LinearAlgebra

function verify_sign()
    N = 6; θ = 0.4; J = cos(θ)/2; h = sin(θ)/2
    σx = [0.0 1; 1 0]; σz = [1.0 0; 0 -1]; I2 = [1.0 0; 0 1]
    kron_chain(ops) = reduce(kron, ops)

    d = 2^N; H = zeros(d, d)
    for i in 1:N
        ops = [I2 for _ in 1:N]; j = i%N+1; ops[i] = σz; ops[j] = σz
        H .+= J * kron_chain(ops)
    end
    for i in 1:N
        ops = [I2 for _ in 1:N]; ops[i] = σx; H .+= h * kron_chain(ops)
    end

    Px = kron_chain([σx for _ in 1:N])
    evals = eigvals(Symmetric(H)); evecs = eigvecs(Symmetric(H))

    # Find GS in even parity sector (Px=+1 → gF=-1 APBC)
    idx = 0; emin = Inf
    for i in 1:d
        v = evecs[:, i]
        px = real(v ⋅ (Px * v))
        if px > 0.5 && evals[i] < emin
            emin = evals[i]; idx = i
        end
    end

    ψ_gs = CoolingTNS.EDStateVector(ComplexF64.(evecs[:, idx]), N)
    ham_params = CoolingTNS.IsingParameters(N, J, h, :periodic)
    k_vals, nk_measured = CoolingTNS.measure_momentum_distribution_ed_clean(ψ_gs, ham_params)

    ks = CoolingTNS.allowed_k_indices(N, -1)

    println("="^70)
    println("σ_z SIGN CONVENTION VERIFICATION")
    println("="^70)
    println("N=$N, θ=$θ, J=$J, h=$h")
    println("Even-sector (Px=+1, gF=-1 APBC) GS energy: $emin")
    println()

    println("ALGEBRAIC CHECK:")
    println("  Notes JW: a = -S·σ⁻, a† = -S·σ⁺")
    println("  → a†a = σ⁺σ⁻ = (I+σ_z)/2")
    println("  → aa† = σ⁻σ⁺ = (I-σ_z)/2")
    println("  → aa† - a†a = -σ_z ≠ σ_z")
    println("  → 1 - 2a†a = 1 - (I+σ_z) = -σ_z ≠ σ_z")
    println()
    println("  CORRECT: σ_z = a†a - aa† = 2a†a - 1")
    println("  NOTES:   σ_z = aa† - a†a = 1 - 2a†a = -σ_z  [WRONG]")
    println()

    println("MODE OCCUPATION TEST (the decisive test):")
    println("  Code uses: w_k = +sinθ - cosθ cos(φk)")
    println("  Notes use: w_k = -sinθ - cosθ cos(φk)")
    println()

    println("k        n_k(ED)    n_k(code)  n_k(notes)  err_code     err_notes")
    println("-"^75)
    total_err_code = 0.0
    total_err_notes = 0.0
    for (i, k) in enumerate(ks)
        kf = Float64(k)
        φ_code = CoolingTNS.bogoliubov_angle(kf, θ, N)
        nk_code = sin(φ_code)^2

        φk = 2π * kf / N
        w_n = -sin(θ) - cos(θ) * cos(φk)
        r_n = -cos(θ) * sin(φk)
        φ_notes = abs(r_n) < 1e-14 ? 0.0 : atan(r_n, w_n) / 2
        nk_notes = sin(φ_notes)^2

        ec = abs(nk_measured[i] - nk_code)
        en = abs(nk_measured[i] - nk_notes)
        total_err_code += ec
        total_err_notes += en
        println("$(rpad(k,8)) $(rpad(round(nk_measured[i],digits=6),10)) $(rpad(round(nk_code,digits=6),10)) $(rpad(round(nk_notes,digits=6),10))  $(rpad(round(ec,sigdigits=3),12)) $(round(en,sigdigits=3))")
    end
    println("-"^75)
    println("TOTAL    $(rpad("",10)) $(rpad("",10)) $(rpad("",10))  $(rpad(round(total_err_code,sigdigits=3),12)) $(round(total_err_notes,sigdigits=3))")

    println()
    if total_err_code < total_err_notes
        println("★★★ CODE formula is correct (w_k = +sinθ - cosθ cos(φk))")
        println("    MapToSpin.tex line 40 has WRONG sign: σ_z ≠ aa† - a†a")
    else
        println("★★★ NOTES formula is correct (w_k = -sinθ - cosθ cos(φk))")
        println("    Code has wrong sign!")
    end
end

verify_sign()
