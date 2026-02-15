using CoolingTNS, LinearAlgebra

function verify_nk()
    N = 6; θ = 0.4; J = cos(θ)/2; h = sin(θ)/2
    
    ham_params = CoolingTNS.IsingParameters(N, J, h, :periodic)
    H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
    E0, ψ0, gap = CoolingTNS.ground_state_ed(H_sys)
    
    parity = CoolingTNS.measure_state_parity(ψ0, N)
    println("GS energy: $E0, parity: $parity")
    
    # GS has Px=+1, so gF=-1 (APBC fermions, half-integer k)
    gF = round(Int, parity) == 1 ? -1 : 1  # gF = -P for spin PBC
    ks = CoolingTNS.allowed_k_indices(N, gF)
    println("Correct k-grid (gF=$gF): $ks")
    
    # Rotate to notes basis (returns plain Vector{ComplexF64})
    ψ_notes_vec = CoolingTNS._rotate_state_to_notes(ψ0)
    
    # Build JW operators  
    a_ops = [Matrix(CoolingTNS.jordan_wigner_transform_complex(n, N)[1]) for n in 1:N]
    a_dag_ops = [Matrix(CoolingTNS.jordan_wigner_transform_complex(n, N)[2]) for n in 1:N]
    
    # Measure n_k = ⟨ã†_k ã_k⟩ with half-integer k
    println("\nMeasuring n_k with CORRECT half-integer k-grid:")
    println("k        n_k(ED)    n_k(code_BdG)  n_k(notes_BdG)  err_code   err_notes")
    println("-"^80)
    
    total_err_code = 0.0
    total_err_notes = 0.0
    
    for k in ks
        kf = Float64(k)
        
        # Measure ⟨ã†_k ã_k⟩ using code's Fourier convention (exp(+inφ_k))
        nk = 0.0 + 0.0im
        for m in 1:N, n in 1:N
            phase = exp(2π * im * kf * (m - n) / N) / N
            nk += phase * dot(ψ_notes_vec, a_dag_ops[m] * a_ops[n] * ψ_notes_vec)
        end
        nk_measured = real(nk)
        
        # Code BdG prediction: w_k = sinθ - cosθ cos(φk)
        φ_code = CoolingTNS.bogoliubov_angle(kf, θ, N)
        nk_code = sin(φ_code)^2
        
        # Notes BdG prediction: w_k = -sinθ - cosθ cos(φk)
        φk = 2π * kf / N
        w_notes = -sin(θ) - cos(θ)*cos(φk)
        r_notes = -cos(θ)*sin(φk)
        φ_notes = abs(r_notes) < 1e-14 ? 0.0 : atan(r_notes, w_notes) / 2
        nk_notes = sin(φ_notes)^2
        
        ec = abs(nk_measured - nk_code)
        en = abs(nk_measured - nk_notes)
        total_err_code += ec
        total_err_notes += en
        
        println("$(rpad(k,8)) $(rpad(round(nk_measured,digits=6),10)) $(rpad(round(nk_code,digits=6),14)) $(rpad(round(nk_notes,digits=6),15)) $(rpad(round(ec,sigdigits=3),10)) $(round(en,sigdigits=3))")
    end
    
    println("-"^80)
    println("TOTAL ERROR:  $(rpad("",10)) $(rpad("",14)) $(rpad("",15)) $(rpad(round(total_err_code,sigdigits=4),10)) $(round(total_err_notes,sigdigits=4))")
    
    println()
    if total_err_code < total_err_notes * 0.1
        println("★★★ CODE formula WINS decisively: w_k = +sinθ - cosθ cos(φk)")
    elseif total_err_notes < total_err_code * 0.1
        println("★★★ NOTES formula WINS decisively: w_k = -sinθ - cosθ cos(φk)")
    else
        println("??? Neither formula wins clearly. Errors are comparable.")
        println("    Code total: $total_err_code, Notes total: $total_err_notes")
    end
end

verify_nk()
