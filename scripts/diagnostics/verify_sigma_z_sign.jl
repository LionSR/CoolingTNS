#!/usr/bin/env julia
# Verify that the MapToSpin Jordan-Wigner sign convention agrees with the ED
# mode diagnostics.

using CoolingTNS
using LinearAlgebra

function _obsolete_opposite_z_occupation(k::Real, theta::Real, N::Int)
    phi_k = 2pi * Float64(k) / N
    w_old = -sin(theta) - cos(theta) * cos(phi_k)
    r_old = -cos(theta) * sin(phi_k)
    phi_old = abs(r_old) < 1e-14 ? 0.0 : atan(r_old, w_old) / 2
    return sin(phi_old)^2
end

function _local_jw_sign_errors()
    a, a_dag = CoolingTNS.jordan_wigner_transform_complex(1, 1)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Z = Matrix(CoolingTNS.pauli_z(1, 1))

    sigma_z_error = norm(Matrix(2 * a_dag * a) - I2 - Z)
    obsolete_sigma_z_error = norm(Matrix(a * a_dag - a_dag * a) - Z)
    return sigma_z_error, obsolete_sigma_z_error
end

function verify_sign(; N::Int=6, theta::Float64=0.4, verbose::Bool=true)
    J, h = CoolingTNS.Jh_from_theta(theta)
    theta_code = CoolingTNS.theta_from_Jh(J, h)
    ham_params = CoolingTNS.IsingParameters(N, J, h, :periodic)

    H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
    E0, psi0, _ = CoolingTNS.ground_state_ed(H_sys)

    parity_value = CoolingTNS.measure_state_parity(psi0, N)
    parity = CoolingTNS._reference_parity_sector(parity_value; atol=1e-8)
    if abs(parity_value - parity) > 1e-8
        error("Ground-state parity is not close to +/-1: parity = $parity_value")
    end

    gF = CoolingTNS.fermionic_bc(ham_params.bc, parity)
    ks = CoolingTNS.allowed_k_indices(N, gF)
    _, tilde_n_measured = CoolingTNS.measure_raw_fourier_occupation_ed(psi0, ham_params; gF=gF)

    tilde_n_canonical = [sin(CoolingTNS.bogoliubov_angle(Float64(k), theta_code, N))^2 for k in ks]
    tilde_n_obsolete = [_obsolete_opposite_z_occupation(k, theta_code, N) for k in ks]

    canonical_errors = abs.(tilde_n_measured .- tilde_n_canonical)
    obsolete_errors = abs.(tilde_n_measured .- tilde_n_obsolete)

    sigma_z_error, obsolete_sigma_z_error = _local_jw_sign_errors()

    result = (
        N=N,
        theta=theta_code,
        J=J,
        h=h,
        energy=E0,
        parity=parity_value,
        gF=gF,
        k_indices=ks,
        measured=tilde_n_measured,
        canonical=tilde_n_canonical,
        obsolete=tilde_n_obsolete,
        canonical_error=sum(canonical_errors),
        obsolete_error=sum(obsolete_errors),
        max_canonical_error=maximum(canonical_errors),
        sigma_z_error=sigma_z_error,
        obsolete_sigma_z_error=obsolete_sigma_z_error,
    )

    if verbose
        println("="^70)
        println("JW SIGN CONVENTION VERIFICATION")
        println("="^70)
        println("N=$(result.N), theta=$(result.theta), J=$(result.J), h=$(result.h)")
        println("Ground-state energy: $(result.energy)")
        println("Parity: $(result.parity), fermionic boundary gF=$(result.gF)")
        println()

        println("ALGEBRAIC CHECK")
        println("  MapToSpin convention: a = -S sigma^-, a^dag = -S sigma^+")
        println("  Canonical identity: sigma_z = 2 a^dag a - I")
        println("  local operator error: $(result.sigma_z_error)")
        println("  obsolete identity aa^dag - a^dag a error: $(result.obsolete_sigma_z_error)")
        println()

        println("MODE OCCUPATION CHECK")
        println("  canonical source: CoolingTNS.bogoliubov_angle")
        println("  negative control: obsolete opposite-sigma_z sign")
        println()
        println("k        tilde_n_k(ED)  tilde_n_k(canonical)  tilde_n_k(obsolete)  err_canonical  err_obsolete")
        println("-"^88)
        for (i, k) in enumerate(ks)
            println("$(rpad(k,8)) " *
                    "$(rpad(round(tilde_n_measured[i], digits=6),10)) " *
                    "$(rpad(round(tilde_n_canonical[i], digits=6),15)) " *
                    "$(rpad(round(tilde_n_obsolete[i], digits=6),14)) " *
                    "$(rpad(round(canonical_errors[i], sigdigits=3),15)) " *
                    "$(round(obsolete_errors[i], sigdigits=3))")
        end
        println("-"^88)
        println("TOTAL    " *
                rpad("", 10) * " " *
                rpad("", 15) * " " *
                rpad("", 14) * " " *
                rpad(round(result.canonical_error, sigdigits=4), 15) * " " *
                "$(round(result.obsolete_error, sigdigits=4))")
        println()
    end

    if result.sigma_z_error > 1e-12 || result.max_canonical_error > 1e-10
        error("Canonical Jordan-Wigner convention does not match ED.")
    end

    # The occupation-number negative control is meaningful only away from
    # sin(theta)=0, where flipping the sigma_z sign changes the BdG block.
    separation_factor = 100.0
    negative_control_is_distinct = abs(sin(theta_code)) > 1e-12
    if negative_control_is_distinct &&
            result.obsolete_error <= separation_factor * max(result.canonical_error, eps(Float64))
        error("Obsolete sign convention is not clearly separated from the canonical one.")
    end

    verbose && println("Canonical convention matches ED and the local JW algebra.")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_sign()
end
