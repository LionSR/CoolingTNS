#!/usr/bin/env julia
# Verify the TDVP time convention used by TN Monte Carlo continuous evolution.

using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

function _tdvp_test_vector(psi::MPS, sites)
    N = length(sites)
    dim = 2^N
    vec = zeros(ComplexF64, dim)
    for idx in 0:(dim - 1)
        config = [((idx >> (site - 1)) & 1) == 0 ? "Up" : "Dn" for site in 1:N]
        basis_state = MPS(sites, config)
        vec[idx + 1] = inner(basis_state, psi)
    end
    return vec
end

function tdvp_convention_check(; te::Float64=0.5, tau::Float64=0.5,
                               verbose::Bool=true)
    N = 2
    sites = siteinds("S=1/2", N)

    terms = OpSum()
    terms += 1.0, "X", 1
    terms += 1.0, "X", 2
    H_mpo = MPO(terms, sites)

    psi0 = MPS(sites, "Up")
    psi0_vec = ComplexF64[1, 0, 0, 0]
    H_mat = ComplexF64[
        0 1 1 0
        1 0 0 1
        1 0 0 1
        0 1 1 0
    ]

    exact_real = exp(CoolingTNS._tdvp_real_time(te) * H_mat) * psi0_vec
    exact_real ./= norm(exact_real)

    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.MonteCarloWavefunction(),
        CoolingTNS.ContinuousEvolution();
        Dmax=100,
        cutoff=1e-14,
        tau=tau,
    )
    ham_params = CoolingTNS.IsingParameters(N, 0.0, 0.0)

    evolved = CoolingTNS.evolve_state(
        ham_params, sim_params, CoolingTNS.TNBackend(), H_mpo, psi0, te, sites
    )
    evolved_vec = _tdvp_test_vector(evolved, sites)
    evolved_vec ./= norm(evolved_vec)

    # Negative control: a real TDVP time parameter is not the Schrodinger
    # evolution used by the package wrapper.
    nsteps = CoolingTNS._tdvp_step_count(te, tau)
    nonunitary = tdvp(H_mpo, te, psi0;
                      nsteps=nsteps, nsite=2, reverse_step=true,
                      normalize=true, maxdim=100, cutoff=1e-14, outputlevel=0)
    nonunitary_vec = _tdvp_test_vector(nonunitary, sites)
    nonunitary_vec ./= norm(nonunitary_vec)

    energy_exact = real(exact_real' * H_mat * exact_real)
    energy_evolved = real(evolved_vec' * H_mat * evolved_vec)
    overlap_real = abs(dot(exact_real, evolved_vec))
    overlap_nonunitary = abs(dot(exact_real, nonunitary_vec))
    norm_error = abs(norm(_tdvp_test_vector(evolved, sites)) - 1.0)

    result = (
        te=te,
        tau=tau,
        overlap_real=overlap_real,
        overlap_nonunitary=overlap_nonunitary,
        energy_exact=energy_exact,
        energy_evolved=energy_evolved,
        norm_error=norm_error,
    )

    if verbose
        println("="^60)
        println("TDVP REAL-TIME CONVENTION CHECK")
        println("="^60)
        println("Hamiltonian: H = X_1 + X_2")
        println(@sprintf("time te = %.6f, tau = %.6f", result.te, result.tau))
        println(@sprintf("overlap with exact exp(-i H t): %.12f", result.overlap_real))
        println(@sprintf("overlap for real tdvp time control: %.12f", result.overlap_nonunitary))
        println(@sprintf("exact energy:   %.12e", result.energy_exact))
        println(@sprintf("evolved energy: %.12e", result.energy_evolved))
        println(@sprintf("norm error:     %.12e", result.norm_error))
    end

    if result.overlap_real < 1 - 1e-8
        error("TN continuous evolution does not match real-time Schrodinger evolution.")
    end
    if abs(result.energy_evolved - result.energy_exact) > 1e-8
        error("TN continuous evolution does not conserve the exact two-spin energy.")
    end
    if result.overlap_nonunitary > 0.95
        error("The real-time TDVP negative control is not well separated.")
    end

    verbose && println("TDVP wrapper uses the correct real-time convention.")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    tdvp_convention_check()
end
