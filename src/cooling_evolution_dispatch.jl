"""
    cooling_evolution_dispatch.jl

Pure dispatch implementation of cooling evolution.
NO FUNCTION NAMES ENCODING METHODS - just run_cooling with type dispatch!
"""

using ITensors
using ITensorMPS
using ITensors: apply
using Yao
using ExponentialUtilities
using LinearAlgebra
using Random
using Statistics

# ============================================================================
# Main Cooling Evolution Interface
# ============================================================================

"""
    run_cooling(problem::CoolingProblem, state::QuantumState, coupling_params, sim_params, ham_params)

Run cooling simulation with pure type dispatch.
"""
function run_cooling(problem::CoolingProblem{B}, state::QuantumState{B,S,E}, 
                    coupling_params, sim_params, 
                    ham_params) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    error("run_cooling not implemented for backend=$B, sim_method=$S, evolution_method=$E")
end

# ============================================================================
# TN Backend + Monte Carlo + Continuous Evolution
# ============================================================================

function run_cooling(problem::CoolingProblem{TNBackend}, 
                    state::QuantumState{TNBackend,MonteCarloWavefunction,ContinuousEvolution}, 
                    coupling_params, sim_params, ham_params)
    
    sites = problem.sites
    H_sys = problem.H_sys
    H_sys_bath = problem.H_sys_bath
    ϕ₀ = problem.ϕ₀
    ψ_s = state.state
    
    steps = coupling_params.steps
    te = coupling_params.te
    cutoff = sim_params.cutoff
    Dmax = sim_params.Dmax
    tau = sim_params.tau
    pe = sim_params.pe
    
    N = length(sites) ÷ 2
    
    # Initialize result arrays
    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    nb_list = zeros(Float64, steps + 1)
    
    # Initial measurements
    E_list[1] = real(inner(ψ_s', H_sys, ψ_s))
    GS_overlap_list[1] = abs2(inner(ψ_s, ϕ₀))
    
    println("Cooling starts")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1])")
    
    # Main cooling loop
    for step in 2:steps+1
        # Append fresh bath qubits
        ψ_sb = appendzeros_MPS(ψ_s, sites)
        
        # Time evolution using TDVP
        ψ_sb_evolved = tdvp(H_sys_bath, -im * te, ψ_sb; 
                           time_step=-im * tau, reverse_step=false, 
                           normalize=true, maxdim=Dmax, cutoff=cutoff, outputlevel=0)
        normalize!(ψ_sb_evolved)
        orthogonalize!(ψ_sb_evolved, 2)
        
        # Apply noise if requested
        if pe > 0
            ψ_sb_evolved = apply_depolarizing_noise(ψ_sb_evolved, sites, pe)
            orthogonalize!(ψ_sb_evolved, 2)
        end
        
        # Sample and discard bath
        v_b, ψ_s = sample_bath(ψ_sb_evolved)
        truncate!(ψ_s; cutoff)
        normalize!(ψ_s)
        
        # Measurements
        E_list[step] = real(inner(ψ_s', H_sys, ψ_s))
        GS_overlap_list[step] = abs2(inner(ψ_s, ϕ₀))
        nb_list[step] = mean(v_b .- 1)
        
        println("Step $step: energy/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step]), " *
                "DmaxSB=$(maxlinkdim(ψ_sb_evolved)), DmaxS=$(maxlinkdim(ψ_s)), <nb>=$(nb_list[step])")
    end
    
    println("After cooling: energy/N=$(E_list[end]/N), overlap=$(GS_overlap_list[end])")
    
    return Dict(
        "E_list" => E_list,
        "GS_overlap_list" => GS_overlap_list, 
        "bath_magnetization_list" => nb_list,
        "final_state" => ψ_s
    )
end

# ============================================================================
# ED Backend + Density Matrix + Continuous Evolution
# ============================================================================

function run_cooling(problem::CoolingProblem{EDBackend}, 
                    state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution}, 
                    coupling_params, sim_params, ham_params)
    
    H_sys = problem.H_sys
    H_full = problem.H_sys_bath
    ϕ₀ = problem.ϕ₀
    
    steps = coupling_params.steps
    te = coupling_params.te
    pe = sim_params.pe
    
    # Get dimensions
    H_sys_mat = Matrix(H_sys)
    N_sys = Int(log2(size(H_sys_mat, 1)))
    N_bath = N_sys
    N_total = N_sys + N_bath
    
    # Initialize density matrix from initial state
    ρ = state.state
    
    # Initialize result arrays
    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    purity_list = zeros(Float64, steps + 1)
    bath_mag_list = zeros(Float64, steps + 1)
    
    # Utility operators
    I_sys = I(2^N_sys)
    I_bath = I(2^N_bath)
    
    # Initial measurements
    ρ_sys = tr_bath(ρ, N_sys, N_bath)
    E_list[1] = real(tr(H_sys_mat * ρ_sys))
    GS_overlap_list[1] = real(ϕ₀' * ρ_sys * ϕ₀)
    purity_list[1] = real(tr(ρ_sys^2))
    
    println("Cooling starts (ED Density Matrix)")
    println("Step 1: energy/N=$(E_list[1]/N_sys), overlap=$(GS_overlap_list[1]), purity=$(purity_list[1])")
    
    # Main cooling loop
    for step in 2:steps+1
        # Reset bath to ground state |111...⟩ (for negative delta)
        ρ_bath_fresh = projector(ArrayReg(bit"1"^N_bath))
        ρ_total = kron(ρ_sys, ρ_bath_fresh)
        
        # Time evolution 
        U = exp(-im * te * Matrix(H_full))
        ρ_evolved = U * ρ_total * U'
        
        # Apply depolarizing noise if requested
        if pe > 0
            ρ_noise = I(2^N_total) / 2^N_total
            ρ_evolved = (1 - pe) * ρ_evolved + pe * ρ_noise
        end
        
        # Trace out bath
        ρ_sys = tr_bath(ρ_evolved, N_sys, N_bath)
        
        # Measurements
        E_list[step] = real(tr(H_sys_mat * ρ_sys))
        GS_overlap_list[step] = real(ϕ₀' * ρ_sys * ϕ₀)
        purity_list[step] = real(tr(ρ_sys^2))
        
        # Bath magnetization (average over bath qubits)
        bath_mag = 0.0
        for i in 1:N_bath
            Z_i = kron(I_sys, kron(I(2^(i-1)), kron(Matrix(Z), I(2^(N_bath-i)))))
            bath_mag += real(tr(Z_i * ρ_evolved))
        end
        bath_mag_list[step] = bath_mag / N_bath
        
        println("Step $step: energy/N=$(E_list[step]/N_sys), overlap=$(GS_overlap_list[step]), " *
                "purity=$(purity_list[step]), <σ_z^bath>=$(bath_mag_list[step])")
        
        # Update state for next iteration
        ρ = ρ_sys
    end
    
    println("After cooling: energy/N=$(E_list[end]/N_sys), overlap=$(GS_overlap_list[end])")
    
    return Dict(
        "E_list" => E_list,
        "GS_overlap_list" => GS_overlap_list,
        "purity_list" => purity_list,
        "bath_magnetization_list" => bath_mag_list,
        "final_state" => ρ_sys
    )
end

# ============================================================================
# ED Backend + Monte Carlo + Continuous Evolution  
# ============================================================================

function run_cooling(problem::CoolingProblem{EDBackend}, 
                    state::QuantumState{EDBackend,MonteCarloWavefunction,ContinuousEvolution}, 
                    coupling_params, sim_params, ham_params)
    
    H_sys = problem.H_sys
    H_full = problem.H_sys_bath
    ϕ₀ = problem.ϕ₀
    
    steps = coupling_params.steps
    te = coupling_params.te
    pe = sim_params.pe
    n_trajectories = sim_params.n_trajectories
    
    # Get dimensions
    H_sys_mat = Matrix(H_sys)
    N_sys = Int(log2(size(H_sys_mat, 1)))
    N_bath = N_sys
    
    # Initialize result arrays for all trajectories
    E_trajectories = zeros(Float64, steps + 1, n_trajectories)
    GS_trajectories = zeros(Float64, steps + 1, n_trajectories)
    bath_mag_trajectories = zeros(Float64, steps + 1, n_trajectories)
    
    println("Cooling starts (ED Monte Carlo, $n_trajectories trajectories)")
    
    # Run trajectories
    for traj in 1:n_trajectories
        # Initial state for this trajectory
        ψ = copy(state.state)
        
        # Measurements for initial state
        ψ_sys = ψ  # Initially just system
        E_trajectories[1, traj] = real(expect(H_sys, ψ_sys))
        GS_trajectories[1, traj] = abs2(vdot(vec(ϕ₀), vec(ψ_sys)))
        
        # Evolution for each step
        for step in 2:steps+1
            # Append fresh bath in ground state
            ψ_bath = ArrayReg(bit"1"^N_bath)
            ψ_total = kron(ψ_sys, ψ_bath)
            
            # Time evolution
            ψ_evolved = apply(ψ_total, time_evolve(H_full, te))
            
            # Apply depolarizing noise stochastically
            if pe > 0 && rand() < pe
                # Complete depolarization - random state
                ψ_evolved = rand_state(N_sys + N_bath)
            end
            
            # Measure and collapse bath
            bath_result = measure(ψ_evolved, (N_sys+1):(N_sys+N_bath); nshots=1)
            ψ_sys = partial_tr(ψ_evolved, (N_sys+1):(N_sys+N_bath))
            normalize!(ψ_sys)
            
            # Measurements
            E_trajectories[step, traj] = real(expect(H_sys, ψ_sys))
            GS_trajectories[step, traj] = abs2(vdot(vec(ϕ₀), vec(ψ_sys)))
            
            # Bath magnetization from measurement results
            bath_mag = sum(2 .* bath_result .- 1) / N_bath
            bath_mag_trajectories[step, traj] = bath_mag
        end
        
        if traj % 10 == 0
            println("Completed trajectory $traj/$n_trajectories")
        end
    end
    
    # Compute averages and standard deviations
    E_list = vec(mean(E_trajectories, dims=2))
    GS_overlap_list = vec(mean(GS_trajectories, dims=2))
    bath_mag_list = vec(mean(bath_mag_trajectories, dims=2))
    
    E_std = vec(std(E_trajectories, dims=2))
    GS_std = vec(std(GS_trajectories, dims=2))
    
    println("After cooling: energy/N=$(E_list[end]/N_sys)±$(E_std[end]/N_sys), " *
            "overlap=$(GS_overlap_list[end])±$(GS_std[end])")
    
    return Dict(
        "E_list" => E_list,
        "GS_overlap_list" => GS_overlap_list,
        "bath_magnetization_list" => bath_mag_list,
        "E_trajectories" => E_trajectories,
        "GS_trajectories" => GS_trajectories,
        "E_std" => E_std,
        "GS_std" => GS_std,
        "n_trajectories" => n_trajectories
    )
end

# ============================================================================
# TN Backend + Density Matrix + Trotter Evolution (formerly "MPO")
# ============================================================================

function run_cooling(problem::CoolingProblem{TNBackend}, 
                    state::QuantumState{TNBackend,DensityMatrix,TrotterEvolution}, 
                    coupling_params, sim_params, ham_params)
    
    sites = problem.sites
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀
    gates = problem.extra.gates
    ρ = state.state  # MPO density matrix
    
    steps = coupling_params.steps
    cutoff = sim_params.cutoff
    Dmax = sim_params.Dmax
    tau = sim_params.tau
    pe = sim_params.pe
    
    N = length(sites) ÷ 2
    
    # Initialize result arrays
    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    purity_list = zeros(Float64, steps + 1)
    
    # Initial measurements
    E_list[1] = real(tr(apply(H_sys, ρ)))
    GS_overlap_list[1] = real(inner(ρ, outer(ϕ₀', ϕ₀)))
    purity_list[1] = real(tr(apply(ρ, ρ)))
    
    println("Cooling starts (TN Density Matrix + Trotter)")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1]), purity=$(purity_list[1])")
    
    # Main cooling loop
    for step in 2:steps+1
        # Trace out old bath and add fresh bath
        ρ_s = rdm_mpo(ρ, sites, 1:2:2N-1)
        ρ_sb = appendbath_MPO(ρ_s, sites)
        
        # Apply Trotter gates
        ρ_evolved = ρ_sb
        for gate in gates
            ρ_evolved = apply(gate, ρ_evolved; cutoff, maxdim=Dmax)
        end
        
        # Apply noise if requested
        if pe > 0
            ρ_evolved = apply_noise_mpo(ρ_evolved, sites, pe)
        end
        
        # Update for next iteration
        ρ = ρ_evolved
        
        # Measurements
        E_list[step] = real(tr(apply(H_sys, rdm_mpo(ρ, sites, 1:2:2N-1))))
        ρ_s_temp = rdm_mpo(ρ, sites, 1:2:2N-1)
        GS_overlap_list[step] = real(inner(ρ_s_temp, outer(ϕ₀', ϕ₀)))
        purity_list[step] = real(tr(apply(ρ_s_temp, ρ_s_temp)))
        
        println("Step $step: energy/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step]), " *
                "purity=$(purity_list[step]), Dmax=$(maxlinkdim(ρ))")
    end
    
    println("After cooling: energy/N=$(E_list[end]/N), overlap=$(GS_overlap_list[end])")
    
    return Dict(
        "E_list" => E_list,
        "GS_overlap_list" => GS_overlap_list,
        "purity_list" => purity_list,
        "final_state" => rdm_mpo(ρ, sites, 1:2:2N-1)
    )
end

# ============================================================================
# TN Backend + Monte Carlo + Trotter Evolution (formerly "TrotterMPS")
# ============================================================================

function run_cooling(problem::CoolingProblem{TNBackend}, 
                    state::QuantumState{TNBackend,MonteCarloWavefunction,TrotterEvolution}, 
                    coupling_params, sim_params, ham_params)
    
    sites = problem.sites
    H_sys = problem.H_sys
    H_total = problem.extra.H_sys_bath
    ϕ₀ = problem.ϕ₀
    gates = problem.extra.gates
    ψ_s = state.state
    
    steps = coupling_params.steps
    te = coupling_params.te
    cutoff = sim_params.cutoff
    Dmax = sim_params.Dmax
    tau = sim_params.tau
    pe = sim_params.pe
    
    N = length(sites) ÷ 2
    trotter_steps = Int(te / tau)
    
    # Initialize result arrays
    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    nb_list = zeros(Float64, steps + 1)
    
    # Initial measurements
    E_list[1] = real(inner(ψ_s', H_sys, ψ_s))
    GS_overlap_list[1] = abs2(inner(ψ_s, ϕ₀))
    
    println("Cooling starts (TN Monte Carlo + Trotter)")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1])")
    
    # Main cooling loop
    for step in 2:steps+1
        # Append fresh bath
        ψ_sb = appendzeros_MPS(ψ_s, sites)
        
        # Apply Trotter gates alternating with exact H_0 evolution
        for _ in 1:trotter_steps
            # H_0 evolution (no coupling)
            ψ_sb = tdvp(-H_total, im * tau/2, ψ_sb; normalize=false, maxdim=Dmax, cutoff=cutoff)
            
            # Apply coupling gates
            for gate in gates
                ψ_sb = apply(gate, ψ_sb; cutoff, maxdim=Dmax)
            end
            
            # H_0 evolution again
            ψ_sb = tdvp(-H_total, im * tau/2, ψ_sb; normalize=false, maxdim=Dmax, cutoff=cutoff)
        end
        
        normalize!(ψ_sb)
        
        # Apply noise if requested
        if pe > 0
            ψ_sb = apply_depolarizing_noise(ψ_sb, sites, pe)
        end
        
        # Sample and discard bath
        v_b, ψ_s = sample_bath(ψ_sb)
        truncate!(ψ_s; cutoff)
        normalize!(ψ_s)
        
        # Measurements
        E_list[step] = real(inner(ψ_s', H_sys, ψ_s))
        GS_overlap_list[step] = abs2(inner(ψ_s, ϕ₀))
        nb_list[step] = mean(v_b .- 1)
        
        println("Step $step: energy/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step]), " *
                "DmaxSB=$(maxlinkdim(ψ_sb)), DmaxS=$(maxlinkdim(ψ_s)), <nb>=$(nb_list[step])")
    end
    
    println("After cooling: energy/N=$(E_list[end]/N), overlap=$(GS_overlap_list[end])")
    
    return Dict(
        "E_list" => E_list,
        "GS_overlap_list" => GS_overlap_list,
        "bath_magnetization_list" => nb_list,
        "final_state" => ψ_s
    )
end

# ============================================================================
# Helper Functions (should also be dispatch-based eventually)
# ============================================================================

function tr_bath(ρ_total::Matrix, N_sys::Int, N_bath::Int)
    # Partial trace over bath degrees of freedom
    dim_sys = 2^N_sys
    dim_bath = 2^N_bath
    ρ_sys = zeros(ComplexF64, dim_sys, dim_sys)
    
    for i in 1:dim_sys, j in 1:dim_sys
        for k in 1:dim_bath
            idx_i = (i-1)*dim_bath + k
            idx_j = (j-1)*dim_bath + k
            ρ_sys[i,j] += ρ_total[idx_i, idx_j]
        end
    end
    
    return ρ_sys
end

function projector(ψ::ArrayReg)
    vec_ψ = vec(ψ)
    return vec_ψ * vec_ψ'
end