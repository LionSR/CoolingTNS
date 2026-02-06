"""
    cooling_evolution.jl

Cooling evolution implementations using multiple dispatch.
"""

using ITensors
using ITensorMPS
using ITensors: apply
using LinearAlgebra
using SparseArrays
using Random
using Statistics

# ED backend types and shared functions are already included by CoolingTNS.jl

# ============================================================================
# Main Cooling Evolution Interface
# ============================================================================

"""
    run_cooling(problem::CoolingProblem, state::QuantumState, coupling_params, sim_params, ham_params)

Run cooling simulation with type dispatch. This is the main entry point that delegates
to specialized implementations based on backend and method types.
"""
function run_cooling(problem::CoolingProblem{B}, state::QuantumState{B,S,E}, 
                    coupling_params, sim_params, 
                    ham_params) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    
    # Common setup
    steps = coupling_params.steps
    te = coupling_params.te
    pe = sim_params.pe
    
    # Initialize measurement arrays
    measurements = initialize_measurements(problem, state, steps)
    
    # Initial measurements
    perform_measurements!(measurements, 1, problem, state, ham_params)
    
    # Print initial status
    print_cooling_status(1, measurements, ham_params, state)
    
    # Main cooling loop
    for step in 2:steps+1
        # Prepare system+bath state
        combined_state = prepare_combined_state(problem, state)
        
        # Evolve the combined system
        evolved_state = evolve_cooling_step(problem, combined_state, te, sim_params, ham_params)
        
        # Apply noise if requested
        if pe > 0
            evolved_state = apply_noise(evolved_state, problem, pe)
        end
        
        # Process bath and update system state
        state_and_bath_info = process_bath_and_update(problem, evolved_state, state, sim_params)
        if isa(state_and_bath_info, Tuple)
            state, bath_info = state_and_bath_info
        else
            state = state_and_bath_info
            bath_info = nothing
        end
        
        # Perform measurements
        perform_measurements!(measurements, step, problem, state, ham_params, bath_info)
        
        # Print progress
        if step % 10 == 0 || step == steps + 1
            print_cooling_status(step, measurements, ham_params, state)
        end
    end
    
    println("Cooling completed")
    
    # Return results
    return compile_results(measurements, sim_params)
end

# ============================================================================
# Helper Functions - Generic Interface
# ============================================================================

"""Initialize measurement arrays based on backend and method"""
function initialize_measurements(problem::CoolingProblem, state::QuantumState, steps::Int)
    # Basic measurements available for all backends
    measurements = Dict{String, Any}(
        "E_list" => zeros(Float64, steps + 1),
        "GS_overlap_list" => zeros(Float64, steps + 1)
    )
    
    # Add backend-specific measurements
    add_backend_measurements!(measurements, problem, state, steps)
    
    return measurements
end

# Default: no additional measurements
add_backend_measurements!(_, _, _, _) = nothing

# TN Density matrix methods get purity
function add_backend_measurements!(measurements, ::CoolingProblem{TNBackend}, ::QuantumState{TNBackend,DensityMatrix,E}, steps) where E
    measurements["purity_list"] = zeros(Float64, steps + 1)
    measurements["bath_mag_list"] = zeros(Float64, steps + 1)
end

# ED Density matrix methods get purity + k-space measurements
function add_backend_measurements!(measurements, problem::CoolingProblem{EDBackend}, ::QuantumState{EDBackend,DensityMatrix,E}, steps) where E
    measurements["purity_list"] = zeros(Float64, steps + 1)
    measurements["bath_mag_list"] = zeros(Float64, steps + 1)

    # Add k-space measurements for ED with periodic/antiperiodic BC (only for Ising model)
    add_kspace_measurements!(measurements, problem, steps)
end

# TN Monte Carlo methods track bath samples
function add_backend_measurements!(measurements, ::CoolingProblem{TNBackend}, ::QuantumState{TNBackend,MonteCarloWavefunction,E}, steps) where E
    measurements["nb_list"] = zeros(Float64, steps + 1)
end

# ED Monte Carlo tracks bath magnetization + k-space
function add_backend_measurements!(measurements, problem::CoolingProblem{EDBackend}, ::QuantumState{EDBackend,MonteCarloWavefunction,E}, steps) where E
    measurements["bath_mag_list"] = zeros(Float64, steps + 1)
    add_kspace_measurements!(measurements, problem, steps)
end

# Helper for k-space measurements (ED only, Ising PBC/APBC)
function add_kspace_measurements!(measurements, problem::CoolingProblem{EDBackend}, steps)
    if haskey(problem.extra, :ham_params)
        ham_params = problem.extra.ham_params
        if ham_params.bc in [:periodic, :antiperiodic] && isa(ham_params.model, IsingModel)
            N = ham_params.N
            measurements["momentum_dist"] = zeros(Float64, steps + 1, N)
            measurements["k_values"] = zeros(Float64, N)
        end
    end
end

"""Perform measurements on current state"""
function perform_measurements!(measurements, step::Int, problem::CoolingProblem, state::QuantumState, ham_params, bath_info=nothing)
    # Delegate to backend-specific measurement
    perform_backend_measurements!(measurements, step, problem, state, ham_params, bath_info)
end

"""Print cooling progress"""
function print_cooling_status(step::Int, measurements, ham_params, state::QuantumState)
    N = ham_params.N
    E = measurements["E_list"][step]
    overlap = measurements["GS_overlap_list"][step]
    
    status = "Step $step: energy/N=$(E/N), overlap=$overlap"
    
    # Add method-specific info
    if haskey(measurements, "purity_list")
        status *= ", purity=$(measurements["purity_list"][step])"
    end
    
    println(status)
end

"""Compile results into standard format"""
function compile_results(measurements, sim_params)
    results = copy(measurements)
    
    # Add simulation metadata
    if sim_params isa UnifiedSimulationParameters{MonteCarloWavefunction, E} where E
        if sim_params.n_trajectories > 1
            results["n_trajectories"] = sim_params.n_trajectories
        end
    end
    
    return results
end

# ============================================================================
# Backend-Specific Implementations
# ============================================================================

# --- Tensor Network + Monte Carlo (unified for both evolution methods) ---

# Unified TN+MC prepare_combined_state for both Continuous and Trotter evolution
function prepare_combined_state(problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,MonteCarloWavefunction,E}) where E<:EvolutionMethod
    sites = problem.extra.sites
    coupling = problem.extra.coupling_params.coupling
    return appendzeros_MPS(state.state, sites, coupling)
end

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ψ_sb::MPS, te::Float64, 
                           sim_params::UnifiedSimulationParameters{MonteCarloWavefunction,ContinuousEvolution}, 
                           ham_params)
    # Use the generic evolve_state from evolution.jl
    sites = problem.extra.sites
    return evolve_state(ham_params, sim_params, problem.backend, problem.H_sys_bath, ψ_sb, te, sites)
end

function apply_noise(ψ::MPS, problem::CoolingProblem{TNBackend}, pe::Float64)
    # Get sites from problem.extra
    sites = problem.extra.sites
    ψ_noisy = apply_depolarizing_noise(ψ, sites, pe)
    orthogonalize!(ψ_noisy, 2)
    return ψ_noisy
end

# Unified TN+MC bath processing for both Continuous and Trotter evolution
function process_bath_and_update(problem::CoolingProblem{TNBackend}, ψ_evolved::MPS,
                               state::QuantumState{TNBackend,MonteCarloWavefunction,E},
                               sim_params) where E<:EvolutionMethod
    sites = problem.extra.sites
    N_sys = length(sites) ÷ 2

    v_b, ψ_s = sample_bath(ψ_evolved)

    if length(ψ_s) != N_sys
        @warn "After sampling bath, MPS has unexpected length" expected=N_sys actual=length(ψ_s)
    end

    truncate!(ψ_s; cutoff=sim_params.cutoff)
    normalize!(ψ_s)

    return QuantumState(state.backend, state.sim_method, state.evolution_method, ψ_s), v_b
end

# Unified TN+MC measurements for both Continuous and Trotter evolution
function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend},
                                     state::QuantumState{TNBackend,MonteCarloWavefunction,E},
                                     ham_params, bath_info=nothing) where E<:EvolutionMethod
    ψ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀

    if length(ψ_s) != length(H_sys)
        @warn "Dimension mismatch in measurements" MPS_length=length(ψ_s) H_sys_length=length(H_sys) phi0_length=length(ϕ₀) step=step
    end

    if step == 1 || length(ψ_s) == length(H_sys)
        measurements["E_list"][step] = real(inner(ψ_s', H_sys, ψ_s))
        measurements["GS_overlap_list"][step] = abs2(inner(ψ_s, ϕ₀))
    else
        @warn "Skipping measurement due to dimension mismatch"
        measurements["E_list"][step] = measurements["E_list"][step-1]
        measurements["GS_overlap_list"][step] = measurements["GS_overlap_list"][step-1]
    end

    if haskey(measurements, "nb_list") && bath_info !== nothing
        measurements["nb_list"][step] = compute_bath_magnetization(problem.backend, state, bath_info, ham_params.N)
    end
end

# --- Exact Diagonalization + Density Matrix (shared for Continuous/Trotter) ---

function prepare_combined_state(problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,DensityMatrix,E}) where E<:EvolutionMethod
    N_bath = problem.extra.ham_params.N
    coupling = problem.extra.coupling_params.coupling
    ρ_sys = state.state.n_qubits == 2*N_bath ? trace_out_bath_ed(state.state, N_bath) : state.state
    return prepare_combined_state_ed(ρ_sys, N_bath, coupling)
end

function process_bath_and_update(::CoolingProblem{EDBackend}, ρ_evolved::EDDensityMatrix,
                               state::QuantumState{EDBackend,DensityMatrix,E}, _) where E<:EvolutionMethod
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_evolved), nothing
end

# Evolution differs by tau
function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ρ_total::EDDensityMatrix, te::Float64,
                           ::UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ρ_total, te, nothing)
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ρ_total::EDDensityMatrix, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,TrotterEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ρ_total, te, sim_params.tau)
end

function apply_noise(ρ::EDDensityMatrix, ::CoolingProblem{EDBackend}, pe::Float64)
    dim = 2^ρ.n_qubits
    ρ_noisy_data = (1 - pe) * ρ.data + pe * Matrix{Float64}(I, dim, dim) / dim
    return EDDensityMatrix(ρ_noisy_data, ρ.n_qubits)
end

# --- Exact Diagonalization + Monte Carlo (shared for Continuous/Trotter) ---

function prepare_combined_state(problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,MonteCarloWavefunction,E}) where E<:EvolutionMethod
    coupling = problem.extra.coupling_params.coupling
    return prepare_combined_state_ed(state.state, problem.extra.ham_params.N, coupling)
end

function process_bath_and_update(problem::CoolingProblem{EDBackend}, ψ_evolved::EDStateVector,
                               state::QuantumState{EDBackend,MonteCarloWavefunction,E}, _) where E<:EvolutionMethod
    ψ_sys, bath_outcomes = process_bath_ed_monte_carlo(ψ_evolved, problem.extra.ham_params.N)
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ψ_sys), bath_outcomes
end

# Evolution differs by tau
function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ψ_total::EDStateVector, te::Float64,
                           ::UnifiedSimulationParameters{MonteCarloWavefunction,ContinuousEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ψ_total, te, nothing)
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ψ_total::EDStateVector, te::Float64,
                           sim_params::UnifiedSimulationParameters{MonteCarloWavefunction,TrotterEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ψ_total, te, sim_params.tau)
end

# --- ED Backend Measurements (unified) ---

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                                     state::QuantumState{EDBackend,S,E}, ham_params, bath_info=nothing) where {S<:SimulationMethod, E<:EvolutionMethod}
    perform_measurements_ed(measurements, step, state.state, problem.H_sys, problem.ϕ₀, ham_params, bath_info)
end

# --- Tensor Network + Monte Carlo + Trotter Evolution ---

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ψ_sb::MPS, te::Float64,
                           sim_params::UnifiedSimulationParameters{MonteCarloWavefunction,TrotterEvolution},
                           ham_params)
    sites = problem.extra.sites
    
    # Get or create Trotter gates
    gates = get(problem.extra, :gates, nothing)
    if gates === nothing
        # Need to split sites into system and bath for build_trotter_circuit
        N = ham_params.N
        sites_sys = sites[1:2:2*N-1]
        sites_bath = sites[2:2:2*N]
        
        # Get coupling parameters from problem
        coupling_params = problem.extra.coupling_params
        
        gates = build_trotter_circuit_bath_coupling(ham_params, problem.backend, sites_sys, sites_bath, coupling_params, sim_params)
        # Note: Cannot modify immutable NamedTuple, gates will be recreated each time
    end
    
    # Use evolve_state with gates
    return evolve_state(ham_params, sim_params, problem.backend, problem.H_sys_bath, ψ_sb, te, sites; gates=gates)
end

# --- Tensor Network + Density Matrix (unified for both evolution methods) ---

# Unified TN+DM prepare_combined_state for both Continuous and Trotter
function prepare_combined_state(problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,DensityMatrix,E}) where E<:EvolutionMethod
    sites = problem.extra.sites
    coupling = problem.extra.coupling_params.coupling
    return appendzeros_MPO(state.state, sites, coupling)
end

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ρ_sb::MPO, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution},
                           ham_params)
    error("Continuous evolution for density matrices (MPO) is not supported by TDVP in ITensors. Please use either:
    1. monte_carlo + continuous (uses MPS with TDVP)
    2. density_matrix + trotter (uses MPO with gates)
    3. monte_carlo + trotter (uses MPS with gates)")
end

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ρ_sb::MPO, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,TrotterEvolution},
                           ham_params)
    sites = problem.extra.sites
    N = ham_params.N
    sites_sys = sites[1:2:2*N-1]
    
    # Get or create Trotter gates
    gates = get(problem.extra, :gates, nothing)
    if gates === nothing
        # Need to split sites into system and bath for build_trotter_circuit
        sites_bath = sites[2:2:2*N]
        
        # Get coupling parameters from problem
        coupling_params = problem.extra.coupling_params
        
        gates = build_trotter_circuit_bath_coupling(ham_params, problem.backend, sites_sys, sites_bath, coupling_params, sim_params)
        # Note: Cannot modify immutable NamedTuple
    end
    
    # System-only Trotter gates (match MCWF splitting)
    system_gates = get(problem.extra, :system_gates, nothing)
    if system_gates === nothing
        system_gates = build_system_trotter_circuit(ham_params, sites_sys, sim_params)
    end

    # Apply system evolution + bath/coupling in small steps (no bath reset per substep)
    steps = max(1, Int(floor(te / sim_params.tau)))
    ρ_evolved = ρ_sb
    # MPO bond dimension grows ~D^2; use a capped expansion to balance cost vs. truncation artifacts
    dm_maxdim = max(sim_params.Dmax, 4 * sim_params.Dmax)
    dm_cutoff = sim_params.cutoff / 10
    for _ in 1:steps
        ρ_evolved = apply(system_gates, ρ_evolved; apply_dag=true, cutoff=dm_cutoff, maxdim=dm_maxdim, move_sites_back=true)
        ρ_evolved = apply(gates, ρ_evolved; apply_dag=true, cutoff=dm_cutoff, maxdim=dm_maxdim, move_sites_back=true)
    end
    ρ_evolved /= tr(ρ_evolved)
    
    return ρ_evolved
end

# Unified TN+DM bath processing for both evolution methods
function process_bath_and_update(problem::CoolingProblem{TNBackend}, ρ_evolved::MPO,
                               state::QuantumState{TNBackend,DensityMatrix,E},
                               _sim_params) where E<:EvolutionMethod
    sites = problem.extra.sites
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]

    ρ_s = partial_trace_bath(ρ_evolved, sites, sites_sys)
    ρ_s = 0.5 * (ρ_s + dag(swapprime(ρ_s, 0, 1)))  # Enforce Hermiticity
    ρ_s /= tr(ρ_s)

    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_s), nothing
end

# Unified TN+DM measurements for both Continuous and Trotter evolution
function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend},
                                     state::QuantumState{TNBackend,DensityMatrix,E},
                                     _ham_params, _bath_info=nothing) where E<:EvolutionMethod
    ρ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀

    measurements["E_list"][step] = real(inner(ρ_s, H_sys))
    measurements["GS_overlap_list"][step] = real(inner(ρ_s, projector_mpo(ϕ₀)))
    measurements["purity_list"][step] = real(tr(apply(ρ_s, ρ_s)))
end

# ============================================================================
# Monte Carlo Trajectory Support
# ============================================================================

"""
Run Monte Carlo trajectories for backends that support it.
"""
function run_cooling_monte_carlo(problem::CoolingProblem, initial_state::QuantumState, 
                               coupling_params, sim_params, ham_params)
    n_trajectories = get(sim_params.extra, :n_trajectories, 1)
    
    if n_trajectories == 1
        # Single trajectory - use standard run_cooling
        return run_cooling(problem, initial_state, coupling_params, sim_params, ham_params)
    end
    
    # Multiple trajectories
    steps = coupling_params.steps
    
    # Initialize arrays for all trajectories
    all_measurements = [initialize_measurements(problem, initial_state, steps) for _ in 1:n_trajectories]
    
    # Run each trajectory
    for traj in 1:n_trajectories
        # Fresh copy of initial state for each trajectory
        traj_state = QuantumState(initial_state.backend, initial_state.sim_method, 
                                initial_state.evolution_method, copy(initial_state.state))
        
        # Run single trajectory
        traj_results = run_cooling(problem, traj_state, coupling_params, sim_params, ham_params)
        
        # Store results
        for (key, value) in traj_results
            all_measurements[traj][key] = value
        end
        
        if traj % 10 == 0
            println("Completed trajectory $traj/$n_trajectories")
        end
    end
    
    # Average results across trajectories
    avg_results = Dict{String, Any}()
    
    for key in keys(all_measurements[1])
        if key in ["E_list", "GS_overlap_list", "purity_list", "bath_mag_list", "nb_list"]
            # Average these measurements
            values = [all_measurements[traj][key] for traj in 1:n_trajectories]
            avg_results[key] = mean(values)
            avg_results[key * "_std"] = std(values)
        end
    end
    
    avg_results["n_trajectories"] = n_trajectories
    
    return avg_results
end

# ============================================================================
# Utility Functions
# ============================================================================

"""Trace out bath degrees of freedom"""
function tr_bath(ρ::Matrix, N_sys::Int, N_bath::Int)
    dim_sys = 2^N_sys
    dim_bath = 2^N_bath
    ρ_sys = zeros(ComplexF64, dim_sys, dim_sys)
    
    for i in 1:dim_sys, j in 1:dim_sys
        for k in 1:dim_bath
            idx_i = (i-1)*dim_bath + k
            idx_j = (j-1)*dim_bath + k
            ρ_sys[i,j] += ρ[idx_i, idx_j]
        end
    end
    
    return ρ_sys
end

# tr_sys function moved to bath_measurements.jl to avoid duplication

"""Compute bath magnetization from density matrix"""
function compute_bath_magnetization(ρ_bath::Matrix, N_bath::Int)
    mag = 0.0
    dim = 2^N_bath
    
    for i in 1:dim
        # Count number of 1s in binary representation
        n_ones = count_ones(i-1)
        mag += real(ρ_bath[i,i]) * (2*n_ones/N_bath - 1)
    end
    
    return mag
end

# Note: ED backend measurements are handled by the unified function above
