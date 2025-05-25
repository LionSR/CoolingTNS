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

# ED backend types are already included by CoolingTNS.jl

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
    measurements = Dict(
        "E_list" => zeros(Float64, steps + 1),
        "GS_overlap_list" => zeros(Float64, steps + 1)
    )
    
    # Add backend-specific measurements
    add_backend_measurements!(measurements, problem, state, steps)
    
    return measurements
end

# Default: no additional measurements
add_backend_measurements!(measurements, problem, state, steps) = nothing

# Density matrix methods get purity
function add_backend_measurements!(measurements, problem, state::QuantumState{B,DensityMatrix,E}, steps) where {B,E}
    measurements["purity_list"] = zeros(Float64, steps + 1)
    measurements["bath_mag_list"] = zeros(Float64, steps + 1)
end

# Monte Carlo methods track bath samples
function add_backend_measurements!(measurements, problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,MonteCarloWavefunction,E}, steps) where E
    measurements["nb_list"] = zeros(Float64, steps + 1)
end

# ED Monte Carlo also tracks bath magnetization
function add_backend_measurements!(measurements, problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,MonteCarloWavefunction,E}, steps) where E
    measurements["bath_mag_list"] = zeros(Float64, steps + 1)
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

# --- Tensor Network + Monte Carlo + Continuous Evolution ---

function prepare_combined_state(problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,MonteCarloWavefunction,ContinuousEvolution})
    # Append fresh bath qubits in ground state
    # Get sites from problem.extra
    sites = problem.extra.sites
    return appendzeros_MPS(state.state, sites)
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

function process_bath_and_update(problem::CoolingProblem{TNBackend}, ψ_evolved::MPS, 
                               state::QuantumState{TNBackend,MonteCarloWavefunction,ContinuousEvolution}, 
                               sim_params)
    # Get N from the sites in problem
    sites = problem.extra.sites
    N_total = length(sites)
    N_sys = N_total ÷ 2
    N_bath = N_sys
    
    # For TN backend, sites are interlaced: sys1, bath1, sys2, bath2, ...
    # After sampling bath, we get back an MPS with only system sites
    # The sample_bath function reduces the MPS to system sites only
    v_b, ψ_s = sample_bath(ψ_evolved)
    
    # The returned MPS should have N_sys sites
    if length(ψ_s) != N_sys
        @warn "After sampling bath, MPS has unexpected length" expected=N_sys actual=length(ψ_s)
    end
    
    truncate!(ψ_s; cutoff=sim_params.cutoff)
    normalize!(ψ_s)
    
    # Return updated state and bath sample
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ψ_s), v_b
end

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend}, 
                                     state::QuantumState{TNBackend,MonteCarloWavefunction,ContinuousEvolution}, 
                                     ham_params, bath_info=nothing)
    ψ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀
    
    # Debug: Check dimensions
    if length(ψ_s) != length(H_sys)
        @warn "Dimension mismatch in measurements" MPS_length=length(ψ_s) H_sys_length=length(H_sys) phi0_length=length(ϕ₀) step=step
    end
    
    # For first step, MPS should match H_sys dimensions
    # For later steps after bath sampling, need to be careful
    if step == 1 || length(ψ_s) == length(H_sys)
        measurements["E_list"][step] = real(inner(ψ_s', H_sys, ψ_s))
        measurements["GS_overlap_list"][step] = abs2(inner(ψ_s, ϕ₀))
    else
        # Skip measurement if dimensions don't match
        @warn "Skipping measurement due to dimension mismatch"
        measurements["E_list"][step] = measurements["E_list"][step-1]
        measurements["GS_overlap_list"][step] = measurements["GS_overlap_list"][step-1]
    end
    
    # Bath magnetization if available from bath_info
    if haskey(measurements, "nb_list") && bath_info !== nothing
        N = ham_params.N
        measurements["nb_list"][step] = compute_bath_magnetization(problem.backend, state, bath_info, N)
    end
end

# --- Exact Diagonalization + Density Matrix + Continuous Evolution ---

function prepare_combined_state(problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution})
    # Get N from problem parameters, not from state size
    N_sys = problem.extra.ham_params.N
    N_bath = N_sys
    
    # Handle both clean ED types and legacy Matrix types
    if isa(state.state, EDDensityMatrix)
        # Clean ED backend
        if state.state.n_qubits == 2*N_sys
            ρ_sys = trace_out_bath_ed(state.state, N_sys)
        else
            ρ_sys = state.state
        end
        ρ_bath = state_to_density_ed(zero_state_ed(N_bath))
        return kron_density_ed(ρ_sys, ρ_bath)
    else
        # Legacy matrix format
        if size(state.state, 1) == 2^(2*N_sys)
            ρ_sys = tr_bath(state.state, N_sys, N_bath)
        else
            ρ_sys = state.state
        end
        
        # Use clean ED backend for bath
        ρ_bath = state_to_density_ed(zero_state_ed(N_bath))
        return kron(ρ_sys, ρ_bath.data)
    end
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ρ_total::EDDensityMatrix, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution},
                           ham_params)
    # Get sparse Hamiltonian matrix
    H_sparse = problem.H_sys_bath
    if !isa(H_sparse, AbstractMatrix)
        error("Expected sparse matrix Hamiltonian for ED backend")
    end
    
    # Time evolution
    return evolve_ed(H_sparse, ρ_total, te)
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ρ_total::Matrix, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution},
                           ham_params)
    # Get sparse Hamiltonian matrix
    # Convert to sparse if not already
    H_sparse = sparse(problem.H_sys_bath)
    
    # Use ExponentialUtilities for efficient sparse matrix exponentiation
    # For density matrix evolution: ρ(t) = U ρ(0) U†, where U = exp(-i H t)
    # We can use expv for ρ_vec evolution: d/dt ρ_vec = -i (H⊗I - I⊗H^T) ρ_vec
    n = size(ρ_total, 1)
    
    # Create the Liouvillian superoperator L = -i(H⊗I - I⊗H^T)
    # For density matrix evolution: d/dt ρ = -i[H,ρ] = -i(Hρ - ρH)
    # In vectorized form: d/dt |ρ⟩⟩ = -i(H⊗I - I⊗H^T)|ρ⟩⟩
    # Note: We need the transpose, NOT conjugate transpose
    L = -im * (kron(H_sparse, I(n)) - kron(I(n), transpose(H_sparse)))
    
    # Vectorize the density matrix
    ρ_vec = vec(ρ_total)
    
    # Use Krylov exponentiation for efficient computation
    ρ_vec_evolved = expv(te, L, ρ_vec)
    
    # Reshape back to matrix form
    return reshape(ρ_vec_evolved, n, n)
end

function apply_noise(ρ::Matrix, problem::CoolingProblem{EDBackend}, pe::Float64)
    N_total = size(ρ, 1) |> x -> Int(log2(x))
    ρ_noise = I(2^N_total) / 2^N_total
    return (1 - pe) * ρ + pe * ρ_noise
end

function process_bath_and_update(problem::CoolingProblem{EDBackend}, ρ_evolved::EDDensityMatrix,
                               state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution},
                               sim_params)
    # For density matrix, we keep the full state and trace out bath during measurements
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_evolved), nothing
end

function process_bath_and_update(problem::CoolingProblem{EDBackend}, ρ_evolved::Matrix,
                               state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution},
                               sim_params)
    # For density matrix, we keep the full state and trace out bath during measurements
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_evolved), nothing
end

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                                     state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution},
                                     ham_params, bath_info=nothing)
    ρ_total = state.state
    N_sys = ham_params.N
    N_bath = N_sys
    
    # Get system density matrix
    if ρ_total.n_qubits == 2*N_sys
        # Full system+bath state - trace out bath
        ρ_sys = trace_out_bath_ed(ρ_total, N_sys)
    else
        # Just system state
        ρ_sys = ρ_total
    end
    
    # Energy
    H_sys_mat = problem.H_sys
    measurements["E_list"][step] = expect_ed(H_sys_mat, ρ_sys)
    
    # Ground state overlap: <ϕ₀|ρ|ϕ₀>
    ϕ₀ = problem.ϕ₀
    measurements["GS_overlap_list"][step] = real(ϕ₀.data' * ρ_sys.data * ϕ₀.data)
    
    # Purity
    measurements["purity_list"][step] = purity_ed(ρ_sys)
    
    # Bath magnetization (only if we have full state and not first step)
    if step > 1 && ρ_total.n_qubits == 2*N_sys
        ρ_bath = trace_out_system_ed(ρ_total, N_sys)
        # Compute magnetization
        mag = 0.0
        for i in 1:N_bath
            Z_i = pauli_z(i, N_bath)
            mag += expect_ed(Z_i, ρ_bath)
        end
        measurements["bath_mag_list"][step] = mag / N_bath
    end
end

# --- Tensor Network + Monte Carlo + Trotter Evolution ---

function prepare_combined_state(problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,MonteCarloWavefunction,TrotterEvolution})
    # Append fresh bath qubits
    sites = problem.extra.sites
    return appendzeros_MPS(state.state, sites)
end

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

function process_bath_and_update(problem::CoolingProblem{TNBackend}, ψ_evolved::MPS, 
                               state::QuantumState{TNBackend,MonteCarloWavefunction,TrotterEvolution}, 
                               sim_params)
    # Get N from the sites in problem
    sites = problem.extra.sites
    N_total = length(sites)
    N_sys = N_total ÷ 2
    N_bath = N_sys
    
    # For TN backend, sites are interlaced: sys1, bath1, sys2, bath2, ...
    # After sampling bath, we get back an MPS with only system sites
    # The sample_bath function reduces the MPS to system sites only
    v_b, ψ_s = sample_bath(ψ_evolved)
    
    # The returned MPS should have N_sys sites
    if length(ψ_s) != N_sys
        @warn "After sampling bath, MPS has unexpected length" expected=N_sys actual=length(ψ_s)
    end
    
    truncate!(ψ_s; cutoff=sim_params.cutoff)
    normalize!(ψ_s)
    
    # Return updated state and bath sample
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ψ_s), v_b
end

# apply_noise for TN+MC+Trotter is the same as TN+MC+Continuous  

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend}, 
                                     state::QuantumState{TNBackend,MonteCarloWavefunction,TrotterEvolution}, 
                                     ham_params, bath_info=nothing)
    ψ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀
    
    # Debug: Check dimensions
    if length(ψ_s) != length(H_sys)
        @warn "Dimension mismatch in measurements" MPS_length=length(ψ_s) H_sys_length=length(H_sys) phi0_length=length(ϕ₀) step=step
    end
    
    # For first step, MPS should match H_sys dimensions
    # For later steps after bath sampling, need to be careful
    if step == 1 || length(ψ_s) == length(H_sys)
        measurements["E_list"][step] = real(inner(ψ_s', H_sys, ψ_s))
        measurements["GS_overlap_list"][step] = abs2(inner(ψ_s, ϕ₀))
    else
        # Skip measurement if dimensions don't match
        @warn "Skipping measurement due to dimension mismatch"
        measurements["E_list"][step] = measurements["E_list"][step-1]
        measurements["GS_overlap_list"][step] = measurements["GS_overlap_list"][step-1]
    end
    
    # Bath magnetization if available from bath_info
    if haskey(measurements, "nb_list") && bath_info !== nothing
        N = ham_params.N
        measurements["nb_list"][step] = compute_bath_magnetization(problem.backend, state, bath_info, N)
    end
end

# --- Tensor Network + Density Matrix + Trotter Evolution (MPO) ---

function prepare_combined_state(problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,DensityMatrix,TrotterEvolution})
    # Append fresh bath in ground state
    sites = problem.extra.sites
    return appendzeros_MPO(state.state, sites)
end

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ρ_sb::MPO, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,TrotterEvolution},
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
        
        gates = build_trotter_circuit(ham_params, problem.backend, sites_sys, sites_bath, coupling_params, sim_params)
        # Note: Cannot modify immutable NamedTuple
    end
    
    # Apply gates for Trotter evolution
    ρ_evolved = ρ_sb
    for _ in 1:sim_params.trotter_steps
        ρ_evolved = apply(gates, ρ_evolved; apply_dag=true, cutoff=sim_params.cutoff, maxdim=sim_params.Dmax)
    end
    
    return ρ_evolved
end

function process_bath_and_update(problem::CoolingProblem{TNBackend}, ρ_evolved::MPO,
                               state::QuantumState{TNBackend,DensityMatrix,TrotterEvolution},
                               sim_params)
    # Partial trace out bath to get system density matrix
    sites = problem.extra.sites
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]
    
    # Get bath density matrix before tracing out (optional for measurements)
    # For now, we don't compute bath properties for MPO to keep it efficient
    
    ρ_s = partial_trace_bath(ρ_evolved, sites, sites_sys)
    ρ_s /= tr(ρ_s)  # Renormalize
    
    # Return updated state without bath info (nothing)
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_s), nothing
end

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend},
                                     state::QuantumState{TNBackend,DensityMatrix,TrotterEvolution},
                                     ham_params, bath_info=nothing)
    ρ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀
    N = ham_params.N
    
    measurements["E_list"][step] = real(inner(ρ_s, H_sys))
    measurements["GS_overlap_list"][step] = real(inner(ρ_s, projector_mpo(ϕ₀)))
    measurements["purity_list"][step] = real(tr(apply(ρ_s, ρ_s)))
    
    # Note: Bath magnetization not easily accessible for MPO method
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
