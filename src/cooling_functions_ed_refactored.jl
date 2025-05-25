"""
Refactored ED cooling implementation using native Yao.jl patterns
and Julia's multiple dispatch for cleaner code.
"""

using Yao
using YaoPlots
using LinearAlgebra
using ExponentialUtilities
using KrylovKit
using Random
using Statistics

# Abstract type for ED simulation states
abstract type AbstractEDState end

# Concrete implementations
struct PureEDState <: AbstractEDState
    reg::AbstractRegister
    nbits::Int
end

struct MixedEDState <: AbstractEDState
    dm::DensityMatrix
    nbits::Int
end

# Helper to get number of bits
nbits(state::AbstractEDState) = state.nbits

# Convert between state types
to_density_matrix(state::PureEDState) = MixedEDState(density_matrix(state.reg), state.nbits)
to_density_matrix(state::MixedEDState) = state

"""
Build Hamiltonian using Yao.jl's elegant syntax
"""
function build_hamiltonian_graceful(problem::String, N::Int, ham_params, coupling_params)
    nbits = 2N
    g = coupling_params["g"]
    Δ = coupling_params["Δ"]
    coupling = coupling_params["coupling"]
    
    # Parse coupling operators
    op1_str, op2_str = parse_coupling(coupling)
    ops = Dict("X" => X, "Y" => Y, "Z" => Z)
    op1, op2 = ops[op1_str], ops[op2_str]
    
    # Define operator shortcuts
    sx = i -> put(nbits, i=>X)
    sy = i -> put(nbits, i=>Y)
    sz = i -> put(nbits, i=>Z)
    
    # System sites (odd) and bath sites (even)
    sys_sites = [2i-1 for i in 1:N]
    bath_sites = [2i for i in 1:N]
    
    # Build Hamiltonian using functional style
    if problem == "niIsing"
        J, hx, hz = ham_params
        
        # System terms
        H_sys = sum([
            # ZZ interactions
            [J * sz(sys_sites[i]) * sz(sys_sites[i+1]) for i in 1:N-1]...,
            # X field
            [hx * sx(s) for s in sys_sites]...,
            # Z field
            [hz * sz(s) for s in sys_sites]...
        ])
        
        # Bath terms
        H_bath = sum([-Δ/2 * sz(b) for b in bath_sites])
        
        # Coupling terms
        H_coupling = sum([g * put(nbits, sys_sites[i]=>op1) * put(nbits, bath_sites[i]=>op2) for i in 1:N])
        
        return H_sys + H_bath + H_coupling
        
    elseif problem == "Ising"
        J, h = ham_params
        
        # System terms
        H_sys = sum([
            [J * sz(sys_sites[i]) * sz(sys_sites[i+1]) for i in 1:N-1]...,
            [h * sx(s) for s in sys_sites]...
        ])
        
        # Bath and coupling
        H_bath_coupling = sum([
            -Δ/2 * sz(bath_sites[i]) + 
            g * put(nbits, sys_sites[i]=>op1) * put(nbits, bath_sites[i]=>op2) 
            for i in 1:N
        ])
        
        return H_sys + H_bath_coupling
    else
        error("Unknown problem type: $problem")
    end
end

"""
Create initial states using native Yao.jl patterns
"""
function create_initial_state_ed(nbits::Int, init_type::String, theta::Float64=0.0, use_dm::Bool=false)
    N = nbits ÷ 2
    
    reg = if init_type == "identity"
        # Random state for pure state representation of maximally mixed
        rand_state(nbits)
    elseif init_type == "theta"
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10  # All down
            product_state(bit"1"^nbits)
        elseif abs(theta - 0.5) < 1e-10  # All up
            product_state(bit"0"^nbits)
        elseif abs(theta) < 1e-10  # X+ state
            uniform_state(nbits)
        else
            # General theta state using chain
            zero_state(nbits) |> chain(nbits, put(i=>Ry(2*theta_rad)) for i in 1:nbits)
        end
    else
        # Default: alternating up/down
        # Create bit pattern: 010101...
        bits = [isodd(i) ? 0 : 1 for i in 1:nbits]
        product_state(nbits, sum(b << (i-1) for (i,b) in enumerate(bits)))
    end
    
    if use_dm
        return MixedEDState(density_matrix(reg), nbits)
    else
        return PureEDState(reg, nbits)
    end
end

"""
Time evolution using multiple dispatch
"""
function evolve_state(state::PureEDState, H::AbstractBlock, t::Real)
    evolved_reg = copy(state.reg) |> time_evolve(H, t)
    return PureEDState(evolved_reg, state.nbits)
end

function evolve_state(state::MixedEDState, H::AbstractBlock, t::Real)
    # Use Yao's density matrix evolution
    U = time_evolve(H, t)
    evolved_dm = U * state.dm * U'
    return MixedEDState(evolved_dm, state.nbits)
end

"""
Trace out bath qubits with multiple dispatch
"""
function trace_bath(state::MixedEDState, N::Int)
    bath_qubits = [2i for i in 1:N]
    ρ_sys = partial_tr(state.dm, bath_qubits)
    return MixedEDState(ρ_sys, N)
end

function trace_bath(state::PureEDState, N::Int)
    # For pure states, measure and project
    bath_qubits = [2i for i in 1:N]
    
    # Copy the state for measurement
    measured_state = copy(state.reg)
    
    # Measure bath qubits
    measured_state |> measure!(bath_qubits)
    bath_results = measured_values(measured_state)
    
    # Extract system state by tracing out bath
    sys_qubits = [2i-1 for i in 1:N]
    sys_state = focus!(measured_state, sys_qubits...) |> relax!(nbits=N)
    
    return PureEDState(sys_state, N), bath_results
end

"""
Apply noise using Yao's channel formalism
"""
function apply_noise(state::MixedEDState, pe::Float64)
    N = state.nbits ÷ 2
    
    # Create depolarizing channel
    noisy_dm = state.dm
    for i in 1:N
        q = 2i - 1  # System qubit
        # Apply depolarizing channel
        noisy_dm = (1 - pe) * noisy_dm + 
                   (pe/3) * (put(state.nbits, q=>X) * noisy_dm * put(state.nbits, q=>X)' +
                            put(state.nbits, q=>Y) * noisy_dm * put(state.nbits, q=>Y)' +
                            put(state.nbits, q=>Z) * noisy_dm * put(state.nbits, q=>Z)')
    end
    
    return MixedEDState(noisy_dm, state.nbits)
end

function apply_noise(state::PureEDState, pe::Float64)
    N = state.nbits ÷ 2
    noisy_reg = copy(state.reg)
    
    for i in 1:N
        q = 2i - 1
        if rand() < pe
            op = rand([X, Y, Z])
            noisy_reg |> put(q=>op)
        end
    end
    
    return PureEDState(noisy_reg, state.nbits)
end

"""
Compute observables using native Yao.jl functions
"""
function compute_observables(state::MixedEDState, H_sys::AbstractBlock, ϕ₀::AbstractRegister)
    # Energy expectation
    energy = real(expect(H_sys, state.dm))
    
    # Ground state fidelity
    gs_overlap = real(fidelity(state.dm, ϕ₀))
    
    # Purity
    purity = real(purity(state.dm))
    
    return energy, gs_overlap, purity
end

function compute_observables(state::PureEDState, H_sys::AbstractBlock, ϕ₀::AbstractRegister)
    energy = real(expect(H_sys, state.reg))
    gs_overlap = abs2(fidelity(state.reg, ϕ₀))
    purity = 1.0
    
    return energy, gs_overlap, purity
end

"""
Compute bath magnetization using Yao's expect
"""
function compute_bath_magnetization(state::AbstractEDState, N::Int)
    bath_z = 0.0
    for i in 1:N
        b = 2i
        Z_op = put(state.nbits, b=>Z)
        bath_z += real(expect(Z_op, state isa PureEDState ? state.reg : state.dm))
    end
    return bath_z / N
end

"""
Main cooling simulation with cleaner structure
"""
function run_cooling_ed_graceful(H_sys, H_full, ϕ₀, initial_state::AbstractEDState, 
                                coupling_params, sim_params)
    steps = coupling_params["steps"]
    te = coupling_params["te"]
    N = nbits(initial_state) ÷ 2
    Δ = coupling_params["Δ"]
    pe = get(sim_params, "pe", 0.0)
    
    # Results storage
    results = Dict(
        "E_list" => zeros(steps + 1),
        "GS_overlap_list" => zeros(steps + 1),
        "purity_list" => zeros(steps + 1),
        "bath_z_list" => zeros(steps + 1)
    )
    
    # For pure states, run Monte Carlo
    if initial_state isa PureEDState && get(sim_params, "n_trajectories", 1) > 1
        return run_monte_carlo_cooling(H_sys, H_full, ϕ₀, initial_state, 
                                     coupling_params, sim_params)
    end
    
    # Convert to density matrix if needed
    current_state = initial_state isa PureEDState ? to_density_matrix(initial_state) : initial_state
    
    println("Running ED cooling with $(initial_state isa PureEDState ? "pure state" : "density matrix")")
    
    # Initial observables
    sys_state = partial_tr(current_state.dm, [2i for i in 1:N])
    sys_state_wrapped = MixedEDState(sys_state, N)
    
    results["E_list"][1], results["GS_overlap_list"][1], results["purity_list"][1] = 
        compute_observables(sys_state_wrapped, H_sys, ϕ₀)
    
    println("Step 1: E/N=$(results["E_list"][1]/N), overlap=$(results["GS_overlap_list"][1])")
    
    for step in 2:steps+1
        # Create fresh bath in ground state
        bath_state = Δ < 0 ? product_state(bit"1"^N) : zero_state(N)
        bath_dm = density_matrix(bath_state)
        
        # Combine system and bath
        full_dm = kron(sys_state, bath_dm)
        current_state = MixedEDState(full_dm, 2N)
        
        # Time evolution
        current_state = evolve_state(current_state, H_full, te)
        
        # Apply noise if present
        if pe > 0
            current_state = apply_noise(current_state, pe)
        end
        
        # Compute bath magnetization before tracing out
        results["bath_z_list"][step] = compute_bath_magnetization(current_state, N)
        
        # Trace out bath
        sys_state_wrapped = trace_bath(current_state, N)
        sys_state = sys_state_wrapped.dm
        
        # Compute observables
        results["E_list"][step], results["GS_overlap_list"][step], results["purity_list"][step] = 
            compute_observables(sys_state_wrapped, H_sys, ϕ₀)
        
        println("Step $step: E/N=$(results["E_list"][step]/N), " *
                "overlap=$(results["GS_overlap_list"][step]), " *
                "⟨Z⟩_bath=$(results["bath_z_list"][step])")
    end
    
    results["method"] = "ED_DensityMatrix"
    return results
end

"""
Monte Carlo wavefunction cooling
"""
function run_monte_carlo_cooling(H_sys, H_full, ϕ₀, initial_state::PureEDState, 
                               coupling_params, sim_params)
    steps = coupling_params["steps"]
    te = coupling_params["te"]
    N = nbits(initial_state) ÷ 2
    Δ = coupling_params["Δ"]
    pe = get(sim_params, "pe", 0.0)
    n_traj = get(sim_params, "n_trajectories", 100)
    
    println("Running Monte Carlo wavefunction cooling with $n_traj trajectories")
    
    # Storage for trajectories
    E_trajs = zeros(steps + 1, n_traj)
    overlap_trajs = zeros(steps + 1, n_traj)
    
    for traj in 1:n_traj
        current = PureEDState(copy(initial_state.reg), N)
        
        # Initial observables
        E_trajs[1, traj], overlap_trajs[1, traj], _ = compute_observables(current, H_sys, ϕ₀)
        
        for step in 2:steps+1
            # Fresh bath
            bath_state = Δ < 0 ? product_state(bit"1"^N) : zero_state(N)
            full_state = join(current.reg, bath_state)
            full_wrapped = PureEDState(full_state, 2N)
            
            # Evolution
            full_wrapped = evolve_state(full_wrapped, H_full, te)
            
            # Noise
            if pe > 0
                full_wrapped = apply_noise(full_wrapped, pe)
            end
            
            # Trace bath
            current, _ = trace_bath(full_wrapped, N)
            normalize!(current.reg)
            
            # Observables
            E_trajs[step, traj], overlap_trajs[step, traj], _ = 
                compute_observables(current, H_sys, ϕ₀)
        end
    end
    
    # Average over trajectories
    return Dict(
        "E_list" => vec(mean(E_trajs, dims=2)),
        "GS_overlap_list" => vec(mean(overlap_trajs, dims=2)),
        "purity_list" => ones(steps + 1),
        "bath_z_list" => zeros(steps + 1),
        "method" => "ED_MonteCarloWavefunction",
        "n_trajectories" => n_traj
    )
end

# Export the graceful versions
export AbstractEDState, PureEDState, MixedEDState
export build_hamiltonian_graceful, create_initial_state_ed
export evolve_state, trace_bath, apply_noise
export compute_observables, compute_bath_magnetization
export run_cooling_ed_graceful