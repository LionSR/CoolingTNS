using Yao
using YaoPlots
using LinearAlgebra
using ExponentialUtilities
using KrylovKit
using Random
using Statistics

# Import parse_coupling from ham.jl (it's already included in the module)
# This will be available when the file is included in CoolingTNS module

# SimulationMethod types are now defined in cooling_interface.jl

# Container for ED-specific quantum states
struct EDState{T}
    state::T  # Can be a density matrix or state vector
    nbits::Int
    method::SimulationMethod
end

"""
    setup_init_state_ed(nbits::Int; init_type="product", theta=0.0, method=MonteCarloWavefunction())

Create initial state for ED simulation. Supports both density matrices and pure states.
"""
function setup_init_state_ed(nbits::Int; init_type="product", theta=0.0, method=MonteCarloWavefunction())
    # For Monte Carlo, we only need system qubits
    # For density matrix, we need system + bath
    if method isa MonteCarloWavefunction
        N = nbits ÷ 2  # Number of system spins
        nbits_to_use = N  # Only system qubits
    else
        N = nbits ÷ 2  # Number of system spins
        nbits_to_use = nbits  # System + bath qubits
    end
    
    if init_type == "identity"
        # Maximally mixed state
        if method isa DensityMatrix
            ρ_mat = Matrix(I, 1<<nbits_to_use, 1<<nbits_to_use) / (1<<nbits_to_use)
            ρ = Yao.DensityMatrix(reshape(ρ_mat, 1<<nbits_to_use, 1<<nbits_to_use, 1))
            return EDState(ρ, nbits, method)
        else
            # For pure states, create a random state
            ψ = rand_state(nbits_to_use)
            return EDState(ψ, nbits_to_use, method)
        end
    elseif init_type == "theta"
        # Parameterized product state: |ψ⟩ = ⊗[cos(θπ)|0⟩ + sin(θπ)|1⟩]
        theta_rad = theta * π
        
        # Create single qubit state
        if abs(theta + 0.5) < 1e-10  # All down
            reg = product_state(bit"1"^nbits_to_use)  # |111...⟩
        elseif abs(theta - 0.5) < 1e-10  # All up  
            reg = product_state(bit"0"^nbits_to_use)  # |000...⟩
        elseif abs(theta) < 1e-10  # X+ state
            reg = zero_state(nbits_to_use)
            reg |> chain(nbits_to_use, put(i=>H) for i in 1:nbits_to_use)
        else
            # General theta state
            reg = zero_state(nbits_to_use)
            for i in 1:nbits_to_use
                reg |> put(i => Ry(2*theta_rad))
            end
        end
        
        if method isa DensityMatrix
            return EDState(density_matrix(reg), nbits, method)
        else
            return EDState(reg, nbits_to_use, method)
        end
    else
        # Default: alternating up/down product state
        config_bits = [isodd(i) ? 0 : 1 for i in 1:nbits_to_use]
        config_int = sum(b << (i-1) for (i, b) in enumerate(config_bits))
        reg = product_state(nbits_to_use, config_int)
        
        if method isa DensityMatrix
            return EDState(density_matrix(reg), nbits, method)
        else
            return EDState(reg, nbits_to_use, method)
        end
    end
end

"""
    build_hamiltonian_ed(problem::String, N::Int, ham_params, coupling_params)

Build system+bath Hamiltonian using Yao.jl operators with graceful functional style.
"""
function build_hamiltonian_ed(problem::String, N::Int, ham_params, coupling_params)
    nbits = 2N  # N system + N bath qubits
    g = coupling_params["g"]
    Δ = coupling_params["Δ"] 
    coupling = coupling_params["coupling"]
    
    # Parse coupling operators
    op1_str, op2_str = parse_coupling(coupling)
    op_map = Dict("X" => X, "Y" => Y, "Z" => Z)
    op1, op2 = op_map[op1_str], op_map[op2_str]
    
    # System sites (odd) and bath sites (even)
    sys_sites = 1:2:2N-1
    bath_sites = 2:2:2N
    
    if problem == "niIsing"
        J, hx, hz = ham_params
        
        # System Hamiltonian using functional style
        H_sys = sum([
            # ZZ interactions between system spins
            map(i -> J * put(nbits, sys_sites[i]=>Z) * put(nbits, sys_sites[i+1]=>Z), 1:N-1)...,
            # X field on system spins
            map(s -> hx * put(nbits, s=>X), sys_sites)...,
            # Z field on system spins
            map(s -> hz * put(nbits, s=>Z), sys_sites)...
        ])
        
        # Bath Hamiltonian
        H_bath = sum(map(b -> -Δ/2 * put(nbits, b=>Z), bath_sites))
        
        # System-bath coupling
        H_coupling = sum(map(i -> g * put(nbits, sys_sites[i]=>op1) * put(nbits, bath_sites[i]=>op2), 1:N))
        
        return H_sys + H_bath + H_coupling
        
    elseif problem == "Ising"
        J, h = ham_params
        
        # Transverse field Ising model
        H_sys = sum([
            map(i -> J * put(nbits, sys_sites[i]=>Z) * put(nbits, sys_sites[i+1]=>Z), 1:N-1)...,
            map(s -> h * put(nbits, s=>X), sys_sites)...
        ])
        
        # Bath and coupling combined
        H_bath_coupling = sum(map(i -> -Δ/2 * put(nbits, bath_sites[i]=>Z) + 
                                       g * put(nbits, sys_sites[i]=>op1) * put(nbits, bath_sites[i]=>op2), 1:N))
        
        return H_sys + H_bath_coupling
    else
        error("Unknown problem type: $problem")
    end
end

"""
    setup_problem_ed(N, problem, ham_params, coupling_params, sim_params)

Setup ED problem including system Hamiltonian and ground state.
"""
function setup_problem_ed(N, problem, ham_params, coupling_params, sim_params)
    # Build system-only Hamiltonian for ground state calculation
    nbits_sys = N
    
    if problem == "niIsing"
        J, hx, hz = ham_params
        # Non-integrable Ising model
        H_sys = sum([
            # ZZ interactions
            map(i -> J * put(nbits_sys, (i, i+1)=>kron(Z, Z)), 1:N-1)...,
            # X field
            map(i -> hx * put(nbits_sys, i=>X), 1:N)...,
            # Z field  
            map(i -> hz * put(nbits_sys, i=>Z), 1:N)...
        ])
    elseif problem == "Ising"
        J, h = ham_params
        # Transverse field Ising model
        H_sys = sum([
            map(i -> J * put(nbits_sys, (i, i+1)=>kron(Z, Z)), 1:N-1)...,
            map(i -> h * put(nbits_sys, i=>X), 1:N)...
        ])
    else
        error("Unknown problem type: $problem")
    end
    
    # Find ground state using KrylovKit
    H_sys_mat = mat(H_sys)
    vals, vecs, info = eigsolve(H_sys_mat, 1, :SR; krylovdim=min(30, size(H_sys_mat, 1)))
    e₀ = real(vals[1])
    ϕ₀_vec = vecs[1]
    ϕ₀ = ArrayReg(normalize!(Complex.(ϕ₀_vec)))
    
    # Convert to dict if needed for backward compatibility
    coupling_dict = coupling_params isa CouplingParameters ? to_dict(coupling_params) : coupling_params
    
    # Set resonant cooling if Δ not specified
    if !haskey(coupling_dict, "Δ")
        # Find gap
        vals2, _, _ = eigsolve(H_sys_mat, 2, :SR; krylovdim=min(30, size(H_sys_mat, 1)))
        gap = real(vals2[2] - vals2[1])
        coupling_dict["Δ"] = -gap  # Resonant cooling
    end
    
    # Build full system+bath Hamiltonian
    H_full = build_hamiltonian_ed(problem, N, ham_params, coupling_dict)
    
    return H_sys, H_full, ϕ₀, e₀
end

"""
    evolve_ed(state::EDState, H, t::Real; method=:exponential)

Evolve quantum state under Hamiltonian H for time t.
"""
function evolve_ed(state::EDState{<:DensityMatrix}, H, t::Real; method=:exponential)
    # Density matrix evolution: ρ(t) = U ρ U†, where U = exp(-iHt)
    H_mat = mat(H)
    U = exp(-1im * H_mat * t)
    # Reshape from 3D to 2D for matrix operations
    ρ_mat_3d = state.state.state
    ρ_mat = reshape(ρ_mat_3d, size(ρ_mat_3d, 1), size(ρ_mat_3d, 2))
    ρ_evolved = U * ρ_mat * U'
    # Convert back to 3D for DensityMatrix
    ρ_evolved_3d = reshape(ρ_evolved, size(ρ_evolved, 1), size(ρ_evolved, 2), 1)
    return EDState(Yao.DensityMatrix(ρ_evolved_3d), state.nbits, state.method)
end

function evolve_ed(state::EDState{<:AbstractRegister}, H, t::Real; method=:exponential)
    # Pure state evolution: |ψ(t)⟩ = exp(-iHt)|ψ⟩
    if method == :exponential
        # Use matrix exponentiation instead of Yao's time evolution
        H_mat = mat(H)
        ψ_vec = state.state |> statevec
        # Convert sparse matrix to dense before exponentiation
        H_dense = Matrix(H_mat)
        U = exp(-1im * H_dense * t)
        ψ_evolved_vec = U * ψ_vec
        evolved_state = ArrayReg(ψ_evolved_vec)
        return EDState(evolved_state, state.nbits, state.method)
    elseif method == :krylov
        # Use Krylov methods for large systems
        H_mat = mat(H)
        ψ_vec = state.state |> statevec
        ψ_evolved_vec = expv(t, -1im * H_mat, ψ_vec)
        evolved_state = ArrayReg(ψ_evolved_vec)
        return EDState(evolved_state, state.nbits, state.method)
    end
end

"""
    trace_bath_ed(state::EDState, N::Int)

Trace out bath qubits, keeping only system qubits.
"""
function trace_bath_ed(state::EDState, N::Int)
    if state.state isa Yao.DensityMatrix
        # Bath qubits are at even positions (2, 4, 6, ...)
        # System qubits are at odd positions (1, 3, 5, ...)
        bath_qubits = [2i for i in 1:N]
        
        # Get the density matrix as an array and reshape from 3D to 2D
        ρ_mat_3d = state.state.state
        ρ_mat = reshape(ρ_mat_3d, size(ρ_mat_3d, 1), size(ρ_mat_3d, 2))
        nbits = state.nbits
        
        # Compute partial trace manually
        dim_sys = 1 << N
        dim_bath = 1 << N
        ρ_sys_mat = zeros(ComplexF64, dim_sys, dim_sys)
        
        for i_sys in 0:dim_sys-1
            for j_sys in 0:dim_sys-1
                for k_bath in 0:dim_bath-1
                    # Map system and bath indices to full index
                    # System bits are at odd positions (1,3,5), bath at even (2,4,6)
                    i_full = 0
                    j_full = 0
                    for bit in 0:N-1
                        # System bit at position 2*bit (0-indexed)
                        if (i_sys >> bit) & 1 == 1
                            i_full |= 1 << (2*bit)
                        end
                        if (j_sys >> bit) & 1 == 1
                            j_full |= 1 << (2*bit)
                        end
                        # Bath bit at position 2*bit + 1
                        if (k_bath >> bit) & 1 == 1
                            i_full |= 1 << (2*bit + 1)
                            j_full |= 1 << (2*bit + 1)
                        end
                    end
                    
                    ρ_sys_mat[i_sys+1, j_sys+1] += ρ_mat[i_full+1, j_full+1]
                end
            end
        end
        
        # Create density matrix for system
        # DensityMatrix expects a 3D array with shape (dim, dim, 1)
        ρ_sys_3d = reshape(ρ_sys_mat, dim_sys, dim_sys, 1)
        return Yao.DensityMatrix(ρ_sys_3d)
    else
        # For pure states in Monte Carlo, we sample the bath
        bath_qubits = [2i for i in 1:N]
        
        # Measure bath qubits
        reg_measured = copy(state.state)
        measured_results = Yao.measure!(reg_measured, bath_qubits)
        
        # Extract system state by focusing on system qubits after measurement
        # System qubits are at odd positions (1, 3, 5, ...)
        sys_qubits = [2i-1 for i in 1:N]
        
        # Focus on system qubits and create new reduced register
        focused_reg = Yao.focus!(reg_measured, sys_qubits)
        sys_state_vec = Yao.statevec(focused_reg)
        
        # Extract only the active part of the state vector (first 2^N elements)
        sys_reg = ArrayReg(sys_state_vec[1:1<<N])
        
        # Debug print (disabled)
        # println("DEBUG: sys_vec length = $(length(sys_state_vec)), expected = $(1<<N)")
        
        # measured_results is a BitVector containing the measurement outcomes
        bath_config = [Int(b) for b in measured_results]
        
        # Return both system state and bath measurement
        return sys_reg, bath_config
    end
end

"""
    apply_noise_ed(state::EDState, noise_model::Dict)

Apply noise to the quantum state.
"""
function apply_noise_ed(state::EDState{<:DensityMatrix}, noise_model::Dict)
    if haskey(noise_model, "depolarizing")
        pe = noise_model["depolarizing"]
        N = state.nbits ÷ 2
        
        # Apply single-qubit depolarizing channel to system qubits
        ρ = state.state.state
        for i in 1:N
            q = 2i - 1  # System qubit position
            # E(ρ) = (1-p)ρ + p/3(XρX + YρY + ZρZ)
            ρ_new = (1 - pe) * ρ
            for op in [X, Y, Z]
                U = mat(put(state.nbits, q=>op))
                ρ_new += (pe/3) * U * ρ * U'
            end
            ρ = ρ_new
        end
        
        return EDState(DensityMatrix(state.nbits, ρ), state.nbits, state.method)
    end
    
    return state
end

function apply_noise_ed(state::EDState{<:AbstractRegister}, noise_model::Dict)
    if haskey(noise_model, "depolarizing")
        pe = noise_model["depolarizing"]
        N = state.nbits ÷ 2
        
        # For pure states, apply Pauli operators stochastically
        for i in 1:N
            q = 2i - 1  # System qubit
            if rand() < pe
                # Randomly apply X, Y, or Z
                op = rand([X, Y, Z])
                state.state |> put(q=>op)
            end
        end
    end
    
    return state
end

"""
    compute_observables_ed(ρ_sys, H_sys, ϕ₀)

Compute energy, ground state overlap, and purity.
"""
function compute_observables_ed(ρ_sys::Yao.DensityMatrix, H_sys, ϕ₀)
    # Compute expectation value manually to avoid ambiguity
    # ⟨H⟩ = Tr(H ρ)
    H_mat = mat(H_sys)
    # DensityMatrix stores as 3D array, reshape to 2D
    ρ_mat_3d = ρ_sys.state
    ρ_mat = reshape(ρ_mat_3d, size(ρ_mat_3d, 1), size(ρ_mat_3d, 2))
    energy = real(tr(H_mat * ρ_mat))
    
    # Ground state overlap: ⟨ϕ₀|ρ|ϕ₀⟩
    ϕ₀_vec = ϕ₀ |> statevec
    gs_overlap = real(ϕ₀_vec' * ρ_mat * ϕ₀_vec)
    
    # Purity: Tr(ρ²)
    purity_val = real(tr(ρ_mat * ρ_mat))
    
    return energy, gs_overlap, purity_val
end

function compute_observables_ed(ψ_sys::Yao.AbstractRegister, H_sys, ϕ₀)
    # Energy: ⟨ψ|H|ψ⟩
    energy = real(Yao.expect(H_sys, ψ_sys))
    
    # Ground state overlap: |⟨ϕ₀|ψ⟩|²
    gs_overlap = abs2(ϕ₀' * ψ_sys)
    
    # Purity is always 1 for pure states
    purity = 1.0
    
    return energy, gs_overlap, purity
end

"""
    run_cooling_ed(H_sys, H_sys_bath, ϕ₀, initial_state, coupling_params, sim_params)

Run ED cooling simulation with either density matrix or Monte Carlo wavefunction method.
Returns typed CoolingResults struct based on simulation method.
"""
function run_cooling_ed(H_sys, H_sys_bath, ϕ₀, initial_state::EDState, coupling_params, sim_params)
    # Handle both dict and struct parameter types
    if coupling_params isa CouplingParameters
        steps = coupling_params.steps
        te = coupling_params.te
        coupling_dict = to_dict(coupling_params)
    else
        steps = coupling_params["steps"]
        te = coupling_params["te"]
        coupling_dict = coupling_params
    end
    
    if sim_params isa SimulationParameters
        sim_dict = to_dict(sim_params)
    else
        sim_dict = sim_params
    end
    
    N = initial_state.nbits ÷ 2
    
    # Noise model
    pe = get(sim_dict, "pe", 0.0)
    noise_model = pe > 0 ? Dict("depolarizing" => pe) : Dict()
    
    # Results storage
    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    purity_list = zeros(Float64, steps + 1)
    bath_z_list = zeros(Float64, steps + 1)
    
    # Monte Carlo specific
    if initial_state.method isa MonteCarloWavefunction
        n_trajectories = get(sim_dict, "n_trajectories", 1)
        return run_cooling_monte_carlo(
            H_sys, H_sys_bath, ϕ₀, initial_state, 
            coupling_dict, sim_dict, noise_model,
            n_trajectories
        )
    end
    
    # Density matrix evolution
    println("Running ED cooling with density matrix method")
    
    # Convert initial pure state to density matrix if needed
    if initial_state.method isa MonteCarloWavefunction
        ρ = density_matrix(initial_state.state)
        current_state = EDState(ρ, 2N, DensityMatrix())
    else
        current_state = initial_state
    end
    
    # Initial observables (system only, first N qubits)
    # For initial state, we only have system qubits, so create a dummy full state
    if current_state.nbits == N
        # Initial state is system-only, use it directly
        ρ_sys = current_state.state
    else
        # Trace out bath if present
        println("DEBUG: state type = $(typeof(current_state.state))")
        println("DEBUG: is DensityMatrix? $(current_state.state isa Yao.DensityMatrix)")
        ρ_sys = trace_bath_ed(current_state, N)
    end
    E_list[1], GS_overlap_list[1], purity_list[1] = compute_observables_ed(
        ρ_sys, H_sys, ϕ₀
    )
    
    println("Step 1: E/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1]), purity=$(purity_list[1])")
    
    for step in 2:steps+1
        # Add fresh bath in ground state
        # Bath ground state: |1⟩ for Δ<0, |0⟩ for Δ>0
        Δ = coupling_params["Δ"]
        bath_state = Δ < 0 ? product_state(bit"1"^N) : zero_state(N)
        ρ_bath = density_matrix(bath_state)
        
        # Combine system and bath
        # Note: For density matrices, we need to extract the matrix and do kron
        # Reshape from 3D to 2D
        ρ_sys_mat = reshape(ρ_sys.state, size(ρ_sys.state, 1), size(ρ_sys.state, 2))
        ρ_bath_mat = reshape(ρ_bath.state, size(ρ_bath.state, 1), size(ρ_bath.state, 2))
        ρ_sys_bath = kron(ρ_sys_mat, ρ_bath_mat)
        # DensityMatrix expects a 3D array
        ρ_sys_bath_3d = reshape(ρ_sys_bath, size(ρ_sys_bath, 1), size(ρ_sys_bath, 2), 1)
        current_state = EDState(Yao.DensityMatrix(ρ_sys_bath_3d), 2N, DensityMatrix())
        
        # Time evolution
        current_state = evolve_ed(current_state, H_full, te)
        
        # Apply noise
        if !isempty(noise_model)
            current_state = apply_noise_ed(current_state, noise_model)
        end
        
        # Trace out bath
        ρ_sys = trace_bath_ed(current_state, N)
        
        # Compute observables
        E_list[step], GS_overlap_list[step], purity_list[step] = compute_observables_ed(
            ρ_sys, H_sys, ϕ₀
        )
        
        # Bath magnetization (before tracing out)
        bath_z = 0.0
        for i in 1:N
            b = 2i
            Z_op = put(2N, b=>Z)
            # Compute expectation value manually: ⟨Z⟩ = Tr(Z ρ)
            Z_mat = mat(Z_op)
            ρ_mat_3d = current_state.state.state
            ρ_mat = reshape(ρ_mat_3d, size(ρ_mat_3d, 1), size(ρ_mat_3d, 2))
            bath_z += real(tr(Z_mat * ρ_mat))
        end
        bath_z_list[step] = bath_z / N
        
        println("Step $step: E/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step]), " *
                "purity=$(purity_list[step]), ⟨Z⟩_bath=$(bath_z_list[step])")
    end
    
    # Calculate von Neumann entropy and trace distance for density matrix results
    von_neumann_entropy = zeros(length(E_list))
    trace_distance = zeros(length(E_list))
    
    # TODO: Implement von Neumann entropy and trace distance calculations
    # These would require computing eigenvalues of density matrices
    
    # Return typed DensityMatrixResults struct
    return create_results(
        DensityMatrix(), E_list, GS_overlap_list, purity_list;
        von_neumann_entropy = von_neumann_entropy,
        trace_distance = trace_distance
    )
end

"""
    run_cooling_monte_carlo(H_sys, H_full, ϕ₀, initial_state, params...)

Run Monte Carlo wavefunction cooling simulation.
"""
function run_cooling_monte_carlo(H_sys, H_full, ϕ₀, initial_state, 
                                coupling_params, sim_params, noise_model, n_trajectories)
    steps = coupling_params["steps"]
    te = coupling_params["te"]
    # For Monte Carlo, initial state has only system qubits
    N = initial_state.nbits
    Δ = coupling_params["Δ"]
    
    println("Running ED cooling with Monte Carlo wavefunction method ($n_trajectories trajectories)")
    
    # Results storage for averaging
    E_trajs = zeros(Float64, steps + 1, n_trajectories)
    GS_overlap_trajs = zeros(Float64, steps + 1, n_trajectories)
    
    for traj in 1:n_trajectories
        # Fresh copy of initial state for each trajectory
        # For Monte Carlo, we start with system state only
        ψ_sys = copy(initial_state.state)
        
        # Initial observables
        E_trajs[1, traj] = real(Yao.expect(H_sys, ψ_sys))
        GS_overlap_trajs[1, traj] = abs2(ϕ₀' * ψ_sys)
        
        for step in 2:steps+1
            # Add fresh bath
            if Δ < 0
                # All bath spins in |1⟩ state (all bits set to 1)
                bath_state = product_state(N, (1<<N) - 1)  # This creates |111...⟩
            else
                # All bath spins in |0⟩ state
                bath_state = zero_state(N)
            end
            
            # Create properly interleaved state (s1, b1, s2, b2, ...)
            # First create a state with all qubits
            ψ_full = zero_state(2N)
            
            # Get state vectors
            sys_vec = ψ_sys |> statevec
            bath_vec = bath_state |> statevec
            
            # println("DEBUG: sys_vec length = $(length(sys_vec)), expected = $(1<<N)")
            # println("DEBUG: bath_vec length = $(length(bath_vec)), expected = $(1<<N)")
            # println("DEBUG: bath_state nqubits = $(nqubits(bath_state))")
            
            # Create the tensor product with proper ordering
            full_vec = zeros(ComplexF64, 1<<(2N))
            for i_sys in 0:(1<<N)-1
                for i_bath in 0:(1<<N)-1
                    # Map to interleaved index
                    i_full = 0
                    for bit in 0:N-1
                        if (i_sys >> bit) & 1 == 1
                            i_full |= 1 << (2*bit)
                        end
                        if (i_bath >> bit) & 1 == 1
                            i_full |= 1 << (2*bit + 1)
                        end
                    end
                    full_vec[i_full+1] = sys_vec[i_sys+1] * bath_vec[i_bath+1]
                end
            end
            
            ψ_full = ArrayReg(full_vec)
            
            # Debug print (disabled)
            # println("DEBUG: N=$N, 2N=$(2N)")
            # println("DEBUG: nqubits(ψ_full) = $(nqubits(ψ_full))")
            # println("DEBUG: length(full_vec) = $(length(full_vec)), expected = $(1<<(2N))")
            # println("DEBUG: nqubits(H_full) = $(nqubits(H_full))")
            
            # Create EDState wrapper
            state = EDState(ψ_full, 2N, MonteCarloWavefunction())
            
            # Time evolution  
            state = evolve_ed(state, H_full, te)
            
            # Apply noise
            if !isempty(noise_model)
                state = apply_noise_ed(state, noise_model)
            end
            
            # Sample bath and get system state
            ψ_sys, bath_config = trace_bath_ed(state, N)
            normalize!(ψ_sys)
            
            # Compute observables
            E_val = Yao.expect(H_sys, ψ_sys)
            E_trajs[step, traj] = real(E_val isa Number ? E_val : E_val[1])
            overlap = Yao.statevec(ϕ₀)' * Yao.statevec(ψ_sys)
            GS_overlap_trajs[step, traj] = abs2(overlap[1])
        end
    end
    
    # Average over trajectories
    E_list = mean(E_trajs, dims=2)[:, 1]
    GS_overlap_list = mean(GS_overlap_trajs, dims=2)[:, 1]
    
    # Purity is always 1 for pure states  
    purity_list = ones(steps + 1)
    
    # Compute standard deviations
    E_std = std(E_trajs, dims=2)[:, 1]
    GS_std = std(GS_overlap_trajs, dims=2)[:, 1]
    
    # Print final results
    for step in 1:steps+1
        println("Step $step: E/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step])")
    end
    
    # Return typed MonteCarloResults struct
    return create_results(
        MonteCarloWavefunction(), E_list, GS_overlap_list, purity_list;
        E_trajectories = E_trajs,
        GS_trajectories = GS_overlap_trajs,
        n_trajectories = n_trajectories,
        E_std = E_std,
        GS_std = GS_std
    )
end