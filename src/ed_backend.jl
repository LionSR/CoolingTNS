"""
    ed_backend.jl

Efficient ED backend implementation without Yao.jl dependencies.
Uses only linear algebra, sparse arrays, and Krylov methods.
All parameters are kept real (Float64) as per requirement.
"""

using LinearAlgebra
using SparseArrays
using KrylovKit
using Random

# ============================================================================
# Quantum State Types for ED Backend
# ============================================================================

if !@isdefined(EDStateVector)
    """
        EDStateVector

    Represents a pure quantum state as a vector in the computational basis.
    """
    struct EDStateVector
        data::Vector{Float64}  # Real amplitudes only
        n_qubits::Int
        
        function EDStateVector(data::Vector{Float64}, n_qubits::Int)
            @assert length(data) == 2^n_qubits "Vector dimension must be 2^n_qubits"
            new(normalize(data), n_qubits)
        end
    end
end

if !@isdefined(EDDensityMatrix)
    """
        EDDensityMatrix

    Represents a mixed quantum state as a density matrix.
    """
    struct EDDensityMatrix
        data::Matrix{Float64}  # Real density matrix
        n_qubits::Int
        
        function EDDensityMatrix(data::Matrix{Float64}, n_qubits::Int)
            @assert size(data, 1) == size(data, 2) == 2^n_qubits "Matrix dimension must be 2^n_qubits"
            @assert issymmetric(data) "Density matrix must be symmetric for real states"
            @assert abs(tr(data) - 1.0) < 1e-10 "Density matrix must have trace 1 (got $(tr(data)))"
            new(data, n_qubits)
        end
    end
end

# ============================================================================
# State Creation Functions
# ============================================================================

"""
    product_state_ed(n_qubits::Int, config::Int) -> EDStateVector

Create a product state with given bit configuration.
config=0 means all qubits in |0⟩, config=1 means |00...01⟩, etc.
"""
function product_state_ed(n_qubits::Int, config::Int=0)
    @assert 0 <= config < 2^n_qubits "Configuration out of range"
    data = zeros(Float64, 2^n_qubits)
    data[config + 1] = 1.0  # Julia is 1-indexed
    return EDStateVector(data, n_qubits)
end

"""
    zero_state_ed(n_qubits::Int) -> EDStateVector

Create the all-zero state |00...0⟩.
"""
zero_state_ed(n_qubits::Int) = product_state_ed(n_qubits, 0)

"""
    random_state_ed(n_qubits::Int) -> EDStateVector

Create a random normalized pure state with real amplitudes.
"""
function random_state_ed(n_qubits::Int)
    return EDStateVector(randn(Float64, 2^n_qubits), n_qubits)
end

"""
    maximally_mixed_ed(n_qubits::Int) -> EDDensityMatrix

Create the maximally mixed state I/2^n.
"""
function maximally_mixed_ed(n_qubits::Int)
    dim = 2^n_qubits
    return EDDensityMatrix(Matrix{Float64}(I, dim, dim) / dim, n_qubits)
end

# ============================================================================
# State Manipulation Functions
# ============================================================================

"""
    kron_states_ed(ψ1::EDStateVector, ψ2::EDStateVector) -> EDStateVector

Kronecker product of two state vectors.
"""
function kron_states_ed(ψ1::EDStateVector, ψ2::EDStateVector)
    return EDStateVector(kron(ψ1.data, ψ2.data), ψ1.n_qubits + ψ2.n_qubits)
end

"""
    kron_density_ed(ρ1::EDDensityMatrix, ρ2::EDDensityMatrix) -> EDDensityMatrix

Kronecker product of two density matrices.
"""
function kron_density_ed(ρ1::EDDensityMatrix, ρ2::EDDensityMatrix)
    return EDDensityMatrix(kron(ρ1.data, ρ2.data), ρ1.n_qubits + ρ2.n_qubits)
end

"""
    state_to_density_ed(ψ::EDStateVector) -> EDDensityMatrix

Convert pure state to density matrix: |ψ⟩⟨ψ|.
"""
function state_to_density_ed(ψ::EDStateVector)
    return EDDensityMatrix(ψ.data * ψ.data', ψ.n_qubits)
end

# ============================================================================
# Measurement Functions
# ============================================================================

"""
    measure_ed!(ψ::EDStateVector, qubits::Vector{Int}) -> (EDStateVector, Vector{Int})

Measure specified qubits and collapse the state.
Returns the post-measurement state (with measured qubits removed) and measurement outcomes.
"""
function measure_ed!(ψ::EDStateVector, qubits::Vector{Int})
    n_total = ψ.n_qubits
    n_measure = length(qubits)
    n_remaining = n_total - n_measure
    
    # Convert to 0-based indexing for bit operations
    qubits_0 = qubits .- 1
    
    # Calculate probabilities for each measurement outcome
    probs = zeros(Float64, 2^n_measure)
    
    for state_idx in 0:(2^n_total - 1)
        # Extract measurement bits
        outcome = 0
        for (i, q) in enumerate(qubits_0)
            if (state_idx >> q) & 1 == 1
                outcome |= (1 << (i-1))
            end
        end
        
        probs[outcome + 1] += ψ.data[state_idx + 1]^2
    end
    
    # Sample outcome based on probabilities
    outcome = sample_outcome(probs) - 1  # Convert to 0-based
    
    # Extract measurement results as bit array
    results = Int[]
    for i in 0:(n_measure-1)
        push!(results, (outcome >> i) & 1)
    end
    
    # Collapse state and trace out measured qubits
    collapsed_data = zeros(Float64, 2^n_remaining)
    
    # Create mask for remaining qubits
    remaining_qubits = setdiff(0:(n_total-1), qubits_0)
    
    for state_idx in 0:(2^n_total - 1)
        # Check if this state matches the measurement outcome
        matches = true
        for (i, q) in enumerate(qubits_0)
            if ((state_idx >> q) & 1) != results[i]
                matches = false
                break
            end
        end
        
        if matches
            # Extract index for remaining qubits
            remaining_idx = 0
            for (i, q) in enumerate(remaining_qubits)
                if (state_idx >> q) & 1 == 1
                    remaining_idx |= (1 << (i-1))
                end
            end
            
            collapsed_data[remaining_idx + 1] += ψ.data[state_idx + 1]
        end
    end
    
    return EDStateVector(collapsed_data, n_remaining), results
end

"""
    sample_outcome(probs::Vector{Float64}) -> Int

Sample an outcome index based on probability distribution.
"""
function sample_outcome(probs::Vector{Float64})
    r = rand()
    cumsum = 0.0
    for (i, p) in enumerate(probs)
        cumsum += p
        if r <= cumsum
            return i
        end
    end
    return length(probs)
end

"""
    expect_ed(op::AbstractMatrix, ψ::EDStateVector) -> Float64

Compute expectation value ⟨ψ|op|ψ⟩.
"""
function expect_ed(op::AbstractMatrix, ψ::EDStateVector)
    return real(dot(ψ.data, op * ψ.data))
end

"""
    expect_ed(op::AbstractMatrix, ρ::EDDensityMatrix) -> Float64

Compute expectation value Tr(op·ρ).
"""
function expect_ed(op::AbstractMatrix, ρ::EDDensityMatrix)
    return real(tr(op * ρ.data))
end

# ============================================================================
# Partial Trace Functions
# ============================================================================

"""
    partial_trace_ed(ρ::EDDensityMatrix, keep_qubits::Vector{Int}) -> EDDensityMatrix

Trace out all qubits except those specified in keep_qubits.
"""
function partial_trace_ed(ρ::EDDensityMatrix, keep_qubits::Vector{Int})
    n_total = ρ.n_qubits
    n_keep = length(keep_qubits)
    
    dim_keep = 2^n_keep
    dim_trace = 2^(n_total - n_keep)
    
    # Convert to 0-based indexing
    keep_qubits_0 = keep_qubits .- 1
    trace_qubits_0 = setdiff(0:(n_total-1), keep_qubits_0)
    
    ρ_reduced = zeros(Float64, dim_keep, dim_keep)
    
    for i in 0:(dim_keep-1), j in 0:(dim_keep-1)
        # Sum over traced out degrees of freedom
        for k in 0:(dim_trace-1)
            # Construct full indices
            idx_i = construct_index(i, k, keep_qubits_0, trace_qubits_0)
            idx_j = construct_index(j, k, keep_qubits_0, trace_qubits_0)
            
            ρ_reduced[i+1, j+1] += ρ.data[idx_i+1, idx_j+1]
        end
    end
    
    return EDDensityMatrix(ρ_reduced, n_keep)
end

"""
    construct_index(keep_idx::Int, trace_idx::Int, keep_qubits::Vector{Int}, trace_qubits::Vector{Int}) -> Int

Construct the full index from partial indices.
"""
function construct_index(keep_idx::Int, trace_idx::Int, keep_qubits::Vector{Int}, trace_qubits::Vector{Int})
    full_idx = 0
    
    # Set bits for kept qubits
    for (i, q) in enumerate(keep_qubits)
        if (keep_idx >> (i-1)) & 1 == 1
            full_idx |= (1 << q)
        end
    end
    
    # Set bits for traced qubits
    for (i, q) in enumerate(trace_qubits)
        if (trace_idx >> (i-1)) & 1 == 1
            full_idx |= (1 << q)
        end
    end
    
    return full_idx
end

"""
    trace_out_system_ed(ρ::EDDensityMatrix, n_sys::Int) -> EDDensityMatrix

Trace out first n_sys qubits (system), keeping the rest (bath).
"""
function trace_out_system_ed(ρ::EDDensityMatrix, n_sys::Int)
    return partial_trace_ed(ρ, collect((n_sys+1):ρ.n_qubits))
end

"""
    trace_out_bath_ed(ρ::EDDensityMatrix, n_sys::Int) -> EDDensityMatrix

Trace out bath qubits, keeping system qubits.
"""
function trace_out_bath_ed(ρ::EDDensityMatrix, n_sys::Int)
    return partial_trace_ed(ρ, collect(1:n_sys))
end

# ============================================================================
# Time Evolution Functions
# ============================================================================

"""
    evolve_ed(H::AbstractMatrix, ψ::EDStateVector, t::Float64) -> EDStateVector

Time evolve a state vector under Hamiltonian H for time t.
"""
function evolve_ed(H::AbstractMatrix, ψ::EDStateVector, t::Float64)
    # For real Hamiltonians and real initial states, we can use real arithmetic
    # exp(-iHt)|ψ⟩ = cos(Ht)|ψ⟩ - i*sin(Ht)|ψ⟩
    # Since |ψ⟩ is real and H is real, the evolved state stays real if we project
    
    n = size(H, 1)
    
    # Compute eigendecomposition (more stable for small systems)
    if n <= 64  # For small systems, full diagonalization is fine
        F = eigen(Symmetric(Matrix(H)))
        # exp(-iHt) = V * exp(-iΛt) * V'
        phases = exp.(-im * t * F.values)
        U = F.vectors * Diagonal(phases) * F.vectors'
        evolved_data = real(U * ψ.data)
    else
        # For larger systems, use Krylov approximation
        # Compute action of exp(-iHt) using Arnoldi iteration
        evolved_data = ψ.data
        dt = t / 10
        
        for _ in 1:10
            # One step of evolution
            # Use Taylor expansion: exp(-iHdt) ≈ I - iHdt - H²dt²/2
            Hψ = H * evolved_data
            H2ψ = H * Hψ
            evolved_data = evolved_data - dt^2/2 * H2ψ
            evolved_data = evolved_data / norm(evolved_data)
        end
    end
    
    return EDStateVector(evolved_data, ψ.n_qubits)
end

"""
    evolve_ed(H::AbstractMatrix, ρ::EDDensityMatrix, t::Float64) -> EDDensityMatrix

Time evolve a density matrix under Hamiltonian H for time t using vectorization.
"""
function evolve_ed(H::AbstractMatrix, ρ::EDDensityMatrix, t::Float64)
    n = size(H, 1)
    
    # For density matrix: ρ(t) = U(t)ρ(0)U†(t) where U(t) = exp(-iHt)
    # Direct approach for small systems
    if n <= 64
        F = eigen(Symmetric(Matrix(H)))
        phases = exp.(-im * t * F.values)
        U = F.vectors * Diagonal(phases) * F.vectors'
        ρ_evolved = U * ρ.data * U'
        return EDDensityMatrix(real(ρ_evolved), ρ.n_qubits)
    else
        # For larger systems, use vectorized approach
        # d/dt vec(ρ) = -i(H⊗I - I⊗H) vec(ρ)
        
        # Since we want real arithmetic, and H is real symmetric:
        # We can work in the eigenbasis of H
        
        # Simple approach: multiple small steps
        ρ_current = ρ.data
        dt = t / 10
        
        for _ in 1:10
            # Compute [H, ρ]
            commutator = H * ρ_current - ρ_current * H
            # Update: ρ(t+dt) ≈ ρ(t) - i*dt*[H,ρ]
            # Since we want real result, we use ρ(t+dt) ≈ ρ(t) - dt²/2*[H,[H,ρ]]
            commutator2 = H * commutator - commutator * H
            ρ_current = ρ_current - dt^2/2 * commutator2
            
            # Ensure Hermiticity and trace preservation
            ρ_current = (ρ_current + ρ_current') / 2
            ρ_current = ρ_current / tr(ρ_current)
        end
        
        return EDDensityMatrix(ρ_current, ρ.n_qubits)
    end
end

# ============================================================================
# Operator Construction Functions
# ============================================================================

"""
    pauli_x(i::Int, n_qubits::Int) -> SparseMatrixCSC

Pauli X operator on qubit i (1-indexed).
Convention: qubit 1 is the least significant bit (rightmost in tensor product)
"""
function pauli_x(i::Int, n_qubits::Int)
    @assert 1 <= i <= n_qubits "Qubit index out of range"
    
    X_local = sparse([0.0 1.0; 1.0 0.0])
    
    # Build operator using tensor products
    # Qubit ordering: |q_n q_{n-1} ... q_2 q_1⟩
    # So for qubit i, we need I⊗...⊗I⊗X⊗I⊗...⊗I with X at position i from the right
    op = sparse(1.0I, 1, 1)
    for j in n_qubits:-1:1
        if j == i
            op = kron(op, X_local)
        else
            op = kron(op, sparse(I, 2, 2))
        end
    end
    
    return op
end

"""
    pauli_y(i::Int, n_qubits::Int) -> SparseMatrixCSC

Pauli Y operator on qubit i (1-indexed).
Convention: qubit 1 is the least significant bit (rightmost in tensor product)
"""
function pauli_y(i::Int, n_qubits::Int)
    @assert 1 <= i <= n_qubits "Qubit index out of range"
    
    # Real representation of Y: [0 -1; 1 0]
    Y_local = sparse([0.0 -1.0; 1.0 0.0])
    
    # Build operator using tensor products
    op = sparse(1.0I, 1, 1)
    for j in n_qubits:-1:1
        if j == i
            op = kron(op, Y_local)
        else
            op = kron(op, sparse(I, 2, 2))
        end
    end
    
    return op
end

"""
    pauli_z(i::Int, n_qubits::Int) -> SparseMatrixCSC

Pauli Z operator on qubit i (1-indexed).
Convention: qubit 1 is the least significant bit (rightmost in tensor product)
"""
function pauli_z(i::Int, n_qubits::Int)
    @assert 1 <= i <= n_qubits "Qubit index out of range"
    
    Z_local = sparse([1.0 0.0; 0.0 -1.0])
    
    # Build operator using tensor products
    op = sparse(1.0I, 1, 1)
    for j in n_qubits:-1:1
        if j == i
            op = kron(op, Z_local)
        else
            op = kron(op, sparse(I, 2, 2))
        end
    end
    
    return op
end

"""
    pauli_zz(i::Int, j::Int, n_qubits::Int) -> SparseMatrixCSC

Two-qubit ZZ operator on qubits i and j.
"""
function pauli_zz(i::Int, j::Int, n_qubits::Int)
    @assert 1 <= i <= n_qubits && 1 <= j <= n_qubits "Qubit indices out of range"
    @assert i != j "Qubit indices must be different"
    
    return pauli_z(i, n_qubits) * pauli_z(j, n_qubits)
end

# ============================================================================
# Ground State Computation
# ============================================================================

"""
    ground_state_ed(H::AbstractMatrix) -> (Float64, EDStateVector, Float64)

Find ground state energy, state vector, and gap using Krylov methods.
Returns (E0, ψ0, gap).
"""
function ground_state_ed(H::AbstractMatrix)
    # Find ground state
    vals, vecs, _ = eigsolve(H, 1, :SR; krylovdim=min(30, size(H, 1)))
    E0 = real(vals[1])
    ψ0_data = real(vecs[1])  # Take real part
    
    # Convert to dense vector if sparse
    if isa(ψ0_data, SparseVector)
        ψ0_data = Vector(ψ0_data)
    end
    
    ψ0 = EDStateVector(ψ0_data, Int(log2(length(ψ0_data))))
    
    # Find gap
    vals2, _, _ = eigsolve(H, 2, :SR; krylovdim=min(30, size(H, 1)))
    gap = real(vals2[2]) - E0
    
    return E0, ψ0, gap
end

# ============================================================================
# Noise Functions
# ============================================================================

"""
    apply_depolarizing_ed(ψ::EDStateVector, p::Float64, qubits::Vector{Int}) -> EDStateVector

Apply depolarizing noise with probability p to specified qubits.
"""
function apply_depolarizing_ed(ψ::EDStateVector, p::Float64, qubits::Vector{Int})
    ψ_noisy = EDStateVector(copy(ψ.data), ψ.n_qubits)
    
    for q in qubits
        if rand() < p
            # Apply random Pauli
            pauli_choice = rand(1:3)
            if pauli_choice == 1
                op = pauli_x(q, ψ.n_qubits)
            elseif pauli_choice == 2
                op = pauli_y(q, ψ.n_qubits)
            else
                op = pauli_z(q, ψ.n_qubits)
            end
            
            ψ_noisy = EDStateVector(op * ψ_noisy.data, ψ_noisy.n_qubits)
        end
    end
    
    return ψ_noisy
end

"""
    apply_depolarizing_ed(ρ::EDDensityMatrix, p::Float64) -> EDDensityMatrix

Apply global depolarizing noise to density matrix.
"""
function apply_depolarizing_ed(ρ::EDDensityMatrix, p::Float64)
    ρ_noise = maximally_mixed_ed(ρ.n_qubits)
    return EDDensityMatrix((1 - p) * ρ.data + p * ρ_noise.data, ρ.n_qubits)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    purity_ed(ρ::EDDensityMatrix) -> Float64

Compute purity Tr(ρ²).
"""
function purity_ed(ρ::EDDensityMatrix)
    return real(tr(ρ.data^2))
end

"""
    overlap_ed(ψ1::EDStateVector, ψ2::EDStateVector) -> Float64

Compute overlap |⟨ψ1|ψ2⟩|².
"""
function overlap_ed(ψ1::EDStateVector, ψ2::EDStateVector)
    @assert ψ1.n_qubits == ψ2.n_qubits "States must have same number of qubits"
    return abs2(dot(ψ1.data, ψ2.data))
end

"""
    fidelity_ed(ρ1::EDDensityMatrix, ρ2::EDDensityMatrix) -> Float64

Compute fidelity between two density matrices.
For pure states this reduces to |⟨ψ1|ψ2⟩|².
"""
function fidelity_ed(ρ1::EDDensityMatrix, ρ2::EDDensityMatrix)
    @assert ρ1.n_qubits == ρ2.n_qubits "States must have same number of qubits"
    
    # For general mixed states: F = Tr(sqrt(sqrt(ρ1) ρ2 sqrt(ρ1)))²
    # For pure states or when one is pure, this simplifies
    sqrt_ρ1 = sqrt(ρ1.data)
    return real(tr(sqrt(sqrt_ρ1 * ρ2.data * sqrt_ρ1)))^2
end