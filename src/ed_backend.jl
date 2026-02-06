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
        data::Vector{ComplexF64}  # Complex amplitudes for proper quantum mechanics
        n_qubits::Int
        
        function EDStateVector(data::Vector{ComplexF64}, n_qubits::Int)
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
        data::Matrix{ComplexF64}  # Complex density matrix
        n_qubits::Int
        
        function EDDensityMatrix(data::Matrix{ComplexF64}, n_qubits::Int)
            @assert size(data, 1) == size(data, 2) == 2^n_qubits "Matrix dimension must be 2^n_qubits"
            @assert ishermitian(data) "Density matrix must be Hermitian"
            @assert abs(tr(data) - 1.0) < 1e-10 "Density matrix must have trace 1 (got $(tr(data)))"
            new(data, n_qubits)
        end
    end
end

# Base function extensions for our types
Base.size(ρ::EDDensityMatrix) = size(ρ.data)
Base.size(ρ::EDDensityMatrix, dim::Int) = size(ρ.data, dim)
Base.size(ψ::EDStateVector) = size(ψ.data)
Base.size(ψ::EDStateVector, dim::Int) = size(ψ.data, dim)

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
    data = zeros(ComplexF64, 2^n_qubits)
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
    # Generate random complex state
    data = randn(ComplexF64, 2^n_qubits)
    return EDStateVector(data, n_qubits)
end

"""
    maximally_mixed_ed(n_qubits::Int) -> EDDensityMatrix

Create the maximally mixed state I/2^n.
"""
function maximally_mixed_ed(n_qubits::Int)
    dim = 2^n_qubits
    return EDDensityMatrix(Matrix{ComplexF64}(I, dim, dim) / dim, n_qubits)
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
    # |ψ⟩⟨ψ| is automatically Hermitian
    ρ = ψ.data * ψ.data'
    return EDDensityMatrix(ρ, ψ.n_qubits)
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
        
        probs[outcome + 1] += abs2(ψ.data[state_idx + 1])
    end
    
    # Sample outcome based on probabilities
    outcome = sample_outcome(probs) - 1  # Convert to 0-based
    
    # Extract measurement results as bit array
    results = Int[]
    for i in 0:(n_measure-1)
        push!(results, (outcome >> i) & 1)
    end
    
    # Collapse state and trace out measured qubits
    collapsed_data = zeros(ComplexF64, 2^n_remaining)
    
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
    # For Hermitian operators, expectation values are real
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
    
    ρ_reduced = zeros(ComplexF64, dim_keep, dim_keep)
    
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

# Cache for time evolution operators exp(-iHt)
const EVOLUTION_OP_CACHE = Dict{Tuple{UInt64, Float64}, Matrix{ComplexF64}}()

"""
    get_evolution_operator(H::AbstractMatrix, t::Float64) -> Matrix{ComplexF64}
    
Get the cached evolution operator U = exp(-iHt) for Hamiltonian H and time t.
"""
function get_evolution_operator(H::AbstractMatrix, t::Float64)
    # Use hash of H and t as cache key
    H_hash = hash(H)
    cache_key = (H_hash, t)
    
    if haskey(EVOLUTION_OP_CACHE, cache_key)
        return EVOLUTION_OP_CACHE[cache_key]
    end
    
    # Compute exp(-iHt) via eigendecomposition
    F = eigen(Symmetric(Matrix(H)))
    phases = exp.(-im * t * F.values)
    U = F.vectors * Diagonal(phases) * F.vectors'
    
    EVOLUTION_OP_CACHE[cache_key] = U
    return U
end

"""
    evolve_ed(H::AbstractMatrix, ψ::EDStateVector, t::Float64) -> EDStateVector

Time evolve a state vector under Hamiltonian H for time t.
"""
function evolve_ed(H::AbstractMatrix, ψ::EDStateVector, t::Float64)
    # Use cached evolution operator
    U = get_evolution_operator(H, t)
    evolved_data = U * ψ.data
    return EDStateVector(evolved_data, ψ.n_qubits)
end

"""
    evolve_ed(H::AbstractMatrix, ρ::EDDensityMatrix, t::Float64) -> EDDensityMatrix

Time evolve a density matrix under Hamiltonian H for time t using vectorization.
"""
function evolve_ed(H::AbstractMatrix, ρ::EDDensityMatrix, t::Float64)
    # Use cached evolution operator
    U = get_evolution_operator(H, t)
    ρ_evolved = U * ρ.data * U'
    # Enforce Hermiticity to avoid numerical errors
    ρ_sym = (ρ_evolved + ρ_evolved') / 2
    return EDDensityMatrix(ρ_sym, ρ.n_qubits)
end

# ============================================================================
# Operator Construction Functions
# ============================================================================

# Local Pauli matrices (constant)
const PAULI_X_LOCAL = sparse([0.0 1.0; 1.0 0.0])
const PAULI_Y_LOCAL = sparse([0.0 -1.0; 1.0 0.0])  # Real representation
const PAULI_Z_LOCAL = sparse([1.0 0.0; 0.0 -1.0])
const IDENTITY_2X2 = sparse(I, 2, 2)

"""
    single_site_operator(local_op::SparseMatrixCSC, i::Int, n_qubits::Int) -> SparseMatrixCSC

Build a single-site operator acting on qubit i in an n_qubit system.
Qubit 1 is the least significant bit (rightmost in tensor product).
"""
function single_site_operator(local_op::SparseMatrixCSC, i::Int, n_qubits::Int)
    @assert 1 <= i <= n_qubits "Qubit index out of range"

    op = sparse(1.0I, 1, 1)
    for j in n_qubits:-1:1
        op = kron(op, j == i ? local_op : IDENTITY_2X2)
    end
    return op
end

"""Pauli X operator on qubit i (1-indexed)."""
pauli_x(i::Int, n_qubits::Int) = single_site_operator(PAULI_X_LOCAL, i, n_qubits)

"""Pauli Y operator on qubit i (1-indexed). Uses real representation [0 -1; 1 0]."""
pauli_y(i::Int, n_qubits::Int) = single_site_operator(PAULI_Y_LOCAL, i, n_qubits)

"""Pauli Z operator on qubit i (1-indexed)."""
pauli_z(i::Int, n_qubits::Int) = single_site_operator(PAULI_Z_LOCAL, i, n_qubits)

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
    # Ground state eigenvector can be complex even for real Hamiltonian
    ψ0_data = ComplexF64.(vecs[1])
    
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
# Pauli operator selectors for random sampling
const PAULI_OPERATORS = (pauli_x, pauli_y, pauli_z)

function apply_depolarizing_ed(ψ::EDStateVector, p::Float64, qubits::Vector{Int})
    ψ_noisy_data = copy(ψ.data)

    for q in qubits
        if rand() < p
            op = PAULI_OPERATORS[rand(1:3)](q, ψ.n_qubits)
            ψ_noisy_data = op * ψ_noisy_data
        end
    end

    return EDStateVector(ψ_noisy_data, ψ.n_qubits)
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

# ============================================================================
# K-Space Measurement Functions (for PBC/APBC)
# ============================================================================

# Cache for Jordan-Wigner operators to avoid recomputation
const JW_CACHE = Dict{Tuple{Int, Int}, Tuple{SparseMatrixCSC{Float64, Int}, SparseMatrixCSC{Float64, Int}}}()

"""
    jordan_wigner_transform(i::Int, n_qubits::Int) -> Tuple{SparseMatrixCSC, SparseMatrixCSC}

Returns the Jordan-Wigner transformed creation and annihilation operators for site i.
a_i = (-1)^(sum_{j<i} n_j) * (X_i - iY_i)/2
a†_i = (-1)^(sum_{j<i} n_j) * (X_i + iY_i)/2
"""
function jordan_wigner_transform(i::Int, n_qubits::Int)
    # Check cache first
    cache_key = (i, n_qubits)
    if haskey(JW_CACHE, cache_key)
        return JW_CACHE[cache_key]
    end
    # Build the string operator (-1)^(sum_{j<i} n_j) = Π_{j<i} Z_j
    string_op = sparse(I, 2^n_qubits, 2^n_qubits)
    for j in 1:i-1
        string_op = string_op * pauli_z(j, n_qubits)
    end
    
    # Build fermionic operators
    X_i = pauli_x(i, n_qubits)
    Y_i = pauli_y(i, n_qubits)
    
    # Standard convention: |↑⟩ = vacuum, |↓⟩ = occupied
    # a† creates fermion: |↑⟩ → |↓⟩, so a† = σ^-
    # a annihilates fermion: |↓⟩ → |↑⟩, so a = σ^+
    # With real Y matrix: Y_real = -i*Y_complex
    # σ^+ = (X + iY)/2 = (X - Y_real)/2
    # σ^- = (X - iY)/2 = (X + Y_real)/2
    a = string_op * (X_i - Y_i) / 2      # σ^+
    a_dag = string_op * (X_i + Y_i) / 2  # σ^-
    
    result = (a, a_dag)
    JW_CACHE[cache_key] = result
    return result
end

"""
    momentum_state_overlap_ed(ψ::EDStateVector, k::Float64, bc::Symbol) -> Float64

Compute |⟨k|ψ⟩|² where |k⟩ is a momentum eigenstate.
k is in units of 2π/N (so k ∈ [-N/2, N/2] for PBC, k ∈ [-(N-1)/2, (N-1)/2] for APBC).
bc is :periodic or :antiperiodic.
"""
function momentum_state_overlap_ed(ψ::EDStateVector, k::Float64, bc::Symbol)
    N = ψ.n_qubits
    
    # Create the momentum eigenstate |k⟩ in real space
    # |k⟩ = (1/√N) Σ_n exp(2πikn/N) a†_n |0⟩
    
    # Start with vacuum state
    vacuum = zeros(ComplexF64, 2^N)
    vacuum[1] = 1.0  # |00...0⟩
    
    # Build momentum state by applying Fourier-transformed creation operator
    k_state = zeros(ComplexF64, 2^N)
    
    for n in 1:N
        phase_real = cos(2π * k * n / N) / sqrt(N)
        phase_imag = sin(2π * k * n / N) / sqrt(N)
        
        # Get Jordan-Wigner transformed operators
        _, a_dag_n = jordan_wigner_transform(n, N)
        
        # Apply a†_n with phase factor
        phase = (phase_real + im * phase_imag)
        k_state += phase * a_dag_n * vacuum
    end
    
    # Compute overlap for complex states
    overlap = dot(k_state, ψ.data)
    
    return abs2(overlap)
end

# Cache for correlation operators a†_m a_n
const CORRELATION_OP_CACHE = Dict{Tuple{Int, Int, Int}, SparseMatrixCSC}()

"""
    get_correlation_operator(m::Int, n::Int, N::Int) -> SparseMatrixCSC
    
Get the cached operator a†_m a_n for sites m and n in a system of N qubits.
"""
function get_correlation_operator(m::Int, n::Int, N::Int)
    cache_key = (m, n, N)
    if haskey(CORRELATION_OP_CACHE, cache_key)
        return CORRELATION_OP_CACHE[cache_key]
    end
    
    # Get Jordan-Wigner operators
    a_n, _ = jordan_wigner_transform(n, N)
    _, a_m_dag = jordan_wigner_transform(m, N)
    
    # a†_m a_n operator
    op = a_m_dag * a_n
    
    CORRELATION_OP_CACHE[cache_key] = op
    return op
end

"""
    get_allowed_k_values(N::Int, bc::Symbol) -> Vector{Int}

Get allowed k indices based on boundary conditions.
"""
function get_allowed_k_values(N::Int, bc::Symbol)
    bc == :periodic && return collect(-div(N,2)+1:div(N,2))
    bc == :antiperiodic && return collect(-div(N-1,2):div(N-1,2))
    error("Momentum distribution only defined for periodic/antiperiodic BC")
end

"""
    compute_real_space_correlations(state::EDStateVector, N::Int) -> Matrix{Float64}

Compute real-space fermionic correlations ⟨a†_m a_n⟩ for a pure state.
"""
function compute_real_space_correlations(state::EDStateVector, N::Int)
    correlations = zeros(Float64, N, N)
    for m in 1:N, n in 1:N
        a_n, _ = jordan_wigner_transform(n, N)
        _, a_m_dag = jordan_wigner_transform(m, N)
        correlations[m, n] = real(dot(state.data, a_m_dag * a_n * state.data))
    end
    return correlations
end

"""
    compute_real_space_correlations(state::EDDensityMatrix, N::Int) -> Matrix{ComplexF64}

Compute real-space fermionic correlations Tr(ρ a†_m a_n) for a density matrix.
"""
function compute_real_space_correlations(state::EDDensityMatrix, N::Int)
    correlations = zeros(ComplexF64, N, N)
    for m in 1:N, n in 1:N
        a_n, _ = jordan_wigner_transform(n, N)
        _, a_m_dag = jordan_wigner_transform(m, N)
        correlations[m, n] = tr(state.data * a_m_dag * a_n)
    end
    return correlations
end

"""
    fourier_transform_correlations(correlations::AbstractMatrix, k_values::Vector{Int}, N::Int) -> Vector{Float64}

Fourier transform real-space correlations to momentum distribution.
"""
function fourier_transform_correlations(correlations::AbstractMatrix, k_values::Vector{Int}, N::Int)
    n_k = zeros(Float64, length(k_values))
    for (ki, k) in enumerate(k_values)
        nk = 0.0 + 0.0im
        for m in 1:N, n in 1:N
            phase = exp(2π * im * k * (m - n) / N)
            nk += phase * correlations[m, n] / N
        end
        n_k[ki] = real(nk)
    end
    return n_k
end

"""
    measure_momentum_distribution_ed(state::Union{EDStateVector, EDDensityMatrix}, ham_params::HamiltonianParameters)

Measure momentum distribution n_k = ⟨a†_k a_k⟩ for all allowed k values.
Returns (k_values, n_k) where k_values are in units of 2π/N.
"""
function measure_momentum_distribution_ed(state::Union{EDStateVector, EDDensityMatrix}, ham_params::HamiltonianParameters)
    N = ham_params.N
    k_indices = get_allowed_k_values(N, ham_params.bc)
    correlations = compute_real_space_correlations(state, N)
    n_k = fourier_transform_correlations(correlations, k_indices, N)
    k_momentum = [2π * k / N for k in k_indices]
    return k_momentum, n_k
end

"""
    projector_ed(bit::Int, qubit::Int, n_qubits::Int) -> SparseMatrixCSC{Float64, Int}

Create projector onto |bit⟩ (0 or 1) for specified qubit.
|0⟩⟨0| = (I + Z)/2, |1⟩⟨1| = (I - Z)/2
"""
function projector_ed(bit::Int, qubit::Int, n_qubits::Int)
    @assert bit in (0, 1) "bit must be 0 or 1"
    @assert 1 <= qubit <= n_qubits "qubit must be in range 1:n_qubits"
    sign = 1 - 2 * bit  # +1 for bit=0, -1 for bit=1
    return (I + sign * pauli_z(qubit, n_qubits)) / 2
end