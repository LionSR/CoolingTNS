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

const ED_DENSITY_TRACE_TOL = 1e-6

function _normalize_density_trace_ed(data::Matrix{ComplexF64})
    # Due to floating-point roundoff (especially after many steps), the trace
    # can drift slightly from 1. We renormalize if the drift is small, and error
    # out only if it is clearly inconsistent with a density matrix.
    trρ = tr(data)
    if abs(trρ - 1.0) > ED_DENSITY_TRACE_TOL
        error("Density matrix must have trace 1 (got $trρ)")
    end
    return data / trρ
end

"""
    _canonical_reduced_density_data_ed(data)

Project an ED reduced-density block to its Hermitian, trace-one representative
after a partial trace, matching the TN MPO post-trace convention.
"""
function _canonical_reduced_density_data_ed(data::Matrix{ComplexF64})
    return _normalize_density_trace_ed(0.5 * (data + data'))
end

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
            data = _normalize_density_trace_ed(data)
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
    results = [(outcome >> i) & 1 for i in 0:(n_measure-1)]
    
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
    
    return EDDensityMatrix(_canonical_reduced_density_data_ed(ρ_reduced), n_keep)
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

Trace out system qubits, keeping bath qubits in the interleaved layout.
"""
function trace_out_system_ed(ρ::EDDensityMatrix, n_sys::Int)
    return partial_trace_ed(ρ, interleaved_bath_sites(n_sys))
end

"""
    trace_out_bath_ed(ρ::EDDensityMatrix, n_sys::Int) -> EDDensityMatrix

Trace out bath qubits, keeping system qubits in the interleaved layout.
"""
function trace_out_bath_ed(ρ::EDDensityMatrix, n_sys::Int)
    return partial_trace_ed(ρ, interleaved_system_sites(n_sys))
end

# ============================================================================
# Time Evolution Functions
# ============================================================================

# Cache for time evolution eigendecompositions of H (keyed by `hash(H)`).
#
# This enables efficient randomized-time protocols, where caching full evolution
# operators `U(t)` for every distinct `t` would otherwise lead to unbounded
# memory growth.
const EVOLUTION_EIG_CACHE = Dict{UInt64, Tuple{Vector{Float64}, Matrix{ComplexF64}}}()

# Cache for selected time evolution operators exp(-iHt) (keyed by (hash(H), t)).
# For randomized-time protocols `t` is typically unique at each step, so we cap
# the cache to keep memory bounded.
const MAX_EVOLUTION_OP_CACHE_SIZE = 64
const EVOLUTION_OP_CACHE = Dict{Tuple{UInt64, Float64}, Matrix{ComplexF64}}()

function _get_eigendecomp(H::AbstractMatrix)
    H_hash = hash(H)
    if haskey(EVOLUTION_EIG_CACHE, H_hash)
        return EVOLUTION_EIG_CACHE[H_hash]
    end

    @assert ishermitian(H) "H must be Hermitian for eigendecomposition"
    F = eigen(Hermitian(Matrix(H)))
    vals = Vector{Float64}(F.values)
    vecs = Matrix{ComplexF64}(F.vectors)
    return EVOLUTION_EIG_CACHE[H_hash] = (vals, vecs)
end

"""
    get_evolution_operator(H::AbstractMatrix, t::Float64) -> Matrix{ComplexF64}

Get an evolution operator U = exp(-iHt). For fixed-time protocols this caches
U by `(hash(H), t)`. For randomized-time protocols, caching is capped to avoid
unbounded memory growth.
"""
function get_evolution_operator(H::AbstractMatrix, t::Float64)
    H_hash = hash(H)
    cache_key = (H_hash, t)

    if haskey(EVOLUTION_OP_CACHE, cache_key)
        return EVOLUTION_OP_CACHE[cache_key]
    end

    vals, vecs = _get_eigendecomp(H)
    phases = exp.(-im * t * vals)
    U = vecs * Diagonal(phases) * vecs'

    if length(EVOLUTION_OP_CACHE) < MAX_EVOLUTION_OP_CACHE_SIZE
        EVOLUTION_OP_CACHE[cache_key] = U
    end

    return U
end

"""
    evolve_ed(H::AbstractMatrix, ψ::EDStateVector, t::Float64) -> EDStateVector

Time evolve a state vector under Hamiltonian H for time t.

Implementation uses the cached eigendecomposition of H to avoid constructing and
storing dense evolution matrices U(t) for each step.
"""
function evolve_ed(H::AbstractMatrix, ψ::EDStateVector, t::Float64)
    vals, vecs = _get_eigendecomp(H)
    coeff = vecs' * ψ.data
    coeff .*= exp.(-im * t * vals)
    evolved_data = vecs * coeff
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

Apply local depolarizing noise to a pure state by sampling independent Pauli
errors on the specified qubits.  Each qubit receives no error with probability
`1-p`, and receives `X`, `Y`, or `Z` with probability `p/3` each.
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
    apply_depolarizing_ed(ρ::EDDensityMatrix, p::Float64, qubits=1:ρ.n_qubits) -> EDDensityMatrix

Apply the deterministic density-matrix average of the local Pauli channel used
by `apply_depolarizing_ed(::EDStateVector, ...)`.  On each specified qubit the
channel is

```
ρ ↦ (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†).
```

The channels are applied independently across qubits.
"""
function apply_depolarizing_ed(ρ::EDDensityMatrix, p::Float64, qubits=1:ρ.n_qubits)
    ρ_noisy_data = copy(ρ.data)
    for q in qubits
        pauli_average = zero(ρ_noisy_data)
        for pauli in PAULI_OPERATORS
            op = pauli(q, ρ.n_qubits)
            pauli_average .+= op * ρ_noisy_data * op'
        end
        ρ_noisy_data = (1 - p) * ρ_noisy_data + (p / 3) * pauli_average
    end
    return EDDensityMatrix(Matrix{ComplexF64}(ρ_noisy_data), ρ.n_qubits)
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

"""
    measure_momentum_distribution_ed(state::Union{EDStateVector, EDDensityMatrix}, ham_params::HamiltonianParameters)

Deprecated compatibility wrapper for the canonical ED raw-Fourier occupation
measurement.

The implementation lives in `measure_raw_fourier_occupation_ed`, which uses the
notes-aligned complex Jordan-Wigner convention and the parity-aware fermionic
momentum grid. Keep this name as a thin wrapper so older call sites do not
silently use a second convention. New code should call
`measure_raw_fourier_occupation_ed` directly.
"""
function measure_momentum_distribution_ed(state::Union{EDStateVector, EDDensityMatrix}, ham_params::HamiltonianParameters)
    return measure_raw_fourier_occupation_ed(state, ham_params)
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
