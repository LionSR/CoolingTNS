"""
    bath_measurements.jl

Dispatched functions for computing bath-related measurements.
"""

using ITensors
using ITensorMPS
using LinearAlgebra

# ============================================================================
# Bath Magnetization Measurement Dispatch
# ============================================================================

"""
    compute_bath_magnetization(backend::CoolingBackend, state::QuantumState, evolved_state, N_bath::Int)

Compute bath magnetization using dispatch on backend and simulation method.
"""
function compute_bath_magnetization(backend::CoolingBackend, state::QuantumState, evolved_state, N_bath::Int)
    error("compute_bath_magnetization not implemented for backend=$(typeof(backend)), sim_method=$(typeof(state.sim_method))")
end

function _pauli_z_from_tn_sample(sample::Int)
    sample == 1 && return 1.0
    sample == 2 && return -1.0
    throw(ArgumentError("TN bath samples are ITensor site indices 1 or 2, got $sample"))
end

function _pauli_z_from_ed_bit(bit::Int)
    bit == 0 && return 1.0
    bit == 1 && return -1.0
    throw(ArgumentError("ED bath measurement bits must be 0 or 1, got $bit"))
end

function _require_bath_sample_length(sample::AbstractVector, N_bath::Int, label::AbstractString)
    N_bath > 0 || throw(ArgumentError("$label requires positive N_bath, got $N_bath"))
    length(sample) == N_bath || throw(ArgumentError(
        "$label length $(length(sample)) does not match N_bath=$N_bath",
    ))
    return nothing
end

function _bath_sample_magnetization(sample::Vector{Int}, N_bath::Int,
                                    label::AbstractString, sample_to_z)
    _require_bath_sample_length(sample, N_bath, label)
    return sum(sample_to_z.(sample)) / N_bath
end

# --- Tensor Network + Monte Carlo ---
# For MPS Monte Carlo, bath magnetization comes from the sampled bath configuration
function compute_bath_magnetization(::TNBackend, ::QuantumState{TNBackend,MonteCarloWavefunction,E}, 
                                  bath_sample::Vector{Int}, N_bath::Int) where E
    # `sample_bath` returns ITensor site indices: 1 = Up (Z=+1), 2 = Dn (Z=-1).
    return _bath_sample_magnetization(
        bath_sample, N_bath, "TN bath sample", _pauli_z_from_tn_sample)
end

# --- ED + Monte Carlo ---
# For ED Monte Carlo, bath magnetization comes from collapsed measurement
function compute_bath_magnetization(::EDBackend, ::QuantumState{EDBackend,MonteCarloWavefunction,E},
                                  bath_result::Vector{Int}, N_bath::Int) where E
    # `measure_ed!` returns computational bits: 0 = Up (Z=+1), 1 = Dn (Z=-1).
    return _bath_sample_magnetization(
        bath_result, N_bath, "ED bath measurement", _pauli_z_from_ed_bit)
end

# --- ED + Density Matrix ---
# For density matrix, compute expectation value from bath reduced density matrix
function compute_bath_magnetization(::EDBackend, ::QuantumState{EDBackend,DensityMatrix,E},
                                  ρ_bath::Matrix, N_bath::Int) where E
    dim = 2^N_bath

    # Each diagonal entry contributes its bath magnetization weighted by its
    # population; n_ones counts the down-spins in the basis-state bitstring.
    # `1 - 2 * n_ones / N_bath` is the popcount form of averaging
    # `_pauli_z_from_ed_bit` over the bits of `i - 1` (bit 0 -> +1, bit 1 -> -1).
    return sum(1:dim; init=0.0) do i
        n_ones = count_ones(i - 1)
        real(ρ_bath[i, i]) * (1 - 2 * n_ones / N_bath)
    end
end

# --- TN + Density Matrix (MPO) ---
# For MPO, we need to compute expectation values differently
function compute_bath_magnetization(::TNBackend, ::QuantumState{TNBackend,DensityMatrix,E},
                                  ρ_bath::MPO, sites_bath::Vector{<:Index}) where E
    # One `Σ_i Z_i` MPO and one `inner`, rather than an MPO and an `inner` per
    # bath site. `MPO(::OpSum)` of single-site terms is exact at bond dimension
    # 2, so this loses no accuracy — but it does move the N-term sum inside the
    # tensor contraction, which reassociates it at the ~1 ulp level. Values
    # written to `RESULT_BATH_MAGNETIZATION` differ from the per-site form in
    # their last digits.
    N_bath = length(sites_bath)

    terms = OpSum()
    for i in eachindex(sites_bath)
        terms += 1.0, "Z", i
    end

    return real(inner(ρ_bath, MPO(terms, sites_bath))) / N_bath
end
