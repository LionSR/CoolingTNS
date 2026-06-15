"""
    state_manipulation.jl

Dispatched functions for state manipulation operations like appending bath,
sampling, and tracing out subsystems.
"""

using ITensors
using ITensorMPS
using LinearAlgebra


# ============================================================================
# Append Bath States
# ============================================================================

"""
    append_bath(backend::CoolingBackend, system_state, bath_state)

Append fresh bath qubits to system state using dispatch.
"""
# TN Backend
append_bath(::TNBackend, ψ_s::MPS, sites::Vector{<:Index}, coupling::String="XX") =
    appendzeros_MPS(ψ_s, sites, coupling)
append_bath(::TNBackend, ρ_s::MPO, sites::Vector{<:Index}, coupling::String="XX") =
    appendzeros_MPO(ρ_s, sites, coupling)

# ED Backend
append_bath(::EDBackend, ψ_s::EDStateVector, N_bath::Int, coupling::String="XX") =
    prepare_combined_state_ed(ψ_s, N_bath, coupling)
append_bath(::EDBackend, ρ_s::EDDensityMatrix, N_bath::Int, coupling::String="XX") =
    prepare_combined_state_ed(ρ_s, N_bath, coupling)

function append_bath(::EDBackend, ρ_s::Matrix, N_bath::Int, coupling::String="XX")
    N_sys = Int(log2(size(ρ_s, 1)))
    return append_bath(EDBackend(), EDDensityMatrix(Matrix{ComplexF64}(ρ_s), N_sys), N_bath, coupling).data
end

# ============================================================================
# Sample/Trace Out Bath
# ============================================================================

"""
    process_bath(backend, sim_method, combined_state, N_sys, N_bath)

Process bath: Monte Carlo samples, Density Matrix traces out.
Returns (system_state, bath_info).
"""
# TN Backend
function process_bath(::TNBackend, ::MonteCarloWavefunction, ψ_sb::MPS, _N_sys::Int, _N_bath::Int)
    v_b, ψ_s = sample_bath(ψ_sb)
    return ψ_s, v_b
end

function process_bath(::TNBackend, ::DensityMatrix, ρ_sb::MPO, N_sys::Int, _N_bath::Int)
    sites = [siteind(ρ_sb, i) for i in 1:length(ρ_sb)]
    sites_sys = sites[1:2:2*N_sys-1]
    ρ_s = partial_trace_bath(ρ_sb, sites, sites_sys)
    return ρ_s / tr(ρ_s), nothing
end

# ED Backend
function process_bath(::EDBackend, ::MonteCarloWavefunction, ψ_evolved::EDStateVector, _N_sys::Int, N_bath::Int)
    return measure_ed!(ψ_evolved, collect(2:2:2*N_bath))
end

# ED Density Matrix: return full state, trace during measurements
process_bath(::EDBackend, ::DensityMatrix, ρ::EDDensityMatrix, _, _) = (ρ, nothing)
process_bath(::EDBackend, ::DensityMatrix, ρ::Matrix, _, _) = (ρ, nothing)

# ============================================================================
# Trace Operations
# ============================================================================

"""Trace out bath degrees of freedom to get system reduced density matrix."""
trace_out_bath(::EDBackend, ρ::EDDensityMatrix, N_sys::Int, _) = trace_out_bath_ed(ρ, N_sys)

function trace_out_bath(::TNBackend, ρ::MPO, N_sys::Int, _)
    sites = [siteind(ρ, i) for i in 1:length(ρ)]
    sites_sys = sites[1:2:2*N_sys-1]
    return partial_trace_bath(ρ, sites, sites_sys)
end

# Matrix support for backward compatibility
function trace_out_bath(::EDBackend, ρ::Matrix, N_sys::Int, _N_bath::Int)
    N_total = Int(log2(size(ρ, 1)))
    return trace_out_bath_ed(EDDensityMatrix(real(ρ), N_total), N_sys).data
end

# ============================================================================
# Get Sites Information
# ============================================================================

"""Get site indices from problem."""
get_sites(problem::CoolingProblem, ::TNBackend) = problem.extra.sites
get_sites(::CoolingProblem, ::EDBackend) = nothing

# ============================================================================
# Utility Functions
# ============================================================================

"""Create projector from EDStateVector"""
projector(ψ::EDStateVector) = state_to_density_ed(ψ).data

"""Create MPO projector from MPS for TN backend"""
projector_mpo(ψ::MPS) = outer(ψ', ψ)
