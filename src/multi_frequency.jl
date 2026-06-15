"""
    multi_frequency.jl

Helpers for multi-frequency cooling protocols: choosing bath detunings Δ and
computing low-lying excitation gaps of interacting spin models.

See `docs/multi_frequency_cooling_plan.md` for the design motivation.
"""

using ITensors
using ITensorMPS
using KrylovKit
using LinearAlgebra


"""
    uniform_delta_grid(delta_min, delta_max, R) -> Vector{Float64}

Uniform grid of `R` detuning values from `delta_min` to `delta_max` (inclusive).
"""
function uniform_delta_grid(delta_min::Real, delta_max::Real, R::Integer)
    R < 1 && throw(ArgumentError("R must be ≥ 1, got $R"))
    return collect(range(Float64(delta_min), Float64(delta_max); length=Int(R)))
end


"""
    compute_excitation_gaps(ham_params, backend; num_excitations=10, kwargs...) -> Vector{Float64}

Compute the lowest `num_excitations` excitation gaps of the **system** Hamiltonian,
i.e. `E_n - E_0` for `n = 1, ..., num_excitations`.

- For [`EDBackend`](@ref), uses KrylovKit's `eigsolve` on the sparse ED Hamiltonian.
- For [`TNBackend`](@ref), uses DMRG with orthogonality constraints to previously
  found states.

This is intended as a heuristic to choose multi-frequency bath detunings in
interacting models, not as a high-precision spectroscopy routine.
"""
function compute_excitation_gaps end

function compute_excitation_gaps(
    ham_params::HamiltonianParameters,
    backend::EDBackend;
    num_excitations::Int=10,
    krylovdim::Int=50,
)
    num_excitations < 1 && throw(ArgumentError("num_excitations must be ≥ 1"))

    H_sys = construct_system_hamiltonian(ham_params, backend, ham_params.N)
    max_excitations = size(H_sys, 1) - 1
    if num_excitations > max_excitations
        throw(ArgumentError(
            "Requested $num_excitations excitation gaps, but the ED Hilbert space " *
            "contains only $max_excitations excited levels."
        ))
    end

    # Request the (num_excitations+1) lowest eigenvalues.
    vals, _, _ = eigsolve(H_sys, num_excitations + 1, :SR; krylovdim=min(krylovdim, size(H_sys, 1)))
    energies = sort(real.(vals))
    E0 = energies[1]

    gaps = energies[2:end] .- E0
    return gaps[1:num_excitations]
end

function compute_excitation_gaps(
    ham_params::HamiltonianParameters,
    backend::TNBackend;
    num_excitations::Int=10,
    nsweeps_ground::Int=5,
    nsweeps_excited::Int=3,
    maxdim_ground::Vector{Int}=[10, 20, 100, 100, 200],
    maxdim_excited::Int=200,
    cutoff::Float64=1e-10,
    weight::Float64=20.0,
    init_linkdims::Int=10,
)
    num_excitations < 1 && throw(ArgumentError("num_excitations must be ≥ 1"))

    N = ham_params.N
    sites = siteinds("S=1/2", N)
    H_sys = construct_system_hamiltonian(ham_params, backend, sites)

    # Ground state
    ψ_init = randomMPS(sites, linkdims=init_linkdims)
    sweeps0 = Sweeps(nsweeps_ground)
    setmaxdim!(sweeps0, maxdim_ground...)
    setcutoff!(sweeps0, cutoff)
    E0, ϕ0 = dmrg(H_sys, ψ_init, sweeps0; outputlevel=0)

    # Excited states (enforce orthogonality to all lower states)
    sweeps1 = Sweeps(nsweeps_excited)
    setmaxdim!(sweeps1, maxdim_excited)
    setcutoff!(sweeps1, cutoff)

    prev_states = MPS[ϕ0]
    ψ_guess = randomMPS(sites, linkdims=init_linkdims)

    gaps = Vector{Float64}(undef, num_excitations)
    for n in 1:num_excitations
        En, ϕn = dmrg(H_sys, prev_states, ψ_guess, sweeps1; outputlevel=0, weight=weight)
        gaps[n] = En - E0
        push!(prev_states, ϕn)
        ψ_guess = ϕn
    end

    return gaps
end

max_available_excitations(::HamiltonianParameters, ::CoolingBackend) = typemax(Int)
max_available_excitations(ham_params::HamiltonianParameters, ::EDBackend) = 2^ham_params.N - 1


"""
    spectral_delta_values(ham_params, backend; R=5, num_excitations=2R) -> Vector{Float64}

Pick `R` detuning values from the low-lying excitation spectrum of the system.

This computes `num_excitations` gaps and then selects `R` roughly evenly spaced
values in the sorted gap list.
"""
function spectral_delta_values(
    ham_params::HamiltonianParameters,
    backend::CoolingBackend;
    R::Int=5,
    num_excitations::Int=2R,
    kwargs...,
)
    R < 1 && throw(ArgumentError("R must be ≥ 1, got $R"))

    available_excitations = max_available_excitations(ham_params, backend)
    requested_excitations = min(num_excitations, available_excitations)
    if requested_excitations < R
        throw(ArgumentError(
            "Need at least R=$R excitation gaps, but only $available_excitations are available."
        ))
    end

    gaps = compute_excitation_gaps(ham_params, backend; num_excitations=requested_excitations, kwargs...)
    gaps = sort(gaps)
    length(gaps) >= R || throw(ArgumentError("Need at least R=$R excitation gaps, got $(length(gaps))"))

    if R == 1
        return [gaps[1]]
    end

    M = length(gaps)
    idxs = [round(Int, 1 + (i - 1) * (M - 1) / (R - 1)) for i in 1:R]
    return gaps[idxs]
end
