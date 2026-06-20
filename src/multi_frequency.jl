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
using Random


"""
    uniform_delta_grid(delta_min, delta_max, R) -> Vector{Float64}

Uniform grid of `R` detuning values from `delta_min` to `delta_max` (inclusive).
"""
function uniform_delta_grid(delta_min::Real, delta_max::Real, R::Integer)
    R < 1 && throw(ArgumentError("R must be ≥ 1, got $R"))
    return collect(range(Float64(delta_min), Float64(delta_max); length=Int(R)))
end

function _multi_frequency_delta_index(
    cycle::Integer,
    R::Integer,
    schedule::Symbol;
    rng::AbstractRNG=Random.default_rng(),
)
    cycle >= 1 || throw(ArgumentError("cycle must be positive, got $cycle"))
    R >= 1 || throw(ArgumentError("R must be positive, got $R"))
    schedule = validate_multi_frequency_schedule(schedule)
    schedule == :round_robin && return mod1(Int(cycle), Int(R))
    schedule == :descending && return Int(R) - mod(Int(cycle) - 1, Int(R))
    schedule == :random && return rand(rng, 1:Int(R))
    error("unreachable multi-frequency schedule $schedule")
end

"""
    multi_frequency_cycle_choice(mf_params, cycle; rng=Random.default_rng())

Return the detuning index, detuning value, and evolution time used in one
physical multi-frequency cooling cycle.  The cycle index starts at one; it is
not the measurement-array index, whose first entry is the initial state.
"""
function multi_frequency_cycle_choice(
    mf_params::MultiFrequencyCouplingParameters,
    cycle::Integer;
    rng::AbstractRNG=Random.default_rng(),
)
    r = _multi_frequency_delta_index(
        cycle,
        length(mf_params.delta_values),
        mf_params.schedule;
        rng=rng,
    )
    te_step = mf_params.randomize_times ? (rand(rng) * 2 * mf_params.te) : mf_params.te
    return (delta_index=r, delta=mf_params.delta_values[r], te=te_step)
end

"""
    multi_frequency_cycle_sequence(mf_params; rng=Random.default_rng())

Return the full detuning/time sequence for a multi-frequency protocol.  The
returned `delta_list` and `te_list` have length `steps + 1`, with `NaN` in the
first entry to match the result arrays produced by `run_cooling_multi_freq`.
The `delta_indices` vector has one entry per physical cooling cycle.

For `schedule=:random` or `randomize_times=true`, this function samples the
sequence from `rng`.  It matches a run of `run_cooling_multi_freq` when that run
is supplied an RNG in the same state.
"""
function multi_frequency_cycle_sequence(
    mf_params::MultiFrequencyCouplingParameters;
    rng::AbstractRNG=Random.default_rng(),
)
    steps = mf_params.steps
    steps >= 0 || throw(ArgumentError("steps must be nonnegative, got $steps"))

    delta_indices = Vector{Int}(undef, steps)
    delta_list = fill(NaN, steps + 1)
    te_list = fill(NaN, steps + 1)

    for cycle in 1:steps
        choice = multi_frequency_cycle_choice(mf_params, cycle; rng=rng)
        delta_indices[cycle] = choice.delta_index
        delta_list[cycle + 1] = choice.delta
        te_list[cycle + 1] = choice.te
    end

    return (delta_indices=delta_indices, delta_list=delta_list, te_list=te_list)
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

function _ed_krylov_excitation_ceiling(hilbert_dim::Int, krylovdim::Int)
    krylovdim < 2 && throw(ArgumentError(
        "krylovdim must allow the ground state and at least one excitation; got $krylovdim."
    ))

    if krylovdim >= hilbert_dim
        return hilbert_dim - 1
    end

    # KrylovKit needs one extra Krylov vector beyond the requested eigenvalues
    # unless the finite ED problem is small enough to span the whole space.
    return max(krylovdim - 2, 0)
end

function compute_excitation_gaps(
    ham_params::HamiltonianParameters,
    backend::EDBackend;
    num_excitations::Int=10,
    krylovdim::Int=50,
)
    num_excitations < 1 && throw(ArgumentError("num_excitations must be ≥ 1"))

    H_sys = construct_system_hamiltonian(ham_params, backend, ham_params.N)
    hilbert_dim = size(H_sys, 1)
    hilbert_excitations = hilbert_dim - 1
    if num_excitations > hilbert_excitations
        throw(ArgumentError(
            "Requested $num_excitations excitation gaps, but the ED Hilbert space " *
            "contains only $hilbert_excitations excited levels."
        ))
    end

    krylov_excitations = _ed_krylov_excitation_ceiling(hilbert_dim, krylovdim)
    if num_excitations > krylov_excitations
        throw(ArgumentError(
            "Requested $num_excitations excitation gaps, but krylovdim=$krylovdim " *
            "can compute at most $krylov_excitations excited levels for this ED system."
        ))
    end

    # Request the (num_excitations+1) lowest eigenvalues.
    effective_krylovdim = min(krylovdim, hilbert_dim)
    vals, _, _ = eigsolve(H_sys, num_excitations + 1, :SR; krylovdim=effective_krylovdim)
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
    ψ_init = random_mps(sites, linkdims=init_linkdims)
    sweeps0 = Sweeps(nsweeps_ground)
    setmaxdim!(sweeps0, maxdim_ground...)
    setcutoff!(sweeps0, cutoff)
    E0, ϕ0 = dmrg(H_sys, ψ_init, sweeps0; outputlevel=0)

    # Excited states (enforce orthogonality to all lower states)
    sweeps1 = Sweeps(nsweeps_excited)
    setmaxdim!(sweeps1, maxdim_excited)
    setcutoff!(sweeps1, cutoff)

    prev_states = MPS[ϕ0]
    ψ_guess = random_mps(sites, linkdims=init_linkdims)

    gaps = Vector{Float64}(undef, num_excitations)
    for n in 1:num_excitations
        En, ϕn = dmrg(H_sys, prev_states, ψ_guess, sweeps1; outputlevel=0, weight=weight)
        gaps[n] = En - E0
        push!(prev_states, ϕn)
        ψ_guess = ϕn
    end

    return gaps
end

max_available_excitations(::HamiltonianParameters, ::CoolingBackend; kwargs...) = nothing
function max_available_excitations(
    ham_params::HamiltonianParameters,
    ::EDBackend;
    krylovdim::Int=50,
    kwargs...,
)
    return _ed_krylov_excitation_ceiling(2^ham_params.N, krylovdim)
end


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
    num_excitations < 1 && throw(ArgumentError("num_excitations must be ≥ 1, got $num_excitations"))
    num_excitations < R && throw(ArgumentError(
        "Need at least R=$R requested excitation gaps, got num_excitations=$num_excitations."
    ))

    available_excitations = max_available_excitations(ham_params, backend; kwargs...)
    requested_excitations = isnothing(available_excitations) ?
                            num_excitations :
                            min(num_excitations, available_excitations)
    if !isnothing(available_excitations) && requested_excitations < R
        throw(ArgumentError(
            "Need at least R=$R excitation gaps, but only $available_excitations are available."
        ))
    end

    gaps = compute_excitation_gaps(ham_params, backend; num_excitations=requested_excitations, kwargs...)
    gaps = sort(gaps)
    @assert length(gaps) >= R "compute_excitation_gaps returned fewer gaps than requested"

    if R == 1
        return [gaps[1]]
    end

    M = length(gaps)
    idxs = [round(Int, 1 + (i - 1) * (M - 1) / (R - 1)) for i in 1:R]
    return gaps[idxs]
end
