"""
    evolution.jl

Time evolution methods using multiple dispatch on SimulationMethod, EvolutionMethod, and backend.
"""

using ITensors
using ITensorMPS

# ============================================================================
# Time Evolution Interface
# ============================================================================

"""
    evolve_state(ham_params, sim_params, backend, H_total, state, t, sites; kwargs...)

Generic evolution interface using triple dispatch on model, simulation method, and backend.
"""
function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters,
                     backend::CoolingBackend, H_total, ψ, t, sites::Union{Nothing, Vector{<:Index}}; kwargs...)
    error("evolve_state not implemented for model=$(typeof(ham_params.model)), " *
          "sim_method=$(typeof(sim_params.sim_method)), " *
          "evolution_method=$(typeof(sim_params.evolution_method)), " *
          "backend=$(typeof(backend))")
end

# ============================================================================
# Monte Carlo + Continuous Evolution + Tensor Networks
# ============================================================================

"""
    _tdvp_real_time(t)

Convert a physical Schrödinger evolution time ``t`` into the ITensorMPS TDVP
parameter. ITensorMPS evolves as ``exp(τ H)``, so real-time evolution
``exp(-i H t)`` requires ``τ = -i t``.
"""
function _tdvp_real_time(t::Float64)
    t < 0 && throw(ArgumentError("TDVP real-time evolution time must be nonnegative; got t=$t."))
    return -1.0im * t
end

"""
    _tdvp_step_count(t, tau)

Choose the number of TDVP substeps for physical time ``t`` and target substep
size ``tau``. TDVP receives the full physical time separately; unlike a Trotter
loop, these substeps only partition that time interval, so `ceil(t/tau)` avoids
oversized substeps when ``t`` is not an integer multiple of ``tau``.
"""
function _tdvp_step_count(t::Float64, tau::Float64)
    t < 0 && throw(ArgumentError("TDVP evolution time must be nonnegative; got t=$t."))
    tau <= 0 && throw(ArgumentError("TDVP step tau must be positive; got tau=$tau."))
    t == 0 && return 0
    return Int(ceil(t / tau))
end

struct TDVPSweepObserver{F}
    callback::F
end

"""
    tdvp_sweep_observer(callback)

Return an ITensorMPS-compatible TDVP observer adapter. When passed as
`tdvp_sweep_observer!`, the callback is called after each TDVP sweep with the
keyword payload emitted by ITensorMPS, including `state`, `sweep`, and
`current_time`.
"""
tdvp_sweep_observer(callback) = TDVPSweepObserver(callback)

function ITensorMPS.update_observer!(observer::TDVPSweepObserver; kwargs...)
    observer.callback(; kwargs...)
    return observer
end

function evolve_state(::HamiltonianParameters, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, ContinuousEvolution},
                     ::TNBackend, H_total, ψ, t::Float64, ::Vector{<:Index};
                     tdvp_outputlevel::Integer=0,
                     tdvp_sweep_observer! = nothing,
                     kwargs...)
    if !isempty(kwargs)
        unknown = join(string.(keys(kwargs)), ", ")
        throw(ArgumentError("Unsupported MCWF+TDVP evolution keyword(s): $unknown."))
    end
    Dmax, cutoff, tau = sim_params.Dmax, sim_params.cutoff, sim_params.tau

    # Pick an integer number of TDVP steps so arbitrary `t` works even when `t/tau`
    # is not an integer (e.g. randomized evolution times in multi-frequency cooling).
    nsteps = _tdvp_step_count(t, tau)
    if nsteps == 0
        ψ_evolved = copy(ψ)
        normalize!(ψ_evolved)
        orthogonalize!(ψ_evolved, min(2, length(ψ_evolved)))
        return ψ_evolved
    end
    @debug "evolve_state MC+Continuous: Dmax=$Dmax, tau=$tau, t=$t, nsteps=$nsteps, nsite=2"

    tdvp_kwargs = (
        nsteps=nsteps,
        nsite=2,
        reverse_step=true,
        normalize=true,
        maxdim=Dmax,
        cutoff=cutoff,
        outputlevel=tdvp_outputlevel,
    )
    if tdvp_sweep_observer! !== nothing
        tdvp_kwargs = merge(tdvp_kwargs, (sweep_observer! = tdvp_sweep_observer!,))
    end

    # Use nsite=2 to allow bond dimension growth from product states
    ψ_evolved = tdvp(H_total, _tdvp_real_time(t), ψ; tdvp_kwargs...)
    normalize!(ψ_evolved)
    orthogonalize!(ψ_evolved, 2)
    return ψ_evolved
end

# ============================================================================
# Monte Carlo + Trotter Evolution + Tensor Networks
# ============================================================================

function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, TrotterEvolution},
                     ::TNBackend, ::Any, ψ, t::Float64, ::Vector{<:Index}; gates=nothing, step_gates=nothing, kwargs...)
    gates === nothing && error("Trotter evolution requires pre-computed gates")

    Dmax = tn_method_maxdim(sim_params.sim_method, sim_params.Dmax)
    cutoff, tau = sim_params.cutoff, sim_params.tau
    steps, dt = trotter_time_slices(t, tau)
    steps == 0 && return ψ
    active_gates = _trotter_gates_for_step(gates, step_gates, dt, tau)

    # `apply(gates, ψ; ...)` returns a new MPS (it does not mutate `ψ`), and in the
    # cooling loop the input `ψ` is freshly constructed and not reused. Avoid an
    # extra full copy here to reduce allocations.
    ψ_evolved = ψ

    # All Hamiltonian terms (system + bath + coupling) are in the interleaved gates,
    # matching the MPO DM+Trotter decomposition for consistency.
    for _ in 1:steps
        ψ_evolved = apply(active_gates, ψ_evolved; cutoff=cutoff, maxdim=Dmax, move_sites_back=true)
        normalize!(ψ_evolved)
    end

    return ψ_evolved
end

# ============================================================================
# Density Matrix + Trotter Evolution + MPO
# ============================================================================

function evolve_state(::HamiltonianParameters, sim_params::UnifiedSimulationParameters{DensityMatrix, TrotterEvolution},
                     ::TNBackend, gates, ρ, t::Float64, ::Vector{<:Index}; step_gates=nothing, kwargs...)
    Dmax = tn_method_maxdim(sim_params.sim_method, sim_params.Dmax)
    cutoff = sim_params.cutoff / 10
    steps, dt = trotter_time_slices(t, sim_params.tau)
    steps == 0 && return ρ
    active_gates = _trotter_gates_for_step(gates, step_gates, dt, sim_params.tau)

    for _ in 1:steps
        ρ = apply(active_gates, ρ; apply_dag=true, cutoff=cutoff, maxdim=Dmax, move_sites_back=true)
    end

    return ρ
end

function _trotter_gates_for_step(gates, step_gates, dt::Float64, tau::Float64)
    if isapprox(dt, tau; rtol=0.0, atol=1e-12)
        return gates
    end
    step_gates === nothing && throw(ArgumentError(
        "Requested Trotter time requires adjusted gate step dt=$dt, but no step_gates builder was provided."
    ))
    return step_gates(dt)
end
