"""
Typed parameter and result structures for cooling simulations.

This module provides type-safe parameter handling and consistent result structures
across different backends and simulation methods.
"""

# ============================================================================
# Backend Types
# ============================================================================

"""
    CoolingBackend

Base abstract type for different simulation backends.
"""
abstract type CoolingBackend end
struct EDBackend <: CoolingBackend end  # Exact Diagonalization
struct TNBackend <: CoolingBackend end  # Tensor Network Backend

"""
    get_backend(method::String) -> CoolingBackend

Convert string method name to backend type.
"""
function get_backend(method::String)
    method == "ED" && return EDBackend()
    method == "TN" && return TNBackend()
    error("Unknown backend: $method. Use 'ED' for exact diagonalization or 'TN' for tensor network")
end

"""
    get_sim_method(method::String) -> SimulationMethod

Convert string to SimulationMethod type.
"""
function get_sim_method(method::String)
    method == "density_matrix" && return DensityMatrix()
    method == "monte_carlo" && return MonteCarloWavefunction()
    error("Unknown simulation method: $method. Use 'density_matrix' or 'monte_carlo'")
end

"""
    get_evolution_method(method::String) -> EvolutionMethod

Convert string to EvolutionMethod type.
"""
function get_evolution_method(method::String)
    method == "continuous" && return ContinuousEvolution()
    method == "trotter" && return TrotterEvolution()
    error("Unknown evolution method: $method. Use 'continuous' or 'trotter'")
end

# Default simulation methods for backends
default_simulation_method(::EDBackend) = DensityMatrix()  # Can be Monte Carlo or Density Matrix
default_simulation_method(::TNBackend) = MonteCarloWavefunction()  # Most common for TN

# Default evolution methods for backends
default_evolution_method(::EDBackend) = ContinuousEvolution()  # Matrix exponentiation
default_evolution_method(::TNBackend) = ContinuousEvolution()  # TDVP is default for TN

# ============================================================================
# Hamiltonian Model Types
# ============================================================================

"""
    HamiltonianModel

Base abstract type for different Hamiltonian models.
"""
abstract type HamiltonianModel end
struct IsingModel <: HamiltonianModel end
struct NiIsingModel <: HamiltonianModel end
struct RydbergModel <: HamiltonianModel end

"""
    HamiltonianParameters

Unified parameter structure for different Hamiltonian models using multiple dispatch.
Now includes N (number of system spins) to avoid passing it separately.
"""
struct HamiltonianParameters{M<:HamiltonianModel}
    model::M
    N::Int  # Number of system spins
    params::NamedTuple
    bc::Symbol  # Boundary conditions: :open, :periodic, :antiperiodic
end

# Convenience constructors with default open BC
IsingParameters(N::Int, J, h) = HamiltonianParameters(IsingModel(), N, (J=J, h=h), :open)
NiIsingParameters(N::Int, J, hx, hz) = HamiltonianParameters(NiIsingModel(), N, (J=J, hx=hx, hz=hz), :open)
RydbergParameters(N::Int, Ω, Δ, V) = HamiltonianParameters(RydbergModel(), N, (Ω=Ω, Δ=Δ, V=V), :open)

# Versions with explicit BC
IsingParameters(N::Int, J, h, bc::Symbol) = HamiltonianParameters(IsingModel(), N, (J=J, h=h), bc)
NiIsingParameters(N::Int, J, hx, hz, bc::Symbol) = HamiltonianParameters(NiIsingModel(), N, (J=J, hx=hx, hz=hz), bc)
RydbergParameters(N::Int, Ω, Δ, V, bc::Symbol) = HamiltonianParameters(RydbergModel(), N, (Ω=Ω, Δ=Δ, V=V), bc)

# Generate name for HamiltonianParameters
function hamiltonian_name(ham_params::HamiltonianParameters{IsingModel})
    return "IsingN$(ham_params.N)bc$(ham_params.bc)J$(ham_params.params.J)h$(ham_params.params.h)"
end

function hamiltonian_name(ham_params::HamiltonianParameters{NiIsingModel})
    return "niIsingN$(ham_params.N)bc$(ham_params.bc)J$(ham_params.params.J)hx$(ham_params.params.hx)hz$(ham_params.params.hz)"
end

function hamiltonian_name(ham_params::HamiltonianParameters{RydbergModel})
    return "RydbergN$(ham_params.N)bc$(ham_params.bc)Omega$(ham_params.params.Ω)Delta$(ham_params.params.Δ)V$(ham_params.params.V)"
end

"""
    parse_hamiltonian_name(name::AbstractString) -> HamiltonianParameters

Inverse of [`hamiltonian_name`](@ref): parse strings produced by `hamiltonian_name`
back into a `HamiltonianParameters` object.

This is primarily intended for plotting and file-name based workflows.
"""
function parse_hamiltonian_name(name::AbstractString)::HamiltonianParameters
    if startswith(name, "Ising")
        m = match(r"^IsingN([0-9]+)bc([A-Za-z]+)J([-+0-9.eE]+)h([-+0-9.eE]+)$", name)
        m === nothing && throw(ArgumentError("Unrecognized Ising hamiltonian name: \"$name\""))

        N = parse(Int, m.captures[1])
        bc = Symbol(m.captures[2])
        J = parse(Float64, m.captures[3])
        h = parse(Float64, m.captures[4])
        return IsingParameters(N, J, h, bc)
    elseif startswith(name, "niIsing")
        m = match(r"^niIsingN([0-9]+)bc([A-Za-z]+)J([-+0-9.eE]+)hx([-+0-9.eE]+)hz([-+0-9.eE]+)$", name)
        m === nothing && throw(ArgumentError("Unrecognized niIsing hamiltonian name: \"$name\""))

        N = parse(Int, m.captures[1])
        bc = Symbol(m.captures[2])
        J = parse(Float64, m.captures[3])
        hx = parse(Float64, m.captures[4])
        hz = parse(Float64, m.captures[5])
        return NiIsingParameters(N, J, hx, hz, bc)
    elseif startswith(name, "Rydberg")
        m = match(r"^RydbergN([0-9]+)bc([A-Za-z]+)Omega([-+0-9.eE]+)Delta([-+0-9.eE]+)V([-+0-9.eE]+)$", name)
        m === nothing && throw(ArgumentError("Unrecognized Rydberg hamiltonian name: \"$name\""))

        N = parse(Int, m.captures[1])
        bc = Symbol(m.captures[2])
        Ω = parse(Float64, m.captures[3])
        Δ = parse(Float64, m.captures[4])
        V = parse(Float64, m.captures[5])
        return RydbergParameters(N, Ω, Δ, V, bc)
    end

    throw(ArgumentError("Unrecognized hamiltonian name: \"$name\""))
end

# ============================================================================
# Coupling Parameters
# ============================================================================

"""
    CouplingParameters

Base abstract type for coupling parameters.
"""
abstract type CouplingParameters end

"""
    BasicCouplingParameters

Standard coupling parameters used by most backends.
"""
struct BasicCouplingParameters <: CouplingParameters
    coupling::String        # Coupling type: "XX", "YY", "ZZ", "XY", "XZ", "YZ"
    g::Float64             # Coupling strength
    steps::Int             # Number of cooling steps
    te::Float64            # Evolution time per step
    delta::Union{Float64, Nothing} # Bath detuning (computed automatically if nothing)
end

"""
    MultiFrequencyCouplingParameters

Coupling parameters for multi-frequency cooling, where the bath detuning Δ is
cycled through a list of values during the cooling loop.

`te` is interpreted as the *mean* evolution time per step. If
`randomize_times=true`, each step uses an independently drawn time
`t_m ~ Uniform(0, 2te)` (see docs/multi_frequency_cooling_plan.md).

The standard single-frequency simulation uses `BasicCouplingParameters`.
"""
struct MultiFrequencyCouplingParameters <: CouplingParameters
    coupling::String              # e.g. "XX"
    g::Float64                    # Coupling strength
    steps::Int                    # Total cooling steps
    te::Float64                   # Mean evolution time per step
    delta_values::Vector{Float64} # Bath detunings to cycle through
    randomize_times::Bool         # Randomize evolution times per step?
    schedule::Symbol              # :round_robin or :random
end

function MultiFrequencyCouplingParameters(
    coupling::String,
    g::Real,
    steps::Integer,
    te::Real,
    delta_values::AbstractVector{<:Real};
    randomize_times::Bool=false,
    schedule::Symbol=:round_robin,
)
    schedule in (:round_robin, :random) ||
        throw(ArgumentError("schedule must be :round_robin or :random, got $schedule"))
    deltas = Float64.(collect(delta_values))
    isempty(deltas) && throw(ArgumentError("delta_values must be nonempty"))
    return MultiFrequencyCouplingParameters(
        coupling,
        Float64(g),
        Int(steps),
        Float64(te),
        deltas,
        randomize_times,
        schedule,
    )
end

"""
    OptimizationCouplingParameters

Extended coupling parameters for optimization tasks.
"""
struct OptimizationCouplingParameters <: CouplingParameters
    coupling::String
    g::Float64
    steps::Int
    te::Float64
    delta::Union{Float64, Nothing} # Bath detuning (computed automatically if nothing)
    # Optimization-specific parameters
    search_method::String   # "Random", "Bayesian", etc.
    num_trials::Int
    bounds::Dict{String, Tuple{Float64, Float64}}
end

"""
    SimulationMethod

Base abstract type for simulation methods (orthogonal to backend choice).
"""
abstract type SimulationMethod end
struct DensityMatrix <: SimulationMethod end     # Can be used with MPO, or MPS+density matrix conversion
struct MonteCarloWavefunction <: SimulationMethod end  # Used with MPS, TrotterMPS, or ED trajectories

"""
    EvolutionMethod

Base abstract type for time evolution methods.
"""  
abstract type EvolutionMethod end
struct ContinuousEvolution <: EvolutionMethod end  # TDVP, matrix exponentiation
struct TrotterEvolution <: EvolutionMethod end     # Gate-based Trotter decomposition

"""
    SimulationParameters

Base abstract type for simulation parameters.
"""
abstract type SimulationParameters end

"""
    UnifiedSimulationParameters

Universal simulation parameters that work with any combination of:
- SimulationMethod: DensityMatrix or MonteCarloWavefunction
- EvolutionMethod: ContinuousEvolution or TrotterEvolution  
- Backend: ED, MPS, MPO, TrotterMPS
"""
struct UnifiedSimulationParameters{S<:SimulationMethod, E<:EvolutionMethod} <: SimulationParameters
    sim_method::S
    evolution_method::E
    
    # Tensor network parameters (used when applicable)
    Dmax::Int
    cutoff::Float64
    tau::Float64  # Time step
    
    # Noise parameters
    pe::Float64  # Depolarizing error probability
    
    # Monte Carlo parameters (used when applicable)
    n_trajectories::Int
    parallel::Bool
    
    # Trotter parameters (used when applicable) 
    trotter_steps::Int
    
    # DMRG parameters (used when applicable)
    maxiter::Int
    normalize::Bool
end

# Simplified constructor with defaults
function UnifiedSimulationParameters(sim_method::S, evolution_method::E; 
                                    Dmax=100, cutoff=1e-10, tau=0.01, pe=0.0, 
                                    n_trajectories=1, parallel=false, trotter_steps=1, 
                                    maxiter=100, normalize=true) where {S<:SimulationMethod, E<:EvolutionMethod}
    UnifiedSimulationParameters(sim_method, evolution_method, Dmax, cutoff, tau, pe, 
                               n_trajectories, parallel, trotter_steps, maxiter, normalize)
end

# Legacy type aliases removed - use UnifiedSimulationParameters directly

# ============================================================================
# Result Structures
# ============================================================================

"""
    CoolingResults

Base abstract type for cooling simulation results.
"""
abstract type CoolingResults end

"""
    DensityMatrixResults

Results from density matrix cooling simulations.
"""
struct DensityMatrixResults <: CoolingResults
    E_list::Vector{Float64}           # Energy evolution
    GS_overlap_list::Vector{Float64}  # Ground state overlap evolution
    purity_list::Vector{Float64}      # Purity evolution
end

"""
    MonteCarloResults

Results from Monte Carlo wavefunction cooling simulations.
"""
struct MonteCarloResults <: CoolingResults
    E_list::Vector{Float64}           # Mean energy evolution
    GS_overlap_list::Vector{Float64}  # Mean ground state overlap evolution
    purity_list::Vector{Float64}      # Purity (always 1 for pure states)
    # Monte Carlo specific data
    E_trajectories::Matrix{Float64}   # Individual trajectory energies (steps × trajectories)
    GS_trajectories::Matrix{Float64}  # Individual trajectory overlaps (steps × trajectories)
    n_trajectories::Int               # Number of trajectories used
    # Statistical measures
    E_std::Vector{Float64}            # Standard deviation of energy
    GS_std::Vector{Float64}           # Standard deviation of overlap
    # todo: add bath magnetization
end

"""
    TensorNetworkResults

Results from tensor network (MPS/MPO) cooling simulations.
"""
struct TensorNetworkResults <: CoolingResults
    energy_list::Vector{Float64}           # Energy evolution
    gs_overlap_list::Vector{Float64}       # Ground state overlap evolution
    purity_list::Vector{Float64}           # Purity evolution (for MPO) or constant 1 (for MPS)
    # Tensor network specific data
    bond_dims::Vector{Vector{Int}}         # Bond dimensions evolution
    truncation_errors::Vector{Float64}     # Truncation errors at each step
    renyi_entropy::Vector{Vector{Float64}}  # Entanglement entropy profile
    # Additional fields for consistency with other backends
    bath_magnetization_list::Vector{Float64}  # Bath magnetization evolution
    final_state::Any                       # Final state (MPS or MPO)
end

# ============================================================================
# Constructor Functions
# ============================================================================

"""
    create_coupling_params(coupling, g, steps, te; kwargs...)

Create appropriate CouplingParameters struct.
"""
function create_coupling_params(coupling::String, g::Float64, steps::Int, te::Float64; 
                               optimization::Bool=false, kwargs...)
    delta = get(kwargs, :delta, nothing)
    
    if optimization
        # Extract optimization parameters
        search_method = get(kwargs, :search_method, "Bayesian")
        num_trials = get(kwargs, :num_trials, 20)
        bounds = get(kwargs, :bounds, Dict("g" => (0.01, 1.0), "te" => (0.1, 10.0)))
        
        return OptimizationCouplingParameters(coupling, g, steps, te, delta, 
                                            search_method, num_trials, bounds)
    else
        return BasicCouplingParameters(coupling, g, steps, te, delta)
    end
end

# Note: create_sim_params with backend dispatch is defined in utils.jl

# ============================================================================
# Utility Functions
# ============================================================================

# All to_dict functions now handled by generic reflection-based implementation above

# Removed specific results functions - use generic to_dict from above

# ============================================================================
# Generic Reflection-Based Conversion Functions  
# ============================================================================

"""
    to_dict(obj)

Generic function to convert any struct to a dictionary using reflection.
Handles special field name mappings and null values automatically.
"""
function to_dict(obj::T) where T
    result = Dict{String,Any}()
    for field_name in fieldnames(T)
        value = getfield(obj, field_name)
        
        # Handle special field name mappings
        dict_key = if field_name == :delta && value !== nothing
            "Δ"  # Use Greek delta in dict
        else
            string(field_name)
        end
        
        # Only add non-nothing values
        if value !== nothing
            result[dict_key] = value
        end
    end
    return result
end


"""
    TensorNetworkResults(E_list, GS_list, bath_mag_list, final_state)

Simplified constructor for TensorNetworkResults with just the essential data.
"""
function TensorNetworkResults(E_list::Vector{Float64}, GS_list::Vector{Float64}, 
                            bath_mag_list::Vector{Float64}, final_state)
    # For simplified construction, use empty vectors for optional fields
    purity_list = ones(Float64, length(E_list))  # MPS states are pure
    bond_dims = Vector{Vector{Int}}()
    truncation_errors = zeros(Float64, length(E_list))
    renyi_entropy = Vector{Vector{Float64}}()
    
    return TensorNetworkResults(E_list, GS_list, purity_list, bond_dims, 
                               truncation_errors, renyi_entropy, 
                               bath_mag_list, final_state)
end

# Remove duplicate definition - already defined above