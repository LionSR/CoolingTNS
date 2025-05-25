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
    if method == "ED"
        return EDBackend()
    elseif method == "TN"
        return TNBackend()
    else
        error("Unknown method: $method. Use 'ED' for exact diagonalization or 'TN' for tensor network")
    end
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
end

# Convenience constructors
IsingParameters(N::Int, J, h) = HamiltonianParameters(IsingModel(), N, (J=J, h=h))
NiIsingParameters(N::Int, J, hx, hz) = HamiltonianParameters(NiIsingModel(), N, (J=J, hx=hx, hz=hz))
RydbergParameters(N::Int, Ω, Δ, V) = HamiltonianParameters(RydbergModel(), N, (Ω=Ω, Δ=Δ, V=V))

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

"""
    create_sim_params(method::SimulationMethod; kwargs...)

Create appropriate SimulationParameters struct based on simulation method.
"""
function create_sim_params(method::DensityMatrix; kwargs...)
    pe = get(kwargs, :pe, 0.0)
    dephasing = get(kwargs, :dephasing, 0.0)
    amplitude_damping = get(kwargs, :amplitude_damping, 0.0)
    
    return DensityMatrixParameters(pe, dephasing, amplitude_damping)
end

function create_sim_params(method::MonteCarloWavefunction; kwargs...)
    n_trajectories = get(kwargs, :n_trajectories, 10)
    pe = get(kwargs, :pe, 0.0)
    parallel = get(kwargs, :parallel, false)
    
    return MonteCarloParameters(n_trajectories, pe, parallel)
end

# Legacy backend dispatch functions - now handled by UnifiedSimulationParameters
# These can be removed as they're no longer used in the new architecture

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