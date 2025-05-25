module CoolingTNS

__precompile__(false)  # Disable precompilation due to LuxurySparse issues

# Import ITensors package
using ITensors
using ITensorMPS

# For convenience, here we should just include all the files that are still being used

include("parameter_types.jl")      # Define parameter types first
include("cooling_types.jl")        # CoolingProblem and QuantumState types

include("utils.jl")
include("utils_mps.jl")
include("coupling_utils.jl")
include("utils_mpo.jl")
include("plotting.jl")
include("policy.jl")
include("argparse.jl")
include("noise.jl")
include("bath_measurements.jl")    # Dispatched bath measurement functions
include("state_manipulation.jl")   # Dispatched state manipulation functions

# Core implementations (all using dispatch by default)
include("system_hamiltonian.jl")    # System Hamiltonian construction
include("ground_state.jl")          # Ground state computation
include("setup_system.jl")          # System setup using the above
include("system_bath_hamiltonian.jl") # System-bath coupling
include("trotter.jl")               # Trotter circuit construction
include("initial_state.jl")         # Initial state preparation
include("cooling_evolution.jl")     # Cooling evolution
include("setup.jl")                 # setup_problem implementations



# these are outdated previous implementions without dispatching that should not be used anymore
# include("ham.jl")
# include("cooling_functions_mps.jl")
# include("cooling_functions_mpo.jl")
# include("cooling_functions_trotter_mps.jl")
# include("cooling_functions_ed.jl")


export setup_problem, run_cooling, setup_initial_state
export CoolingProblem, QuantumState
export CoolingBackend, EDBackend, TNBackend
export SimulationMethod, DensityMatrix, MonteCarloWavefunction
export EvolutionMethod, ContinuousEvolution, TrotterEvolution
export plot_data
# Export new parameter types and functions
export CouplingParameters, SimulationParameters, CoolingResults
export BasicCouplingParameters, OptimizationCouplingParameters
export UnifiedSimulationParameters
export HamiltonianParameters, IsingParameters, NiIsingParameters, RydbergParameters
export HamiltonianModel, IsingModel, NiIsingModel, RydbergModel
# Legacy parameter types removed - use UnifiedSimulationParameters
export DensityMatrixResults, MonteCarloResults, TensorNetworkResults
export create_coupling_params, create_sim_params, create_results
export to_dict
export setup_common_parameters, create_filename, save_results
export get_backend, mean_last_window

end
