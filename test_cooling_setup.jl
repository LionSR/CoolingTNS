#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays

# Include necessary files
include("src/CoolingTNS.jl")
using .CoolingTNS

# Test the actual cooling setup for a small system
N = 2  # 2 system qubits, 2 bath qubits
args = [
    "--N", "$N",
    "--problem", "Ising", 
    "--backend", "ED",
    "--sim_method", "density_matrix",
    "--evolution_method", "continuous",
    "--coupling", "XX",
    "--g", "0.1",
    "--te", "0.5",
    "--steps", "1"
]

println("=== Testing Cooling Setup ===")
println("System: N=$N ($(2*N) total qubits)")

# Parse arguments and setup
parsed = CoolingTNS.parse_commandline(args)
common_params = CoolingTNS.setup_common_parameters(parsed)

# Get parameters
problem_type, ham_params, ham_name, coupling_params = common_params
backend = CoolingTNS.get_backend(parsed["backend"])

# Create simulation parameters using the keyword constructor
sim_params = CoolingTNS.UnifiedSimulationParameters(
    CoolingTNS.DensityMatrix(),
    CoolingTNS.ContinuousEvolution();
    Dmax=parsed["Dmax"],
    cutoff=parsed["cutoff"],
    tau=parsed["tau"],
    pe=parsed["peInt"]*1e-3,
    n_trajectories=parsed["n_trajectories"],
    trotter_steps=Int(parsed["te"] / parsed["tau"])
)

# Setup problem
cooling_problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)

println("\nSystem Hamiltonian eigenvalues:")
H_sys_matrix = Matrix(cooling_problem.H_sys)
vals_sys = eigvals(H_sys_matrix)
for (i, E) in enumerate(vals_sys)
    println("  E$i = $E")
end

println("\nGround state energy: $(cooling_problem.e₀)")
println("Computed Δ: $(coupling_params.delta)")

# Check system+bath Hamiltonian
println("\nSystem+Bath Hamiltonian size: $(size(cooling_problem.H_sys_bath))")
H_sb_matrix = Matrix(cooling_problem.H_sys_bath)
vals_sb = eigvals(H_sb_matrix)
println("Lowest eigenvalues of system+bath:")
for i in 1:min(8, length(vals_sb))
    println("  E$i = $(vals_sb[i])")
end

# Setup initial state
initial_state = CoolingTNS.setup_initial_state(
    cooling_problem,
    sim_params,
    parsed["init_state"],
    parsed["theta"]
)

println("\n\nInitial state type: $(typeof(initial_state))")

# Get initial density matrix
if isa(initial_state.state, CoolingTNS.EDDensityMatrix)
    ρ_init = initial_state.state.data
else
    ρ_init = initial_state.state
end

println("Initial density matrix size: $(size(ρ_init))")
println("Trace of initial density matrix: $(tr(ρ_init))")

# Measure initial energy
E_init = CoolingTNS.measure_energy(initial_state, cooling_problem)
println("\nInitial system energy: $E_init")
println("Initial system energy per site: $(E_init/N)")

# Check if initial state is correct
# For product state, system should be in |00⟩ (all spins up)
# Bath should be in |00⟩ (ground state of bath Hamiltonian)
println("\nInitial state populations:")
for i in 1:min(16, size(ρ_init, 1))
    if abs(ρ_init[i,i]) > 1e-10
        # Convert index to binary representation
        state_str = string(i-1, base=2, pad=2*N)
        println("  |$state_str⟩: $(ρ_init[i,i])")
    end
end

# Now run one cooling step
println("\n\n=== Running one cooling step ===")
final_state = CoolingTNS.run_cooling_step(initial_state, cooling_problem, common_params)

# Measure final energy
E_final = CoolingTNS.measure_energy(final_state, cooling_problem)
println("\nFinal system energy: $E_final")
println("Final system energy per site: $(E_final/N)")
println("Energy change: $(E_final - E_init)")

if E_final > E_init
    println("\n❌ System is heating! Energy increased by $(E_final - E_init)")
    
    # Debug: Check the bath energies
    println("\nDebug: Bath qubit energies (Δ/2 * Z):")
    for i in 1:N
        println("  Bath qubit $i: $(coupling_params.delta/2)")
    end
else
    println("\n✅ System is cooling! Energy decreased by $(E_init - E_final)")
end