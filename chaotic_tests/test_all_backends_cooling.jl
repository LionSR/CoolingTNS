#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays
using ITensors
using ITensorMPS

# Include necessary files
include("src/CoolingTNS.jl")
using .CoolingTNS

# Test configuration
N = 3  # Small enough to compute exactly
coupling = "XX"
g = 0.1
te = 0.5
steps = 5

println("=== Comprehensive Backend Comparison ===")
println("System: N=$N, Ising model")
println("Coupling: $coupling, g=$g")
println("Evolution time per step: $te")
println("Total steps: $steps\n")

# Store results for comparison
results = Dict()

# Test all combinations
backends = ["ED", "TN"]
sim_methods = ["density_matrix", "monte_carlo"]
evolution_methods = ["continuous"]  # Focus on continuous first

for backend in backends
    for sim_method in sim_methods
        for evolution_method in evolution_methods
            key = "$backend-$sim_method-$evolution_method"
            println("\n--- Testing $key ---")
            
            # Create arguments
            args = [
                "--N", "$N",
                "--problem", "Ising",
                "--backend", backend,
                "--sim_method", sim_method,
                "--evolution_method", evolution_method,
                "--coupling", coupling,
                "--g", "$g",
                "--te", "$te",
                "--steps", "$steps",
                "--Dmax", "64"  # High bond dimension for accuracy
            ]
            
            try
                # Parse arguments
                parsed = CoolingTNS.parse_commandline(args)
                common_params = CoolingTNS.setup_common_parameters(parsed)
                
                # Get backend
                backend_obj = CoolingTNS.get_backend(parsed["backend"])
                
                # Create simulation parameters
                sim_method_obj = if sim_method == "density_matrix"
                    CoolingTNS.DensityMatrix()
                else
                    CoolingTNS.MonteCarloWavefunction()
                end
                
                evolution_method_obj = if evolution_method == "continuous"
                    CoolingTNS.ContinuousEvolution()
                else
                    CoolingTNS.TrotterEvolution()
                end
                
                sim_params = CoolingTNS.create_sim_params(
                    backend_obj; 
                    sim_method=sim_method_obj, 
                    evolution_method=evolution_method_obj,
                    Dmax=parsed["Dmax"], 
                    cutoff=parsed["cutoff"],
                    tau=parsed["tau"], 
                    pe=parsed["peInt"]*1e-3,
                    n_trajectories=parsed["n_trajectories"]
                )
                
                # Setup problem
                problem_type, ham_params, ham_name, coupling_params = common_params
                cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
                
                # Setup initial state
                initial_state = CoolingTNS.setup_initial_state(
                    cooling_problem,
                    sim_params,
                    parsed["init_state"],
                    parsed["theta"]
                )
                
                # Measure initial energy
                E_init = if backend == "ED"
                    if sim_method == "density_matrix"
                        CoolingTNS.expect_ed(cooling_problem.H_sys, initial_state.state)
                    else
                        CoolingTNS.expect_ed(cooling_problem.H_sys, CoolingTNS.state_to_density_ed(initial_state.state))
                    end
                else
                    real(inner(initial_state.state', cooling_problem.H_sys, initial_state.state))
                end
                
                println("Initial energy/N: $(E_init/N)")
                
                # Run cooling
                final_state = initial_state
                energies = [E_init]
                
                for step in 1:steps
                    final_state = CoolingTNS.run_cooling_step(final_state, cooling_problem, common_params)
                    
                    # Measure energy
                    E = if backend == "ED"
                        if sim_method == "density_matrix"
                            # For density matrix, need to trace out bath
                            if isa(final_state.state, CoolingTNS.EDDensityMatrix) && final_state.state.n_qubits == 2*N
                                ρ_sys = CoolingTNS.trace_out_bath_ed(final_state.state, N)
                                CoolingTNS.expect_ed(cooling_problem.H_sys, ρ_sys)
                            else
                                CoolingTNS.expect_ed(cooling_problem.H_sys, final_state.state)
                            end
                        else
                            # For Monte Carlo, convert state vector to density matrix
                            ρ = CoolingTNS.state_to_density_ed(final_state.state)
                            CoolingTNS.expect_ed(cooling_problem.H_sys, ρ)
                        end
                    else
                        real(inner(final_state.state', cooling_problem.H_sys, final_state.state))
                    end
                    
                    push!(energies, E)
                    println("Step $step: energy/N = $(E/N)")
                end
                
                # Store results
                results[key] = (
                    initial = E_init/N,
                    final = energies[end]/N,
                    energies = energies ./ N,
                    ground_state = cooling_problem.e₀/N
                )
                
                # Check if cooling
                if energies[end] > E_init
                    println("❌ HEATING! Energy increased from $(E_init/N) to $(energies[end]/N)")
                else
                    println("✅ Cooling: Energy decreased from $(E_init/N) to $(energies[end]/N)")
                end
                
            catch e
                println("ERROR in $key: $e")
                if isa(e, MethodError)
                    println("Method signature: $(e.f)($(join(typeof.(e.args), ", ")))")
                end
                # Print first few lines of stacktrace
                for (i, frame) in enumerate(stacktrace(catch_backtrace())[1:min(5, end)])
                    println("  at $frame")
                end
            end
        end
    end
end

# Summary comparison
println("\n\n=== SUMMARY ===")
println("Ground state energy/N ≈ $(first(values(results)).ground_state)")
println("\nMethod                        Initial    Final     Change    Status")
println("─" ^ 70)
for (key, res) in sort(collect(results))
    status = res.final < res.initial ? "✅ Cool" : "❌ Heat"
    @printf("%-28s  %7.4f  %7.4f  %+7.4f  %s\n", 
            key, res.initial, res.final, res.final - res.initial, status)
end

# Check if ED and TN match
println("\n\nED vs TN Comparison:")
for sim_method in sim_methods
    for evolution_method in evolution_methods
        ed_key = "ED-$sim_method-$evolution_method"
        tn_key = "TN-$sim_method-$evolution_method"
        
        if haskey(results, ed_key) && haskey(results, tn_key)
            ed_res = results[ed_key]
            tn_res = results[tn_key]
            
            diff_init = abs(ed_res.initial - tn_res.initial)
            diff_final = abs(ed_res.final - tn_res.final)
            
            println("\n$sim_method-$evolution_method:")
            println("  Initial difference: $diff_init")
            println("  Final difference: $diff_final")
            
            if diff_init < 1e-6 && diff_final < 0.01
                println("  ✅ ED and TN match well")
            else
                println("  ❌ ED and TN differ significantly")
            end
        end
    end
end