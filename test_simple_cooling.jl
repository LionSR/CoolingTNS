#!/usr/bin/env julia

using ITensors
using ITensorMPS
using LinearAlgebra
using SparseArrays
using KrylovKit
using HDF5

# Include the main module
include("src/CoolingTNS.jl")
using .CoolingTNS

# Simple test to compare backends
function test_cooling_step()
    # Parameters for small test
    N = 3
    problem_type = "Ising"
    
    # Common parameters
    base_args = [
        "--N", "$N",
        "--problem", problem_type,
        "--coupling", "XX",
        "--g", "0.1",
        "--te", "0.5",
        "--steps", "1",
        "--init_state", "product",
        "--sim_method", "density_matrix",
        "--evolution_method", "continuous"
    ]
    
    println("=== Testing Cooling Step ===")
    println("System: N=$N, $problem_type model")
    println()
    
    # Test ED backend
    println("--- ED Backend ---")
    args_ed = vcat(base_args, ["--backend", "ED"])
    parsed_ed = CoolingTNS.parse_commandline(args_ed)
    common_params_ed = CoolingTNS.setup_common_parameters(parsed_ed)
    
    # Setup problem
    problem_ed = CoolingTNS.setup_problem_unified(common_params_ed)
    initial_state_ed = CoolingTNS.setup_initial_state(parsed_ed["init_state"], problem_ed, common_params_ed)
    
    # Initial measurements
    E_initial_ed = CoolingTNS.measure_energy(initial_state_ed, problem_ed)
    overlap_initial_ed = CoolingTNS.compute_overlap(initial_state_ed, problem_ed.ground_state)
    
    println("Initial: E/N = $(E_initial_ed/N), overlap = $overlap_initial_ed")
    
    # Run one cooling step
    final_state_ed = CoolingTNS.run_cooling_step(initial_state_ed, problem_ed, common_params_ed)
    
    # Final measurements
    E_final_ed = CoolingTNS.measure_energy(final_state_ed, problem_ed)
    overlap_final_ed = CoolingTNS.compute_overlap(final_state_ed, problem_ed.ground_state)
    
    println("Final: E/N = $(E_final_ed/N), overlap = $overlap_final_ed")
    println("Energy change: $(E_final_ed - E_initial_ed)")
    
    # Test TN backend
    println("\n--- TN Backend ---")
    args_tn = vcat(base_args, ["--backend", "TN", "--Dmax", "16"])
    parsed_tn = CoolingTNS.parse_commandline(args_tn)
    common_params_tn = CoolingTNS.setup_common_parameters(parsed_tn)
    
    # Setup problem
    problem_tn = CoolingTNS.setup_problem_unified(common_params_tn)
    initial_state_tn = CoolingTNS.setup_initial_state(parsed_tn["init_state"], problem_tn, common_params_tn)
    
    # Initial measurements
    E_initial_tn = CoolingTNS.measure_energy(initial_state_tn, problem_tn)
    overlap_initial_tn = CoolingTNS.compute_overlap(initial_state_tn, problem_tn.ground_state)
    
    println("Initial: E/N = $(E_initial_tn/N), overlap = $overlap_initial_tn")
    
    # Run one cooling step
    final_state_tn = CoolingTNS.run_cooling_step(initial_state_tn, problem_tn, common_params_tn)
    
    # Final measurements
    E_final_tn = CoolingTNS.measure_energy(final_state_tn, problem_tn)
    overlap_final_tn = CoolingTNS.compute_overlap(final_state_tn, problem_tn.ground_state)
    
    println("Final: E/N = $(E_final_tn/N), overlap = $overlap_final_tn")
    println("Energy change: $(E_final_tn - E_initial_tn)")
    
    # Summary
    println("\n=== Summary ===")
    println("Ground state energy/N ≈ $(problem_ed.ground_energy/N)")
    println("\nED Backend: E/N: $(E_initial_ed/N) → $(E_final_ed/N)")
    println("TN Backend: E/N: $(E_initial_tn/N) → $(E_final_tn/N)")
    
    if E_final_ed > E_initial_ed
        println("\n❌ ED Backend is HEATING (energy increased by $(E_final_ed - E_initial_ed))")
    else
        println("\n✅ ED Backend is cooling (energy decreased by $(E_initial_ed - E_final_ed))")
    end
    
    if E_final_tn > E_initial_tn
        println("❌ TN Backend is HEATING (energy increased by $(E_final_tn - E_initial_tn))")
    else
        println("✅ TN Backend is cooling (energy decreased by $(E_initial_tn - E_final_tn))")
    end
end

# Run the test
test_cooling_step()