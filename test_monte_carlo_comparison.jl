#!/usr/bin/env julia

using ITensors
using ITensorMPS
using LinearAlgebra
using SparseArrays
using Statistics

# Include main module
include("src/CoolingTNS.jl")
using .CoolingTNS

# Test Monte Carlo methods for both backends
N = 3
n_trajectories = 10  # Multiple trajectories for better statistics
println("=== Monte Carlo Method Comparison (N=$N) ===")
println("Testing monte_carlo + continuous evolution")
println("Initial state: all up |000⟩")
println("Number of trajectories: $n_trajectories")

# Common arguments
base_args = [
    "--N", "$N",
    "--problem", "Ising",
    "--coupling", "XX",
    "--g", "0.1",
    "--te", "0.5",
    "--steps", "10",
    "--sim_method", "monte_carlo",
    "--evolution_method", "continuous",
    "--init_state", "product",
    "--n_trajectories", "$n_trajectories"
]

# Test ED Backend
println("\n--- Running ED Backend (Monte Carlo) ---")
args_ed = vcat(base_args, ["--backend", "ED"])
parsed_ed = CoolingTNS.parse_commandline(args_ed)

# Run full simulation
run(`julia --project=. Cooling.jl $(args_ed[2:end])`)

# Test TN Backend
println("\n--- Running TN Backend (Monte Carlo) ---")
args_tn = vcat(base_args, ["--backend", "TN", "--Dmax", "64"])
parsed_tn = CoolingTNS.parse_commandline(args_tn)

# Run full simulation
run(`julia --project=. Cooling.jl $(args_tn[2:end])`)

println("\n\nNote: Check the output files for detailed results")
println("ED results in: IsingJ1.0h2.0_XX_g0.1_t0.5_s10_ED.h5")
println("TN results in: IsingJ1.0h2.0_XX_g0.1_t0.5_s10_Dmax64_TN.h5")