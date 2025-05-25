using ArgParse

function parse_commandline(args=ARGS)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--search_method"
        help = "method for hyperparameter search (valid choices: Random, Grid, Bayesian)"
        default = "Random"
        "--window_size"
        help = "window size for moving average"
        default = 50
        arg_type = Int
        "--problem"
        help = "type of problem to solve (valid choices: Ising, niIsing)"
        default = "niIsing"
        "--N"
        help = "number of spins in the system"
        arg_type = Int
        default = 20
        "--J"
        help = "coupling constant J"
        arg_type = Float64
        default = 1.0
        "--h"
        help = "magnetic field h"
        arg_type = Float64
        default = 2.0
        "--hx"
        help = "x-component of the magnetic field"
        arg_type = Float64
        default = -1.05
        "--hz"
        help = "z-component of the magnetic field"
        arg_type = Float64
        default = 0.5
        "--steps"
        help = "number of cooling steps"
        arg_type = Int
        default = 100
        "--g"
        help = "coupling strength g"
        arg_type = Float64
        default = 0.3
        "--te"
        help = "total evolution time"
        arg_type = Float64
        default = 2.0
        "--tau"
        help = "Time step size for TDVP/Trotter evolution"
        arg_type = Float64
        default = 0.1
        "--cutoff"
        help = "truncation error cutoff"
        arg_type = Float64
        default = 1E-6
        "--num_trials"
        help = "number of trials for the search"
        default = 10
        arg_type = Int
        "--Dmax"
        help = "maximum bond dimension"
        arg_type = Int
        default = 20
        "--method"
        help = "simulation method for cooling (valid choices: MPS, MPO, TrotterMPS, ED)"
        default = "MPS"
        "--ed_method"
        help = "ED simulation method (valid choices: density_matrix, monte_carlo)"
        default = "density_matrix"
        "--n_trajectories"
        help = "number of trajectories for Monte Carlo wavefunction method"
        arg_type = Int
        default = 100
        "--peInt"
        help = "pe: noise strength (will be times by 1e-3)"
        arg_type = Int
        default = 0
        "--coupling"
        help = "coupling type"
        arg_type = String
        default = "XX"
        "--init_state"
        help = "initial state type: 'product' (default), 'identity' (maximally mixed), 'theta' (use --theta value)"
        arg_type = String
        default = "product"
        "--theta"
        help = "theta angle for initial state (in units of pi, e.g., -0.5 for all down)"
        arg_type = Float64
        default = 0.0
    end
    return parse_args(args, s)
end
