using ArgParse

function _validate_initial_state_args(parsed_args)
    init_state = get(parsed_args, "init_state", "product")
    sim_method = get(parsed_args, "sim_method", "monte_carlo")

    if init_state == "identity" && sim_method == "monte_carlo"
        throw(ArgumentError(
            "--init_state identity denotes the maximally mixed density matrix " *
            "and requires --sim_method density_matrix. For Monte Carlo " *
            "wavefunction simulations, choose a pure initial state such as " *
            "product, theta, or ground."
        ))
    end

    return parsed_args
end

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
        help = "type of problem to solve (valid choices: Ising, niIsing, Rydberg)"
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
        default = -2.0
        "--hx"
        help = "x-component of the magnetic field"
        arg_type = Float64
        default = -1.05
        "--hz"
        help = "z-component of the magnetic field"
        arg_type = Float64
        default = 0.5
        "--Omega"
        help = "Rydberg Rabi frequency Ω"
        arg_type = Float64
        default = 1.0
        "--Delta"
        help = "Rydberg detuning Δ"
        arg_type = Float64
        default = 0.0
        "--V"
        help = "Rydberg van der Waals interaction scale"
        arg_type = Float64
        default = 1.0
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
        "--bc"
        help = "boundary conditions (valid choices: open, periodic, antiperiodic)"
        default = "open"
        "--backend"
        help = "simulation backend (valid choices: TN, ED)"
        default = "TN"
        "--sim_method"
        help = "simulation method (valid choices: density_matrix, monte_carlo)"
        default = "monte_carlo"
        "--evolution_method"
        help = "evolution method (valid choices: continuous, trotter)"
        default = "continuous"
        "--n_trajectories"
        help = "number of trajectories for Monte Carlo wavefunction method"
        arg_type = Int
        default = 1
        "--peInt"
        help = "pe: noise strength (will be times by 1e-3)"
        arg_type = Int
        default = 0
        "--coupling"
        help = "coupling type"
        arg_type = String
        default = "XX"
        "--init_state"
        help = "initial state type: 'product' (default), 'identity' (maximally mixed; density matrix only), 'theta' (use --theta value), or 'ground' (system ground state)"
        arg_type = String
        default = "product"
        "--theta"
        help = "theta code parameter for initial state: -0.5 -> |0>, 0 -> |+>, 0.5 -> |1>"
        arg_type = Float64
        default = 0.0
        "--measure_modes"
        help = "record Bogoliubov mode observables h_k for Ising periodic/antiperiodic k-space diagnostics"
        action = :store_true
    end

    parsed_args = normalize_initial_state_args!(
        normalize_method_token_args!(parse_args(args, s))
    )
    return _validate_initial_state_args(parsed_args)
end
