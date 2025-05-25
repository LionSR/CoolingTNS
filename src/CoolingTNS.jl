module CoolingTNS

# Import ITensors package
using ITensors
using ITensorMPS

include("ham.jl")
include("dmrg.jl")
include("utils.jl")
include("utils_mps.jl")
include("utils_mpo.jl")
include("cooling_interface.jl")  # Define abstract types first
include("cooling_functions_mps.jl")
include("cooling_functions_mpo.jl")
include("cooling_functions_trotter_mps.jl")
include("cooling_functions_ed.jl")
include("plotting.jl")
include("policy.jl")
include("argparse.jl")
include("noise.jl")

export setup_problem, run_cooling_mps, run_cooling_trotter_mps, run_cooling_ed
export setup_problem_ed, setup_init_state_ed
export plot_data
export ham_ising_sys_bath, ham_niising_sys_bath

end
