include("optCooling.jl")

parsed_args = CoolingTNS.parse_commandline()
parsed_args["method"] = "MPO"
run_optimization(parsed_args)
