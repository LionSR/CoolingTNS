include("optCooling.jl")

parsed_args = CoolingTNS.parse_commandline()
parsed_args["method"] = "MPS"
run_optimization(parsed_args)

