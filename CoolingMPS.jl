include("Cooling.jl")

parsed_args = CoolingTNS.parse_commandline()
parsed_args["method"] = "MPS"
run_cooling(parsed_args)
