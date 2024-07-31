include("Cooling.jl")

parsed_args = CoolingTNS.parse_commandline()
parsed_args["method"] = "MPO"
run_cooling(parsed_args)
