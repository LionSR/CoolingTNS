using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
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
        default = 50
        "--g"
        help = "coupling strength g"
        arg_type = Float64
        default = 0.3
        "--te"
        help = "total evolution time"
        arg_type = Float64
        default = 2.0
        "--tau"
        help = "Trotter step size"
        arg_type = Float64
        default = 0.1
        "--cutoff"
        help = "truncation error cutoff"
        arg_type = Float64
        default = 1E-4
        "--Dmax"
        help = "maximum bond dimension"
        arg_type = Int
        default = 20
        "--method"
        help = "simulation method for cooling (valid choices: MPS and MPO)"
        default = "MPS"
    end
    return parse_args(s)
end