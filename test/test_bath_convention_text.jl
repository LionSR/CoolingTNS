using Test

_bath_normalize_ws(s::AbstractString) = replace(s, r"\s+" => " ")

@testset "Bath convention text consistency" begin
    plan_path = joinpath(@__DIR__, "..", "docs", "multi_frequency_cooling_plan.md")
    coupling_path = joinpath(@__DIR__, "..", "src", "coupling_utils.jl")
    setup_path = joinpath(@__DIR__, "..", "src", "setup_system.jl")
    system_bath_path = joinpath(@__DIR__, "..", "src", "system_bath_hamiltonian.jl")
    mps_path = joinpath(@__DIR__, "..", "src", "utils_mps.jl")
    mpo_path = joinpath(@__DIR__, "..", "src", "utils_mpo.jl")
    module_path = joinpath(@__DIR__, "..", "src", "CoolingTNS.jl")

    plan_flat = _bath_normalize_ws(read(plan_path, String))
    coupling_text = read(coupling_path, String)
    setup_text = read(setup_path, String)
    system_bath_text = read(system_bath_path, String)
    mps_text = read(mps_path, String)
    mpo_text = read(mpo_path, String)
    module_text = read(module_path, String)

    @test occursin("The bath field is the Pauli operator returned by `get_bath_operator(coupling)`", plan_flat)
    @test occursin("For mixed symmetric labels, the bath-side set contains two Pauli operators", plan_flat)
    @test occursin("`XY`/`YX` use `Z`, `YZ`/`ZY` use `X`, and `XZ`/`ZX` use `Y`", plan_flat)
    @test occursin("guarantees noncommutation with every bath-side term", plan_flat)
    @test occursin("selected by `bath_ground_state_amplitudes`", plan_flat)

    @test occursin("get_bath_ground_state(coupling::String) = bath_ground_state_amplitudes(coupling)", coupling_text)
    @test !occursin("get_bath_ground_state(coupling::String) = bath_ground_state_amplitudes(coupling)", mps_text)
    @test occursin("coupling_utils.jl", mpo_text)
    @test occursin("export bath_ground_state_amplitudes, get_bath_ground_state", module_text)
    @test occursin("get_bath_operator(coupling)", setup_text)
    @test occursin("get_bath_operator is the source of truth", system_bath_text)

    @test !occursin("the bath field is chosen from the bath-side operator `B`", plan_flat)
    @test !occursin("present convention uses a Z bath field for bath-side X or Y couplings and an X", plan_flat)
    @test !occursin("bath ground state is Z=-1", setup_text)
end
