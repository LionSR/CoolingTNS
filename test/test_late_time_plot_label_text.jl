using Test

normalize_ws(s::AbstractString) = replace(s, r"\s+" => " ")

@testset "Finite-window plotting labels are not steady-state claims" begin
    repo_root = joinpath(@__DIR__, "..")

    script_paths = [
        joinpath(repo_root, "scripts", "multi_freq_cooling.jl"),
        joinpath(repo_root, "scripts", "plotting", "randomized_time_resonance_figure.jl"),
        joinpath(repo_root, "scripts", "plotting", "plot_ground_state_cooling_ising_ed_dm.jl"),
        joinpath(repo_root, "scripts", "plotting", "plot_ground_state_cooling_ising_tn_mcwf_trotter.jl"),
        joinpath(repo_root, "scripts", "plotting", "run_ground_state_cooling_ising_tn_mcwf_trotter_multi_long.jl"),
        joinpath(repo_root, "scripts", "plotting", "scan_randomized_times_resonances_tn_ising_mcwf_trotter_N20.jl"),
    ]

    texts = Dict(path => normalize_ws(read(path, String)) for path in script_paths)
    combined_lower = lowercase(join(values(texts), "\n"))

    @test occursin("finite-window late-time averages", texts[joinpath(repo_root, "scripts", "multi_freq_cooling.jl")])
    @test occursin("tail-mean(last %d)/N", texts[joinpath(repo_root, "scripts", "multi_freq_cooling.jl")])
    @test occursin("finite-window late-time energy density", texts[joinpath(repo_root, "scripts", "plotting", "randomized_time_resonance_figure.jl")])
    @test occursin("late-time mean energy density", texts[joinpath(repo_root, "scripts", "plotting", "randomized_time_resonance_figure.jl")])
    @test occursin("tail-mean e(last %d)", texts[joinpath(repo_root, "scripts", "plotting", "plot_ground_state_cooling_ising_ed_dm.jl")])
    @test occursin("tail-mean e(last %d)", texts[joinpath(repo_root, "scripts", "plotting", "plot_ground_state_cooling_ising_tn_mcwf_trotter.jl")])
    @test occursin("tail-mean e(last %d)", texts[joinpath(repo_root, "scripts", "plotting", "run_ground_state_cooling_ising_tn_mcwf_trotter_multi_long.jl")])

    forbidden = [
        "steady-state energy density",
        "steady-state relative energy",
        "steady e(last",
        "steady(last",
        "strongly improves the steady-state energy",
        "summary comparing energies and (optionally) steady-state averages",
    ]
    @test all(!occursin(phrase, combined_lower) for phrase in forbidden)
end
