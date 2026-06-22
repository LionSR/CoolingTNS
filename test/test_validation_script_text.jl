using Test

_validation_normalize_ws(s::AbstractString) = replace(s, r"\s+" => " ")

@testset "Validation script Trotter convention text" begin
    script_path = joinpath(@__DIR__, "..", "scripts", "validation", "plot_validation.jl")
    script_flat = _validation_normalize_ws(read(script_path, String))

    @test occursin("Figure 2: TN Gate Trotter Convergence", script_flat)
    @test occursin("function figure2_tn_trotter_convergence()", script_flat)
    @test occursin("Running ED continuous reference", script_flat)
    @test occursin("delta_ref = prob_ref.extra.coupling_params.delta", script_flat)
    @test occursin("BasicCouplingParameters(\"XX\", G, steps_trotter, TE, delta_ref)", script_flat)
    @test occursin("tau_values = [0.5, 0.2, 0.1, 0.05]", script_flat)
    @test occursin("Using shared bath detuning Delta=", script_flat)
    @test occursin("Running TN DM+Trotter tau=", script_flat)
    @test occursin("ED continuous reference", script_flat)
    @test occursin("TN gate approximation error", script_flat)
    @test occursin("validation_tn_trotter_convergence.pdf", script_flat)
    @test occursin("figure2_tn_trotter_convergence()", script_flat)
    @test occursin("Figure 3: Cross-Backend ED vs TN", script_flat)
    @test occursin("cp_cross_auto = CoolingTNS.BasicCouplingParameters(\"XX\", G, STEPS, TE, nothing)", script_flat)
    @test occursin("delta_cross = prob_cross.extra.coupling_params.delta", script_flat)
    @test occursin("BasicCouplingParameters(\"XX\", G, STEPS, TE, delta_cross)", script_flat)
    @test occursin("Using shared cross-backend bath detuning Delta=", script_flat)
    @test occursin("ham_params=ham_cross, coupling_params=cp_cross, Dmax=100", script_flat)

    @test !occursin("Figure 2: Trotter Convergence (ED DM", script_flat)
    @test !occursin("Running reference ED DM+Continuous", script_flat)
    @test !occursin("tau_values = [0.4, 0.2, 0.1, 0.05]", script_flat)
    @test !occursin("Running ED DM+Trotter tau=", script_flat)
    @test !occursin("cp = CoolingTNS.BasicCouplingParameters(\"XX\", G, steps_trotter, TE, nothing)", script_flat)
    @test !occursin("Trotter convergence order", script_flat)
    @test !occursin("Final energy vs Trotter step", script_flat)
    @test !occursin("validation_trotter_convergence.pdf", script_flat)
    @test !occursin("cp = CoolingTNS.BasicCouplingParameters(\"XX\", G, STEPS, TE, nothing)", script_flat)
    @test !occursin("ham_params=ham_cross, coupling_params=cp, Dmax=100", script_flat)
end
