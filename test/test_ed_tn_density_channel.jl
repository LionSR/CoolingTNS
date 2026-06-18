using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

@isdefined(test_mpo_to_matrix) || include("test_helpers.jl")

# XZ selects the Y-bath ground state, which has complex amplitudes. Setting
# J = 0 removes system-system Trotter splitting error, so this test isolates
# row/column conjugation errors in MPO bath-density assembly.
@testset "ED and TN density-channel convention" begin
    ham_params = CoolingTNS.NiIsingParameters(2, 0.0, -0.7, 0.3)
    coupling_params = CoolingTNS.BasicCouplingParameters("XZ", 0.23, 1, 0.4, 0.9)

    sim_ed = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(),
        CoolingTNS.TrotterEvolution();
        tau=coupling_params.te,
        pe=0.0,
    )
    sim_tn = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(),
        CoolingTNS.TrotterEvolution();
        tau=coupling_params.te,
        Dmax=32,
        cutoff=1e-14,
        pe=0.0,
    )
    sim_mps = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.MonteCarloWavefunction(),
        CoolingTNS.TrotterEvolution();
        tau=coupling_params.te,
        Dmax=32,
        cutoff=1e-14,
        pe=0.0,
    )

    problem_ed = CoolingTNS.setup_problem(
        CoolingTNS.EDBackend(), ham_params, coupling_params, sim_ed
    )
    problem_tn = CoolingTNS.setup_problem(
        CoolingTNS.TNBackend(), ham_params, coupling_params, sim_tn
    )

    @test problem_ed.extra.coupling_params.delta == coupling_params.delta
    @test problem_tn.extra.coupling_params.delta == coupling_params.delta
    @test problem_ed.e₀ ≈ problem_tn.e₀ atol=1e-10

    state_ed = CoolingTNS.setup_initial_state(problem_ed, sim_ed, "theta", -0.2)
    state_tn = CoolingTNS.setup_initial_state(problem_tn, sim_tn, "theta", -0.2)
    ψ_sys_tn = CoolingTNS._theta_product_mps(siteinds(problem_tn.ϕ₀), -0.2)

    ρ_sb_ed = CoolingTNS.prepare_combined_state(problem_ed, state_ed)
    ρ_sb_tn = CoolingTNS.prepare_combined_state(problem_tn, state_tn)
    ψ_sb_tn = CoolingTNS.appendzeros_MPS(
        ψ_sys_tn, problem_tn.extra.sites, coupling_params.coupling
    )
    @test test_mpo_to_matrix(ρ_sb_tn) ≈ ρ_sb_ed.data atol=1e-12

    ρ_evolved_ed = CoolingTNS.evolve_cooling_step(
        problem_ed, ρ_sb_ed, coupling_params.te, sim_ed, ham_params
    )
    ρ_evolved_tn = CoolingTNS.evolve_cooling_step(
        problem_tn, ρ_sb_tn, coupling_params.te, sim_tn, ham_params
    )
    ψ_evolved_tn = CoolingTNS.evolve_state(
        ham_params,
        sim_mps,
        CoolingTNS.TNBackend(),
        nothing,
        ψ_sb_tn,
        coupling_params.te,
        problem_tn.extra.sites;
        gates=problem_tn.extra.interleaved_gates,
    )
    @test test_mpo_to_matrix(ρ_evolved_tn) ≈ ρ_evolved_ed.data atol=1e-10
    @test real(inner(ψ_evolved_tn', ρ_evolved_tn, ψ_evolved_tn)) ≈ 1.0 atol=1e-10

    ρ_s_ed = CoolingTNS.trace_out_bath_ed(ρ_evolved_ed, ham_params.N)
    state_s_tn, bath_mag_tn = CoolingTNS.process_bath_and_update(
        problem_tn, ρ_evolved_tn, state_tn, sim_tn
    )
    ρ_s_tn = test_mpo_to_matrix(state_s_tn.state)

    @test ρ_s_tn ≈ ρ_s_ed.data atol=1e-10
    @test tr(ρ_s_tn) ≈ 1.0 atol=1e-12
    @test ρ_s_tn ≈ ρ_s_tn' atol=1e-12
    @test isfinite(bath_mag_tn)

    H_sys_ed = Matrix(problem_ed.H_sys)
    E_ed = real(tr(H_sys_ed * ρ_s_ed.data))
    E_tn = real(inner(state_s_tn.state, problem_tn.H_sys))
    @test E_tn ≈ E_ed atol=1e-10
end
