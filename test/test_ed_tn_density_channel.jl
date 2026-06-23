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

@testset "Ground-state fixed-point defect of the cooling channel" begin
    ham_params = CoolingTNS.NiIsingParameters(2, 1.0, -1.05, 0.5; bc=:open)
    coupling_params = CoolingTNS.BasicCouplingParameters("ZZ", 0.2, 1, 1.0, nothing)

    function ground_density_channel_diagnostics(backend, evolution_method)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            evolution_method;
            Dmax=64,
            cutoff=1e-13,
            tau=0.02,
            pe=0.0,
        )
        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "ground", 0.0)
        ρ_sb = CoolingTNS.prepare_combined_state(problem, state0)
        ρ_evolved = CoolingTNS.evolve_cooling_step(
            problem,
            ρ_sb,
            problem.extra.coupling_params.te,
            sim_params,
            ham_params,
        )
        state1, _ = CoolingTNS.process_bath_and_update(problem, ρ_evolved, state0, sim_params)

        ρ0, ρ1, H = if backend isa CoolingTNS.EDBackend
            (
                CoolingTNS.state_to_density_ed(problem.ϕ₀).data,
                state1.state.data,
                Matrix(problem.H_sys),
            )
        else
            (
                test_mpo_to_matrix(state0.state),
                test_mpo_to_matrix(state1.state),
                test_mpo_to_matrix(problem.H_sys),
            )
        end
        processed_system_sites = backend isa CoolingTNS.EDBackend ?
            state1.state.n_qubits :
            length(state1.state)

        return (
            problem=problem,
            processed_system_sites=processed_system_sites,
            initial_trace=tr(ρ0),
            final_trace=tr(ρ1),
            initial_energy=real(tr(H * ρ0)),
            final_energy=real(tr(H * ρ1)),
            initial_purity=real(tr(ρ0 * ρ0)),
            final_overlap=real(tr(ρ0 * ρ1)),
            trace_norm_defect=sum(svdvals(ρ1 - ρ0)),
        )
    end

    diag_ed = ground_density_channel_diagnostics(
        CoolingTNS.EDBackend(),
        CoolingTNS.ContinuousEvolution(),
    )
    diag_tn = ground_density_channel_diagnostics(
        CoolingTNS.TNBackend(),
        CoolingTNS.TrotterEvolution(),
    )

    problem_ed = diag_ed.problem
    problem_tn = diag_tn.problem

    @test problem_ed.e₀ ≈ problem_tn.e₀ atol=1e-10
    @test diag_ed.processed_system_sites == ham_params.N
    @test diag_tn.processed_system_sites == ham_params.N
    @test diag_ed.initial_trace ≈ 1.0 + 0.0im atol=1e-12
    @test diag_tn.initial_trace ≈ 1.0 + 0.0im atol=1e-10
    @test diag_ed.final_trace ≈ 1.0 + 0.0im atol=1e-12
    @test diag_tn.final_trace ≈ 1.0 + 0.0im atol=1e-10
    @test diag_ed.initial_energy ≈ problem_ed.e₀ atol=1e-12
    @test diag_tn.initial_energy ≈ problem_tn.e₀ atol=1e-10
    @test diag_ed.initial_purity ≈ 1.0 atol=1e-12
    @test diag_tn.initial_purity ≈ 1.0 atol=1e-10

    drift_ed = diag_ed.final_energy - problem_ed.e₀
    drift_tn = diag_tn.final_energy - problem_tn.e₀
    overlap_loss_ed = 1 - diag_ed.final_overlap
    overlap_loss_tn = 1 - diag_tn.final_overlap

    # This is a map diagnostic, not a numerical target for cooling.  For these
    # parameters the exact density channel does not leave the system ground
    # state invariant, and the TN channel reproduces the same one-cycle defect.
    @test diag_ed.trace_norm_defect > 1e-2
    @test drift_ed > 1e-2
    @test overlap_loss_ed > 1e-2

    # The TN-ED tolerance is calibrated for this N=2, g=0.2, tau=0.02
    # diagnostic.  It is meant to sit well below the O(1e-2) fixed-point defect,
    # not to give a model-independent Trotter error bound.
    @test diag_tn.trace_norm_defect ≈ diag_ed.trace_norm_defect atol=5e-5
    @test drift_tn ≈ drift_ed atol=5e-5
    @test overlap_loss_tn ≈ overlap_loss_ed atol=5e-5
end
