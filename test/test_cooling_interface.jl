using Test
using CoolingTNS

@testset "Cooling Interface Tests" begin
    # Test parameters
    N = 4
    problem = "niIsing"
    ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)  # N, J, hx, hz
    coupling_params = CoolingTNS.BasicCouplingParameters(
        "XX",    # coupling
        0.1,     # g
        5,       # steps
        1.0,     # te
        nothing  # delta (auto-compute)
    )

    @testset "Backend Creation" begin
        # Test new backend system
        @test CoolingTNS.get_backend("ED") isa CoolingTNS.EDBackend
        @test CoolingTNS.get_backend("TN") isa CoolingTNS.TNBackend
        @test_throws ErrorException CoolingTNS.get_backend("InvalidMethod")
    end

    @testset "Default Simulation Methods" begin
        @test CoolingTNS.default_simulation_method(CoolingTNS.EDBackend()) isa CoolingTNS.DensityMatrix
        @test CoolingTNS.default_simulation_method(CoolingTNS.TNBackend()) isa CoolingTNS.MonteCarloWavefunction
    end
    
    @testset "Default Evolution Methods" begin
        @test CoolingTNS.default_evolution_method(CoolingTNS.EDBackend()) isa CoolingTNS.ContinuousEvolution
        @test CoolingTNS.default_evolution_method(CoolingTNS.TNBackend()) isa CoolingTNS.ContinuousEvolution
    end

    @testset "Result Key Constants" begin
        @test CoolingTNS.RESULT_ENERGY == "E_list"
        @test CoolingTNS.RESULT_GROUND_STATE_OVERLAP == "GS_overlap_list"
        @test CoolingTNS.RESULT_PURITY == "purity_list"
        @test CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION == "momentum_dist"
        @test CoolingTNS.RESULT_K_VALUES == "k_values"
        @test CoolingTNS.RESULT_MODE_HK == "mode_hk"
        @test CoolingTNS.RESULT_MODE_NK == "mode_nk"
        @test CoolingTNS.RESULT_DELTA_LIST == "delta_list"
        @test CoolingTNS.RESULT_TE_LIST == "te_list"

        @test CoolingTNS.RESULT_KEYS isa Tuple
        @test all(key -> key isa String, CoolingTNS.RESULT_KEYS)
        @test length(unique(CoolingTNS.RESULT_KEYS)) == length(CoolingTNS.RESULT_KEYS)
        @test CoolingTNS.RESULT_ENERGY in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MODE_HK in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MODE_NK in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_DELTA_LIST in CoolingTNS.RESULT_KEYS
    end

    @testset "Problem Setup for Different Backends" begin
        # Create simulation parameters for each backend/method combination
        sim_params_ed = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0, n_trajectories=1
        )
        
        sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            Dmax=20, cutoff=1e-6, tau=0.1, pe=0.0, n_trajectories=10
        )
        
        backends_and_params = [
            (CoolingTNS.EDBackend(), sim_params_ed),
            (CoolingTNS.TNBackend(), sim_params_tn)
        ]
        
        for (backend, sim_params) in backends_and_params
            @testset "$(typeof(backend))" begin
                # Skip ED for larger systems
                test_N = backend isa CoolingTNS.EDBackend ? 3 : N
                test_ham_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
                
                problem_setup = CoolingTNS.setup_problem(
                    backend, test_ham_params, coupling_params, sim_params
                )
                
                @test problem_setup isa CoolingTNS.CoolingProblem
                @test problem_setup.backend === backend
                @test problem_setup.e₀ < 0  # Ground state energy should be negative
                @test !isnothing(problem_setup.H_sys)
                @test !isnothing(problem_setup.ϕ₀)
                
                # Check backend-specific fields
                if backend isa CoolingTNS.EDBackend
                    # ED backend doesn't use sites
                    @test !haskey(problem_setup.extra, :sites)
                else
                    # TN backend stores sites in extra
                    @test haskey(problem_setup.extra, :sites)
                    @test !isnothing(problem_setup.extra.sites)
                    @test length(problem_setup.extra.sites) == 2 * test_N
                end
            end
        end
    end

    @testset "Initial State Setup" begin
        # Test with TN backend
        backend = CoolingTNS.TNBackend()
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            Dmax=20, cutoff=1e-6, tau=0.1
        )
        
        problem_setup = CoolingTNS.setup_problem(
            backend, ham_params, coupling_params, sim_params
        )
        
        init_types = ["product", "identity", "theta"]
        theta_values = [0.0, -0.5, 0.5]
        
        for init_type in init_types
            for theta in theta_values
                if init_type != "theta" && theta != 0.0
                    continue  # Only test theta values with theta init type
                end
                
                initial_state = CoolingTNS.setup_initial_state(
                    problem_setup, sim_params, init_type, theta
                )
                
                @test initial_state isa CoolingTNS.QuantumState
                @test initial_state.backend === backend
                @test !isnothing(initial_state.state)
            end
        end
    end

    @testset "Full Cooling Simulation" begin
        # Test with small system using ED
        backend = CoolingTNS.EDBackend()
        test_N = 3
        test_ham_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
        
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0
        )
        
        problem_setup = CoolingTNS.setup_problem(
            backend, test_ham_params, coupling_params, sim_params
        )
        
        initial_state = CoolingTNS.setup_initial_state(
            problem_setup, sim_params, "product", 0.0
        )
        
        # Run short simulation
        short_coupling_params = CoolingTNS.BasicCouplingParameters(
            coupling_params.coupling,
            coupling_params.g,
            2,  # steps
            coupling_params.te,
            coupling_params.delta
        )
        
        results = CoolingTNS.run_cooling(
            problem_setup,
            initial_state,
            short_coupling_params,
            sim_params,
            test_ham_params
        )
        
        @test haskey(results, CoolingTNS.RESULT_ENERGY)
        @test haskey(results, CoolingTNS.RESULT_GROUND_STATE_OVERLAP)
        @test length(results[CoolingTNS.RESULT_ENERGY]) == short_coupling_params.steps + 1
        @test all(isfinite, results[CoolingTNS.RESULT_ENERGY])
        @test all(0 .<= results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] .<= 1)
        
        # Energy should decrease (cooling)
        @test results[CoolingTNS.RESULT_ENERGY][end] <= results[CoolingTNS.RESULT_ENERGY][1] + 1e-10
    end

    # Cross-backend cooling comparisons are covered in `test_correctness.jl`.
    # They are intentionally gated behind `ENV["COOLINGTNS_FULL_TESTS"]` since
    # Monte Carlo trajectories can be slow and inherently stochastic.
end
