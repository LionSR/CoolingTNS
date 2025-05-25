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
                    @test isnothing(problem_setup.sites)
                else
                    @test !isnothing(problem_setup.sites)
                    @test length(problem_setup.sites) == 2 * test_N
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
        
        @test haskey(results, "E_list")
        @test haskey(results, "GS_overlap_list")
        @test length(results["E_list"]) == short_coupling_params.steps + 1
        @test all(isfinite, results["E_list"])
        @test all(0 .<= results["GS_overlap_list"] .<= 1)
        
        # Energy should decrease (cooling)
        @test results["E_list"][end] <= results["E_list"][1] + 1e-10
    end

    @testset "Cross-Backend Consistency" begin
        # Compare results between ED and TN for very small system
        test_N = 2
        test_steps = 3
        test_ham_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
        
        short_coupling_params = CoolingTNS.BasicCouplingParameters(
            coupling_params.coupling,
            coupling_params.g,
            test_steps,
            coupling_params.te,
            coupling_params.delta
        )
        
        # Set up backends with their appropriate simulation parameters
        backends_and_params = [
            (CoolingTNS.EDBackend(), CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.DensityMatrix(),
                CoolingTNS.ContinuousEvolution();
                pe=0.0
            )),
            (CoolingTNS.TNBackend(), CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.MonteCarloWavefunction(),
                CoolingTNS.ContinuousEvolution();
                Dmax=20, cutoff=1e-6, tau=0.1, pe=0.0, n_trajectories=1
            ))
        ]
        
        results_dict = Dict()
        
        for (backend, sim_params) in backends_and_params
            problem_setup = CoolingTNS.setup_problem(
                backend, test_ham_params, short_coupling_params, sim_params
            )
            
            initial_state = CoolingTNS.setup_initial_state(
                problem_setup, sim_params, "product", 0.0
            )
            
            results = CoolingTNS.run_cooling(
                problem_setup,
                initial_state,
                short_coupling_params,
                sim_params,
                test_ham_params
            )
            
            results_dict[typeof(backend)] = results
        end
        
        # Compare ground state energies
        ed_results = results_dict[CoolingTNS.EDBackend]
        tn_results = results_dict[CoolingTNS.TNBackend]
        
        # Ground state energies should be very close
        @test abs(ed_results["E_list"][1] - tn_results["E_list"][1]) < 0.1
        
        # Both should show cooling (energy decrease)
        @test ed_results["E_list"][end] < ed_results["E_list"][1]
        @test tn_results["E_list"][end] < tn_results["E_list"][1]
    end
end