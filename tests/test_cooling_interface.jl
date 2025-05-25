using Test
using CoolingTNS

@testset "Cooling Interface Tests" begin
    # Test parameters
    N = 4
    problem = "niIsing"
    ham_params = (1.0, -1.05, 0.5)  # J, hx, hz
    coupling_params = Dict(
        "coupling" => "XX",
        "g" => 0.1,
        "te" => 1.0,
        "steps" => 5
    )
    sim_params = Dict(
        "pe" => 0.0,
        "cutoff" => 1e-6,
        "Dmax" => 20,
        "tau" => 0.1,
        "n_trajectories" => 10
    )

    @testset "Backend Creation" begin
        @test CoolingTNS.get_backend("ED") isa CoolingTNS.EDBackend
        @test CoolingTNS.get_backend("MPS") isa CoolingTNS.MPSBackend
        @test CoolingTNS.get_backend("MPO") isa CoolingTNS.MPOBackend
        @test CoolingTNS.get_backend("TrotterMPS") isa CoolingTNS.TrotterMPSBackend
        @test_throws ErrorException CoolingTNS.get_backend("InvalidMethod")
    end

    @testset "Simulation Method Mapping" begin
        @test CoolingTNS.simulation_method(CoolingTNS.EDBackend()) isa CoolingTNS.DensityMatrix
        @test CoolingTNS.simulation_method(CoolingTNS.MPSBackend()) isa CoolingTNS.MonteCarloWavefunction
        @test CoolingTNS.simulation_method(CoolingTNS.MPOBackend()) isa CoolingTNS.DensityMatrix
        @test CoolingTNS.simulation_method(CoolingTNS.TrotterMPSBackend()) isa CoolingTNS.MonteCarloWavefunction
    end

    @testset "Problem Setup for Different Backends" begin
        backends = [
            CoolingTNS.EDBackend(),
            CoolingTNS.MPSBackend(),
            CoolingTNS.MPOBackend(),
            CoolingTNS.TrotterMPSBackend()
        ]
        
        for backend in backends
            @testset "$(typeof(backend))" begin
                # Skip ED for larger systems
                test_N = backend isa CoolingTNS.EDBackend ? 3 : N
                
                problem_setup = CoolingTNS.setup_problem(
                    backend, test_N, problem, ham_params, coupling_params, sim_params
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
        # Test with MPS backend
        backend = CoolingTNS.MPSBackend()
        problem_setup = CoolingTNS.setup_problem(
            backend, N, problem, ham_params, coupling_params, sim_params
        )
        
        init_types = ["product", "identity", "theta"]
        theta_values = [0.0, -0.5, 0.5]
        
        for init_type in init_types
            for theta in theta_values
                if init_type != "theta" && theta != 0.0
                    continue  # Only test theta values with theta init type
                end
                
                initial_state = CoolingTNS.setup_initial_state(
                    problem_setup, init_type, theta
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
        
        problem_setup = CoolingTNS.setup_problem(
            backend, test_N, problem, ham_params, coupling_params, sim_params
        )
        
        initial_state = CoolingTNS.setup_initial_state(
            problem_setup, "product", 0.0
        )
        
        # Run short simulation
        short_coupling_params = copy(coupling_params)
        short_coupling_params["steps"] = 2
        
        results = CoolingTNS.run_cooling(
            problem_setup,
            initial_state,
            short_coupling_params,
            sim_params
        )
        
        @test haskey(results, "E_list")
        @test haskey(results, "GS_overlap_list")
        @test length(results["E_list"]) == short_coupling_params["steps"] + 1
        @test all(isfinite, results["E_list"])
        @test all(0 .<= results["GS_overlap_list"] .<= 1)
        
        # Energy should decrease (cooling)
        @test results["E_list"][end] <= results["E_list"][1] + 1e-10
    end

    @testset "Cross-Backend Consistency" begin
        # Compare results between ED and MPS for very small system
        test_N = 2
        test_steps = 3
        
        short_coupling_params = copy(coupling_params)
        short_coupling_params["steps"] = test_steps
        
        backends = [CoolingTNS.EDBackend(), CoolingTNS.MPSBackend()]
        results_dict = Dict()
        
        for backend in backends
            problem_setup = CoolingTNS.setup_problem(
                backend, test_N, problem, ham_params, short_coupling_params, sim_params
            )
            
            initial_state = CoolingTNS.setup_initial_state(
                problem_setup, "product", 0.0
            )
            
            results = CoolingTNS.run_cooling(
                problem_setup,
                initial_state,
                short_coupling_params,
                sim_params,
                ham_params
            )
            
            results_dict[typeof(backend)] = results
        end
        
        # Compare ground state energies
        ed_results = results_dict[CoolingTNS.EDBackend]
        mps_results = results_dict[CoolingTNS.MPSBackend]
        
        # Ground state energies should be very close
        @test abs(ed_results["E_list"][1] - mps_results["E_list"][1]) < 0.1
        
        # Both should show cooling (energy decrease)
        @test ed_results["E_list"][end] < ed_results["E_list"][1]
        @test mps_results["E_list"][end] < mps_results["E_list"][1]
    end
end