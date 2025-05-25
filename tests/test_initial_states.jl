using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Initial State Tests" begin
    N = 4
    
    @testset "Initial State Dispatch Tests" begin
        # Create test problem setups for different backends
        ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 10, 1.0, nothing)
        
        @testset "TN Backend - Monte Carlo" begin
            backend = CoolingTNS.TNBackend()
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.MonteCarloWavefunction(),
                CoolingTNS.ContinuousEvolution();
                Dmax=20, cutoff=1e-6
            )
            
            problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
            
            @testset "Product State" begin
                state = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
                @test state isa CoolingTNS.QuantumState
                @test state.state isa MPS
                @test length(state.state) == N
                @test maxlinkdim(state.state) == 1  # Product state has bond dimension 1
            end
            
            @testset "Identity State" begin
                state = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
                @test state isa CoolingTNS.QuantumState
                @test state.state isa MPS
                @test length(state.state) == N
            end
            
            @testset "Theta States" begin
                # All down state (theta = -0.5π)
                state_down = CoolingTNS.setup_initial_state(problem, sim_params, "theta", -0.5)
                
                # All up state (theta = 0.5π)
                state_up = CoolingTNS.setup_initial_state(problem, sim_params, "theta", 0.5)
                
                # X+ state (theta = 0)
                state_plus = CoolingTNS.setup_initial_state(problem, sim_params, "theta", 0.0)
                
                @test state_down.state isa MPS
                @test state_up.state isa MPS
                @test state_plus.state isa MPS
            end
        end
        
        @testset "TN Backend - Density Matrix" begin
            backend = CoolingTNS.TNBackend()
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.DensityMatrix(),
                CoolingTNS.TrotterEvolution();
                Dmax=20, cutoff=1e-6, tau=0.1
            )
            
            problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
            
            @testset "Product State MPO" begin
                state = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
                @test state isa CoolingTNS.QuantumState
                @test state.state isa MPO
                @test length(state.state) == N
            end
            
            @testset "Identity State MPO" begin
                state = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
                @test state isa CoolingTNS.QuantumState
                @test state.state isa MPO
                @test length(state.state) == N
            end
            
            @testset "Theta States MPO" begin
                # Test different theta values
                for theta in [-0.5, 0.0, 0.5]
                    state = CoolingTNS.setup_initial_state(problem, sim_params, "theta", theta)
                    @test state.state isa MPO
                end
            end
        end
        
        @testset "ED Backend - Monte Carlo" begin
            backend = CoolingTNS.EDBackend()
            test_N = 3  # Smaller for ED
            ham_params_ed = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
            
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.MonteCarloWavefunction(),
                CoolingTNS.ContinuousEvolution();
                pe=0.0
            )
            
            problem = CoolingTNS.setup_problem(backend, ham_params_ed, coupling_params, sim_params)
            
            @testset "Product State" begin
                state = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
                @test state isa CoolingTNS.QuantumState
                @test state.state isa CoolingTNS.Yao.ArrayReg
            end
            
            @testset "Theta States" begin
                # All down
                state_down = CoolingTNS.setup_initial_state(problem, sim_params, "theta", -0.5)
                @test state_down.state isa CoolingTNS.Yao.ArrayReg
                
                # All up
                state_up = CoolingTNS.setup_initial_state(problem, sim_params, "theta", 0.5)
                @test state_up.state isa CoolingTNS.Yao.ArrayReg
                
                # X+ state
                state_plus = CoolingTNS.setup_initial_state(problem, sim_params, "theta", 0.0)
                @test state_plus.state isa CoolingTNS.Yao.ArrayReg
            end
        end
        
        @testset "ED Backend - Density Matrix" begin
            backend = CoolingTNS.EDBackend()
            test_N = 3  # Smaller for ED
            ham_params_ed = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
            
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.DensityMatrix(),
                CoolingTNS.ContinuousEvolution();
                pe=0.0
            )
            
            problem = CoolingTNS.setup_problem(backend, ham_params_ed, coupling_params, sim_params)
            
            @testset "Identity State" begin
                state = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
                @test state isa CoolingTNS.QuantumState
                # For ED density matrix, it returns a special state type
                @test !isnothing(state.state)
            end
            
            @testset "Product State" begin
                state = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
                @test state isa CoolingTNS.QuantumState
                @test state.state isa CoolingTNS.Yao.DensityMatrix
            end
        end
    end
    
    @testset "State Properties" begin
        # Test that initial states have expected properties
        test_N = 3
        ham_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 10, 1.0, nothing)
        
        @testset "TN Backend Properties" begin
            backend = CoolingTNS.TNBackend()
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.MonteCarloWavefunction(),
                CoolingTNS.ContinuousEvolution();
                Dmax=20, cutoff=1e-6
            )
            
            problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
            
            # Product state should be normalized
            state = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
            ψ = state.state
            @test abs(inner(ψ, ψ) - 1.0) < 1e-10
            
            # Identity state (for MPS) should be normalized
            state_id = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
            ψ_id = state_id.state
            @test abs(inner(ψ_id, ψ_id) - 1.0) < 1e-10
        end
    end
    
    @testset "Dispatch Consistency" begin
        # Verify that dispatch correctly routes to appropriate implementations
        test_N = 2
        ham_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 10, 1.0, nothing)
        
        # Create different simulation parameter combinations
        sim_params_combinations = [
            (CoolingTNS.MonteCarloWavefunction(), CoolingTNS.ContinuousEvolution()),
            (CoolingTNS.MonteCarloWavefunction(), CoolingTNS.TrotterEvolution()),
            (CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution()),
            (CoolingTNS.DensityMatrix(), CoolingTNS.TrotterEvolution())
        ]
        
        for (sim_method, evo_method) in sim_params_combinations
            @testset "$(typeof(sim_method)) + $(typeof(evo_method))" begin
                # TN Backend
                sim_params = CoolingTNS.UnifiedSimulationParameters(
                    sim_method, evo_method;
                    Dmax=10, cutoff=1e-6, tau=0.1
                )
                
                backend = CoolingTNS.TNBackend()
                problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
                state = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
                
                @test state isa CoolingTNS.QuantumState
                @test state.sim_method === sim_method
                @test state.evolution_method === evo_method
                
                # Check state type based on simulation method
                if sim_method isa CoolingTNS.MonteCarloWavefunction
                    @test state.state isa MPS
                else  # DensityMatrix
                    @test state.state isa MPO
                end
            end
        end
    end
end