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
                # Computational |0...0> state (theta = -0.5)
                state_zero = CoolingTNS.setup_initial_state(problem, sim_params, "theta", -0.5)
                
                # Computational |1...1> state (theta = 0.5)
                state_one = CoolingTNS.setup_initial_state(problem, sim_params, "theta", 0.5)
                
                # X+ state (theta = 0)
                state_plus = CoolingTNS.setup_initial_state(problem, sim_params, "theta", 0.0)
                
                @test state_zero.state isa MPS
                @test state_one.state isa MPS
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
                for theta in [-0.5, 0.0, 0.25, 0.5]
                    state = CoolingTNS.setup_initial_state(problem, sim_params, "theta", theta)
                    @test state.state isa MPO
                end
            end
        end
    end

    @testset "Theta Product Convention" begin
        test_N = 3
        θ_values = [-0.5, 0.0, 0.25, 0.5]
        sites = siteinds("S=1/2", test_N)

        for θ in θ_values
            amp0, amp1 = CoolingTNS._theta_site_amplitudes(θ)
            expected_z = amp0^2 - amp1^2
            expected_x = 2 * amp0 * amp1

            ψ_ed = CoolingTNS.create_theta_state_ed(test_N, "theta", θ)
            ψ_tn = CoolingTNS._theta_product_mps(sites, θ)

            @test abs(inner(ψ_tn, ψ_tn) - 1.0) < 1e-12
            @test maxlinkdim(ψ_tn) == 1

            @test CoolingTNS.expect_ed(CoolingTNS.pauli_z(1, test_N), ψ_ed) ≈ expected_z atol=1e-12
            @test CoolingTNS.expect_ed(CoolingTNS.pauli_x(1, test_N), ψ_ed) ≈ expected_x atol=1e-12
            @test expect(ψ_tn, "Z")[1] ≈ expected_z atol=1e-12
            @test expect(ψ_tn, "X")[1] ≈ expected_x atol=1e-12

            ρ_ed = CoolingTNS.state_to_density_ed(ψ_ed)
            ρ_tn = outer(ψ_tn', ψ_tn)
            z_terms = OpSum()
            x_terms = OpSum()
            z_terms += 1.0, "Z", 1
            x_terms += 1.0, "X", 1
            Z_tn = MPO(z_terms, sites)
            X_tn = MPO(x_terms, sites)

            @test real(inner(ρ_tn, Z_tn)) ≈ CoolingTNS.expect_ed(CoolingTNS.pauli_z(1, test_N), ρ_ed) atol=1e-12
            @test real(inner(ρ_tn, X_tn)) ≈ CoolingTNS.expect_ed(CoolingTNS.pauli_x(1, test_N), ρ_ed) atol=1e-12
        end
    end

    @testset "Theta Initial Energy Matches ED and TN" begin
        test_N = 4
        θ = 0.25
        ham_params = CoolingTNS.IsingParameters(test_N, 1.0, 0.7)

        ψ_ed = CoolingTNS.create_theta_state_ed(test_N, "theta", θ)
        H_ed = CoolingTNS.construct_system_hamiltonian(
            ham_params, CoolingTNS.EDBackend(), test_N
        )
        E_ed = CoolingTNS.expect_ed(H_ed, ψ_ed)

        sites = siteinds("S=1/2", test_N)
        ψ_tn = CoolingTNS._theta_product_mps(sites, θ)
        H_tn = CoolingTNS.construct_system_hamiltonian(
            ham_params, CoolingTNS.TNBackend(), sites
        )
        E_tn = real(inner(ψ_tn', H_tn, ψ_tn))

        @test E_tn ≈ E_ed atol=1e-10
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
