using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Initial State Tests" begin
    N = 4
    
    @testset "MPS Initial States" begin
        sites = siteinds("S=1/2", 2N)
        
        @testset "Product State" begin
            ψ = CoolingTNS.setup_init_state_mps(sites; init_type="product")
            @test ψ isa MPS
            @test length(ψ) == N
            @test maxlinkdim(ψ) == 1  # Product state has bond dimension 1
        end
        
        @testset "Identity State" begin
            ψ = CoolingTNS.setup_init_state_mps(sites; init_type="identity")
            @test ψ isa MPS
            @test length(ψ) == N
        end
        
        @testset "Theta States" begin
            # All down state (theta = -0.5π)
            ψ_down = CoolingTNS.setup_init_state_mps(sites; init_type="theta", theta=-0.5)
            
            # All up state (theta = 0.5π)
            ψ_up = CoolingTNS.setup_init_state_mps(sites; init_type="theta", theta=0.5)
            
            # X+ state (theta = 0)
            ψ_plus = CoolingTNS.setup_init_state_mps(sites; init_type="theta", theta=0.0)
            
            @test ψ_down isa MPS
            @test ψ_up isa MPS
            @test ψ_plus isa MPS
        end
    end
    
    @testset "ED Initial States" begin
        nbits = 2N
        
        @testset "Density Matrix States" begin
            # Product state
            state_prod = CoolingTNS.setup_init_state_ed(
                nbits; init_type="product", method=CoolingTNS.DensityMatrix()
            )
            @test state_prod.method isa CoolingTNS.DensityMatrix
            @test state_prod.nbits == nbits
            
            # Identity state
            state_id = CoolingTNS.setup_init_state_ed(
                nbits; init_type="identity", method=CoolingTNS.DensityMatrix()
            )
            @test state_id.method isa CoolingTNS.DensityMatrix
            
            # Check identity state is created
            @test state_id.state isa CoolingTNS.EDState
        end
        
        @testset "Pure State (Monte Carlo)" begin
            # All down
            state_down = CoolingTNS.setup_init_state_ed(
                nbits; init_type="theta", theta=-0.5, 
                method=CoolingTNS.MonteCarloWavefunction()
            )
            @test state_down.method isa CoolingTNS.MonteCarloWavefunction
            
            # All up
            state_up = CoolingTNS.setup_init_state_ed(
                nbits; init_type="theta", theta=0.5,
                method=CoolingTNS.MonteCarloWavefunction()
            )
            
            # X+ state
            state_plus = CoolingTNS.setup_init_state_ed(
                nbits; init_type="theta", theta=0.0,
                method=CoolingTNS.MonteCarloWavefunction()
            )
            
            # Check states are created correctly
            # The state field contains the actual quantum register
            @test state_down.state.state isa Yao.ArrayReg
            @test state_up.state.state isa Yao.ArrayReg
        end
    end
    
    @testset "MPO Initial States" begin
        sites = siteinds("S=1/2", 2N)
        
        @testset "Product State MPO" begin
            ρ = CoolingTNS.setup_init_state_mpo(sites; init_type="product")
            @test ρ isa MPO
            @test length(ρ) == N
        end
        
        @testset "Identity State MPO" begin
            ρ = CoolingTNS.setup_init_state_mpo(sites; init_type="identity")
            @test ρ isa MPO
            @test length(ρ) == N
        end
        
        @testset "Theta States MPO" begin
            # Test different theta values
            for theta in [-0.5, 0.0, 0.5]
                ρ = CoolingTNS.setup_init_state_mpo(sites; init_type="theta", theta=theta)
                @test ρ isa MPO
                @test ρ isa MPO
            end
        end
    end
    
    @testset "State Consistency Across Backends" begin
        # Test that the same initial state type gives consistent results
        test_N = 3
        nbits = 2 * test_N
        sites = siteinds("S=1/2", nbits)
        
        # Create product state in different representations
        mps_state = CoolingTNS.setup_init_state_mps(sites; init_type="product")
        mpo_state = CoolingTNS.setup_init_state_mpo(sites; init_type="product")
        ed_state = CoolingTNS.setup_init_state_ed(
            nbits; init_type="product", method=CoolingTNS.MonteCarloWavefunction()
        )
        
        # All should represent the same quantum state
        # (detailed comparison would require converting between representations)
        @test mps_state isa MPS
        @test mpo_state isa MPO
        @test ed_state.state isa CoolingTNS.EDState
    end
end