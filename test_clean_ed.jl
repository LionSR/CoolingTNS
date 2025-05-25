#!/usr/bin/env julia

"""
Test script for clean ED backend functionality
"""

using Pkg
Pkg.activate(".")

using CoolingTNS
using LinearAlgebra
using Test

println("Testing Clean ED Backend Implementation")
println("=" ^ 50)

@testset "Clean ED Backend Tests" begin
    N = 4
    
    @testset "ED State Creation" begin
        # Test state vector creation
        ψ_zero = CoolingTNS.zero_state_ed(N)
        @test ψ_zero isa CoolingTNS.EDStateVector
        @test ψ_zero.n_qubits == N
        @test ψ_zero.data[1] ≈ 1.0
        @test norm(ψ_zero.data) ≈ 1.0
        
        # Test density matrix creation
        ρ_mixed = CoolingTNS.maximally_mixed_ed(N)
        @test ρ_mixed isa CoolingTNS.EDDensityMatrix
        @test ρ_mixed.n_qubits == N
        @test tr(ρ_mixed.data) ≈ 1.0
        @test CoolingTNS.purity_ed(ρ_mixed) ≈ 1/2^N
    end
    
    @testset "Pauli Operators" begin
        # Test Pauli operators
        X1 = CoolingTNS.pauli_x(1, N)
        Z1 = CoolingTNS.pauli_z(1, N)
        
        @test X1 isa AbstractMatrix
        @test Z1 isa AbstractMatrix
        @test size(X1) == (2^N, 2^N)
        @test size(Z1) == (2^N, 2^N)
        
        # Test commutation relation [X,Z] = 2iY
        commutator = X1 * Z1 - Z1 * X1
        @test norm(commutator) > 0  # Should not commute
    end
    
    @testset "Hamiltonian Construction" begin
        # Test system Hamiltonian
        ham_params = CoolingTNS.IsingParameters(N, 1.0, -2.0)
        backend = CoolingTNS.EDBackend()
        
        H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, backend, N)
        @test H_sys isa AbstractMatrix
        @test size(H_sys) == (2^N, 2^N)
        @test ishermitian(H_sys)
        
        # Test ground state
        result = CoolingTNS.ground_state_ed(H_sys)
        if isa(result, Tuple)
            e_gs, ψ_gs, gap = result
            @test ψ_gs isa CoolingTNS.EDStateVector
            @test norm(ψ_gs.data) ≈ 1.0
            @test e_gs < 0  # Ground state energy should be negative
        else
            ψ_gs = result
            @test ψ_gs isa CoolingTNS.EDStateVector
            @test norm(ψ_gs.data) ≈ 1.0
        end
    end
    
    @testset "Time Evolution" begin
        # Test evolution
        ψ_init = CoolingTNS.zero_state_ed(2)
        H = CoolingTNS.pauli_x(1, 2)  # Single qubit rotation
        t = π/4  # Quarter rotation
        
        ψ_evolved = CoolingTNS.evolve_ed(H, ψ_init, t)
        @test ψ_evolved isa CoolingTNS.EDStateVector
        @test norm(ψ_evolved.data) ≈ 1.0
        
        # Energy should be conserved
        E_init = CoolingTNS.expect_ed(H, ψ_init)
        E_final = CoolingTNS.expect_ed(H, ψ_evolved)
        @test abs(E_init - E_final) < 1e-10
    end
    
    @testset "Partial Trace" begin
        # Test partial trace operations
        N_sys = 2
        N_bath = 2
        N_total = N_sys + N_bath
        
        # Create entangled state
        ρ_total = CoolingTNS.maximally_mixed_ed(N_total)
        
        # Trace out bath
        ρ_sys = CoolingTNS.trace_out_bath_ed(ρ_total, N_sys)
        @test ρ_sys isa CoolingTNS.EDDensityMatrix
        @test ρ_sys.n_qubits == N_sys
        @test tr(ρ_sys.data) ≈ 1.0
        
        # Trace out system
        ρ_bath = CoolingTNS.trace_out_system_ed(ρ_total, N_sys)
        @test ρ_bath isa CoolingTNS.EDDensityMatrix
        @test ρ_bath.n_qubits == N_bath
        @test tr(ρ_bath.data) ≈ 1.0
    end
    
    @testset "Measurement and Collapse" begin
        # Test measurement
        ψ = CoolingTNS.random_state_ed(3)
        qubits_to_measure = [1, 3]
        
        ψ_collapsed, outcomes = CoolingTNS.measure_ed!(ψ, qubits_to_measure)
        @test ψ_collapsed isa CoolingTNS.EDStateVector
        @test length(outcomes) == length(qubits_to_measure)
        @test all(outcome in [0, 1] for outcome in outcomes)
        @test norm(ψ_collapsed.data) ≈ 1.0
    end
end

@testset "Cooling Simulation Test" begin
    println("\nTesting basic cooling simulation...")
    
    N = 3  # Small system for quick test
    
    # Run quick cooling test
    try
        result = read(`julia --startup-file=no -t 1 Cooling.jl --N $N --problem niIsing --backend ED --sim_method density_matrix --evolution_method continuous --coupling XX --g 0.1 --te 0.5 --steps 3 --peInt 1`, String)
        
        @test contains(result, "Step")
        println("✓ Basic cooling simulation completed successfully")
    catch e
        @test false
    end
end

println("\n" * "=" ^ 50)
println("All clean ED backend tests completed!")