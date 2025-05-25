using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Hamiltonian Construction Tests" begin
    N = 4
    
    @testset "Parse Coupling" begin
        @test CoolingTNS.parse_coupling("XX") == ("X", "X")
        @test CoolingTNS.parse_coupling("YZ") == ("Y", "Z")
        @test CoolingTNS.parse_coupling("ZY") == ("Z", "Y")
        @test_throws ArgumentError CoolingTNS.parse_coupling("XXX")
        @test_throws ArgumentError CoolingTNS.parse_coupling("XA")
    end
    
    @testset "System Hamiltonians" begin
        sites = siteinds("S=1/2", N)
        
        @testset "Ising Model" begin
            J, h = 1.0, -2.0
            ham_params = (J, h)
            H = CoolingTNS.ham_ising(N, sites, ham_params)
            
            @test H isa MPO
            @test length(H) == N
            
            # Check energy scale is reasonable
            # Create a random state and compute energy
            ψ = randomMPS(sites, linkdims=4)
            normalize!(ψ)
            E = real(inner(ψ', H, ψ))
            @test abs(E) < N * (abs(J) + abs(h))  # Energy should be bounded
        end
        
        @testset "Non-integrable Ising Model" begin
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = (J, hx, hz)
            H = CoolingTNS.ham_niising(N, sites, ham_params)
            
            @test H isa MPO
            @test length(H) == N
            
            # Test with specific product states
            # All up state |0000⟩
            ψ_up = MPS(sites, "Up")
            E_up = real(inner(ψ_up', H, ψ_up))
            # Energy should be: (N-1)*J + N*hz (since Z|0⟩ = |0⟩)
            expected_E_up = (N-1) * J + N * hz
            @test abs(E_up - expected_E_up) < 1e-10
            
            # All down state |1111⟩
            ψ_down = MPS(sites, "Dn")
            E_down = real(inner(ψ_down', H, ψ_down))
            # Energy should be: (N-1)*J - N*hz (since Z|1⟩ = -|1⟩)
            expected_E_down = (N-1) * J - N * hz
            @test abs(E_down - expected_E_down) < 1e-10
        end
    end
    
    @testset "System-Bath Hamiltonians" begin
        sites = siteinds("S=1/2", 2N)  # N system + N bath
        J, hx, hz = 1.0, -1.05, 0.5
        ham_params = (J, hx, hz)
        
        coupling_params = Dict(
            "g" => 0.1,
            "Δ" => -1.0,
            "coupling" => "XX"
        )
        
        @testset "Ising System-Bath" begin
            J, h = 1.0, -2.0
            ham_params = (J, h)
            H = CoolingTNS.ham_ising_sys_bath(N, sites, ham_params, coupling_params)
            
            @test H isa MPO
            @test length(H) == 2N
        end
        
        @testset "Non-integrable Ising System-Bath" begin
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params_ni = (J, hx, hz)
            H = CoolingTNS.ham_niising_sys_bath(N, sites, ham_params_ni, coupling_params)
            
            @test H isa MPO
            @test length(H) == 2N
        end
        
        @testset "Different Coupling Types" begin
            coupling_types = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params_ni = (J, hx, hz)
            
            for coupling in coupling_types
                coupling_params_test = copy(coupling_params)
                coupling_params_test["coupling"] = coupling
                
                H = CoolingTNS.ham_niising_sys_bath(N, sites, ham_params_ni, coupling_params_test)
                @test H isa MPO
                @test length(H) == 2N
            end
        end
    end
    
    @testset "Ground State Calculation" begin
        sites = siteinds("S=1/2", N)
        J, hx, hz = 1.0, -1.05, 0.5
        ham_params = (J, hx, hz)
        
        H_sys, Δ, e₀, ϕ₀ = CoolingTNS.setup_system(N, "niIsing", sites, ham_params)
        
        @test H_sys isa MPO
        @test ϕ₀ isa MPS
        @test e₀ < 0  # Ground state energy should be negative for these parameters
        @test Δ > 0   # Gap should be positive
        
        # Check that ϕ₀ is indeed the ground state
        E_gs = real(inner(ϕ₀', H_sys, ϕ₀))
        @test abs(E_gs - e₀) < 1e-10
        
        # Check normalization
        @test abs(inner(ϕ₀, ϕ₀) - 1.0) < 1e-10
    end
    
    @testset "ED Hamiltonian Construction" begin
        test_N = 3  # Small system for ED
        J, hx, hz = 1.0, -1.05, 0.5
        ham_params = (J, hx, hz)
        coupling_params = Dict(
            "g" => 0.1,
            "Δ" => -1.0,
            "coupling" => "XX"
        )
        
        H = CoolingTNS.build_hamiltonian_ed("niIsing", test_N, ham_params, coupling_params)
        
        # Check that it's a valid Yao block
        @test H isa CoolingTNS.Yao.AbstractBlock
        @test CoolingTNS.Yao.nqubits(H) == 2 * test_N
        
        # Check Hermiticity
        H_mat = CoolingTNS.Yao.mat(H)
        @test ishermitian(H_mat)
        
        # Check dimension
        @test size(H_mat) == (2^(2test_N), 2^(2test_N))
    end
end