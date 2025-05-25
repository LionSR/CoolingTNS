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
    
    @testset "System Hamiltonians - Tensor Network Backend" begin
        backend = CoolingTNS.TNBackend()
        sites = siteinds("S=1/2", N)
        
        @testset "Ising Model" begin
            J, h = 1.0, -2.0
            ham_params = CoolingTNS.IsingParameters(N, J, h)
            H = CoolingTNS.construct_system_hamiltonian(ham_params, backend, sites)
            
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
            ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz)
            H = CoolingTNS.construct_system_hamiltonian(ham_params, backend, sites)
            
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
    
    @testset "System Hamiltonians - ED Backend" begin
        backend = CoolingTNS.EDBackend()
        test_N = 3  # Small system for ED
        
        @testset "Ising Model" begin
            J, h = 1.0, -2.0
            ham_params = CoolingTNS.IsingParameters(test_N, J, h)
            H = CoolingTNS.construct_system_hamiltonian(ham_params, backend, test_N)
            
            # Check that it's a valid Yao block
            @test H isa CoolingTNS.Yao.AbstractBlock
            @test CoolingTNS.Yao.nqubits(H) == test_N
            
            # Check Hermiticity
            H_mat = Matrix(H)
            @test ishermitian(H_mat)
        end
        
        @testset "Non-integrable Ising Model" begin
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(test_N, J, hx, hz)
            H = CoolingTNS.construct_system_hamiltonian(ham_params, backend, test_N)
            
            @test H isa CoolingTNS.Yao.AbstractBlock
            @test CoolingTNS.Yao.nqubits(H) == test_N
        end
    end
    
    @testset "System-Bath Hamiltonians" begin
        sites = siteinds("S=1/2", 2N)  # N system + N bath
        
        coupling_params = CoolingTNS.BasicCouplingParameters(
            "XX",   # coupling type
            0.1,    # g
            10,     # steps
            1.0,    # te
            -1.0    # Δ
        )
        
        @testset "TN Backend - Ising System-Bath" begin
            backend = CoolingTNS.TNBackend()
            J, h = 1.0, -2.0
            ham_params = CoolingTNS.IsingParameters(N, J, h)
            H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
            
            @test H isa MPO
            @test length(H) == 2N
        end
        
        @testset "TN Backend - Non-integrable Ising System-Bath" begin
            backend = CoolingTNS.TNBackend()
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz)
            H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
            
            @test H isa MPO
            @test length(H) == 2N
        end
        
        @testset "ED Backend - System-Bath" begin
            backend = CoolingTNS.EDBackend()
            test_N = 3
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(test_N, J, hx, hz)
            H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, 2*test_N, coupling_params)
            
            @test H isa CoolingTNS.Yao.AbstractBlock
            @test CoolingTNS.Yao.nqubits(H) == 2 * test_N
            
            # Check Hermiticity
            H_mat = Matrix(H)
            @test ishermitian(H_mat)
        end
        
        @testset "Different Coupling Types" begin
            coupling_types = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]
            backend = CoolingTNS.TNBackend()
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz)
            
            for coupling in coupling_types
                test_coupling_params = CoolingTNS.BasicCouplingParameters(
                    coupling, 0.1, 10, 1.0, -1.0
                )
                
                H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, sites, test_coupling_params)
                @test H isa MPO
                @test length(H) == 2N
            end
        end
    end
    
    @testset "Ground State Calculation" begin
        @testset "TN Backend" begin
            backend = CoolingTNS.TNBackend()
            sites = siteinds("S=1/2", N)
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz)
            
            H_sys, Δ, e₀, ϕ₀ = CoolingTNS.setup_system(ham_params, backend, sites)
            
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
        
        @testset "ED Backend" begin
            backend = CoolingTNS.EDBackend()
            test_N = 3
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(test_N, J, hx, hz)
            
            H_sys, Δ, e₀, ϕ₀ = CoolingTNS.setup_system(ham_params, backend)
            
            @test H_sys isa CoolingTNS.Yao.AbstractBlock
            @test ϕ₀ isa CoolingTNS.Yao.ArrayReg
            @test e₀ < 0
            @test Δ > 0
        end
    end
    
    @testset "Dispatch Pattern Tests" begin
        # Test that dispatch correctly routes to different implementations
        test_N = 3
        
        # Different model types
        ising_params = CoolingTNS.IsingParameters(test_N, 1.0, -2.0)
        ni_ising_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
        
        # Different backends
        ed_backend = CoolingTNS.EDBackend()
        tn_backend = CoolingTNS.TNBackend()
        
        # Test system Hamiltonian dispatch
        @test CoolingTNS.construct_system_hamiltonian(ising_params, ed_backend, test_N) isa CoolingTNS.Yao.AbstractBlock
        
        sites = siteinds("S=1/2", test_N)
        @test CoolingTNS.construct_system_hamiltonian(ising_params, tn_backend, sites) isa MPO
        @test CoolingTNS.construct_system_hamiltonian(ni_ising_params, tn_backend, sites) isa MPO
    end
end