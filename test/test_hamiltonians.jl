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

    @testset "Bath Operator Convention" begin
        local_paulis = Dict(
            "X" => ComplexF64[0 1; 1 0],
            "Y" => ComplexF64[0 -im; im 0],
            "Z" => ComplexF64[1 0; 0 -1],
        )

        for sys_op in ["X", "Y", "Z"], bath_coupling_op in ["X", "Y", "Z"]
            coupling = sys_op * bath_coupling_op
            bath_op = CoolingTNS.get_bath_operator(coupling)
            expected_bath_op = bath_coupling_op == "Z" ? "X" : "Z"

            @test bath_op == expected_bath_op
            commutator = local_paulis[bath_op] * local_paulis[bath_coupling_op] -
                         local_paulis[bath_coupling_op] * local_paulis[bath_op]
            @test norm(commutator) > 1e-12
        end
    end

    @testset "Bath Ground State Convention" begin
        xz_label, xz_amps = CoolingTNS.bath_ground_state_amplitudes("XZ")
        tn_xz_label, tn_xz_amps = CoolingTNS.get_bath_ground_state("XZ")
        @test xz_label == "X-"
        @test tn_xz_label == xz_label
        @test tn_xz_amps ≈ xz_amps

        xy_label, xy_amps = CoolingTNS.bath_ground_state_amplitudes("XY")
        @test xy_label == "Dn"
        @test xy_amps ≈ ComplexF64[0, 1]

        ψ_bath_xz = CoolingTNS.get_bath_ground_state_ed(1, "XZ")
        ψ_bath_xy = CoolingTNS.get_bath_ground_state_ed(1, "XY")
        @test CoolingTNS.expect_ed(CoolingTNS.pauli_x(1, 1), ψ_bath_xz) ≈ -1.0 atol=1e-12
        @test CoolingTNS.expect_ed(CoolingTNS.pauli_z(1, 1), ψ_bath_xy) ≈ -1.0 atol=1e-12
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

        @testset "XZ bath field uses X" begin
            ham_params = CoolingTNS.IsingParameters(1, 0.0, 0.0)
            xz_coupling_params = CoolingTNS.BasicCouplingParameters(
                "XZ", 0.0, 1, 1.0, 2.0
            )
            H = Matrix(CoolingTNS.construct_system_bath_hamiltonian(
                ham_params, CoolingTNS.EDBackend(), 2, xz_coupling_params
            ))

            expected = Matrix((xz_coupling_params.delta / 2) * CoolingTNS.pauli_x(2, 2))
            commuting_field = Matrix((xz_coupling_params.delta / 2) * CoolingTNS.pauli_z(2, 2))

            @test H ≈ expected atol=1e-12
            @test norm(H - commuting_field) > 1e-6

            sites_xz = siteinds("S=1/2", 2)
            H_tn = CoolingTNS.construct_system_bath_hamiltonian(
                ham_params, CoolingTNS.TNBackend(), sites_xz, xz_coupling_params
            )
            ψ_up_up = MPS(sites_xz, ["Up", "Up"])
            ψ_up_dn = MPS(sites_xz, ["Up", "Dn"])

            @test inner(ψ_up_up', H_tn, ψ_up_dn) ≈ xz_coupling_params.delta / 2 atol=1e-12
            @test inner(ψ_up_up', H_tn, ψ_up_up) ≈ 0.0 atol=1e-12
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
    end
end
