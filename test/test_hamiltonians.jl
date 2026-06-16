using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

function product_labels_from_bits(N::Int, state::Int)
    return [((state >> (i - 1)) & 1) == 0 ? "Up" : "Dn" for i in 1:N]
end

function product_mps_from_bits(sites, state::Int)
    return MPS(sites, product_labels_from_bits(length(sites), state))
end

function mpo_matrix_in_computational_basis(H::MPO, sites)
    dim = 2^length(sites)
    basis = [product_mps_from_bits(sites, state) for state in 0:(dim - 1)]
    return [inner(basis[i]', H, basis[j]) for i in 1:dim, j in 1:dim]
end

function interleaved_labels_from_system_bits(N::Int, system_state::Int; bath_label::String="Up")
    system_labels = product_labels_from_bits(N, system_state)
    return [isodd(site) ? system_labels[(site + 1) ÷ 2] : bath_label for site in 1:(2 * N)]
end

@testset "Hamiltonian Construction Tests" begin
    N = 4

    function zz_product_energy(N, J, bc)
        if bc == :open
            return (N - 1) * J
        elseif bc == :periodic
            return N * J
        elseif bc == :antiperiodic
            return (N - 2) * J
        end
        error("unsupported test boundary condition $bc")
    end
    
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

        @testset "Ising Model Boundary Conditions" begin
            J, h = 1.0, 0.0
            ψ_up = MPS(sites, "Up")

            for bc in [:open, :periodic, :antiperiodic]
                ham_params = CoolingTNS.IsingParameters(N, J, h, bc)
                H = CoolingTNS.construct_system_hamiltonian(ham_params, backend, sites)
                E_tn = real(inner(ψ_up', H, ψ_up))
                ψ_ed = CoolingTNS.product_state_ed(N, 0)
                H_ed = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
                E_ed = CoolingTNS.expect_ed(H_ed, ψ_ed)

                @test E_tn ≈ zz_product_energy(N, J, bc) atol=1e-10
                @test E_tn ≈ E_ed atol=1e-10
            end
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

        @testset "Non-integrable Ising Boundary Conditions" begin
            J, hx, hz = 1.0, 0.0, 0.25
            ψ_up = MPS(sites, "Up")

            for bc in [:open, :periodic, :antiperiodic]
                ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz, bc)
                H = CoolingTNS.construct_system_hamiltonian(ham_params, backend, sites)
                E_tn = real(inner(ψ_up', H, ψ_up))
                expected_E = zz_product_energy(N, J, bc) + N * hz

                @test E_tn ≈ expected_E atol=1e-10
            end
        end
    end

    @testset "Rydberg Rabi Convention" begin
        Ω = 1.3
        Δ = 0.7
        V = 0.2

        @test CoolingTNS.rydberg_rabi_x_coefficient(Ω) == Ω / 2
        @test CoolingTNS.rydberg_number_identity_shift(1, Δ, V) ≈ -Δ / 2
        @test CoolingTNS.rydberg_number_identity_shift(2, Δ, V) ≈ -Δ + V / 4

        H_one = CoolingTNS.construct_system_hamiltonian(
            CoolingTNS.RydbergParameters(1, Ω, 0.0, 0.0),
            CoolingTNS.EDBackend(),
            1,
        )
        @test sort(real(eigvals(Matrix(H_one)))) ≈ [-Ω / 2, Ω / 2]

        N_rydberg = 3
        ham_params = CoolingTNS.RydbergParameters(N_rydberg, Ω, Δ, V)
        H_ed = CoolingTNS.construct_system_hamiltonian(
            ham_params, CoolingTNS.EDBackend(), N_rydberg
        )

        ψ_ed = CoolingTNS.create_theta_state_ed(N_rydberg, "theta", 0.0)
        E_ed = real(dot(ψ_ed.data, H_ed * ψ_ed.data))

        sites = siteinds("S=1/2", N_rydberg)
        H_tn = CoolingTNS.construct_system_hamiltonian(
            ham_params, CoolingTNS.TNBackend(), sites
        )
        ψ_tn = MPS(sites, "X+")
        E_tn = real(inner(ψ_tn', H_tn, ψ_tn))

        @test E_tn ≈ E_ed atol=1e-10

        @testset "ED and TN matrices agree exactly" begin
            for N_matrix in [1, 2, 3]
                ham_matrix = CoolingTNS.RydbergParameters(N_matrix, Ω, Δ, V)
                H_ed_matrix = Matrix(CoolingTNS.construct_system_hamiltonian(
                    ham_matrix, CoolingTNS.EDBackend(), N_matrix
                ))

                sites_matrix = siteinds("S=1/2", N_matrix)
                H_tn_matrix = mpo_matrix_in_computational_basis(
                    CoolingTNS.construct_system_hamiltonian(
                        ham_matrix, CoolingTNS.TNBackend(), sites_matrix
                    ),
                    sites_matrix,
                )

                @test H_tn_matrix ≈ H_ed_matrix atol=1e-12
            end
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

        @testset "TN Backend - Ising System-Bath Boundary Conditions" begin
            backend = CoolingTNS.TNBackend()
            J, h = 1.0, 0.0
            ψ_up = MPS(sites, "Up")
            zero_coupling = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 1.0, 0.0)

            for bc in [:open, :periodic, :antiperiodic]
                ham_params = CoolingTNS.IsingParameters(N, J, h, bc)
                H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, sites, zero_coupling)
                E_tn = real(inner(ψ_up', H, ψ_up))
                H_ed = CoolingTNS.construct_system_bath_hamiltonian(
                    ham_params, CoolingTNS.EDBackend(), 2N, zero_coupling
                )
                E_ed = CoolingTNS.expect_ed(H_ed, CoolingTNS.product_state_ed(2N, 0))

                @test E_tn ≈ zz_product_energy(N, J, bc) atol=1e-10
                @test E_tn ≈ E_ed atol=1e-10
            end
        end
        
        @testset "TN Backend - Non-integrable Ising System-Bath" begin
            backend = CoolingTNS.TNBackend()
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz)
            H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
            
            @test H isa MPO
            @test length(H) == 2N
        end

        @testset "TN Backend - Non-integrable Ising System-Bath Boundary Conditions" begin
            backend = CoolingTNS.TNBackend()
            J, hx, hz = 1.0, 0.0, 0.25
            ψ_up = MPS(sites, "Up")
            zero_coupling = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 1.0, 0.0)

            for bc in [:open, :periodic, :antiperiodic]
                ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz, bc)
                H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, sites, zero_coupling)
                E_tn = real(inner(ψ_up', H, ψ_up))
                H_ed = CoolingTNS.construct_system_bath_hamiltonian(
                    ham_params, CoolingTNS.EDBackend(), 2N, zero_coupling
                )
                E_ed = CoolingTNS.expect_ed(H_ed, CoolingTNS.product_state_ed(2N, 0))
                expected_E = zz_product_energy(N, J, bc) + N * hz

                @test E_tn ≈ expected_E atol=1e-10
                @test E_tn ≈ E_ed atol=1e-10
            end
        end
        @testset "TN Backend - Rydberg System-Bath" begin
            backend = CoolingTNS.TNBackend()
            Ω, Δ, V = 1.0, 0.3, 0.2
            ham_params = CoolingTNS.RydbergParameters(N, Ω, Δ, V)
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

        @testset "System terms agree in isolated and interleaved TN MPOs" begin
            backend = CoolingTNS.TNBackend()
            test_N = 3
            product_labels = ["Up", "Dn", "Up"]
            zero_coupling = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.0, 0.0)
            cases = [
                CoolingTNS.IsingParameters(test_N, 1.1, -0.7),
                CoolingTNS.NiIsingParameters(test_N, 1.1, -0.7, 0.2),
                CoolingTNS.RydbergParameters(test_N, 0.9, 0.4, 0.3),
            ]

            for ham_params in cases
                sites_sys = siteinds("S=1/2", test_N)
                sites_sb = siteinds("S=1/2", 2 * test_N)

                H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, backend, sites_sys)
                H_sb = CoolingTNS.construct_system_bath_hamiltonian(
                    ham_params, backend, sites_sb, zero_coupling
                )

                ψ_sys = MPS(sites_sys, product_labels)
                interleaved_labels = [
                    isodd(site) ? product_labels[(site + 1) ÷ 2] : "Up"
                    for site in 1:(2 * test_N)
                ]
                ψ_sb = MPS(sites_sb, interleaved_labels)

                E_sys = real(inner(ψ_sys', H_sys, ψ_sys))
                E_sb = real(inner(ψ_sb', H_sb, ψ_sb))
                @test E_sb ≈ E_sys atol=1e-10
            end
        end

        @testset "System-term matrix elements agree in isolated and interleaved TN MPOs" begin
            backend = CoolingTNS.TNBackend()
            test_N = 2
            zero_coupling = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.0, 0.0)
            cases = [
                CoolingTNS.IsingParameters(test_N, 1.1, -0.7),
                CoolingTNS.NiIsingParameters(test_N, 1.1, -0.7, 0.2),
                CoolingTNS.RydbergParameters(test_N, 0.9, 0.4, 0.3),
            ]

            for ham_params in cases
                sites_sys = siteinds("S=1/2", test_N)
                sites_sb = siteinds("S=1/2", 2 * test_N)

                H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, backend, sites_sys)
                H_sb = CoolingTNS.construct_system_bath_hamiltonian(
                    ham_params, backend, sites_sb, zero_coupling
                )

                basis_sys = [
                    MPS(sites_sys, product_labels_from_bits(test_N, state))
                    for state in 0:(2^test_N - 1)
                ]
                basis_sb = [
                    MPS(sites_sb, interleaved_labels_from_system_bits(test_N, state))
                    for state in 0:(2^test_N - 1)
                ]

                for bra in 1:length(basis_sys), ket in 1:length(basis_sys)
                    H_sys_element = inner(basis_sys[bra]', H_sys, basis_sys[ket])
                    H_sb_element = inner(basis_sb[bra]', H_sb, basis_sb[ket])
                    @test H_sb_element ≈ H_sys_element atol=1e-10
                end
            end
        end

        @testset "TN Trotter rejects non-open Ising boundary conditions" begin
            backend = CoolingTNS.TNBackend()
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.DensityMatrix(), CoolingTNS.TrotterEvolution(); tau=0.1
            )
            trotter_coupling = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 1.0, 0.5)

            for bc in [:periodic, :antiperiodic]
                ham_params = CoolingTNS.IsingParameters(N, 1.0, -0.5, bc)
                @test_throws ArgumentError CoolingTNS.build_trotter_circuit_interleaved(
                    ham_params, backend, sites, trotter_coupling, sim_params
                )

                ni_ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -0.5, 0.2, bc)
                @test_throws ArgumentError CoolingTNS.build_trotter_circuit_interleaved(
                    ni_ham_params, backend, sites, trotter_coupling, sim_params
                )
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
    end
end
