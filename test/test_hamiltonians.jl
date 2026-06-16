using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

function hamiltonian_test_mpo_to_matrix(H::MPO)
    N_sites = length(H)
    Tfull = H[1]
    for i in 2:N_sites
        Tfull *= H[i]
    end

    all_inds = inds(Tfull)
    s_inds = Index[]
    sp_inds = Index[]
    for idx in all_inds
        if hastags(idx, "Site") && plev(idx) == 0
            push!(s_inds, idx)
        elseif hastags(idx, "Site") && plev(idx) == 1
            push!(sp_inds, idx)
        end
    end
    sort!(s_inds, by=x -> parse(Int, match(r"n=(\d+)", string(tags(x))).captures[1]))
    sort!(sp_inds, by=x -> parse(Int, match(r"n=(\d+)", string(tags(x))).captures[1]))

    dim = 2^N_sites
    mat = zeros(ComplexF64, dim, dim)
    for i in 0:dim-1, j in 0:dim-1
        vals = Dict{Index, Int}()
        for k in 1:N_sites
            vals[sp_inds[k]] = ((i >> (k-1)) & 1) + 1
            vals[s_inds[k]] = ((j >> (k-1)) & 1) + 1
        end
        mat[i+1, j+1] = Tfull[vals...]
    end
    return mat
end

function hamiltonian_test_mps_to_vector(ψ::MPS, sites)
    dim = 2^length(sites)
    vec = zeros(ComplexF64, dim)
    for idx in 0:(dim - 1)
        config = [((idx >> (site - 1)) & 1) == 0 ? "Up" : "Dn" for site in eachindex(sites)]
        vec[idx + 1] = inner(MPS(sites, config), ψ)
    end
    return vec
end

@testset "Hamiltonian Construction Tests" begin
    N = 4
    
    @testset "Parse Coupling" begin
        @test CoolingTNS.parse_coupling("XX") == ("X", "X")
        @test CoolingTNS.parse_coupling("YZ") == ("Y", "Z")
        @test CoolingTNS.parse_coupling("ZY") == ("Z", "Y")
        @test CoolingTNS.coupling_operator_terms("XX") == (("X", "X"),)
        @test CoolingTNS.coupling_operator_terms("XY") == (("X", "Y"), ("Y", "X"))
        @test CoolingTNS.coupling_operator_terms("ZY") == (("Z", "Y"), ("Y", "Z"))
        @test CoolingTNS.get_bath_operator("YZ") == "X"
        @test CoolingTNS.get_bath_operator("ZY") == "X"
        @test CoolingTNS.get_bath_operator("XZ") == "Y"
        @test CoolingTNS.get_bath_operator("ZX") == "Y"
        @test_throws ArgumentError CoolingTNS.parse_coupling("XXX")
        @test_throws ArgumentError CoolingTNS.parse_coupling("XA")
    end

    @testset "Bath Operator Convention" begin
        local_paulis = Dict(
            "X" => ComplexF64[0 1; 1 0],
            "Y" => ComplexF64[0 -im; im 0],
            "Z" => ComplexF64[1 0; 0 -1],
        )

        expected_bath_ops = Dict(
            "XX" => "Z",
            "YY" => "Z",
            "ZZ" => "X",
            "XY" => "Z",
            "YX" => "Z",
            "XZ" => "Y",
            "ZX" => "Y",
            "YZ" => "X",
            "ZY" => "X",
        )

        for coupling in sort(collect(keys(expected_bath_ops)))
            bath_op = CoolingTNS.get_bath_operator(coupling)
            @test bath_op == expected_bath_ops[coupling]

            for (_, bath_coupling_op) in CoolingTNS.coupling_operator_terms(coupling)
                commutator = local_paulis[bath_op] * local_paulis[bath_coupling_op] -
                             local_paulis[bath_coupling_op] * local_paulis[bath_op]
                @test norm(commutator) > 1e-12
            end
        end
    end

    @testset "Bath Ground State Convention" begin
        xz_label, xz_amps = CoolingTNS.bath_ground_state_amplitudes("XZ")
        tn_xz_label, tn_xz_amps = CoolingTNS.get_bath_ground_state("XZ")
        @test xz_label == "Y-"
        @test xz_amps ≈ ComplexF64[1 / sqrt(2), -im / sqrt(2)]
        @test tn_xz_label == xz_label
        @test tn_xz_amps ≈ xz_amps

        xy_label, xy_amps = CoolingTNS.bath_ground_state_amplitudes("XY")
        @test xy_label == "Dn"
        @test xy_amps ≈ ComplexF64[0, 1]

        ψ_bath_xz = CoolingTNS.get_bath_ground_state_ed(1, "XZ")
        ψ_bath_xy = CoolingTNS.get_bath_ground_state_ed(1, "XY")
        @test CoolingTNS.expect_ed(CoolingTNS.pauli_y_complex(1, 1), ψ_bath_xz) ≈ -1.0 atol=1e-12
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
                H_tn_matrix = hamiltonian_test_mpo_to_matrix(
                    CoolingTNS.construct_system_hamiltonian(
                        ham_matrix, CoolingTNS.TNBackend(), sites_matrix
                    )
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
        
        @testset "TN Backend - Non-integrable Ising System-Bath" begin
            backend = CoolingTNS.TNBackend()
            J, hx, hz = 1.0, -1.05, 0.5
            ham_params = CoolingTNS.NiIsingParameters(N, J, hx, hz)
            H = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
            
            @test H isa MPO
            @test length(H) == 2N
        end
        
        
        @testset "Different Coupling Types" begin
            coupling_types = ["XX", "YY", "ZZ", "XY", "YX", "YZ", "ZY", "XZ", "ZX"]
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

        @testset "XZ bath field uses Y" begin
            ham_params = CoolingTNS.IsingParameters(1, 0.0, 0.0)
            xz_coupling_params = CoolingTNS.BasicCouplingParameters(
                "XZ", 0.0, 1, 1.0, 2.0
            )
            H = Matrix(CoolingTNS.construct_system_bath_hamiltonian(
                ham_params, CoolingTNS.EDBackend(), 2, xz_coupling_params
            ))

            expected = Matrix((xz_coupling_params.delta / 2) * CoolingTNS.pauli_y_complex(2, 2))
            x_field = Matrix((xz_coupling_params.delta / 2) * CoolingTNS.pauli_x(2, 2))
            z_field = Matrix((xz_coupling_params.delta / 2) * CoolingTNS.pauli_z(2, 2))

            @test H ≈ expected atol=1e-12
            @test norm(H - x_field) > 1e-6
            @test norm(H - z_field) > 1e-6

            sites_xz = siteinds("S=1/2", 2)
            H_tn = CoolingTNS.construct_system_bath_hamiltonian(
                ham_params, CoolingTNS.TNBackend(), sites_xz, xz_coupling_params
            )
            @test hamiltonian_test_mpo_to_matrix(H_tn) ≈ expected atol=1e-12
        end

        @testset "ED and TN system-bath couplings agree" begin
            small_N = 2
            coupling_types = ["XX", "YY", "ZZ", "XY", "YX", "YZ", "ZY", "XZ", "ZX"]
            sites_small = siteinds("S=1/2", 2small_N)
            ham_params = CoolingTNS.IsingParameters(small_N, 0.8, -0.4)

            for coupling in coupling_types
                test_coupling_params = CoolingTNS.BasicCouplingParameters(
                    coupling, 0.17, 1, 0.3, 0.6
                )

                H_ed = Matrix(CoolingTNS.construct_system_bath_hamiltonian(
                    ham_params, CoolingTNS.EDBackend(), 2small_N, test_coupling_params
                ))
                H_tn = hamiltonian_test_mpo_to_matrix(
                    CoolingTNS.construct_system_bath_hamiltonian(
                        ham_params, CoolingTNS.TNBackend(), sites_small, test_coupling_params
                    )
                )

                @test H_ed ≈ H_ed' atol=1e-12
                @test H_tn ≈ H_tn' atol=1e-12
                @test H_tn ≈ H_ed atol=1e-12
            end

            xy_coupling_params = CoolingTNS.BasicCouplingParameters(
                "XY", 0.17, 1, 0.3, 0.6
            )
            H_xy = CoolingTNS.construct_system_bath_hamiltonian(
                ham_params, CoolingTNS.EDBackend(), 2small_N, xy_coupling_params
            )
            ψ0 = CoolingTNS.product_state_ed(2small_N, 0)
            raw_ψt = exp(-im * Matrix(H_xy) * 0.37) * ψ0.data
            @test norm(raw_ψt) ≈ norm(ψ0.data) atol=1e-12
            ψt = CoolingTNS.evolve_ed(H_xy, ψ0, 0.37)
            @test ψt.data ≈ raw_ψt atol=1e-12
        end

        @testset "TN Trotter mixed coupling uses symmetric local term" begin
            ham_params = CoolingTNS.IsingParameters(1, 0.0, 0.0)
            coupling_params = CoolingTNS.BasicCouplingParameters("XY", 0.23, 1, 0.4, 0.0)
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                CoolingTNS.MonteCarloWavefunction(),
                CoolingTNS.TrotterEvolution();
                tau=0.4,
                Dmax=20,
                cutoff=1e-14,
            )
            sites_pair = siteinds("S=1/2", 2)

            gates = CoolingTNS.build_trotter_circuit_interleaved(
                ham_params, CoolingTNS.TNBackend(), sites_pair, coupling_params, sim_params
            )
            ψ_tn = apply(gates, MPS(sites_pair, ["Up", "Up"]); cutoff=1e-14, maxdim=20, move_sites_back=true)
            tn_vec = hamiltonian_test_mps_to_vector(ψ_tn, sites_pair)

            H_ed = CoolingTNS.construct_system_bath_hamiltonian(
                ham_params, CoolingTNS.EDBackend(), 2, coupling_params
            )
            ψ_ed = CoolingTNS.evolve_ed(H_ed, CoolingTNS.product_state_ed(2, 0), sim_params.tau)

            @test norm(tn_vec - ψ_ed.data) < 1e-10
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
