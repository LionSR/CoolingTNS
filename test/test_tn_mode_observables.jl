using Test
using CoolingTNS
using ITensors
using ITensorMPS
using Random

function _ed_state_vector_to_mps(ψ_ed::CoolingTNS.EDStateVector, sites)
    N = length(sites)
    if ψ_ed.n_qubits != N
        throw(ArgumentError(
            "ED state has $(ψ_ed.n_qubits) qubits but $(N) ITensor sites were provided"
        ))
    end
    # ED stores site 1 as the least significant bit. Julia column-major
    # reshape makes the first tensor index fastest varying, so it matches
    # the ITensor site order without reversing the axes.
    ψ_tensor = ITensor(reshape(ψ_ed.data, ntuple(_ -> 2, N)), sites...)
    ψ_tn = MPS(ψ_tensor, sites; cutoff=1e-14)
    normalize!(ψ_tn)
    return ψ_tn
end

function _test_argument_error_contains(f, text::AbstractString)
    err = try
        f()
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    if err isa ArgumentError
        @test occursin(text, err.msg)
    end
    return err
end

@testset "TN Mode Observables" begin
    @testset "Notes-to-code Pauli map follows Ry(pi/2) convention" begin
        coeff, op = CoolingTNS._notes_pauli_to_code(:Z)
        @test coeff ≈ 1.0 + 0.0im atol=0
        @test op == :X

        coeff, op = CoolingTNS._notes_pauli_to_code(:X)
        @test coeff ≈ -1.0 + 0.0im atol=0
        @test op == :Z

        coeff, op = CoolingTNS._notes_pauli_to_code(:Y)
        @test coeff ≈ 1.0 + 0.0im atol=0
        @test op == :Y

        @test_throws ArgumentError CoolingTNS._notes_pauli_to_code(:I)
    end

    @testset "ED-to-MPS helper preserves site order" begin
        N = 4
        sites = siteinds("S=1/2", N)

        ψ_ed = CoolingTNS.product_state_ed(N, 1)
        ψ_tn = _ed_state_vector_to_mps(ψ_ed, sites)

        site_1_down = MPS(sites, ["Dn", "Up", "Up", "Up"])
        site_4_down = MPS(sites, ["Up", "Up", "Up", "Dn"])
        @test abs(inner(site_1_down, ψ_tn)) ≈ 1.0 atol=1e-12
        @test abs(inner(site_4_down, ψ_tn)) ≈ 0.0 atol=1e-12
    end

    @testset "MPS h_k agrees with ED for X+ product state" begin
        N = 4
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        sites = siteinds("S=1/2", N)

        ψ_tn = MPS(sites, "X+")
        ψ_ed = CoolingTNS.create_theta_state_ed(N, "theta", 0.0)

        px_tn = measure_state_parity(ψ_tn, N)
        px_ed = measure_state_parity(ψ_ed, N)
        @test px_tn ≈ 1.0 atol=1e-10
        @test px_tn ≈ px_ed atol=1e-10

        gF = fermionic_bc(:periodic, 1)
        ks_tn, hk_tn, εk_tn = measure_all_mode_observables(ψ_tn, ham_params; gF=gF)
        ks_ed, hk_ed, εk_ed = measure_all_mode_observables(ψ_ed, ham_params; gF=gF)
        ks_compat, hk_compat, εk_compat = measure_all_mode_energies(ψ_tn, ham_params; gF=gF)

        @test ks_tn == ks_ed
        @test ks_compat == ks_tn
        @test εk_tn ≈ εk_ed atol=1e-12
        @test εk_compat ≈ εk_tn atol=1e-12
        @test hk_tn ≈ hk_ed atol=1e-10
        @test hk_compat ≈ hk_tn atol=1e-10

        for k in ks_tn
            @test measure_hk(ψ_tn, k, ham_params) ≈ measure_hk(ψ_ed, k, ham_params) atol=1e-10
        end
    end

    @testset "MPO h_k agrees with ED density matrix for X+ product state" begin
        N = 4
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        sites = siteinds("S=1/2", N)

        ψ_tn = MPS(sites, "X+")
        ρ_tn = outer(ψ_tn', ψ_tn)
        ψ_ed = CoolingTNS.create_theta_state_ed(N, "theta", 0.0)
        ρ_ed = CoolingTNS.state_to_density_ed(ψ_ed)

        px_tn = measure_state_parity(ρ_tn, N)
        px_ed = measure_state_parity(ρ_ed, N)
        @test px_tn ≈ 1.0 atol=1e-10
        @test px_tn ≈ px_ed atol=1e-10

        gF = fermionic_bc(:periodic, 1)
        ks_tn, hk_tn, εk_tn = measure_all_mode_observables(ρ_tn, ham_params; gF=gF)
        ks_ed, hk_ed, εk_ed = measure_all_mode_observables(ρ_ed, ham_params; gF=gF)
        ks_compat, hk_compat, εk_compat = measure_all_mode_energies(ρ_tn, ham_params; gF=gF)

        @test ks_tn == ks_ed
        @test ks_compat == ks_tn
        @test εk_tn ≈ εk_ed atol=1e-12
        @test εk_compat ≈ εk_tn atol=1e-12
        @test hk_tn ≈ hk_ed atol=1e-10
        @test hk_compat ≈ hk_tn atol=1e-10

        for k in ks_tn
            @test measure_hk(ρ_tn, k, ham_params) ≈ measure_hk(ρ_ed, k, ham_params) atol=1e-10
        end
    end

    @testset "MPS h_k agrees with ED for all-up product state on explicit grids" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        sites = siteinds("S=1/2", N)

        ψ_tn = MPS(sites, "Up")
        ψ_ed = CoolingTNS.product_state_ed(N, 0)

        @test abs(measure_state_parity(ψ_tn, N)) < 1e-10
        @test abs(measure_state_parity(ψ_ed, N)) < 1e-10

        for gF in [-1, 1]
            ks_tn, hk_tn, εk_tn = measure_all_mode_observables(ψ_tn, ham_params; gF=gF)
            ks_ed, hk_ed, εk_ed = measure_all_mode_observables(ψ_ed, ham_params; gF=gF)

            @test ks_tn == ks_ed
            @test εk_tn ≈ εk_ed atol=1e-12
            @test hk_tn ≈ hk_ed atol=1e-10
        end
    end

    @testset "Automatic mixed-parity MPS grid uses ED reference sector" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        sites = siteinds("S=1/2", N)

        ψ_tn = MPS(sites, "Up")
        ψ_ed = CoolingTNS.product_state_ed(N, 0)

        @test abs(measure_state_parity(ψ_tn, N)) < 1e-10
        @test abs(measure_state_parity(ψ_ed, N)) < 1e-10

        expected_gF = fermionic_bc(:periodic, 1)
        expected_ks = allowed_k_indices(N, expected_gF)

        ks_tn, hk_tn, εk_tn = @test_logs (:warn, r"no definite P_x parity") begin
            measure_all_mode_observables(ψ_tn, ham_params)
        end
        ks_ed, hk_ed, εk_ed = @test_logs (:warn, r"no definite P_x parity") begin
            measure_all_mode_observables(ψ_ed, ham_params)
        end

        @test ks_tn == expected_ks
        @test ks_tn == ks_ed
        @test εk_tn ≈ εk_ed atol=1e-12
        @test hk_tn ≈ hk_ed atol=1e-10
    end

    @testset "MPS h_k agrees with ED for the exact ground state" begin
        N = 4
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        sites = siteinds("S=1/2", N)

        H_ed = CoolingTNS.construct_system_hamiltonian(ham_params, EDBackend(), N)
        E0_ed, ψ_ed, _ = CoolingTNS.ground_state_ed(H_ed)
        ψ_tn = _ed_state_vector_to_mps(ψ_ed, sites)

        px_ed = measure_state_parity(ψ_ed, N)
        px_tn = measure_state_parity(ψ_tn, N)
        @test abs(abs(px_ed) - 1) < 1e-10
        @test px_tn ≈ px_ed atol=1e-10

        gF = fermionic_bc(:periodic, CoolingTNS._reference_parity_sector(px_ed))
        ks_tn, hk_tn, εk_tn = measure_all_mode_observables(ψ_tn, ham_params; gF=gF)
        ks_ed, hk_ed, εk_ed = measure_all_mode_observables(ψ_ed, ham_params; gF=gF)

        @test ks_tn == ks_ed
        @test εk_tn ≈ εk_ed atol=1e-12
        @test hk_tn ≈ hk_ed atol=1e-10
        @test hk_tn ≈ fill(-1.0, N) atol=1e-10
        @test mode_occupation_from_hk(hk_tn) ≈ zeros(N) atol=1e-10
        @test ising_energy_from_mode_hk(ks_tn, hk_tn, ham_params) ≈ E0_ed atol=1e-10
    end

    @testset "TN h_k agrees with ED for the APBC exact ground state" begin
        N = 4
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :antiperiodic)
        sites = siteinds("S=1/2", N)

        H_ed = CoolingTNS.construct_system_hamiltonian(ham_params, EDBackend(), N)
        E0_ed, ψ_ed, _ = CoolingTNS.ground_state_ed(H_ed)
        ρ_ed = CoolingTNS.state_to_density_ed(ψ_ed)
        ψ_tn = _ed_state_vector_to_mps(ψ_ed, sites)
        ρ_tn = outer(ψ_tn', ψ_tn)

        px_ed = measure_state_parity(ψ_ed, N)
        px_mps = measure_state_parity(ψ_tn, N)
        px_mpo = measure_state_parity(ρ_tn, N)
        @test abs(abs(px_ed) - 1) < 1e-10
        @test px_mps ≈ px_ed atol=1e-10
        @test px_mpo ≈ px_ed atol=1e-10

        parity = CoolingTNS._reference_parity_sector(px_ed)
        @test parity == 1
        gF = fermionic_bc(:antiperiodic, parity)
        @test gF == 1
        @test allowed_k_indices(N, gF) == [-1, 0, 1, 2]

        ks_mps, hk_mps, εk_mps = measure_all_mode_observables(ψ_tn, ham_params; gF=gF)
        ks_mpo, hk_mpo, εk_mpo = measure_all_mode_observables(ρ_tn, ham_params; gF=gF)
        ks_ed, hk_ed, εk_ed = measure_all_mode_observables(ψ_ed, ham_params; gF=gF)
        ks_ed_dm, hk_ed_dm, εk_ed_dm = measure_all_mode_observables(ρ_ed, ham_params; gF=gF)

        @test ks_mps == ks_ed == ks_mpo == ks_ed_dm
        @test εk_mps ≈ εk_ed atol=1e-12
        @test εk_mpo ≈ εk_ed atol=1e-12
        @test εk_ed_dm ≈ εk_ed atol=1e-12
        @test hk_mps ≈ hk_ed atol=1e-10
        @test hk_mpo ≈ hk_ed atol=1e-10
        @test hk_ed_dm ≈ hk_ed atol=1e-10
        @test hk_mps ≈ fill(-1.0, N) atol=1e-10
        @test hk_mpo ≈ fill(-1.0, N) atol=1e-10
        @test mode_occupation_from_hk(hk_mps) ≈ zeros(N) atol=1e-10
        @test mode_occupation_from_hk(hk_mpo) ≈ zeros(N) atol=1e-10
        @test ising_energy_from_mode_hk(ks_mps, hk_mps, ham_params) ≈ E0_ed atol=1e-10
        @test ising_energy_from_mode_hk(ks_mpo, hk_mpo, ham_params) ≈ E0_ed atol=1e-10
    end

    @testset "APBC special-mode sector ground state agrees across ED and TN" begin
        N = 4
        J, h = 1.0, -1.05
        ham_params = IsingParameters(N, J, h, :antiperiodic)
        sites = siteinds("S=1/2", N)

        H_ed = CoolingTNS.construct_system_hamiltonian(ham_params, EDBackend(), N)
        E0_ed, ψ_ed, _ = CoolingTNS.ground_state_ed(H_ed)
        ρ_ed = CoolingTNS.state_to_density_ed(ψ_ed)
        ψ_tn = _ed_state_vector_to_mps(ψ_ed, sites)
        ρ_tn = outer(ψ_tn', ψ_tn)

        px_ed = measure_state_parity(ψ_ed, N)
        px_mps = measure_state_parity(ψ_tn, N)
        px_mpo = measure_state_parity(ρ_tn, N)
        @test abs(abs(px_ed) - 1) < 1e-10
        @test px_mps ≈ px_ed atol=1e-10
        @test px_mpo ≈ px_ed atol=1e-10

        parity = CoolingTNS._reference_parity_sector(px_ed)
        @test parity == 1
        gF = fermionic_bc(:antiperiodic, parity)
        expected_ks = [-1, 0, 1, 2]
        @test gF == 1
        @test allowed_k_indices(N, gF) == expected_ks

        ks_mps, hk_mps, εk_mps = measure_all_mode_observables(ψ_tn, ham_params; gF=gF)
        ks_mpo, hk_mpo, εk_mpo = measure_all_mode_observables(ρ_tn, ham_params; gF=gF)
        ks_ed, hk_ed, εk_ed = measure_all_mode_observables(ψ_ed, ham_params; gF=gF)
        ks_ed_dm, hk_ed_dm, εk_ed_dm = measure_all_mode_observables(ρ_ed, ham_params; gF=gF)

        expected_hk = [-1.0, 1.0, -1.0, 1.0]
        expected_nk = [0.0, 1.0, 0.0, 1.0]

        @test ks_mps == expected_ks
        @test ks_mps == ks_ed == ks_mpo == ks_ed_dm
        @test εk_mps ≈ εk_ed atol=1e-12
        @test εk_mpo ≈ εk_ed atol=1e-12
        @test εk_ed_dm ≈ εk_ed atol=1e-12
        @test hk_mps ≈ hk_ed atol=1e-10
        @test hk_mpo ≈ hk_ed atol=1e-10
        @test hk_ed_dm ≈ hk_ed atol=1e-10
        @test hk_mps ≈ expected_hk atol=1e-10
        @test hk_mpo ≈ expected_hk atol=1e-10
        @test mode_occupation_from_hk(hk_mps) ≈ expected_nk atol=1e-10
        @test mode_occupation_from_hk(hk_mpo) ≈ expected_nk atol=1e-10
        @test ising_energy_from_mode_hk(ks_mps, hk_mps, ham_params) ≈ E0_ed atol=1e-10
        @test ising_energy_from_mode_hk(ks_mpo, hk_mpo, ham_params) ≈ E0_ed atol=1e-10
    end

    @testset "MPS h_k agrees with ED for an entangled state" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        sites = siteinds("S=1/2", N)
        θ = 0.37

        gate = exp(-1.0im * θ * op("X", sites[1]) * op("X", sites[2]))
        ψ_tn = apply([gate], MPS(sites, "Up"); cutoff=1e-14, maxdim=20, move_sites_back=true)
        normalize!(ψ_tn)

        data = zeros(ComplexF64, 2^N)
        data[1] = cos(θ)
        data[4] = -1.0im * sin(θ)
        ψ_ed = CoolingTNS.EDStateVector(data, N)

        @test abs(measure_state_parity(ψ_tn, N)) < 1e-10
        @test abs(measure_state_parity(ψ_ed, N)) < 1e-10

        for gF in [-1, 1]
            ks_tn, hk_tn, εk_tn = measure_all_mode_observables(ψ_tn, ham_params; gF=gF)
            ks_ed, hk_ed, εk_ed = measure_all_mode_observables(ψ_ed, ham_params; gF=gF)

            @test ks_tn == ks_ed
            @test εk_tn ≈ εk_ed atol=1e-12
            @test hk_tn ≈ hk_ed atol=1e-10
        end
    end

    @testset "MPS mode observables reject open spin boundaries" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :open)
        sites = siteinds("S=1/2", N)
        ψ_tn = MPS(sites, "X+")

        @test_throws ArgumentError measure_hk(ψ_tn, 1//2, ham_params)
        @test_throws ArgumentError measure_all_mode_observables(ψ_tn, ham_params; gF=1)
    end

    @testset "MPS mode observables identify odd system sizes" begin
        N = 3
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        sites = siteinds("S=1/2", N)
        ψ_tn = MPS(sites, "X+")

        err = try
            measure_hk(ψ_tn, 1//2, ham_params)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        @test occursin("even N", err.msg)
    end

    @testset "MPO mode observables reject open spin boundaries" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :open)
        sites = siteinds("S=1/2", N)
        ψ_tn = MPS(sites, "X+")
        ρ_tn = outer(ψ_tn', ψ_tn)

        @test_throws ArgumentError measure_hk(ρ_tn, 1//2, ham_params)
    end

    @testset "TN mode observables reject unsupported Hamiltonian models through shared guard" begin
        N = 4
        sites = siteinds("S=1/2", N)
        ψ_tn = MPS(sites, "X+")
        ρ_tn = outer(ψ_tn', ψ_tn)

        unsupported_hamiltonians = [
            NiIsingParameters(N, 1.0, -1.05, 0.5, :periodic),
            RydbergParameters(N, 1.0, 0.0, 1.0, :open),
        ]

        for ham_params in unsupported_hamiltonians
            _test_argument_error_contains(
                () -> measure_hk(ψ_tn, 1//2, ham_params),
                "integrable transverse-field Ising",
            )
            _test_argument_error_contains(
                () -> measure_hk(ρ_tn, 1//2, ham_params),
                "integrable transverse-field Ising",
            )
            _test_argument_error_contains(
                () -> measure_all_mode_observables(ψ_tn, ham_params; gF=1),
                "integrable transverse-field Ising",
            )
            _test_argument_error_contains(
                () -> measure_all_mode_observables(ρ_tn, ham_params; gF=1),
                "integrable transverse-field Ising",
            )
            _test_argument_error_contains(
                () -> measure_all_mode_energies(ψ_tn, ham_params; gF=1),
                "integrable transverse-field Ising",
            )
            _test_argument_error_contains(
                () -> measure_all_mode_energies(ρ_tn, ham_params; gF=1),
                "integrable transverse-field Ising",
            )
        end

        short_sites = siteinds("S=1/2", N - 1)
        short_ψ = MPS(short_sites, "X+")
        _test_argument_error_contains(
            () -> measure_hk(short_ψ, 1//2, first(unsupported_hamiltonians)),
            "integrable transverse-field Ising",
        )
    end

    @testset "TN MCWF continuous cooling records MPS Fourier modes" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(MonteCarloWavefunction(), ContinuousEvolution(); maxiter=20)

        problem_tn = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state_tn = setup_initial_state(problem_tn, sim_params, "theta", 0.0)
        problem_ed = setup_problem(EDBackend(), ham_params, coupling_params, sim_params)
        state_ed = setup_initial_state(problem_ed, sim_params, "theta", 0.0)

        results_tn = redirect_stdout(devnull) do
            run_cooling(problem_tn, state_tn, coupling_params, sim_params, ham_params; measure_modes=true)
        end
        results_ed = redirect_stdout(devnull) do
            run_cooling(problem_ed, state_ed, coupling_params, sim_params, ham_params; measure_modes=true)
        end

        @test haskey(results_tn, RESULT_MODE_GF)
        @test haskey(results_tn, RESULT_MODE_GF_SOURCE)
        @test haskey(results_tn, RESULT_MODE_HK)
        @test haskey(results_tn, RESULT_MODE_NK)
        @test haskey(results_tn, RESULT_MODE_K_INDICES)
        @test haskey(results_tn, RESULT_MODE_ENERGIES)
        @test results_tn[RESULT_MODE_GF] == results_ed[RESULT_MODE_GF]
        @test results_tn[RESULT_MODE_GF_SOURCE] == results_ed[RESULT_MODE_GF_SOURCE]
        @test results_tn[RESULT_MODE_GF_SOURCE] == "state"
        @test results_tn[RESULT_MODE_K_INDICES] == results_ed[RESULT_MODE_K_INDICES]
        @test results_tn[RESULT_MODE_ENERGIES] ≈ results_ed[RESULT_MODE_ENERGIES] atol=1e-12
        @test results_tn[RESULT_MODE_HK] ≈ results_ed[RESULT_MODE_HK] atol=1e-8
        @test results_tn[RESULT_MODE_NK] ≈ results_ed[RESULT_MODE_NK] atol=1e-8
    end

    @testset "TN and ED cooling record reference source for mixed-parity mode grid" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(MonteCarloWavefunction(), ContinuousEvolution(); maxiter=20)

        problem_tn = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state_tn = setup_initial_state(problem_tn, sim_params, "product", 0.0)
        problem_ed = setup_problem(EDBackend(), ham_params, coupling_params, sim_params)
        state_ed = setup_initial_state(problem_ed, sim_params, "product", 0.0)

        @test abs(measure_state_parity(state_tn.state, N)) < 1e-10
        @test abs(measure_state_parity(state_ed.state, N)) < 1e-10

        results_tn = redirect_stdout(devnull) do
            run_cooling(problem_tn, state_tn, coupling_params, sim_params, ham_params; measure_modes=true)
        end
        results_ed = redirect_stdout(devnull) do
            run_cooling(problem_ed, state_ed, coupling_params, sim_params, ham_params; measure_modes=true)
        end

        expected_gF = fermionic_bc(:periodic, 1)
        @test results_tn[RESULT_MODE_GF_SOURCE] == "reference"
        @test results_ed[RESULT_MODE_GF_SOURCE] == "reference"
        @test results_tn[RESULT_MODE_GF] == expected_gF
        @test results_ed[RESULT_MODE_GF] == expected_gF
        @test results_tn[RESULT_MODE_K_INDICES] == results_ed[RESULT_MODE_K_INDICES]
        @test results_tn[RESULT_MODE_ENERGIES] ≈ results_ed[RESULT_MODE_ENERGIES] atol=1e-12
        @test results_tn[RESULT_MODE_HK] ≈ results_ed[RESULT_MODE_HK] atol=1e-8
        @test results_tn[RESULT_MODE_NK] ≈ results_ed[RESULT_MODE_NK] atol=1e-8
    end

    @testset "TN MCWF mode rows reject dimension-mismatch fallback" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 1, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(MonteCarloWavefunction(), ContinuousEvolution(); maxiter=20)

        problem = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state = setup_initial_state(problem, sim_params, "theta", 0.0)
        measurements = CoolingTNS.initialize_measurements(
            problem,
            state,
            coupling_params.steps;
            measure_modes=true,
            ham_params=ham_params,
        )
        CoolingTNS.perform_backend_measurements!(measurements, 1, problem, state, ham_params)

        wrong_length_state = QuantumState(
            problem.backend,
            sim_params.sim_method,
            sim_params.evolution_method,
            MPS(problem.extra.sites, "Up"),
        )
        bath_sample = [1, 2, 1, 2]
        @test_logs (:warn, r"Dimension mismatch") (:warn, r"Skipping measurement") begin
            CoolingTNS.perform_backend_measurements!(
                measurements,
                2,
                problem,
                wrong_length_state,
                ham_params,
                bath_sample,
            )
        end

        @test isfinite(measurements[RESULT_ENERGY][1])
        @test isfinite(measurements[RESULT_GROUND_STATE_OVERLAP][1])
        @test isnan(measurements[RESULT_ENERGY][2])
        @test isnan(measurements[RESULT_GROUND_STATE_OVERLAP][2])
        @test isnan(measurements[RESULT_BATH_SAMPLE_MAGNETIZATION][2])
        @test measurements[RESULT_MODE_GF_SOURCE] == "state"
        @test all(isfinite, measurements[RESULT_MODE_HK][1, :])
        @test all(isfinite, measurements[RESULT_MODE_NK][1, :])
        @test all(isnan, measurements[RESULT_MODE_HK][2, :])
        @test all(isnan, measurements[RESULT_MODE_NK][2, :])
        @test_throws ArgumentError validate_mode_measurement_rows(
            measurements[RESULT_MODE_HK],
            measurements[RESULT_MODE_NK],
            measurements[RESULT_MODE_MEASUREMENT_CYCLES];
            energy=measurements[RESULT_ENERGY],
        )
    end

    @testset "TN MCWF mode stride keeps skipped mismatch rows as NaN" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 2, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(MonteCarloWavefunction(), ContinuousEvolution(); maxiter=20)

        problem = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state = setup_initial_state(problem, sim_params, "theta", 0.0)
        measurements = CoolingTNS.initialize_measurements(
            problem,
            state,
            coupling_params.steps;
            measure_modes=true,
            ham_params=ham_params,
            mode_measurement_stride=2,
        )
        CoolingTNS.perform_backend_measurements!(measurements, 1, problem, state, ham_params)

        wrong_length_state = QuantumState(
            problem.backend,
            sim_params.sim_method,
            sim_params.evolution_method,
            MPS(problem.extra.sites, "Up"),
        )
        @test_logs (:warn, r"Dimension mismatch") (:warn, r"Skipping measurement") begin
            CoolingTNS.perform_backend_measurements!(
                measurements,
                2,
                problem,
                wrong_length_state,
                ham_params,
            )
        end

        @test measurements[RESULT_MODE_MEASUREMENT_CYCLES] == [0, 2]
        @test all(isnan, measurements[RESULT_MODE_HK][2, :])
        @test all(isnan, measurements[RESULT_MODE_NK][2, :])
    end

    @testset "TN MCWF continuous cooling records nonzero-step mode rows" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.05, 2, 0.05, 0.5)
        sim_params = UnifiedSimulationParameters(
            MonteCarloWavefunction(),
            ContinuousEvolution();
            maxiter=20,
            cutoff=1e-12,
            Dmax=64,
        )

        problem_tn = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state_tn = setup_initial_state(problem_tn, sim_params, "theta", 0.0)
        problem_ed = setup_problem(EDBackend(), ham_params, coupling_params, sim_params)
        state_ed = setup_initial_state(problem_ed, sim_params, "theta", 0.0)

        Random.seed!(1234)
        results_tn = redirect_stdout(devnull) do
            run_cooling(problem_tn, state_tn, coupling_params, sim_params, ham_params; measure_modes=true)
        end
        Random.seed!(1234)
        results_ed = redirect_stdout(devnull) do
            run_cooling(problem_ed, state_ed, coupling_params, sim_params, ham_params; measure_modes=true)
        end

        n_steps_total = coupling_params.steps + 1
        @test size(results_tn[RESULT_MODE_HK]) == (n_steps_total, N)
        @test size(results_tn[RESULT_MODE_NK]) == (n_steps_total, N)
        @test all(isfinite, results_tn[RESULT_MODE_HK])
        @test all(isfinite, results_tn[RESULT_MODE_NK])
        @test results_tn[RESULT_MODE_GF] == results_ed[RESULT_MODE_GF]
        @test results_tn[RESULT_MODE_GF_SOURCE] == results_ed[RESULT_MODE_GF_SOURCE]
        @test results_tn[RESULT_MODE_GF_SOURCE] == "state"
        @test results_tn[RESULT_MODE_K_INDICES] == results_ed[RESULT_MODE_K_INDICES]
        @test results_tn[RESULT_MODE_ENERGIES] ≈ results_ed[RESULT_MODE_ENERGIES] atol=1e-12
        @test results_tn[RESULT_MODE_HK] ≈ results_ed[RESULT_MODE_HK] atol=1e-6
        @test results_tn[RESULT_MODE_NK] ≈ results_ed[RESULT_MODE_NK] atol=1e-6
        @test results_tn[RESULT_MODE_NK] ≈
            mode_occupation_from_hk(results_tn[RESULT_MODE_HK]) atol=1e-12
        @test all(nk -> -1e-12 <= nk <= 1 + 1e-12, results_tn[RESULT_MODE_NK])
    end

    @testset "TN MCWF mode measurement stride records selected cycles" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 3, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(
            MonteCarloWavefunction(),
            ContinuousEvolution();
            maxiter=20,
            cutoff=1e-12,
            Dmax=64,
        )

        problem = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state = setup_initial_state(problem, sim_params, "theta", 0.0)
        results = redirect_stdout(devnull) do
            run_cooling(
                problem,
                state,
                coupling_params,
                sim_params,
                ham_params;
                measure_modes=true,
                mode_measurement_stride=2,
            )
        end

        @test results[RESULT_MODE_MEASUREMENT_CYCLES] == [0, 2, 3]
        hk = results[RESULT_MODE_HK]
        nk = results[RESULT_MODE_NK]
        @test size(hk) == (4, N)
        @test all(isfinite, hk[[1, 3, 4], :])
        @test all(isfinite, nk[[1, 3, 4], :])
        @test all(isnan, hk[2, :])
        @test all(isnan, nk[2, :])
    end

    @testset "TN MCWF early stop records final off-stride mode row" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = CoolingTNS.MultiFrequencyCouplingParameters(
            "XX",
            0.0,
            3,
            0.0,
            [0.5];
            randomize_times=false,
            schedule=:round_robin,
        )
        sim_params = UnifiedSimulationParameters(
            MonteCarloWavefunction(),
            ContinuousEvolution();
            maxiter=20,
            cutoff=1e-12,
            Dmax=64,
        )

        problem = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state = setup_initial_state(problem, sim_params, "theta", 0.0)
        results = redirect_stdout(devnull) do
            run_cooling(
                problem,
                state,
                coupling_params,
                sim_params,
                ham_params;
                measure_modes=true,
                mode_measurement_stride=3,
                stop_condition=info -> info.step == 3 ? "unit_test_stop" : nothing,
            )
        end

        @test results[CoolingTNS.RESULT_REQUESTED_STEPS] == 3
        @test results[CoolingTNS.RESULT_COMPLETED_STEPS] == 2
        @test results[CoolingTNS.RESULT_STOP_REASON] == "unit_test_stop"
        @test results[RESULT_MODE_MEASUREMENT_CYCLES] == [0, 2]

        hk = results[RESULT_MODE_HK]
        nk = results[RESULT_MODE_NK]
        @test size(hk) == (3, N)
        @test size(nk) == (3, N)
        @test all(isfinite, hk[[1, 3], :])
        @test all(isfinite, nk[[1, 3], :])
        @test all(isnan, hk[2, :])
        @test all(isnan, nk[2, :])

        measured = validate_mode_measurement_rows(
            hk,
            nk,
            results[RESULT_MODE_MEASUREMENT_CYCLES];
            energy=results[RESULT_ENERGY],
        )
        @test measured.cycles == [0, 2]
        @test measured.rows == [1, 3]
    end

    @testset "TN density-matrix zero-step cooling records MPO Fourier modes" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(DensityMatrix(), ContinuousEvolution())

        problem_tn = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state_tn = setup_initial_state(problem_tn, sim_params, "theta", 0.0)
        problem_ed = setup_problem(EDBackend(), ham_params, coupling_params, sim_params)
        state_ed = setup_initial_state(problem_ed, sim_params, "theta", 0.0)

        results_tn = redirect_stdout(devnull) do
            run_cooling(problem_tn, state_tn, coupling_params, sim_params, ham_params; measure_modes=true)
        end
        results_ed = redirect_stdout(devnull) do
            run_cooling(problem_ed, state_ed, coupling_params, sim_params, ham_params; measure_modes=true)
        end

        @test haskey(results_tn, RESULT_MODE_GF)
        @test haskey(results_tn, RESULT_MODE_GF_SOURCE)
        @test haskey(results_tn, RESULT_MODE_HK)
        @test haskey(results_tn, RESULT_MODE_NK)
        @test haskey(results_tn, RESULT_MODE_K_INDICES)
        @test haskey(results_tn, RESULT_MODE_ENERGIES)
        @test results_tn[RESULT_MODE_GF] == results_ed[RESULT_MODE_GF]
        @test results_tn[RESULT_MODE_GF_SOURCE] == results_ed[RESULT_MODE_GF_SOURCE]
        @test results_tn[RESULT_MODE_GF_SOURCE] == "state"
        @test results_tn[RESULT_MODE_K_INDICES] == results_ed[RESULT_MODE_K_INDICES]
        @test results_tn[RESULT_MODE_ENERGIES] ≈ results_ed[RESULT_MODE_ENERGIES] atol=1e-12
        @test results_tn[RESULT_MODE_HK] ≈ results_ed[RESULT_MODE_HK] atol=1e-8
        @test results_tn[RESULT_MODE_NK] ≈ results_ed[RESULT_MODE_NK] atol=1e-8
    end

    @testset "TN density-matrix Trotter rejects non-open Fourier cooling" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(DensityMatrix(), TrotterEvolution(); tau=0.1)

        @test_throws ArgumentError setup_problem(TNBackend(), ham_params, coupling_params, sim_params)

        sim_params_mcwf = UnifiedSimulationParameters(MonteCarloWavefunction(), TrotterEvolution(); tau=0.1)
        @test_throws ArgumentError setup_problem(TNBackend(), ham_params, coupling_params, sim_params_mcwf)
    end
end
