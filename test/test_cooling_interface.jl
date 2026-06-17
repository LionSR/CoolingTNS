using Test
using CoolingTNS
using Random
using HDF5

@testset "Cooling Interface Tests" begin
    # Test parameters
    N = 4
    problem = "niIsing"
    ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)  # N, J, hx, hz
    coupling_params = CoolingTNS.BasicCouplingParameters(
        "XX",    # coupling
        0.1,     # g
        5,       # steps
        1.0,     # te
        nothing  # delta (auto-compute)
    )

    @testset "Backend Creation" begin
        # Test new backend system
        @test CoolingTNS.get_backend("ED") isa CoolingTNS.EDBackend
        @test CoolingTNS.get_backend("TN") isa CoolingTNS.TNBackend
        @test_throws ErrorException CoolingTNS.get_backend("InvalidMethod")
    end

    @testset "Default Simulation Methods" begin
        @test CoolingTNS.default_simulation_method(CoolingTNS.EDBackend()) isa CoolingTNS.DensityMatrix
        @test CoolingTNS.default_simulation_method(CoolingTNS.TNBackend()) isa CoolingTNS.MonteCarloWavefunction
    end
    
    @testset "Default Evolution Methods" begin
        @test CoolingTNS.default_evolution_method(CoolingTNS.EDBackend()) isa CoolingTNS.ContinuousEvolution
        @test CoolingTNS.default_evolution_method(CoolingTNS.TNBackend()) isa CoolingTNS.ContinuousEvolution
    end

    @testset "Command-Line Parameter Helpers" begin
        parsed_args = Dict{String, Any}(
            "backend" => "ED",
            "sim_method" => "density_matrix",
            "evolution_method" => "trotter",
            "Dmax" => 40,
            "cutoff" => 1e-7,
            "tau" => 0.2,
            "peInt" => 3,
            "n_trajectories" => 5,
        )

        sim_params = CoolingTNS.create_sim_params_from_args(parsed_args)
        @test sim_params.sim_method isa CoolingTNS.DensityMatrix
        @test sim_params.evolution_method isa CoolingTNS.TrotterEvolution
        @test sim_params.Dmax == 40
        @test sim_params.cutoff == 1e-7
        @test sim_params.tau == 0.2
        @test sim_params.pe == 0.003
        @test sim_params.n_trajectories == 5

        legacy_mps = Dict{String, Any}("method" => "MPS")
        CoolingTNS.normalize_optimization_args!(legacy_mps)
        @test legacy_mps["backend"] == "TN"
        @test legacy_mps["sim_method"] == "monte_carlo"
        @test legacy_mps["evolution_method"] == "continuous"

        legacy_mpo = Dict{String, Any}("method" => "MPO")
        CoolingTNS.normalize_optimization_args!(legacy_mpo)
        @test legacy_mpo["backend"] == "TN"
        @test legacy_mpo["sim_method"] == "density_matrix"
        @test legacy_mpo["evolution_method"] == "trotter"

        explicit = Dict{String, Any}(
            "backend" => "ED",
            "sim_method" => "density_matrix",
            "evolution_method" => "continuous",
        )
        CoolingTNS.normalize_optimization_args!(explicit)
        @test explicit["backend"] == "ED"
        @test explicit["sim_method"] == "density_matrix"
        @test explicit["evolution_method"] == "continuous"
    end

    @testset "Result Key Constants" begin
        @test CoolingTNS.RESULT_ENERGY == "E_list"
        @test CoolingTNS.RESULT_GROUND_STATE_OVERLAP == "GS_overlap_list"
        @test CoolingTNS.RESULT_PURITY == "purity_list"
        @test CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION == "momentum_dist"
        @test CoolingTNS.RESULT_K_VALUES == "k_values"
        @test CoolingTNS.RESULT_MOMENTUM_GF == "momentum_gF"
        @test CoolingTNS.RESULT_MOMENTUM_GF_SOURCE == "momentum_gF_source"
        @test CoolingTNS.RESULT_MODE_HK == "mode_hk"
        @test CoolingTNS.RESULT_MODE_NK == "mode_nk"
        @test CoolingTNS.RESULT_DELTA_LIST == "delta_list"
        @test CoolingTNS.RESULT_TE_LIST == "te_list"

        @test CoolingTNS.RESULT_KEYS isa Tuple
        @test all(key -> key isa String, CoolingTNS.RESULT_KEYS)
        @test length(unique(CoolingTNS.RESULT_KEYS)) == length(CoolingTNS.RESULT_KEYS)
        @test CoolingTNS.RESULT_ENERGY in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MOMENTUM_GF in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MOMENTUM_GF_SOURCE in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MODE_HK in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MODE_NK in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_DELTA_LIST in CoolingTNS.RESULT_KEYS
    end

    @testset "Result save skips duplicate parsed-argument keys" begin
        mktempdir() do dir
            cd(dir) do
                result = Dict{String,Any}(
                    CoolingTNS.RESULT_ENERGY => [1.0, 0.5],
                    CoolingTNS.RESULT_N_TRAJECTORIES => 3,
                )
                parsed_args = Dict{String,Any}(
                    "n_trajectories" => 7,
                    "steps" => 1,
                )

                CoolingTNS.save_results(
                    "duplicate_metadata",
                    result,
                    -1.0,
                    "TestHamiltonian",
                    parsed_args,
                )

                h5open(joinpath("Results", "duplicate_metadata.h5"), "r") do file
                    @test read(file[CoolingTNS.RESULT_N_TRAJECTORIES]) == 3
                    @test read(file["steps"]) == 1
                end
            end
        end
    end

    @testset "Command-line Rydberg Parameters" begin
        parsed = CoolingTNS.parse_commandline([
            "--problem", "Rydberg",
            "--N", "3",
            "--Omega", "1.2",
            "--Delta", "-0.4",
            "--V", "2.5",
            "--bc", "periodic",
            "--backend", "ED",
            "--coupling", "ZZ",
            "--g", "0.05",
            "--steps", "7",
            "--te", "1.5",
        ])

        problem_name, parsed_ham_params, ham_name, parsed_coupling =
            CoolingTNS.setup_common_parameters(parsed)

        @test problem_name == "Rydberg"
        @test parsed_ham_params.model isa CoolingTNS.RydbergModel
        @test parsed_ham_params.N == 3
        @test parsed_ham_params.bc == :periodic
        @test parsed_ham_params.params.Ω == 1.2
        @test parsed_ham_params.params.Δ == -0.4
        @test parsed_ham_params.params.V == 2.5
        @test ham_name == "RydbergN3bcperiodicOmega1.2Delta-0.4V2.5"

        roundtrip_ham_params = CoolingTNS.parse_hamiltonian_name(ham_name)
        @test roundtrip_ham_params.model isa CoolingTNS.RydbergModel
        @test roundtrip_ham_params.N == parsed_ham_params.N
        @test roundtrip_ham_params.bc == parsed_ham_params.bc
        @test roundtrip_ham_params.params == parsed_ham_params.params

        @test parsed_coupling.coupling == "ZZ"
        @test parsed_coupling.g == 0.05
        @test parsed_coupling.steps == 7
        @test parsed_coupling.te == 1.5
    end

    @testset "Command-line initial-state validation" begin
        parsed = CoolingTNS.parse_commandline([
            "--sim_method", "density_matrix",
            "--init_state", "identity",
        ])
        @test parsed["sim_method"] == "density_matrix"
        @test parsed["init_state"] == "identity"

        @test_throws ArgumentError CoolingTNS.parse_commandline([
            "--sim_method", "monte_carlo",
            "--init_state", "identity",
        ])
    end

    @testset "Common parameter boundary conditions" begin
        base_args = Dict{String,Any}(
            "N" => 4,
            "problem" => "Ising",
            "J" => 1.0,
            "h" => 0.5,
            "coupling" => "XX",
            "g" => 0.1,
            "steps" => 2,
            "te" => 1.0,
        )

        for (backend, bc) in [("TN", "periodic"), ("ED", "antiperiodic")]
            args = copy(base_args)
            args["backend"] = backend
            args["bc"] = bc

            parsed_problem, parsed_ham, ham_name, parsed_coupling =
                CoolingTNS.setup_common_parameters(args)

            @test parsed_problem == "Ising"
            @test parsed_ham.bc == Symbol(bc)
            @test occursin("bc$(bc)", ham_name)
            @test parsed_coupling.coupling == "XX"
        end

        default_args = copy(base_args)
        default_args["backend"] = "TN"
        _, default_ham, default_name, _ = CoolingTNS.setup_common_parameters(default_args)
        @test default_ham.bc == :open
        @test occursin("bcopen", default_name)
    end

    @testset "Problem Setup for Different Backends" begin
        # Create simulation parameters for each backend/method combination
        sim_params_ed = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0, n_trajectories=1
        )
        
        sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            Dmax=20, cutoff=1e-6, tau=0.1, pe=0.0, n_trajectories=10
        )
        
        backends_and_params = [
            (CoolingTNS.EDBackend(), sim_params_ed),
            (CoolingTNS.TNBackend(), sim_params_tn)
        ]
        
        for (backend, sim_params) in backends_and_params
            @testset "$(typeof(backend))" begin
                # Skip ED for larger systems
                test_N = backend isa CoolingTNS.EDBackend ? 3 : N
                test_ham_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
                
                problem_setup = CoolingTNS.setup_problem(
                    backend, test_ham_params, coupling_params, sim_params
                )
                
                @test problem_setup isa CoolingTNS.CoolingProblem
                @test problem_setup.backend === backend
                @test problem_setup.e₀ < 0  # Ground state energy should be negative
                @test !isnothing(problem_setup.H_sys)
                @test !isnothing(problem_setup.ϕ₀)
                
                # Check backend-specific fields
                if backend isa CoolingTNS.EDBackend
                    # ED backend doesn't use sites
                    @test !haskey(problem_setup.extra, :sites)
                else
                    # TN backend stores sites in extra
                    @test haskey(problem_setup.extra, :sites)
                    @test !isnothing(problem_setup.extra.sites)
                    @test length(problem_setup.extra.sites) == 2 * test_N
                end
            end
        end
    end

    @testset "Initial State Setup" begin
        # Test with TN backend
        backend = CoolingTNS.TNBackend()
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            Dmax=20, cutoff=1e-6, tau=0.1
        )
        
        problem_setup = CoolingTNS.setup_problem(
            backend, ham_params, coupling_params, sim_params
        )
        
        init_types = ["product", "theta"]
        theta_values = [0.0, -0.5, 0.5]
        
        for init_type in init_types
            for theta in theta_values
                if init_type != "theta" && theta != 0.0
                    continue  # Only test theta values with theta init type
                end
                
                initial_state = CoolingTNS.setup_initial_state(
                    problem_setup, sim_params, init_type, theta
                )
                
                @test initial_state isa CoolingTNS.QuantumState
                @test initial_state.backend === backend
                @test !isnothing(initial_state.state)
            end
        end

        @test_throws ArgumentError CoolingTNS.setup_initial_state(
            problem_setup, sim_params, "identity", 0.0
        )
    end

    @testset "Full Cooling Simulation" begin
        # Test with small system using ED
        backend = CoolingTNS.EDBackend()
        test_N = 3
        test_ham_params = CoolingTNS.NiIsingParameters(test_N, 1.0, -1.05, 0.5)
        
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0
        )
        
        problem_setup = CoolingTNS.setup_problem(
            backend, test_ham_params, coupling_params, sim_params
        )
        
        initial_state = CoolingTNS.setup_initial_state(
            problem_setup, sim_params, "product", 0.0
        )
        
        # Run short simulation
        short_coupling_params = CoolingTNS.BasicCouplingParameters(
            coupling_params.coupling,
            coupling_params.g,
            2,  # steps
            coupling_params.te,
            coupling_params.delta
        )
        
        results = CoolingTNS.run_cooling(
            problem_setup,
            initial_state,
            short_coupling_params,
            sim_params,
            test_ham_params
        )
        
        @test haskey(results, CoolingTNS.RESULT_ENERGY)
        @test haskey(results, CoolingTNS.RESULT_GROUND_STATE_OVERLAP)
        @test length(results[CoolingTNS.RESULT_ENERGY]) == short_coupling_params.steps + 1
        @test all(isfinite, results[CoolingTNS.RESULT_ENERGY])
        @test all(0 .<= results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] .<= 1)
        
        # Energy should decrease (cooling)
        @test results[CoolingTNS.RESULT_ENERGY][end] <= results[CoolingTNS.RESULT_ENERGY][1] + 1e-10
    end

    @testset "Odd ED Ising chains skip Fourier k-space measurements" begin
        backend = CoolingTNS.EDBackend()
        ham_params_odd = CoolingTNS.IsingParameters(3, 1.0, 0.5, :antiperiodic)
        coupling_params_odd = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.1, nothing)
        sim_params_odd = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0,
            n_trajectories=1,
        )
        problem_odd = CoolingTNS.setup_problem(
            backend,
            ham_params_odd,
            coupling_params_odd,
            sim_params_odd,
        )
        state_odd = CoolingTNS.setup_initial_state(
            problem_odd,
            sim_params_odd,
            "product",
            0.0,
        )
        results_odd = redirect_stdout(devnull) do
            CoolingTNS.run_cooling(
                problem_odd,
                state_odd,
                coupling_params_odd,
                sim_params_odd,
                ham_params_odd,
            )
        end

        @test haskey(results_odd, CoolingTNS.RESULT_ENERGY)
        @test haskey(results_odd, CoolingTNS.RESULT_GROUND_STATE_OVERLAP)
        @test !haskey(results_odd, CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION)
        @test !haskey(results_odd, CoolingTNS.RESULT_K_VALUES)
    end

    @testset "ED Monte Carlo Noise Dispatch" begin
        Random.seed!(1234)

        backend = CoolingTNS.EDBackend()
        test_N = 2
        test_ham_params = CoolingTNS.IsingParameters(test_N, 1.0, -2.0)
        noisy_coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.05, 1, 0.2, nothing)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            pe=0.2,
            n_trajectories=1,
        )

        problem_setup = CoolingTNS.setup_problem(
            backend, test_ham_params, noisy_coupling_params, sim_params
        )
        initial_state = CoolingTNS.setup_initial_state(
            problem_setup, sim_params, "product", 0.0
        )

        combined_state = CoolingTNS.prepare_combined_state(problem_setup, initial_state)
        noisy_combined = CoolingTNS.apply_noise(combined_state, problem_setup, 1.0)

        @test noisy_combined isa CoolingTNS.EDStateVector
        @test noisy_combined.n_qubits == 2 * test_N
        @test sum(abs2, noisy_combined.data) ≈ 1.0 atol=1e-12

        results = CoolingTNS.run_cooling(
            problem_setup,
            initial_state,
            problem_setup.extra.coupling_params,
            sim_params,
            test_ham_params,
        )

        @test length(results[CoolingTNS.RESULT_ENERGY]) == noisy_coupling_params.steps + 1
        @test all(isfinite, results[CoolingTNS.RESULT_ENERGY])
        @test all(isfinite, results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP])
    end

    @testset "Rydberg TN continuous setup" begin
        backend = CoolingTNS.TNBackend()
        test_ham_params = CoolingTNS.RydbergParameters(2, 1.0, 0.3, 0.2)
        test_coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 0.2, 0.5)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            Dmax=8, cutoff=1e-8, tau=0.1, pe=0.0, n_trajectories=1
        )

        problem_setup = CoolingTNS.setup_problem(
            backend, test_ham_params, test_coupling_params, sim_params
        )

        @test problem_setup isa CoolingTNS.CoolingProblem
        @test !isnothing(problem_setup.H_sys_bath)
        @test haskey(problem_setup.extra, :sites)
        @test length(problem_setup.extra.sites) == 2 * test_ham_params.N
    end

    # Cross-backend cooling comparisons are covered in `test_correctness.jl`.
    # They are intentionally gated behind `ENV["COOLINGTNS_FULL_TESTS"]` since
    # Monte Carlo trajectories can be slow and inherently stochastic.
end
