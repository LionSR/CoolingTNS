using Test
using CoolingTNS
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

    @testset "HDF5 Result Metadata Namespace" begin
        mktempdir() do dir
            cd(dir) do
                result = Dict{String, Any}(
                    "E_list" => [1.0, 0.5],
                    "search_method" => "result-owned",
                )
                parsed_args = Dict{String, Any}(
                    "N" => 2,
                    "search_method" => "Random",
                    "num_trials" => 4,
                )

                CoolingTNS.save_results(
                    "collision_test",
                    result,
                    -1.0,
                    "IsingN2bcopenJ1.0h1.0",
                    parsed_args;
                    is_optimization=true,
                )

                h5open(joinpath("ResultsOpt", "collision_test.h5"), "r") do file
                    @test read(file, "E_list") == [1.0, 0.5]
                    @test read(file, "search_method") == "result-owned"
                    @test read(file, "N") == 2
                    @test haskey(file, CoolingTNS.HDF5_PARSED_ARGS_GROUP)

                    parsed = file[CoolingTNS.HDF5_PARSED_ARGS_GROUP]
                    @test read(parsed, "search_method") == "Random"
                    @test read(parsed, "num_trials") == 4
                    @test read(parsed, "N") == 2
                end
            end
        end
    end

    @testset "ED density-matrix example driver root" begin
        example_utils = normpath(joinpath(@__DIR__, "..", "examples", "ed_dm_example_utils.jl"))
        example_text = read(example_utils, String)

        @test occursin("result_path=joinpath(ED_DM_EXAMPLE_ROOT, \"Results\", filename * \".h5\")", example_text)
        @test occursin("run(Cmd(cmd; dir=ED_DM_EXAMPLE_ROOT))", example_text)

        include(example_utils)
        integer_params = ed_dm_ising_example(N=4, J=1, h=2, g=1, te=2)
        float_params = ed_dm_ising_example(N=4, J=1.0, h=2.0, g=1.0, te=2.0)

        @test integer_params.filename == float_params.filename
        @test integer_params.result_path == float_params.result_path
        @test integer_params.J === 1.0
        @test integer_params.h === 2.0
        @test integer_params.g === 1.0
        @test integer_params.te === 2.0
    end

    @testset "Result Key Constants" begin
        @test CoolingTNS.RESULT_ENERGY == "E_list"
        @test CoolingTNS.RESULT_GROUND_STATE_OVERLAP == "GS_overlap_list"
        @test CoolingTNS.RESULT_PURITY == "purity_list"
        @test CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION == "momentum_dist"
        @test CoolingTNS.RESULT_K_VALUES == "k_values"
        @test CoolingTNS.RESULT_MODE_HK == "mode_hk"
        @test CoolingTNS.RESULT_MODE_NK == "mode_nk"
        @test CoolingTNS.RESULT_DELTA_LIST == "delta_list"
        @test CoolingTNS.RESULT_TE_LIST == "te_list"

        @test CoolingTNS.RESULT_KEYS isa Tuple
        @test all(key -> key isa String, CoolingTNS.RESULT_KEYS)
        @test length(unique(CoolingTNS.RESULT_KEYS)) == length(CoolingTNS.RESULT_KEYS)
        @test CoolingTNS.RESULT_ENERGY in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MODE_HK in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_MODE_NK in CoolingTNS.RESULT_KEYS
        @test CoolingTNS.RESULT_DELTA_LIST in CoolingTNS.RESULT_KEYS
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
        
        init_types = ["product", "identity", "theta"]
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

    @testset "ED k-space smoke example stays on current interface" begin
        example_path = normpath(joinpath(@__DIR__, "..", "examples", "test_ed_kspace.jl"))
        example_text = read(example_path, String)

        @test occursin("IsingParameters(N, J, h, bc)", example_text)
        @test !occursin("NiIsingParameters", example_text)
        @test !occursin("%3d", example_text)
        @test !occursin("plot_momentum_distribution", example_text)

        include(example_path)
        @test isdefined(@__MODULE__, :test_ed_kspace)
    end

    @testset "ED antiperiodic k-space allocation follows allowed momentum grid" begin
        backend = CoolingTNS.EDBackend()
        ham_params_apbc = CoolingTNS.IsingParameters(4, 1.0, 0.5, :antiperiodic)
        coupling_params_apbc = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 0.5, nothing)
        sim_params_apbc = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0,
            n_trajectories=1,
        )
        problem_apbc = CoolingTNS.setup_problem(
            backend,
            ham_params_apbc,
            coupling_params_apbc,
            sim_params_apbc,
        )
        state_apbc = CoolingTNS.setup_initial_state(
            problem_apbc,
            sim_params_apbc,
            "product",
            0.0,
        )
        results_apbc = CoolingTNS.run_cooling(
            problem_apbc,
            state_apbc,
            coupling_params_apbc,
            sim_params_apbc,
            ham_params_apbc,
        )

        @test haskey(results_apbc, CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION)
        @test haskey(results_apbc, CoolingTNS.RESULT_K_VALUES)
        @test length(results_apbc[CoolingTNS.RESULT_K_VALUES]) ==
              size(results_apbc[CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION], 2)
        @test length(results_apbc[CoolingTNS.RESULT_K_VALUES]) ==
              length(CoolingTNS.allowed_k_indices(ham_params_apbc.N, :antiperiodic))
        @test results_apbc[CoolingTNS.RESULT_K_VALUES] ≈
              [2π * Float64(k) / ham_params_apbc.N for k in
               CoolingTNS.allowed_k_indices(ham_params_apbc.N, :antiperiodic)]
    end

    # Cross-backend cooling comparisons are covered in `test_correctness.jl`.
    # They are intentionally gated behind `ENV["COOLINGTNS_FULL_TESTS"]` since
    # Monte Carlo trajectories can be slow and inherently stochastic.
end
