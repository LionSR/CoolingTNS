using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plotting.jl"))

@testset "Optimization result lookup uses saved metadata" begin
    mktempdir() do dir
        cd(dir) do
            ham_params = CoolingTNS.IsingParameters(2, 1.0, 0.5; bc=:open)
            initial_coupling = CoolingTNS.BasicCouplingParameters("XX", 0.1, 2, 1.0, nothing)
            optimized_coupling = CoolingTNS.BasicCouplingParameters("XX", 0.5, 8, 3.0, nothing)
            backend = CoolingTNS.EDBackend()
            sim_params = CoolingTNS.create_sim_params(
                backend;
                sim_method=CoolingTNS.DensityMatrix(),
                evolution_method=CoolingTNS.ContinuousEvolution(),
                Dmax=20,
                cutoff=1e-6,
                tau=0.1,
                pe=0.0,
                n_trajectories=1,
            )
            search_params = Dict{String, Any}(
                "search_method" => "Random",
                "num_trials" => 3,
            )

            saved_prefix = "Optimize" * CoolingTNS.create_filename(
                ham_params,
                optimized_coupling,
                sim_params,
                backend,
            )
            saved_name = saved_prefix * "_" * CoolingTNS.create_search_name_part(search_params)
            parsed_args = Dict{String, Any}(
                "N" => ham_params.N,
                "problem" => "Ising",
                "J" => ham_params.params.J,
                "h" => ham_params.params.h,
                "bc" => string(ham_params.bc),
                "coupling" => initial_coupling.coupling,
                "g" => initial_coupling.g,
                "steps" => initial_coupling.steps,
                "te" => initial_coupling.te,
                "backend" => "ED",
                "sim_method" => "density_matrix",
                "evolution_method" => "continuous",
                "Dmax" => sim_params.Dmax,
                "cutoff" => sim_params.cutoff,
                "tau" => sim_params.tau,
                "peInt" => 0,
                "n_trajectories" => sim_params.n_trajectories,
                "search_method" => search_params["search_method"],
                "num_trials" => search_params["num_trials"],
                "window_size" => 2,
            )
            result = Dict{String, Any}(
                CoolingTNS.RESULT_ENERGY => [1.0, 0.8],
                CoolingTNS.RESULT_GROUND_STATE_OVERLAP => [0.1, 0.2],
                "Edensity_final" => 0.4,
                "GS_overlap_final" => 0.2,
                "best_g" => optimized_coupling.g,
                "best_te" => optimized_coupling.te,
            )

            CoolingTNS.save_results(
                saved_name,
                result,
                -1.0,
                CoolingTNS.hamiltonian_name(ham_params),
                parsed_args;
                is_optimization=true,
            )

            initial_prefix = "Optimize" * CoolingTNS.create_filename(
                ham_params,
                initial_coupling,
                sim_params,
                backend,
            )
            initial_exact_path = joinpath(
                "ResultsOpt",
                initial_prefix * "_" * CoolingTNS.create_search_name_part(search_params) * ".h5",
            )
            saved_path = joinpath("ResultsOpt", saved_name * ".h5")

            @test !isfile(initial_exact_path)
            @test isfile(saved_path)
            @test _results_h5_path(
                "ResultsOpt",
                initial_prefix;
                is_optimization=true,
                search_params=search_params,
            ) === nothing

            metadata_filter = _optimization_lookup_metadata(
                ham_params,
                initial_coupling,
                sim_params,
                backend,
            )
            @test _results_h5_path(
                "ResultsOpt",
                initial_prefix;
                is_optimization=true,
                search_params=search_params,
                metadata_filter=metadata_filter,
            ) == saved_path
            @test _results_h5_path(
                "ResultsOpt",
                initial_prefix;
                is_optimization=true,
                metadata_filter=metadata_filter,
            ) == saved_path
            ed_continuous_plot_params = CoolingTNS.create_sim_params(
                backend;
                sim_method=CoolingTNS.DensityMatrix(),
                evolution_method=CoolingTNS.ContinuousEvolution(),
                Dmax=99,
                cutoff=1e-3,
                tau=0.7,
                pe=0.0,
                n_trajectories=1,
            )
            @test _results_h5_path(
                "ResultsOpt",
                initial_prefix;
                is_optimization=true,
                search_params=search_params,
                metadata_filter=_optimization_lookup_metadata(
                    ham_params,
                    initial_coupling,
                    ed_continuous_plot_params,
                    backend,
                ),
            ) == saved_path

            wrong_metadata = copy(metadata_filter)
            wrong_metadata["g"] = 0.2
            @test _results_h5_path(
                "ResultsOpt",
                initial_prefix;
                is_optimization=true,
                search_params=search_params,
                metadata_filter=wrong_metadata,
            ) === nothing

            tn_backend = CoolingTNS.TNBackend()
            tn_sim_params = CoolingTNS.create_sim_params(
                tn_backend;
                sim_method=CoolingTNS.MonteCarloWavefunction(),
                evolution_method=CoolingTNS.TrotterEvolution(),
                Dmax=42,
                cutoff=1e-7,
                tau=0.25,
                pe=0.0,
                n_trajectories=5,
            )
            tn_saved_prefix = "Optimize" * CoolingTNS.create_filename(
                ham_params,
                optimized_coupling,
                tn_sim_params,
                tn_backend,
            )
            tn_saved_name = tn_saved_prefix * "_" * CoolingTNS.create_search_name_part(search_params)
            tn_parsed_args = Dict{String, Any}(
                "N" => ham_params.N,
                "problem" => "Ising",
                "J" => ham_params.params.J,
                "h" => ham_params.params.h,
                "bc" => string(ham_params.bc),
                "coupling" => initial_coupling.coupling,
                "g" => initial_coupling.g,
                "steps" => initial_coupling.steps,
                "te" => initial_coupling.te,
                "backend" => "TN",
                "sim_method" => "monte_carlo",
                "evolution_method" => "trotter",
                "Dmax" => tn_sim_params.Dmax,
                "cutoff" => tn_sim_params.cutoff,
                "tau" => tn_sim_params.tau,
                "peInt" => 0,
                "n_trajectories" => tn_sim_params.n_trajectories,
                "search_method" => search_params["search_method"],
                "num_trials" => search_params["num_trials"],
                "window_size" => 2,
            )

            CoolingTNS.save_results(
                tn_saved_name,
                result,
                -1.0,
                CoolingTNS.hamiltonian_name(ham_params),
                tn_parsed_args;
                is_optimization=true,
            )

            tn_initial_prefix = "Optimize" * CoolingTNS.create_filename(
                ham_params,
                initial_coupling,
                tn_sim_params,
                tn_backend,
            )
            tn_saved_path = joinpath("ResultsOpt", tn_saved_name * ".h5")
            tn_metadata_filter = _optimization_lookup_metadata(
                ham_params,
                initial_coupling,
                tn_sim_params,
                tn_backend,
            )

            @test _results_h5_path(
                "ResultsOpt",
                tn_initial_prefix;
                is_optimization=true,
                search_params=search_params,
            ) === nothing
            @test _results_h5_path(
                "ResultsOpt",
                tn_initial_prefix;
                is_optimization=true,
                search_params=search_params,
                metadata_filter=tn_metadata_filter,
            ) == tn_saved_path

            tn_dmax_mismatch = CoolingTNS.create_sim_params(
                tn_backend;
                sim_method=CoolingTNS.MonteCarloWavefunction(),
                evolution_method=CoolingTNS.TrotterEvolution(),
                Dmax=43,
                cutoff=tn_sim_params.cutoff,
                tau=tn_sim_params.tau,
                pe=0.0,
                n_trajectories=tn_sim_params.n_trajectories,
            )
            @test _results_h5_path(
                "ResultsOpt",
                tn_initial_prefix;
                is_optimization=true,
                search_params=search_params,
                metadata_filter=_optimization_lookup_metadata(
                    ham_params,
                    initial_coupling,
                    tn_dmax_mismatch,
                    tn_backend,
                ),
            ) === nothing

            tn_cutoff_mismatch = CoolingTNS.create_sim_params(
                tn_backend;
                sim_method=CoolingTNS.MonteCarloWavefunction(),
                evolution_method=CoolingTNS.TrotterEvolution(),
                Dmax=tn_sim_params.Dmax,
                cutoff=1e-6,
                tau=tn_sim_params.tau,
                pe=0.0,
                n_trajectories=tn_sim_params.n_trajectories,
            )
            @test _results_h5_path(
                "ResultsOpt",
                tn_initial_prefix;
                is_optimization=true,
                search_params=search_params,
                metadata_filter=_optimization_lookup_metadata(
                    ham_params,
                    initial_coupling,
                    tn_cutoff_mismatch,
                    tn_backend,
                ),
            ) === nothing

            tn_tau_mismatch = CoolingTNS.create_sim_params(
                tn_backend;
                sim_method=CoolingTNS.MonteCarloWavefunction(),
                evolution_method=CoolingTNS.TrotterEvolution(),
                Dmax=tn_sim_params.Dmax,
                cutoff=tn_sim_params.cutoff,
                tau=0.3,
                pe=0.0,
                n_trajectories=tn_sim_params.n_trajectories,
            )
            @test _results_h5_path(
                "ResultsOpt",
                tn_initial_prefix;
                is_optimization=true,
                search_params=search_params,
                metadata_filter=_optimization_lookup_metadata(
                    ham_params,
                    initial_coupling,
                    tn_tau_mismatch,
                    tn_backend,
                ),
            ) === nothing
        end
    end
end
