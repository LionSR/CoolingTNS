#!/usr/bin/env julia
"""
Run a large-N multi-frequency tensor-network cooling scaling campaign.

This driver compares two tensor-network representations of the same open-chain
cooling channel:

  - `mpo`: TN density-matrix evolution, represented as an MPO.
  - `mcwf`: TN Monte-Carlo wavefunction trajectories, represented as MPS states.

For each system size and each number of bath detunings R, the script records
energy density, relative energy above the DMRG ground state, ground-state
overlap, runtime, the explicit detuning protocol, effective bond dimensions,
and the first cooling cycle where the method-specific bond threshold is reached.

Example N=64 campaign:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 1,2,5,10 --steps 40 --Dmax 40 --M-mcwf 2

Fixed-detuning Dmax comparison:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 2 --methods mcwf --steps 20 --Dmax 80 \
        --delta-min 0.5051167496264384 --delta-max 3.0307004977586303

Fixed-detuning Dmax ladder:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 1,2,5,10 --methods mcwf --steps 4 \
        --Dmax-values 160,320,640 \
        --delta-min 0.5051167496264384 --delta-max 3.0307004977586303

MCWF+TDVP large-N diagnostic:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 1,2,5,10 --methods mcwf --evolution-method continuous \
        --steps 40 --Dmax 80 \
        --delta-min 0.5051167496264384 --delta-max 3.0307004977586303

Fast path check:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl --quick
"""

using CoolingTNS
using Dates
using HDF5
using ITensors
using ITensorMPS
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "largeN_scaling_helpers.jl"))

const DEFAULT_OUTDIR = joinpath(@__DIR__, "Data", "largeN_multifrequency")

parse_int_list(s::AbstractString) =
    [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]

parse_method_list(s::AbstractString) =
    [lowercase(strip(x)) for x in split(s, ",") if !isempty(strip(x))]

function parse_args(args)
    cfg = Dict{String,Any}(
        "Ns" => [64],
        "R_values" => [1, 2, 5, 10],
        "methods" => ["mpo", "mcwf"],
        "evolution_method" => "trotter",
        "steps" => 40,
        "Dmax" => 40,
        "Dmax_values" => nothing,
        "cutoff" => 1e-7,
        "tau" => 0.2,
        "J" => 1.0,
        "hx" => -1.05,
        "hz" => 0.5,
        "coupling" => "XX",
        "g" => 0.3,
        "te" => 2.0,
        "delta_max_factor" => 6.0,
        "delta_min" => nothing,
        "delta_max" => nothing,
        "schedule" => "round_robin",
        "M_mcwf" => 1,
        "M_mpo" => 1,
        "seed" => 20260617,
        "outdir" => get(ENV, "COOLINGTNS_DATADIR", DEFAULT_OUTDIR),
        "output" => nothing,
        "verbose" => false,
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--quick"
            cfg["Ns"] = [8]
            cfg["R_values"] = [1, 2]
            cfg["steps"] = 2
            cfg["Dmax"] = 12
            cfg["M_mcwf"] = 1
            cfg["M_mpo"] = 1
            i += 1
        elseif a == "--Ns"
            cfg["Ns"] = parse_int_list(args[i + 1]); i += 2
        elseif a == "--R-values"
            cfg["R_values"] = parse_int_list(args[i + 1]); i += 2
        elseif a == "--methods"
            cfg["methods"] = parse_method_list(args[i + 1]); i += 2
        elseif a in ("--steps", "--Dmax", "--M-mcwf", "--M-mpo", "--seed")
            key = replace(a[3:end], "-" => "_")
            cfg[key] = parse(Int, args[i + 1]); i += 2
        elseif a == "--Dmax-values"
            cfg["Dmax_values"] = parse_int_list(args[i + 1]); i += 2
        elseif a in ("--cutoff", "--tau", "--J", "--hx", "--hz", "--g", "--te", "--delta-max-factor",
                     "--delta-min", "--delta-max")
            key = replace(a[3:end], "-" => "_")
            cfg[key] = parse(Float64, args[i + 1]); i += 2
        elseif a in ("--coupling", "--schedule", "--outdir", "--output", "--evolution-method")
            cfg[replace(a[3:end], "-" => "_")] = args[i + 1]; i += 2
        elseif a == "--verbose"
            cfg["verbose"] = true; i += 1
        else
            error("unknown argument: $a")
        end
    end

    all(R -> R >= 1, cfg["R_values"]) || error("all R values must be positive")
    all(N -> N >= 2, cfg["Ns"]) || error("all N values must be at least 2")
    cfg["steps"] >= 1 || error("--steps must be at least 1")
    all(D -> D >= 1, campaign_dmax_values(cfg)) ||
        error("all Dmax values must be positive")
    if cfg["output"] !== nothing && length(campaign_dmax_values(cfg)) > 1
        error("--output names a single HDF5 file and cannot be used with multiple --Dmax-values")
    end
    for method in cfg["methods"]
        method in ("mpo", "mcwf") || error("unknown method '$method'; use mpo or mcwf")
    end
    cfg["evolution_method"] in ("trotter", "continuous") ||
        error("--evolution-method must be trotter or continuous")
    if cfg["evolution_method"] == "continuous" && "mpo" in cfg["methods"]
        error("--evolution-method continuous is only supported for --methods mcwf")
    end
    cfg["schedule_symbol"] = Symbol(cfg["schedule"])
    cfg["schedule_symbol"] in (:round_robin, :random) ||
        error("--schedule must be round_robin or random")
    if (cfg["delta_min"] === nothing) != (cfg["delta_max"] === nothing)
        error("--delta-min and --delta-max must be supplied together")
    end
    if cfg["delta_min"] !== nothing && cfg["delta_max"] < cfg["delta_min"]
        error("--delta-max must be at least --delta-min")
    end
    if cfg["Dmax_values"] !== nothing && cfg["delta_min"] === nothing
        error(
            "--Dmax-values requires --delta-min and --delta-max so every Dmax " *
            "run uses the same physical detuning protocol"
        )
    end
    return cfg
end

function campaign_dmax_values(cfg)
    values = cfg["Dmax_values"]
    values === nothing && return [cfg["Dmax"]]
    isempty(values) && error("--Dmax-values must contain at least one integer")
    length(unique(values)) == length(values) ||
        error("--Dmax-values must not repeat a cap; repeated caps would overwrite output files")
    return values
end

function campaign_dmax_configs(cfg)
    values = campaign_dmax_values(cfg)
    return [merge(copy(cfg), Dict{String,Any}("Dmax" => D)) for D in values]
end

function sim_params_for(method::AbstractString, cfg)
    sim_method = if method == "mpo"
        DensityMatrix()
    elseif method == "mcwf"
        MonteCarloWavefunction()
    else
        error("unknown method '$method'; use mpo or mcwf")
    end
    evolution_method = cfg["evolution_method"] == "continuous" ?
        ContinuousEvolution() : TrotterEvolution()
    if sim_method isa DensityMatrix && evolution_method isa ContinuousEvolution
        error("MPO density-matrix evolution is supported only with TrotterEvolution")
    end
    n_trajectories = method == "mcwf" ? cfg["M_mcwf"] : cfg["M_mpo"]
    return UnifiedSimulationParameters(
        sim_method,
        evolution_method;
        Dmax=cfg["Dmax"],
        cutoff=cfg["cutoff"],
        tau=cfg["tau"],
        pe=0.0,
        n_trajectories=n_trajectories,
    )
end

function mps_or_mpo_link_dims(state)
    n = length(state)
    n <= 1 && return Int[]
    return [linkdim(state, i) for i in 1:(n - 1)]
end

function bond_summary(state)
    dims = mps_or_mpo_link_dims(state)
    isempty(dims) && return (dims=dims, max=1, mean=1.0)
    return (dims=dims, max=maximum(dims), mean=mean(dims))
end

function run_one_trajectory(problem, ham_params, cp_multi, sim_params, cfg, seed)
    steps = cp_multi.steps
    Random.seed!(seed)
    state = setup_initial_state(problem, sim_params, "product", 0.0)

    sys_maxbond = zeros(Int, steps + 1)
    sys_meanbond = zeros(Float64, steps + 1)
    # The initial row has no evolved system-bath state; exclude this sentinel
    # from peak evolved-bond summaries.
    evolved_maxbond = fill(0, steps + 1)
    evolved_meanbond = fill(NaN, steps + 1)
    final_dims = Ref(Int[])

    step_observer = info -> begin
        step = info.step
        if info.stage === :evolved
            evolved_bs = bond_summary(info.evolved_state)
            evolved_maxbond[step] = evolved_bs.max
            evolved_meanbond[step] = evolved_bs.mean
            return nothing
        end

        bs = bond_summary(info.state.state)
        sys_maxbond[step] = bs.max
        sys_meanbond[step] = bs.mean
        final_dims[] = bs.dims

        if cfg["verbose"] && step > 1
            E = info.measurements[RESULT_ENERGY][step]
            @printf("    step %d/%d delta=%.6g E/N=%.8f sysD=%d evolvedD=%d\n",
                    step - 1, steps, info.delta, E / ham_params.N,
                    sys_maxbond[step], evolved_maxbond[step])
            flush(stdout)
        end
    end

    result = nothing
    elapsed = @elapsed begin
        result = run_cooling_multi_freq(
            problem,
            state,
            cp_multi,
            sim_params,
            ham_params;
            step_observer=step_observer,
        )
    end

    E = result[RESULT_ENERGY]
    overlap = result[RESULT_GROUND_STATE_OVERLAP]
    purity = get(result, RESULT_PURITY, fill(NaN, steps + 1))
    delta_list = result[RESULT_DELTA_LIST]

    return Dict{String,Any}(
        "E" => E,
        "overlap" => overlap,
        "purity" => purity,
        "sys_maxbond" => sys_maxbond,
        "sys_meanbond" => sys_meanbond,
        "evolved_maxbond" => evolved_maxbond,
        "evolved_meanbond" => evolved_meanbond,
        "delta_list" => delta_list,
        "final_bond_dims" => final_dims[],
        "elapsed" => elapsed,
    )
end

function output_path(cfg)
    mkpath(cfg["outdir"])
    if cfg["output"] !== nothing
        return cfg["output"]
    end
    Ns = join(cfg["Ns"], "-")
    Rs = join(cfg["R_values"], "-")
    methods = join(cfg["methods"], "-")
    evolution_suffix = cfg["evolution_method"] == "trotter" ? "" : "_$(cfg["evolution_method"])"
    return joinpath(
        cfg["outdir"],
        @sprintf(
            "largeN_multifrequency_tn_N%s_R%s_%s%s_steps%d_Dmax%d_tau%.3g_seed%d.h5",
            Ns,
            Rs,
            methods,
            evolution_suffix,
            cfg["steps"],
            cfg["Dmax"],
            cfg["tau"],
            cfg["seed"],
        ),
    )
end

function write_run_group(parent, name, traj_rows, E0, saturation_threshold,
                         detuning_protocol, delta_values)
    g = create_group(parent, name)
    nsteps = length(traj_rows[1]["E"])
    M = length(traj_rows)

    E = reduce(hcat, [row["E"] for row in traj_rows])
    overlap = reduce(hcat, [row["overlap"] for row in traj_rows])
    purity = reduce(hcat, [row["purity"] for row in traj_rows])
    sys_maxbond = reduce(hcat, [row["sys_maxbond"] for row in traj_rows])
    sys_meanbond = reduce(hcat, [row["sys_meanbond"] for row in traj_rows])
    evolved_maxbond = reduce(hcat, [row["evolved_maxbond"] for row in traj_rows])
    evolved_meanbond = reduce(hcat, [row["evolved_meanbond"] for row in traj_rows])
    delta_lists = reduce(hcat, [row["delta_list"] for row in traj_rows])
    common_delta_list = all(j -> isequal(delta_lists[:, j], delta_lists[:, 1]), 1:M)

    E_mean = vec(mean(E; dims=2))
    E_stderr = M == 1 ? zeros(nsteps) : vec(std(E; dims=2)) ./ sqrt(M)
    rel_mean = relative_energy.(E_mean, Ref(E0))
    system_saturation_cycles = Int[
        first_bond_saturation_cycle(row["sys_maxbond"], saturation_threshold)
        for row in traj_rows
    ]
    evolved_saturation_cycles = Int[
        first_bond_saturation_cycle(row["evolved_maxbond"], saturation_threshold)
        for row in traj_rows
    ]

    write(g, "M", M)
    write(g, "E_trajectories", E)
    write(g, "E_mean", E_mean)
    write(g, "E_stderr", E_stderr)
    write(g, "relative_energy_mean", rel_mean)
    write(g, "GS_overlap_trajectories", overlap)
    write(g, "GS_overlap_mean", vec(mean(overlap; dims=2)))
    write(g, "purity_trajectories", purity)
    write(g, "system_max_bond", sys_maxbond)
    write(g, "system_mean_bond", sys_meanbond)
    write(g, "evolved_max_bond", evolved_maxbond)
    write(g, "evolved_mean_bond", evolved_meanbond)
    write(g, "bond_saturation_threshold", saturation_threshold)
    write(g, "system_saturation_cycle", system_saturation_cycles)
    write(g, "evolved_saturation_cycle", evolved_saturation_cycles)
    write(g, "elapsed_seconds", Float64[row["elapsed"] for row in traj_rows])
    write(g, "delta_lists", delta_lists)
    write(g, "delta_list_first_trajectory", delta_lists[:, 1])
    write(g, "delta_list_is_common", common_delta_list)
    common_delta_list && write(g, "delta_list", delta_lists[:, 1])
    write(g, "delta_values", Float64.(delta_values))
    write_largeN_detuning_protocol(g, detuning_protocol)

    bd = create_group(g, "final_bond_dims")
    for (j, row) in enumerate(traj_rows)
        write(bd, "trajectory_$j", row["final_bond_dims"])
    end

    return (
        final_E_mean=E_mean[end],
        final_rel=rel_mean[end],
        final_overlap=mean(overlap[end, :]),
        final_sys_maxbond=final_system_max_bond(sys_maxbond),
        final_sys_meanbond=final_system_mean_bond(sys_meanbond),
        peak_evolved_maxbond=peak_evolved_max_bond(evolved_maxbond),
        peak_evolved_meanbond=peak_evolved_mean_bond(evolved_meanbond),
        first_system_saturation_cycle=first_recorded_saturation_cycle(system_saturation_cycles),
        first_evolved_saturation_cycle=first_recorded_saturation_cycle(evolved_saturation_cycles),
        elapsed=sum(row["elapsed"] for row in traj_rows),
    )
end

function run_campaign(cfg)
    backend = TNBackend()
    path = output_path(cfg)
    mkpath(dirname(path))

    summaries = Vector{NamedTuple}()

    h5open(path, "w") do f
        write(f, "generated_at", Dates.format(now(), Dates.ISODateTimeFormat))
        write(f, "Ns", Int.(cfg["Ns"]))
        write(f, "R_values", Int.(cfg["R_values"]))
        write(f, "evolution_method", cfg["evolution_method"])
        write(f, "steps", cfg["steps"])
        write(f, "Dmax", cfg["Dmax"])
        write(f, "cutoff", cfg["cutoff"])
        write(f, "tau", cfg["tau"])
        write(f, "J", cfg["J"])
        write(f, "hx", cfg["hx"])
        write(f, "hz", cfg["hz"])
        write(f, "coupling", cfg["coupling"])
        write(f, "g", cfg["g"])
        write(f, "te", cfg["te"])
        write(f, "delta_max_factor", cfg["delta_max_factor"])
        write(f, "delta_min_override", cfg["delta_min"] === nothing ? NaN : cfg["delta_min"])
        write(f, "delta_max_override", cfg["delta_max"] === nothing ? NaN : cfg["delta_max"])
        write(f, "schedule", cfg["schedule"])
        write(f, "seed", cfg["seed"])

        for N in cfg["Ns"]
            ham_params = NiIsingParameters(N, cfg["J"], cfg["hx"], cfg["hz"])
            gn = create_group(f, "N$N")
            write(gn, "N", N)

            for method in cfg["methods"]
                sim_params = sim_params_for(method, cfg)
                cp_base = BasicCouplingParameters(cfg["coupling"], cfg["g"], 1, cfg["te"], nothing)
                @printf("\nsetup N=%d method=%s\n", N, method)
                Random.seed!(cfg["seed"] + 1000 * N)
                base_problem = setup_problem(backend, ham_params, cp_base, sim_params)
                gap = Float64(base_problem.extra.coupling_params.delta)
                E0 = Float64(base_problem.e₀)
                @printf("  E0/N=%.10f, gap=%.8f (reused across R)\n", E0 / N, gap)

                gm = create_group(gn, method)
                write(gm, "E0", E0)
                write(gm, "gap", gap)
                write(gm, "evolution_method", cfg["evolution_method"])
                write(gm, "system_solve_reused_across_R", true)
                detuning_protocol = largeN_detuning_protocol(gap, cfg)
                write_largeN_detuning_protocol(gm, detuning_protocol)

                M = method == "mcwf" ? cfg["M_mcwf"] : cfg["M_mpo"]
                saturation_threshold = CoolingTNS.tn_method_maxdim(
                    sim_params.sim_method, cfg["Dmax"]
                )
                write(gm, "bond_saturation_threshold", saturation_threshold)
                for R in cfg["R_values"]
                    delta_values = largeN_delta_values(detuning_protocol, R)
                    cp_multi = MultiFrequencyCouplingParameters(
                        cfg["coupling"],
                        cfg["g"],
                        cfg["steps"],
                        cfg["te"],
                        delta_values;
                        randomize_times=false,
                        schedule=cfg["schedule_symbol"],
                    )
                    problem = CoolingTNS.setup_tn_multifrequency_problem_from_system(
                        backend,
                        ham_params,
                        cp_multi,
                        sim_params,
                        base_problem.extra.sites,
                        base_problem.H_sys,
                        gap,
                        base_problem.e₀,
                        base_problem.ϕ₀,
                    )
                    @printf("N=%d method=%s R=%d M=%d delta=[%.6g, %.6g]\n",
                            N, method, R, M, minimum(delta_values), maximum(delta_values))

                    traj_rows = Vector{Dict{String,Any}}()
                    for m in 1:M
                        seed = cfg["seed"] + 1_000_000 * N + 10_000 * R + m
                        row = run_one_trajectory(problem, ham_params, cp_multi, sim_params,
                                                 cfg, seed)
                        push!(traj_rows, row)
                        peak_evolved_maxbond = maximum(row["evolved_maxbond"][2:end])
                        system_saturation_cycle = first_bond_saturation_cycle(
                            row["sys_maxbond"], saturation_threshold
                        )
                        evolved_saturation_cycle = first_bond_saturation_cycle(
                            row["evolved_maxbond"], saturation_threshold
                        )
                        @printf("  traj %d/%d seed=%d final E/N=%.8f rel=%.6g final_sysD=%d peak_evolvedD=%d sysSat=%s evolvedSat=%s elapsed=%.1fs\n",
                                m, M, seed, row["E"][end] / N,
                                relative_energy(row["E"][end], E0),
                                row["sys_maxbond"][end],
                                peak_evolved_maxbond,
                                saturation_cycle_label(system_saturation_cycle),
                                saturation_cycle_label(evolved_saturation_cycle),
                                row["elapsed"])
                    end

                    summary = write_run_group(
                        gm, "R$R", traj_rows, E0, saturation_threshold,
                        detuning_protocol, delta_values
                    )
                    push!(summaries, (
                        N=N,
                        method=method,
                        R=R,
                        E0=E0,
                        final_E=summary.final_E_mean,
                        final_rel=summary.final_rel,
                        final_overlap=summary.final_overlap,
                        final_sys_maxbond=summary.final_sys_maxbond,
                        final_sys_meanbond=summary.final_sys_meanbond,
                        peak_evolved_maxbond=summary.peak_evolved_maxbond,
                        peak_evolved_meanbond=summary.peak_evolved_meanbond,
                        first_system_saturation_cycle=summary.first_system_saturation_cycle,
                        first_evolved_saturation_cycle=summary.first_evolved_saturation_cycle,
                        elapsed=summary.elapsed,
                    ))
                    @printf("  mean final E/N=%.8f rel=%.6g overlap=%.6g final_sysD=%d mean_sysD=%.2f peak_evolvedD=%d mean_evolvedD=%.2f sysSat=%s evolvedSat=%s elapsed_total=%.1fs\n",
                            summary.final_E_mean / N,
                            summary.final_rel,
                            summary.final_overlap,
                            summary.final_sys_maxbond,
                            summary.final_sys_meanbond,
                            summary.peak_evolved_maxbond,
                            summary.peak_evolved_meanbond,
                            saturation_cycle_label(summary.first_system_saturation_cycle),
                            saturation_cycle_label(summary.first_evolved_saturation_cycle),
                            summary.elapsed)
                    GC.gc()
                end
            end
        end
    end

    println("\nwrote $path")
    println("summary:")
    for s in summaries
        @printf("  N=%d %-4s R=%2d final E/N=%.8f rel=%.6g overlap=%.6g final_sysD=%d mean_sysD=%.2f peak_evolvedD=%d mean_evolvedD=%.2f sysSat=%s evolvedSat=%s elapsed=%.1fs\n",
                s.N, s.method, s.R, s.final_E / s.N, s.final_rel,
                s.final_overlap, s.final_sys_maxbond, s.final_sys_meanbond,
                s.peak_evolved_maxbond, s.peak_evolved_meanbond,
                saturation_cycle_label(s.first_system_saturation_cycle),
                saturation_cycle_label(s.first_evolved_saturation_cycle),
                s.elapsed)
    end
    return path, summaries
end

function run_campaign_ladder(cfg)
    outputs = Tuple{String,Vector{NamedTuple}}[]
    for run_cfg in campaign_dmax_configs(cfg)
        push!(outputs, run_campaign(run_cfg))
    end
    return outputs
end

function run_largeN_multifrequency_tn_scaling_main()
    cfg = parse_args(ARGS)
    Dmax_values = campaign_dmax_values(cfg)
    @printf("large-N multi-frequency TN campaign\n")
    @printf("  Ns=%s R=%s methods=%s evolution=%s steps=%d Dmax=%s tau=%.3g cutoff=%.1e\n",
            join(cfg["Ns"], ","),
            join(cfg["R_values"], ","),
            join(cfg["methods"], ","),
            cfg["evolution_method"],
            cfg["steps"],
            join(Dmax_values, ","),
            cfg["tau"],
            cfg["cutoff"])
    run_campaign_ladder(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_largeN_multifrequency_tn_scaling_main()
end
