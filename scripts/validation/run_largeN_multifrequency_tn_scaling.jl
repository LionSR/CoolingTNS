#!/usr/bin/env julia
"""
Run a large-N multi-frequency tensor-network cooling scaling campaign.

This driver supports two tensor-network representations of the same open-chain
cooling channel:

  - `mpo`: TN density-matrix evolution, represented as an MPO.
  - `mcwf`: TN Monte-Carlo wavefunction trajectories, represented as MPS states.

The default large-N campaign method is `mcwf`, because the present N=64
diagnostics show that the MPO density-matrix route is a small-system validation
path rather than the practical production path.  Select `--methods mpo` or
`--methods mpo,mcwf` explicitly for density-matrix channel checks.

For each system size and each number of bath detunings R, the script records
energy density, relative energy above the DMRG ground state, ground-state
overlap, runtime, the explicit detuning protocol, effective bond dimensions,
the first cooling cycle where the method-specific bond threshold is reached,
and whether measured truncation-error histories were recorded.

Example N=64 campaign:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 1,2,5,10 --methods mcwf --steps 40 --Dmax 40 \
        --M-mcwf 2

Small-N MPO/MCWF channel check:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --quick

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

Mode-resolved integrable-Ising campaign:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --model ising --bc periodic --Ns 64 --R-values 1,2,5,10 \
        --methods mcwf --evolution-method continuous --steps 40 --Dmax 80 \
        --h -1.05 --init-state theta --theta 0.0 --measure-modes \
        --delta-min 0.5051167496264384 --delta-max 3.0307004977586303

If `--measure-modes` is used without an explicit detuning interval, the
gap-scaled interval for the default periodic, parity-preserving `XX` coupling
is referenced to the lowest generic analytic two-quasiparticle energy
`2 min_{sin φ_k != 0} ε_k` on the deterministic `parity=+1` setup grid
selected before any trajectory-specific state parity is measured, not to the
generic TN excited-state DMRG estimate and not to a reconstruction from
`mode_ek_values`.  Later `mode_gF` metadata remains the source of truth for
the measured mode-observable grid.  Automatic analytic detuning is disabled on
reference grids containing special modes, such as the default antiperiodic
reference sector; use an explicit `--delta-min/--delta-max` there.
For mode-resolved Ising runs with this periodic parity-preserving coupling, the
stored `gap` and `detuning_reference_gap` fields record this analytic reference
even when an explicit fixed detuning interval is supplied.
For long large-`N` mode diagnostics, `--mode-measurement-stride s` measures
the Bogoliubov mode rows only at cooling cycles `0, s, 2s, ...` and at the
requested final cycle; unmeasured rows in `mode_hk` and `mode_nk` are stored as
`NaN`, with the measured cycles recorded in `mode_measurement_cycles`.

Long TDVP runs can also write a per-observer-event CSV trace. The trace includes
the `initial`, `prepared`, `evolved`, and `updated` stages, so partial energy
and bond-dimension diagnostics survive if an expensive trajectory is
interrupted:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 2 --methods mcwf --evolution-method continuous \
        --steps 40 --Dmax 80 --progress-csv /tmp/tdvp_progress.csv

Add `--tdvp-sweep-progress` to install the TDVP sweep observer used by the HDF5
`tdvp_sweep_max_bond` diagnostic. When `--progress-csv` is also supplied, the
same observer additionally writes one CSV row after each TDVP sweep/substep. To
let ITensorMPS print its own TDVP sweep summary, use `--tdvp-outputlevel 1`.

For diagnostic runs where the objective is only to locate the first
bond-dimension cap event, add `--stop-on-bond-cap`.  This stops a trajectory
after the first completed cycle whose retained system state, evolved
system-bath state, or, with `--tdvp-sweep-progress`, TDVP sweep-observer
history reaches the method-specific cap.  It is intended for single-trajectory
diagnostics; use independent jobs for ensemble members whose partial
trajectories may stop at different cycle counts.  Unless `--output` is given
explicitly, the HDF5
filename receives a `_stopcap` suffix so these partial diagnostic files do not
overwrite full benchmark files with the same physical parameters.  The HDF5
group also records `tdvp_sweep_max_bond` and
`tdvp_sweep_saturation_cycle`, so sweep-level cap events are recoverable
without the progress CSV.

Add `--randomize-times` to draw each cycle time independently from
`Uniform(0, 2te)`.  The HDF5 output records the resulting `te_list` or
trajectory-resolved `te_lists`, and the default filename receives a `_randtime`
suffix so randomized-time diagnostics do not overwrite fixed-time runs with the
same mean `te`.

Use `--schedule descending` to cycle through the same fixed detuning grid from
largest to smallest before repeating.  This is a deterministic high-detuning
first probe; it is distinct from `--schedule random`, which samples one
detuning index independently at each cooling cycle.

To prepare independent commands for process-level parallel execution, use
`--print-parallel-plan`.  The plan splits the campaign into one command for
each `(N, method, R, Dmax)` tuple and assigns distinct HDF5 and progress CSV
paths.  The driver does not launch these jobs itself; process scheduling remains
external so each Julia process owns its HDF5 output, progress CSV, and
deterministic trajectory seed stream.

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 1,2,5,10 --methods mcwf --evolution-method continuous \
        --steps 5 --Dmax-values 96,128 \
        --delta-min 0.5051167496264384 --delta-max 3.0307004977586303 \
        --progress-csv /tmp/tdvp_progress.csv --tdvp-sweep-progress \
        --print-parallel-plan

For reproducible core-scaling benchmarks, the printed commands can also pin
Julia and BLAS thread counts:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
        --Ns 64 --R-values 2,5 --methods mcwf --evolution-method continuous \
        --steps 5 --Dmax 96 \
        --delta-min 0.5051167496264384 --delta-max 3.0307004977586303 \
        --print-parallel-plan --plan-julia-threads 1 --plan-blas-threads 1

Fast path check, using the explicit small-N MPO/MCWF validation path:

    julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl --quick

The `--quick` flag is an argument-stream preset: it sets `N=8`, `R=1,2`,
`steps=2`, `Dmax=12`, and `methods=mpo,mcwf` when it is encountered.  Explicit
flags written after `--quick` may override those preset values; flags written
before `--quick` are overwritten by the preset.
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

function require_unique_values(values, flag::AbstractString)
    length(unique(values)) == length(values) && return nothing
    error("$flag must not repeat values; repeated campaign axes would overwrite output files")
end

function parse_args(args)
    cfg = Dict{String,Any}(
        "Ns" => [64],
        "R_values" => [1, 2, 5, 10],
        "methods" => ["mcwf"],
        "evolution_method" => "trotter",
        "model" => "niising",
        "bc" => "open",
        "steps" => 40,
        "Dmax" => 40,
        "Dmax_values" => nothing,
        "cutoff" => 1e-7,
        "tau" => 0.2,
        "J" => 1.0,
        "h" => nothing,
        "hx" => -1.05,
        "hz" => 0.5,
        "coupling" => "XX",
        "g" => 0.3,
        "te" => 2.0,
        "init_state" => "product",
        "theta" => 0.0,
        "delta_max_factor" => 6.0,
        "delta_min" => nothing,
        "delta_max" => nothing,
        "schedule" => "round_robin",
        "randomize_times" => false,
        "M_mcwf" => 1,
        "M_mpo" => 1,
        "seed" => 20260617,
        "outdir" => get(ENV, "COOLINGTNS_DATADIR", DEFAULT_OUTDIR),
        "output" => nothing,
        "progress_csv" => nothing,
        "tdvp_outputlevel" => 0,
        "tdvp_sweep_progress" => false,
        "stop_on_bond_cap" => false,
        "print_parallel_plan" => false,
        "plan_julia_threads" => nothing,
        "plan_blas_threads" => nothing,
        "measure_modes" => false,
        "mode_measurement_stride" => 1,
        "verbose" => false,
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--quick"
            # `--quick` is an argument-stream preset. It intentionally selects
            # the small-N MPO/MCWF channel comparison unless later flags
            # override the individual fields.
            cfg["Ns"] = [8]
            cfg["R_values"] = [1, 2]
            cfg["methods"] = ["mpo", "mcwf"]
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
        elseif a in ("--steps", "--Dmax", "--M-mcwf", "--M-mpo", "--seed",
                     "--mode-measurement-stride",
                     "--tdvp-outputlevel", "--plan-julia-threads", "--plan-blas-threads")
            key = replace(a[3:end], "-" => "_")
            cfg[key] = parse(Int, args[i + 1]); i += 2
        elseif a == "--Dmax-values"
            cfg["Dmax_values"] = parse_int_list(args[i + 1]); i += 2
        elseif a in ("--cutoff", "--tau", "--J", "--h", "--hx", "--hz", "--g", "--te", "--theta", "--delta-max-factor",
                     "--delta-min", "--delta-max")
            key = replace(a[3:end], "-" => "_")
            cfg[key] = parse(Float64, args[i + 1]); i += 2
        elseif a in ("--model", "--bc", "--coupling", "--schedule", "--outdir", "--output", "--init-state",
                     "--evolution-method", "--progress-csv")
            cfg[replace(a[3:end], "-" => "_")] = args[i + 1]; i += 2
        elseif a == "--verbose"
            cfg["verbose"] = true; i += 1
        elseif a in ("--measure-modes", "--measure_modes")
            cfg["measure_modes"] = true; i += 1
        elseif a == "--tdvp-sweep-progress"
            cfg["tdvp_sweep_progress"] = true; i += 1
        elseif a == "--stop-on-bond-cap"
            cfg["stop_on_bond_cap"] = true; i += 1
        elseif a in ("--randomize-times", "--randomize_times")
            cfg["randomize_times"] = true; i += 1
        elseif a == "--print-parallel-plan"
            cfg["print_parallel_plan"] = true; i += 1
        else
            error("unknown argument: $a")
        end
    end

    all(R -> R >= 1, cfg["R_values"]) || error("all R values must be positive")
    all(N -> N >= 2, cfg["Ns"]) || error("all N values must be at least 2")
    require_unique_values(cfg["Ns"], "--Ns")
    require_unique_values(cfg["R_values"], "--R-values")
    require_unique_values(cfg["methods"], "--methods")
    cfg["steps"] >= 1 || error("--steps must be at least 1")
    all(D -> D >= 1, campaign_dmax_values(cfg)) ||
        error("all Dmax values must be positive")
    if cfg["output"] !== nothing && length(campaign_dmax_values(cfg)) > 1
        error("--output names a single HDF5 file and cannot be used with multiple --Dmax-values")
    end
    for method in cfg["methods"]
        method in ("mpo", "mcwf") || error("unknown method '$method'; use mpo or mcwf")
    end
    cfg["model"] in ("niising", "ising") ||
        error("--model must be niising or ising")
    Symbol(cfg["bc"]) in (:open, :periodic, :antiperiodic) ||
        error("--bc must be open, periodic, or antiperiodic")
    if cfg["model"] == "niising" && cfg["h"] !== nothing
        error("--h is only used with --model ising; use --hx for niising")
    end
    if cfg["model"] == "ising" && cfg["h"] === nothing
        # Backward compatibility for the large-N driver: before `--model ising`,
        # `--hx` was the only transverse-field flag.  After parsing, `cfg["h"]`
        # is the effective Ising field, not a record of whether `--h` was
        # explicitly supplied.
        cfg["h"] = cfg["hx"]
    end
    cfg["evolution_method"] in ("trotter", "continuous") ||
        error("--evolution-method must be trotter or continuous")
    cfg["init_state"] in ("product", "theta", "identity") ||
        error("--init-state must be product, theta, or identity")
    if cfg["init_state"] == "identity" && "mcwf" in cfg["methods"]
        error("--init-state identity is only valid for density-matrix/MPO methods")
    end
    if cfg["evolution_method"] == "trotter" && Symbol(cfg["bc"]) != :open
        error("--evolution-method trotter currently requires --bc open in this TN campaign")
    end
    if cfg["evolution_method"] == "continuous" && "mpo" in cfg["methods"]
        error("--evolution-method continuous is only supported for --methods mcwf")
    end
    if cfg["measure_modes"]
        cfg["model"] == "ising" ||
            error("--measure-modes requires --model ising")
        Symbol(cfg["bc"]) in (:periodic, :antiperiodic) ||
            error("--measure-modes requires --bc periodic or antiperiodic")
        all(iseven, cfg["Ns"]) ||
            error("--measure-modes requires even system sizes")
        all(method -> method == "mcwf", cfg["methods"]) ||
            error("--measure-modes is currently supported in this TN campaign only for --methods mcwf")
        cfg["evolution_method"] == "continuous" ||
            error("--measure-modes requires --evolution-method continuous in this TN campaign")
    end
    cfg["mode_measurement_stride"] >= 1 ||
        error("--mode-measurement-stride must be at least 1")
    if !cfg["measure_modes"] && cfg["mode_measurement_stride"] != 1
        error("--mode-measurement-stride requires --measure-modes")
    end
    try
        cfg["schedule_symbol"] = parse_multi_frequency_schedule(cfg["schedule"])
    catch e
        e isa ArgumentError || rethrow()
        error("--schedule: $(e.msg)")
    end
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
    if cfg["tdvp_sweep_progress"] && cfg["evolution_method"] != "continuous"
        error("--tdvp-sweep-progress requires --evolution-method continuous")
    end
    if cfg["stop_on_bond_cap"]
        for method in cfg["methods"]
            ntraj = method == "mcwf" ? cfg["M_mcwf"] : cfg["M_mpo"]
            ntraj == 1 || error(
                "--stop-on-bond-cap is currently restricted to one trajectory " *
                "per selected method; run ensemble members as independent jobs"
            )
        end
    end
    for key in ("plan_julia_threads", "plan_blas_threads")
        value = cfg[key]
        value === nothing || value >= 1 ||
            error("--$(replace(key, "_" => "-")) must be at least 1")
    end
    if !cfg["print_parallel_plan"] &&
       (cfg["plan_julia_threads"] !== nothing || cfg["plan_blas_threads"] !== nothing)
        error("--plan-julia-threads and --plan-blas-threads only apply with --print-parallel-plan")
    end
    return cfg
end

function campaign_hamiltonian_parameters(N::Int, cfg)
    bc = Symbol(cfg["bc"])
    if cfg["model"] == "ising"
        return IsingParameters(N, cfg["J"], cfg["h"], bc)
    elseif cfg["model"] == "niising"
        return NiIsingParameters(N, cfg["J"], cfg["hx"], cfg["hz"], bc)
    end
    error("unknown model '$(cfg["model"])'")
end

function campaign_base_detuning_reference(ham_params, cfg)
    if cfg["measure_modes"] && supports_ising_fourier_observables(ham_params)
        if !ising_mode_detuning_preserves_px(cfg["coupling"])
            cfg["delta_min"] === nothing && error(
                "--measure-modes automatic detuning currently assumes a " *
                "parity-preserving system coupling in the Ising code basis. " *
                "Use an explicit --delta-min/--delta-max for coupling " *
                "$(cfg["coupling"])."
            )
            # This source labels the reference gap stored in HDF5; the actual
            # detuning range is still the explicit fixed interval from cfg.
            return (delta=nothing, source="setup_gap")
        end
        if ising_mode_detuning_has_special_modes(ham_params)
            cfg["delta_min"] === nothing && error(
                "--measure-modes automatic detuning currently assumes the " *
                "reference Fourier grid has no special modes. Use an explicit " *
                "--delta-min/--delta-max for $(ham_params.bc) boundary conditions."
            )
            return (delta=nothing, source="setup_gap")
        end
        return (
            delta=ising_mode_detuning_reference(ham_params),
            source="ising_mode_pair_reference",
        )
    end
    return (delta=nothing, source="setup_gap")
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

function shell_word(x)
    s = string(x)
    isempty(s) && return "''"
    if occursin(r"[^A-Za-z0-9_@%+=:,./~-]", s)
        return "'" * replace(s, "'" => "'\\''") * "'"
    end
    return s
end

function parallel_progress_csv_path(base_path::AbstractString, run_cfg)
    stem, ext = splitext(basename(base_path))
    suffix = @sprintf(
        "_N%d_%s_R%d_Dmax%d",
        only(run_cfg["Ns"]),
        only(run_cfg["methods"]),
        only(run_cfg["R_values"]),
        run_cfg["Dmax"],
    )
    return joinpath(dirname(base_path), stem * suffix * ext)
end

function parallel_plan_configs(cfg)
    dmax_values = campaign_dmax_values(cfg)
    njobs = length(cfg["Ns"]) * length(cfg["methods"]) *
            length(cfg["R_values"]) * length(dmax_values)
    if cfg["output"] !== nothing && njobs > 1
        error("--output cannot be used with a multi-job --print-parallel-plan")
    end

    jobs = Dict{String,Any}[]
    for D in dmax_values, N in cfg["Ns"], method in cfg["methods"], R in cfg["R_values"]
        run_cfg = merge(copy(cfg), Dict{String,Any}(
            "Ns" => [N],
            "methods" => [method],
            "R_values" => [R],
            "Dmax" => D,
            "Dmax_values" => nothing,
            "print_parallel_plan" => false,
        ))
        if cfg["progress_csv"] !== nothing && njobs > 1
            run_cfg["progress_csv"] = parallel_progress_csv_path(cfg["progress_csv"], run_cfg)
        end
        push!(jobs, run_cfg)
    end
    return jobs
end

function command_args_for_config(cfg; script_path=joinpath("scripts", "validation",
                                                           "run_largeN_multifrequency_tn_scaling.jl"))
    args = String[
        "julia",
        "--project=.",
        script_path,
        "--Ns", join(cfg["Ns"], ","),
        "--R-values", join(cfg["R_values"], ","),
        "--methods", join(cfg["methods"], ","),
        "--evolution-method", cfg["evolution_method"],
        "--model", cfg["model"],
        "--bc", cfg["bc"],
        "--steps", string(cfg["steps"]),
        "--Dmax", string(cfg["Dmax"]),
        "--cutoff", string(cfg["cutoff"]),
        "--tau", string(cfg["tau"]),
        "--J", string(cfg["J"]),
        "--coupling", cfg["coupling"],
        "--g", string(cfg["g"]),
        "--te", string(cfg["te"]),
        "--init-state", cfg["init_state"],
        "--theta", string(cfg["theta"]),
        "--delta-max-factor", string(cfg["delta_max_factor"]),
        "--schedule", cfg["schedule"],
        "--M-mcwf", string(cfg["M_mcwf"]),
        "--M-mpo", string(cfg["M_mpo"]),
        "--seed", string(cfg["seed"]),
        "--outdir", cfg["outdir"],
    ]
    if cfg["delta_min"] !== nothing
        append!(args, ["--delta-min", string(cfg["delta_min"]),
                       "--delta-max", string(cfg["delta_max"])])
    end
    if cfg["model"] == "ising"
        append!(args, ["--h", string(cfg["h"])])
    else
        append!(args, ["--hx", string(cfg["hx"]), "--hz", string(cfg["hz"])])
    end
    if cfg["output"] !== nothing
        append!(args, ["--output", cfg["output"]])
    end
    if cfg["progress_csv"] !== nothing
        append!(args, ["--progress-csv", cfg["progress_csv"]])
    end
    cfg["tdvp_outputlevel"] == 0 ||
        append!(args, ["--tdvp-outputlevel", string(cfg["tdvp_outputlevel"])])
    cfg["tdvp_sweep_progress"] && push!(args, "--tdvp-sweep-progress")
    cfg["stop_on_bond_cap"] && push!(args, "--stop-on-bond-cap")
    cfg["randomize_times"] && push!(args, "--randomize-times")
    cfg["measure_modes"] && push!(args, "--measure-modes")
    cfg["mode_measurement_stride"] == 1 ||
        append!(args, ["--mode-measurement-stride", string(cfg["mode_measurement_stride"])])
    cfg["verbose"] && push!(args, "--verbose")
    return args
end

function parallel_plan_environment_assignments(cfg)
    assignments = String[]
    if cfg["plan_julia_threads"] !== nothing
        push!(assignments, "JULIA_NUM_THREADS=$(cfg["plan_julia_threads"])")
    end
    if cfg["plan_blas_threads"] !== nothing
        blas_threads = cfg["plan_blas_threads"]
        append!(
            assignments,
            [
                "OPENBLAS_NUM_THREADS=$blas_threads",
                "MKL_NUM_THREADS=$blas_threads",
                "BLIS_NUM_THREADS=$blas_threads",
            ],
        )
    end
    return assignments
end

function parallel_plan_command(run_cfg)
    words = vcat(
        parallel_plan_environment_assignments(run_cfg),
        command_args_for_config(run_cfg),
    )
    return join(shell_word.(words), " ")
end

parallel_plan_commands(cfg) =
    [parallel_plan_command(run_cfg) for run_cfg in parallel_plan_configs(cfg)]

function print_parallel_plan(cfg; io=stdout)
    commands = parallel_plan_commands(cfg)
    println(io, "# large-N campaign parallel job plan")
    println(io, "# jobs: $(length(commands))")
    println(io, "# This driver does not launch jobs concurrently.")
    println(io, "# Run these commands with an external process scheduler.")
    println(io, "# Each command has a distinct HDF5 output path; progress CSV paths are distinct when a base --progress-csv is supplied.")
    for command in commands
        println(io, command)
    end
    return commands
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

"""Return the diagnostic stop reason when any recorded bond cap is reached."""
function bond_cap_stop_reason(step, saturation_threshold, sys_maxbond,
                              evolved_maxbond, tdvp_sweep_maxbond)
    system_hit = sys_maxbond[step] >= saturation_threshold
    evolved_hit = evolved_maxbond[step] >= saturation_threshold
    tdvp_sweep_hit = tdvp_sweep_maxbond[step] >= saturation_threshold
    return (system_hit || evolved_hit || tdvp_sweep_hit) ? "bond_cap" : nothing
end

const PROGRESS_CSV_COLUMNS = (
    "timestamp",
    "N",
    "method",
    "evolution",
    "R",
    "trajectory",
    "seed",
    "Dmax",
    "cutoff",
    "tau",
    "stage",
    "step",
    "cycle",
    "delta",
    "te",
    "energy_per_site",
    "relative_energy",
    "overlap",
    "system_max_bond",
    "system_mean_bond",
    "evolved_max_bond",
    "evolved_mean_bond",
    "tdvp_sweep",
    "tdvp_time",
    "elapsed_seconds",
)

"""Return a single RFC-4180-compatible CSV cell for scalar progress data."""
function csv_cell(x)
    x === nothing && return ""
    s = x isa AbstractFloat && isnan(x) ? "NaN" : string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

"""Append one flushed progress row, creating the header when the file is new."""
function append_progress_csv_row(path::AbstractString, row)
    mkpath(dirname(path))
    expected_header = join(PROGRESS_CSV_COLUMNS, ",")
    needs_header = !isfile(path) || filesize(path) == 0
    if !needs_header
        existing_header = open(readline, path)
        if existing_header != expected_header
            throw(ArgumentError(
                "Progress CSV header in $path does not match the current schema. " *
                "Use a new --progress-csv path or remove the stale file before appending."
            ))
        end
    end
    open(path, "a") do io
        needs_header && println(io, expected_header)
        println(io, join((csv_cell(get(row, col, "")) for col in PROGRESS_CSV_COLUMNS), ","))
        flush(io)
    end
    return nothing
end

"""
    progress_row(context, info, ham_params, E0, sys_bs, evolved_bs, elapsed)

Build one progress CSV row from a `run_cooling_multi_freq` observer event.
Energy and overlap are defined only for `:initial` and `:updated`, where the
system has been measured. For `:prepared` and `:evolved`, these columns are
`NaN`; the `system_*_bond` columns describe the pre-update system state, while
`evolved_*_bond` describes the current evolved system-bath state.
"""
function progress_base_row(context, ham_params; stage, step, cycle, delta, te,
                           energy_per_site, relative_energy_value, overlap,
                           sys_bs, evolved_bs, tdvp_sweep=NaN, tdvp_time=NaN,
                           elapsed)
    return Dict{String,Any}(
        "timestamp" => Dates.format(now(), Dates.ISODateTimeFormat),
        "N" => ham_params.N,
        "method" => context.method,
        "evolution" => context.evolution,
        "R" => context.R,
        "trajectory" => context.trajectory,
        "seed" => context.seed,
        "Dmax" => context.Dmax,
        "cutoff" => context.cutoff,
        "tau" => context.tau,
        "stage" => stage,
        "step" => step,
        "cycle" => cycle,
        "delta" => delta,
        "te" => te,
        "energy_per_site" => energy_per_site,
        "relative_energy" => relative_energy_value,
        "overlap" => overlap,
        "system_max_bond" => sys_bs.max,
        "system_mean_bond" => sys_bs.mean,
        "evolved_max_bond" => evolved_bs.max,
        "evolved_mean_bond" => evolved_bs.mean,
        "tdvp_sweep" => tdvp_sweep,
        "tdvp_time" => tdvp_time,
        "elapsed_seconds" => elapsed,
    )
end

function progress_row(context, info, ham_params, E0, sys_bs, evolved_bs, elapsed)
    has_energy = info.stage === :initial || info.stage === :updated
    E = has_energy ? info.measurements[RESULT_ENERGY][info.step] : NaN
    overlap = has_energy ? info.measurements[RESULT_GROUND_STATE_OVERLAP][info.step] : NaN
    return progress_base_row(
        context, ham_params;
        stage=string(info.stage),
        step=info.step,
        cycle=info.step - 1,
        delta=info.delta,
        te=info.te,
        energy_per_site=has_energy ? E / ham_params.N : NaN,
        relative_energy_value=has_energy ? relative_energy(E, E0) : NaN,
        overlap=overlap,
        sys_bs=sys_bs,
        evolved_bs=evolved_bs,
        elapsed=elapsed,
    )
end

tdvp_physical_time(t::Real) = Float64(t)
tdvp_physical_time(t::Complex) = -Float64(imag(t))

function tdvp_sweep_progress_row(context, tdvp_context, sweep, current_time,
                                 ham_params, evolved_bs, elapsed)
    return progress_base_row(
        context, ham_params;
        stage="tdvp_sweep",
        step=tdvp_context.step,
        cycle=tdvp_context.step - 1,
        delta=tdvp_context.delta,
        te=tdvp_context.te,
        energy_per_site=NaN,
        relative_energy_value=NaN,
        overlap=NaN,
        sys_bs=tdvp_context.sys_bs,
        evolved_bs=evolved_bs,
        tdvp_sweep=sweep,
        tdvp_time=tdvp_physical_time(current_time),
        elapsed=elapsed,
    )
end

"""
    validate_mode_measurement_result(result, energy)

Validate the Bogoliubov mode-observable payload produced by one large-`N`
trajectory.  The wrapper first requires the complete set of mode datasets, then
delegates the cycle rows, matrix shapes, measured-row finiteness, energy-length
agreement, and `mode_nk = mode_occupation_from_hk(mode_hk)` relation to
`validate_mode_measurement_rows`.  On success, it returns the same
`(cycles, rows)` named tuple.
"""
function validate_mode_measurement_result(result, energy)
    all(k -> haskey(result, k) && result[k] !== nothing,
        RESULT_MODE_OBSERVABLE_PAYLOAD_KEYS) || error(
        "--measure-modes was requested, but the run did not produce a complete " *
        "Ising Fourier-mode measurement set"
    )
    return validate_mode_measurement_rows(
        result[RESULT_MODE_HK],
        result[RESULT_MODE_NK],
        result[RESULT_MODE_MEASUREMENT_CYCLES];
        energy=energy,
    )
end

function run_one_trajectory(problem, ham_params, cp_multi, sim_params, cfg, seed;
                            method, R, trajectory, E0)
    steps = cp_multi.steps
    Random.seed!(seed)
    state = setup_initial_state(problem, sim_params, cfg["init_state"], cfg["theta"])

    sys_maxbond = zeros(Int, steps + 1)
    sys_meanbond = zeros(Float64, steps + 1)
    # The initial row has no evolved system-bath state; exclude this sentinel
    # from peak evolved-bond summaries.
    evolved_maxbond = fill(0, steps + 1)
    evolved_meanbond = fill(NaN, steps + 1)
    tdvp_sweep_maxbond = zeros(Int, steps + 1)
    final_dims = Ref(Int[])
    progress_csv = cfg["progress_csv"]
    progress_context = (
        method=method,
        evolution=cfg["evolution_method"],
        R=R,
        trajectory=trajectory,
        seed=seed,
        Dmax=cfg["Dmax"],
        cutoff=cfg["cutoff"],
        tau=cfg["tau"],
    )
    start_time = time()
    tdvp_context = Ref{Any}(nothing)
    records_tdvp_sweeps = cfg["tdvp_sweep_progress"] &&
                          cfg["evolution_method"] == "continuous" && method == "mcwf"
    saturation_threshold = CoolingTNS.tn_method_maxdim(sim_params.sim_method, cfg["Dmax"])

    step_observer = info -> begin
        step = info.step
        elapsed = time() - start_time
        if info.stage === :prepared || info.stage === :evolved
            evolved_bs = (info.stage === :evolved || progress_csv !== nothing) ?
                bond_summary(info.evolved_state) : nothing
            sys_bs = progress_csv !== nothing ? bond_summary(info.state.state) : nothing
            if info.stage === :prepared
                if records_tdvp_sweeps
                    tdvp_sweep_maxbond[step] = 0
                    tdvp_context[] = (
                        step=step,
                        delta=info.delta,
                        te=info.te,
                        sys_bs=sys_bs,
                    )
                end
            end
            if info.stage === :evolved
                evolved_maxbond[step] = evolved_bs.max
                evolved_meanbond[step] = evolved_bs.mean
                tdvp_context[] = nothing
            end
            if progress_csv !== nothing
                append_progress_csv_row(
                    progress_csv,
                    progress_row(progress_context, info, ham_params, E0,
                                 sys_bs, evolved_bs, elapsed),
                )
            end
            return nothing
        end

        bs = bond_summary(info.state.state)
        sys_maxbond[step] = bs.max
        sys_meanbond[step] = bs.mean
        final_dims[] = bs.dims
        if progress_csv !== nothing
            evolved_bs = step == 1 ?
                (max=NaN, mean=NaN) :
                (max=evolved_maxbond[step], mean=evolved_meanbond[step])
            append_progress_csv_row(
                progress_csv,
                progress_row(
                    progress_context, info, ham_params, E0, bs, evolved_bs, elapsed,
                ),
            )
        end

        if cfg["verbose"] && step > 1
            E = info.measurements[RESULT_ENERGY][step]
            @printf("    step %d/%d delta=%.6g E/N=%.8f sysD=%d evolvedD=%d\n",
                    step - 1, steps, info.delta, E / ham_params.N,
                    sys_maxbond[step], evolved_maxbond[step])
            flush(stdout)
        end
    end

    tdvp_observer = if records_tdvp_sweeps
        tdvp_sweep_observer((; state, sweep, current_time, kwargs...) -> begin
            ctx = tdvp_context[]
            ctx === nothing && return nothing
            elapsed = time() - start_time
            evolved_bs = bond_summary(state)
            tdvp_sweep_maxbond[ctx.step] = max(
                tdvp_sweep_maxbond[ctx.step], evolved_bs.max,
            )
            if progress_csv !== nothing
                append_progress_csv_row(
                    progress_csv,
                    tdvp_sweep_progress_row(
                        progress_context, ctx, sweep, current_time, ham_params,
                        evolved_bs, elapsed,
                    ),
                )
            end
            return nothing
        end)
    else
        nothing
    end
    evolution_kwargs = if cfg["evolution_method"] == "continuous" && method == "mcwf"
        kwargs = (tdvp_outputlevel=cfg["tdvp_outputlevel"],)
        tdvp_observer === nothing ? kwargs :
            merge(kwargs, (tdvp_sweep_observer! = tdvp_observer,))
    else
        (;)
    end

    stop_condition = if cfg["stop_on_bond_cap"]
        info -> begin
            step = info.step
            return bond_cap_stop_reason(
                step, saturation_threshold, sys_maxbond, evolved_maxbond,
                tdvp_sweep_maxbond,
            )
        end
    else
        nothing
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
            evolution_kwargs=evolution_kwargs,
            measure_modes=cfg["measure_modes"],
            stop_condition=stop_condition,
            mode_measurement_stride=cfg["mode_measurement_stride"],
        )
    end

    completed_steps = get(result, RESULT_COMPLETED_STEPS, steps)
    completed_index = completed_steps + 1
    E = result[RESULT_ENERGY]
    overlap = result[RESULT_GROUND_STATE_OVERLAP]
    purity = get(result, RESULT_PURITY, fill(NaN, completed_index))
    delta_list = result[RESULT_DELTA_LIST]
    te_list = result[RESULT_TE_LIST]
    sys_maxbond_completed = sys_maxbond[1:completed_index]
    sys_meanbond_completed = sys_meanbond[1:completed_index]
    evolved_maxbond_completed = evolved_maxbond[1:completed_index]
    evolved_meanbond_completed = evolved_meanbond[1:completed_index]
    tdvp_sweep_maxbond_completed = tdvp_sweep_maxbond[1:completed_index]
    if cfg["measure_modes"]
        validate_mode_measurement_result(result, E)
    end

    return Dict{String,Any}(
        "E" => E,
        "overlap" => overlap,
        "purity" => purity,
        "sys_maxbond" => sys_maxbond_completed,
        "sys_meanbond" => sys_meanbond_completed,
        "evolved_maxbond" => evolved_maxbond_completed,
        "evolved_meanbond" => evolved_meanbond_completed,
        "tdvp_sweep_maxbond" => tdvp_sweep_maxbond_completed,
        RESULT_DELTA_LIST => delta_list,
        RESULT_TE_LIST => te_list,
        "final_bond_dims" => final_dims[],
        "elapsed" => elapsed,
        "seed" => seed,
        RESULT_REQUESTED_STEPS => get(result, RESULT_REQUESTED_STEPS, steps),
        RESULT_COMPLETED_STEPS => completed_steps,
        "stop_reason" => get(result, RESULT_STOP_REASON, ""),
        RESULT_MODE_GF => get(result, RESULT_MODE_GF, nothing),
        RESULT_MODE_GF_SOURCE => get(result, RESULT_MODE_GF_SOURCE, nothing),
        RESULT_MODE_HK => get(result, RESULT_MODE_HK, nothing),
        RESULT_MODE_NK => get(result, RESULT_MODE_NK, nothing),
        RESULT_MODE_K_INDICES => get(result, RESULT_MODE_K_INDICES, nothing),
        RESULT_MODE_ENERGIES => get(result, RESULT_MODE_ENERGIES, nothing),
        RESULT_MODE_MEASUREMENT_CYCLES => get(result, RESULT_MODE_MEASUREMENT_CYCLES, nothing),
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
    model_suffix = cfg["model"] == "niising" && cfg["bc"] == "open" ? "" :
        "_$(cfg["model"])_bc$(cfg["bc"])"
    stop_suffix = cfg["stop_on_bond_cap"] ? "_stopcap" : ""
    schedule_suffix = cfg["schedule_symbol"] == :round_robin ? "" :
        "_sched$(multi_frequency_schedule_token(cfg["schedule_symbol"]))"
    random_time_suffix = cfg["randomize_times"] ? "_randtime" : ""
    mode_suffix = if !cfg["measure_modes"]
        ""
    elseif cfg["mode_measurement_stride"] == 1
        "_modes"
    else
        "_modestride$(cfg["mode_measurement_stride"])"
    end
    init_suffix = if cfg["init_state"] == "product"
        ""
    elseif cfg["init_state"] == "identity"
        "_initidentity"
    else
        @sprintf("_init%s_theta%.12g", cfg["init_state"], cfg["theta"])
    end
    # `te` is a scanned physical protocol parameter, so keep more digits than
    # the legacy `tau` token to avoid collisions between nearby evolution times.
    return joinpath(
        cfg["outdir"],
        @sprintf(
            "largeN_multifrequency_tn_N%s_R%s_%s%s%s%s%s%s%s_steps%d_Dmax%d_te%.12g_tau%.3g_seed%d.h5",
            Ns,
            Rs,
            methods,
            evolution_suffix,
            model_suffix,
            stop_suffix,
            schedule_suffix * random_time_suffix,
            mode_suffix,
            init_suffix,
            cfg["steps"],
            cfg["Dmax"],
            cfg["te"],
            cfg["tau"],
            cfg["seed"],
        ),
    )
end

mode_dataset_present(row) = get(row, RESULT_MODE_HK, nothing) !== nothing

function common_mode_metadata(traj_rows, key::AbstractString)
    value = traj_rows[1][key]
    for row in traj_rows[2:end]
        isequal(row[key], value) ||
            error("mode metadata '$key' differs across trajectories")
    end
    return value
end

"""
Write ensemble-mean, standard-error, and trajectory-resolved Bogoliubov mode
measurements for one `(N, method, R)` campaign group.
"""
function write_mode_measurement_group!(g, traj_rows)
    any_mode = any(mode_dataset_present, traj_rows)
    any_mode || return nothing
    all(mode_dataset_present, traj_rows) ||
        error("mode measurements are present for only part of the trajectory ensemble")

    for key in RESULT_MODE_OBSERVABLE_PAYLOAD_KEYS
        all(row -> get(row, key, nothing) !== nothing, traj_rows) ||
            error("incomplete mode measurement set: missing $key")
    end

    k_indices = common_mode_metadata(traj_rows, RESULT_MODE_K_INDICES)
    mode_energies = common_mode_metadata(traj_rows, RESULT_MODE_ENERGIES)
    mode_cycles = common_mode_metadata(traj_rows, RESULT_MODE_MEASUREMENT_CYCLES)
    mode_gF = common_mode_metadata(traj_rows, RESULT_MODE_GF)
    mode_gF_source = common_mode_metadata(traj_rows, RESULT_MODE_GF_SOURCE)

    mode_hk = cat([Float64.(row[RESULT_MODE_HK]) for row in traj_rows]...; dims=3)
    mode_nk = cat([Float64.(row[RESULT_MODE_NK]) for row in traj_rows]...; dims=3)
    M = size(mode_hk, 3)
    mode_hk_mean = dropdims(mean(mode_hk; dims=3); dims=3)
    mode_nk_mean = dropdims(mean(mode_nk; dims=3); dims=3)
    measured = validate_mode_measurement_rows(mode_hk_mean, mode_nk_mean, mode_cycles)
    mode_hk_stderr = M == 1 ? zeros(size(mode_hk_mean)) :
        dropdims(std(mode_hk; dims=3); dims=3) ./ sqrt(M)
    mode_nk_stderr = M == 1 ? zeros(size(mode_nk_mean)) :
        dropdims(std(mode_nk; dims=3); dims=3) ./ sqrt(M)
    measured_rows = measured.rows
    unmeasured_rows = setdiff(1:size(mode_hk_mean, 1), measured_rows)
    if !isempty(unmeasured_rows)
        mode_hk_stderr[unmeasured_rows, :] .= NaN
        mode_nk_stderr[unmeasured_rows, :] .= NaN
    end

    write(g, RESULT_MODE_HK, mode_hk_mean)
    write(g, RESULT_MODE_NK, mode_nk_mean)
    write(g, RESULT_MODE_K_INDICES, Float64.(k_indices))
    write(g, RESULT_MODE_ENERGIES, Float64.(mode_energies))
    write(g, RESULT_MODE_MEASUREMENT_CYCLES, Int.(mode_cycles))
    write(g, RESULT_MODE_GF, Int(mode_gF))
    write(g, RESULT_MODE_GF_SOURCE, String(mode_gF_source))
    write(g, RESULT_MODE_HK_TRAJECTORIES, mode_hk)
    write(g, RESULT_MODE_NK_TRAJECTORIES, mode_nk)
    write(g, RESULT_MODE_HK_STDERR, mode_hk_stderr)
    write(g, RESULT_MODE_NK_STDERR, mode_nk_stderr)
    return nothing
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
    tdvp_sweep_maxbond = reduce(hcat, [row["tdvp_sweep_maxbond"] for row in traj_rows])
    delta_lists = reduce(hcat, [row[RESULT_DELTA_LIST] for row in traj_rows])
    te_lists = reduce(hcat, [row[RESULT_TE_LIST] for row in traj_rows])
    common_delta_list = all(j -> isequal(delta_lists[:, j], delta_lists[:, 1]), 1:M)
    common_te_list = all(j -> isequal(te_lists[:, j], te_lists[:, 1]), 1:M)

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
    tdvp_sweep_saturation_cycles = Int[
        first_bond_saturation_cycle(row["tdvp_sweep_maxbond"], saturation_threshold)
        for row in traj_rows
    ]

    write(g, "M", M)
    write(g, RESULT_ENERGY_TRAJECTORIES, E)
    write(g, RESULT_ENERGY, E_mean)
    write(g, "E_stderr", E_stderr)
    write(g, RESULT_RELATIVE_ENERGY, rel_mean)
    write(g, RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES, overlap)
    write(g, RESULT_GROUND_STATE_OVERLAP, vec(mean(overlap; dims=2)))
    write(g, "purity_trajectories", purity)
    write(g, "system_max_bond", sys_maxbond)
    write(g, "system_mean_bond", sys_meanbond)
    write(g, "evolved_max_bond", evolved_maxbond)
    write(g, "evolved_mean_bond", evolved_meanbond)
    write(g, "tdvp_sweep_max_bond", tdvp_sweep_maxbond)
    write(g, "bond_saturation_threshold", saturation_threshold)
    write(g, "system_saturation_cycle", system_saturation_cycles)
    write(g, "evolved_saturation_cycle", evolved_saturation_cycles)
    write(g, "tdvp_sweep_saturation_cycle", tdvp_sweep_saturation_cycles)
    write(g, RESULT_TRUNCATION_ERROR_HISTORY_STATUS, TRUNCATION_ERROR_HISTORY_NOT_RECORDED)
    write(g, "elapsed_seconds", Float64[row["elapsed"] for row in traj_rows])
    write(g, "trajectory_seeds", Int[row["seed"] for row in traj_rows])
    write(
        g,
        RESULT_REQUESTED_STEPS,
        Int[row[RESULT_REQUESTED_STEPS] for row in traj_rows],
    )
    write(
        g,
        RESULT_COMPLETED_STEPS,
        Int[row[RESULT_COMPLETED_STEPS] for row in traj_rows],
    )
    write(g, "stop_reasons", String[row["stop_reason"] for row in traj_rows])
    write(g, "delta_lists", delta_lists)
    write(g, "delta_list_first_trajectory", delta_lists[:, 1])
    write(g, "delta_list_is_common", common_delta_list)
    common_delta_list && write(g, RESULT_DELTA_LIST, delta_lists[:, 1])
    write(g, "te_lists", te_lists)
    write(g, "te_list_first_trajectory", te_lists[:, 1])
    write(g, "te_list_is_common", common_te_list)
    common_te_list && write(g, RESULT_TE_LIST, te_lists[:, 1])
    write(g, RESULT_DELTA_VALUES, Float64.(delta_values))
    write_largeN_detuning_protocol(g, detuning_protocol)
    write_mode_measurement_group!(g, traj_rows)

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
        first_tdvp_sweep_saturation_cycle=first_recorded_saturation_cycle(
            tdvp_sweep_saturation_cycles
        ),
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
        write(f, "model", cfg["model"])
        write(f, "bc", cfg["bc"])
        write(f, "evolution_method", cfg["evolution_method"])
        write(f, "steps", cfg["steps"])
        write(f, "Dmax", cfg["Dmax"])
        write(f, "cutoff", cfg["cutoff"])
        write(f, "tau", cfg["tau"])
        write(f, "J", cfg["J"])
        write(f, "h", cfg["model"] == "ising" ? cfg["h"] : NaN)
        write(f, "hx", cfg["model"] == "niising" ? cfg["hx"] : NaN)
        write(f, "hz", cfg["model"] == "niising" ? cfg["hz"] : NaN)
        write(f, "measure_modes", cfg["measure_modes"])
        write(f, "mode_measurement_stride", cfg["mode_measurement_stride"])
        write(f, "tdvp_sweep_progress", cfg["tdvp_sweep_progress"])
        write(f, "coupling", cfg["coupling"])
        write(f, "g", cfg["g"])
        write(f, "te", cfg["te"])
        write(f, "init_state", cfg["init_state"])
        write(f, "theta", cfg["theta"])
        write(f, RESULT_RANDOMIZE_TIMES, cfg["randomize_times"])
        write(f, "delta_max_factor", cfg["delta_max_factor"])
        write(f, "delta_min_override", cfg["delta_min"] === nothing ? NaN : cfg["delta_min"])
        write(f, "delta_max_override", cfg["delta_max"] === nothing ? NaN : cfg["delta_max"])
        write(f, RESULT_SCHEDULE, cfg["schedule"])
        write(f, "stop_on_bond_cap", cfg["stop_on_bond_cap"])
        write(f, "seed", cfg["seed"])
        write(f, "trajectory_seed_rule", LARGE_N_TRAJECTORY_SEED_RULE)

        for N in cfg["Ns"]
            ham_params = campaign_hamiltonian_parameters(N, cfg)
            if cfg["measure_modes"] && !supports_ising_fourier_observables(ham_params)
                error(
                    "--measure-modes requires an even-size integrable Ising chain " *
                    "with periodic or antiperiodic spin boundary conditions"
                )
            end
            gn = create_group(f, "N$N")
            write(gn, "N", N)

            for method in cfg["methods"]
                sim_params = sim_params_for(method, cfg)
                base_detuning = campaign_base_detuning_reference(ham_params, cfg)
                cp_base = BasicCouplingParameters(
                    cfg["coupling"], cfg["g"], 1, cfg["te"], base_detuning.delta
                )
                @printf("\nsetup N=%d method=%s\n", N, method)
                Random.seed!(cfg["seed"] + 1000 * N)
                base_problem = setup_problem(backend, ham_params, cp_base, sim_params)
                gap = Float64(base_problem.extra.coupling_params.delta)
                E0 = Float64(base_problem.e₀)
                @printf(
                    "  E0/N=%.10f, detuning reference=%.8f (%s; reused across R)\n",
                    E0 / N,
                    gap,
                    base_detuning.source,
                )

                gm = create_group(gn, method)
                write(gm, "E0", E0)
                write(gm, "gap", gap)
                write(gm, "detuning_reference_gap_source", base_detuning.source)
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
                        randomize_times=cfg["randomize_times"],
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
                        seed = largeN_trajectory_seed(cfg["seed"], N, R, m)
                        row = run_one_trajectory(problem, ham_params, cp_multi, sim_params,
                                                 cfg, seed; method=method, R=R,
                                                 trajectory=m, E0=E0)
                        push!(traj_rows, row)
                        peak_evolved_maxbond = maximum(row["evolved_maxbond"][2:end])
                        system_saturation_cycle = first_bond_saturation_cycle(
                            row["sys_maxbond"], saturation_threshold
                        )
                        evolved_saturation_cycle = first_bond_saturation_cycle(
                            row["evolved_maxbond"], saturation_threshold
                        )
                        tdvp_sweep_saturation_cycle = first_bond_saturation_cycle(
                            row["tdvp_sweep_maxbond"], saturation_threshold
                        )
                        @printf("  traj %d/%d seed=%d final E/N=%.8f rel=%.6g final_sysD=%d peak_evolvedD=%d sysSat=%s evolvedSat=%s tdvpSweepSat=%s elapsed=%.1fs\n",
                                m, M, seed, row["E"][end] / N,
                                relative_energy(row["E"][end], E0),
                                row["sys_maxbond"][end],
                                peak_evolved_maxbond,
                                saturation_cycle_label(system_saturation_cycle),
                                saturation_cycle_label(evolved_saturation_cycle),
                                saturation_cycle_label(tdvp_sweep_saturation_cycle),
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
                        first_tdvp_sweep_saturation_cycle=summary.first_tdvp_sweep_saturation_cycle,
                        elapsed=summary.elapsed,
                    ))
                    @printf("  mean final E/N=%.8f rel=%.6g overlap=%.6g final_sysD=%d mean_sysD=%.2f peak_evolvedD=%d mean_evolvedD=%.2f sysSat=%s evolvedSat=%s tdvpSweepSat=%s elapsed_total=%.1fs\n",
                            summary.final_E_mean / N,
                            summary.final_rel,
                            summary.final_overlap,
                            summary.final_sys_maxbond,
                            summary.final_sys_meanbond,
                            summary.peak_evolved_maxbond,
                            summary.peak_evolved_meanbond,
                            saturation_cycle_label(summary.first_system_saturation_cycle),
                            saturation_cycle_label(summary.first_evolved_saturation_cycle),
                            saturation_cycle_label(summary.first_tdvp_sweep_saturation_cycle),
                            summary.elapsed)
                    GC.gc()
                end
            end
        end
    end

    println("\nwrote $path")
    println("summary:")
    for s in summaries
        @printf("  N=%d %-4s R=%2d final E/N=%.8f rel=%.6g overlap=%.6g final_sysD=%d mean_sysD=%.2f peak_evolvedD=%d mean_evolvedD=%.2f sysSat=%s evolvedSat=%s tdvpSweepSat=%s elapsed=%.1fs\n",
                s.N, s.method, s.R, s.final_E / s.N, s.final_rel,
                s.final_overlap, s.final_sys_maxbond, s.final_sys_meanbond,
                s.peak_evolved_maxbond, s.peak_evolved_meanbond,
                saturation_cycle_label(s.first_system_saturation_cycle),
                saturation_cycle_label(s.first_evolved_saturation_cycle),
                saturation_cycle_label(s.first_tdvp_sweep_saturation_cycle),
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
    if cfg["print_parallel_plan"]
        print_parallel_plan(cfg)
        return nothing
    end
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
