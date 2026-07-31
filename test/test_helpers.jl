"""
    test_helpers.jl

Shared helpers for CoolingTNS tests.

The default test suite is designed to run quickly. Slow/stochastic tests
(Monte Carlo trajectory averaging, long tensor-network cooling runs, etc.)
are gated behind the environment variable `COOLINGTNS_FULL_TESTS`.

Tests that reach `PythonCall`/matplotlib — directly or transitively through
`scripts/plotting/PlotUtils.jl` — are gated behind `COOLINGTNS_PLOT_TESTS`,
so the suite runs on a machine with no Python environment provisioned.

Set, for example:

    COOLINGTNS_FULL_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
    COOLINGTNS_PLOT_TESTS=1 julia --project=. test/runtests.jl
"""

using Test
using CoolingTNS
using ITensors
using ITensorMPS

"""Return a boolean from an environment variable.

Truthy values: `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive).
Falsy values:  `"0"`, `"false"`, `"no"`, `"off"`.

If the variable is unset or unrecognized, returns `default`.
"""
function env_bool(name::AbstractString; default::Bool=false)
    val = get(ENV, name, nothing)
    val === nothing && return default

    sval = lowercase(strip(val))
    sval in ("1", "true", "yes", "on") && return true
    sval in ("0", "false", "no", "off") && return false

    return default
end

"""Whether slow/stochastic tests should run."""
full_tests_enabled() = env_bool("COOLINGTNS_FULL_TESTS"; default=false)

"""Whether tests that need `PythonCall`/matplotlib should run.

`using PythonCall` resolves the CondaPkg environment at module initialization,
so it fails (or downloads a conda environment) on a machine that has no Python
provisioned. Gating keeps `julia --project=. test/runtests.jl` runnable there.

Unlike `COOLINGTNS_FULL_TESTS` this defaults to **on**, because the two gates
exist for different reasons. The full-tests gate hides tests that are slow
everywhere, so opting in is the right default. These tests are fast; they are
only unrunnable where Python is absent. A developer working from this repo has
the environment (it is provisioned from `CondaPkg.toml`), so defaulting off
would silently drop 139 assertions from every local run to buy nothing. CI
sets `COOLINGTNS_PLOT_TESTS=0` explicitly, and the nightly job turns it back on.
"""
plot_tests_enabled() = env_bool("COOLINGTNS_PLOT_TESTS"; default=true)

"""Whether tests should print verbose simulation output."""
test_verbose() = env_bool("COOLINGTNS_TEST_VERBOSE"; default=false)

"""Run `f` with the simulation's own stdout/stderr chatter muted.

Honours `test_verbose()`, which the bare `redirect_stdout(devnull) do ... end`
blocks this replaces did not: `COOLINGTNS_TEST_VERBOSE=1` now actually shows the
cooling output. `quiet` consumes no randomness, so it is safe to wrap a run that
sits directly under a `Random.seed!` pin.
"""
function quiet(f)
    test_verbose() && return f()
    return redirect_stdout(devnull) do
        redirect_stderr(f, devnull)
    end
end

"""Assert that a required phrase occurs in flattened text."""
function require_phrase(text::AbstractString, phrase::AbstractString)
    @test occursin(phrase, text)
end

"""Assert that every required phrase occurs in flattened text."""
function require_phrases(text::AbstractString, phrases)
    for phrase in phrases
        require_phrase(text, phrase)
    end
end

"""Assert that a stale or forbidden phrase does not occur in flattened text."""
function forbid_phrase(text::AbstractString, phrase::AbstractString)
    @test !occursin(phrase, text)
end

"""Assert that no stale or forbidden phrase occurs in flattened text."""
function forbid_phrases(text::AbstractString, phrases)
    for phrase in phrases
        forbid_phrase(text, phrase)
    end
end

"""Convert an ITensor MPO to a dense matrix in the ED bit ordering.

The test convention matches `appendzeros_MPO`: primed site indices are matrix
rows and unprimed site indices are matrix columns. Site tag `n=1` is the least
significant ED bit.
"""
function test_mpo_to_matrix(O::MPO)
    nsites = length(O)
    tensor = O[1]
    for site in 2:nsites
        tensor *= O[site]
    end

    row_inds = Index[]
    col_inds = Index[]
    for ind in inds(tensor)
        if hastags(ind, "Site") && plev(ind) == 1
            push!(row_inds, ind)
        elseif hastags(ind, "Site") && plev(ind) == 0
            push!(col_inds, ind)
        end
    end

    site_number(ind) = parse(Int, match(r"n=(\d+)", string(tags(ind))).captures[1])
    sort!(row_inds; by=site_number)
    sort!(col_inds; by=site_number)

    dim_total = 2^nsites
    matrix = zeros(ComplexF64, dim_total, dim_total)
    for row in 0:(dim_total - 1), col in 0:(dim_total - 1)
        vals = Dict{Index, Int}()
        for site in 1:nsites
            vals[row_inds[site]] = ((row >> (site - 1)) & 1) + 1
            vals[col_inds[site]] = ((col >> (site - 1)) & 1) + 1
        end
        matrix[row + 1, col + 1] = tensor[vals...]
    end
    return matrix
end

"""Run a full cooling simulation and return `(results, problem, sim_params)`.

This is a convenience helper used by cross-validation tests.
"""
function run_cooling_case(
    backend,
    sim_method,
    evolution_method;
    ham_params,
    coupling_params,
    Dmax=50,
    cutoff=1e-10,
    tau=0.1,
    pe=0.0,
    n_trajectories=1,
    init_type="product",
    theta=0.0,
    measure_modes=false,
    mode_measurement_stride=nothing,
)
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        sim_method,
        evolution_method;
        Dmax=Dmax,
        cutoff=cutoff,
        tau=tau,
        pe=pe,
        n_trajectories=n_trajectories,
    )

    problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    state0 = CoolingTNS.setup_initial_state(problem, sim_params, init_type, theta)
    results = quiet() do
        CoolingTNS.run_cooling(
            problem, state0, coupling_params, sim_params, ham_params;
            measure_modes=measure_modes,
            mode_measurement_stride=mode_measurement_stride,
        )
    end

    return results, problem, sim_params
end

"""String-based wrapper for [`run_cooling_case`](@ref)."""
function run_cooling_case(
    ;
    backend_str::AbstractString,
    sim_method_str::AbstractString,
    evolution_method_str::AbstractString,
    ham_params,
    coupling_params,
    kwargs...,
)
    backend = CoolingTNS.get_backend(backend_str)
    sim_method = CoolingTNS.get_sim_method(sim_method_str)
    evolution_method = CoolingTNS.get_evolution_method(evolution_method_str)

    return run_cooling_case(
        backend,
        sim_method,
        evolution_method;
        ham_params=ham_params,
        coupling_params=coupling_params,
        kwargs...,
    )
end
