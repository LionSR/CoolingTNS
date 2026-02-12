"""
    test_helpers.jl

Shared helpers for CoolingTNS tests.

The default test suite is designed to run quickly. Slow/stochastic tests
(Monte Carlo trajectory averaging, long tensor-network cooling runs, etc.)
are gated behind the environment variable `COOLINGTNS_FULL_TESTS`.

Set, for example:

    COOLINGTNS_FULL_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
"""

using CoolingTNS

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

"""Whether tests should print verbose simulation output."""
test_verbose() = env_bool("COOLINGTNS_TEST_VERBOSE"; default=false)

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
    results = if test_verbose()
        CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
    else
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
            end
        end
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
