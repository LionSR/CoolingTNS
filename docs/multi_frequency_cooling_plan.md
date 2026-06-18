# Multi-Frequency Cooling for Interacting Spin Systems

## Motivation

Paper 1 (arXiv:2503.24330) showed that for free-fermion chains, cooling with a
single bath frequency Δ only cools modes near resonance (ε_k ≈ Δ), and can
even heat modes where (ε_k − Δ)t ≈ 2πr (accidental resonances). Two key
strategies fix this:

1. **Multi-frequency cooling**: cycle through R bath frequencies
   Δ_1, …, Δ_R, each targeting a different part of the spectrum.
2. **Randomized cycle times**: draw t_m from Uniform(0, 2t) to wash out
   accidental resonances.

These results were derived analytically for the free-fermion (Gaussian) case.
The purpose of our follow-up paper is to demonstrate and study these strategies
for **interacting (non-integrable) spin models** using tensor network
simulations, where the Gaussian tricks do not apply.

## Model

Non-integrable Ising chain (open boundary conditions):

    H_S = J Σ σ^z_i σ^z_{i+1} + h_z Σ σ^z_i + h_x Σ σ^x_i

with J = 1, h_x = −1.05, h_z = 0.5. This is the same model used throughout
the CoolingAlgTN paper.

Bath: N independent spin-1/2 sites with H_B = (Δ/2) Σ bath_op_i. For a
system-bath coupling label `AB`, the bath field is chosen from the bath-side
operator `B`, so that it does not commute with the local bath coupling. The
present convention uses a Z bath field for bath-side X or Y couplings and an X
bath field for bath-side Z couplings.

System-bath coupling labels use one local product for identical operators and
the symmetric Hermitian convention for mixed operators. Thus `XX` denotes
V_SB = g Σ σ^x_{S,i} σ^x_{B,i}, while `XY` denotes
V_SB = g Σ (σ^x_{S,i} σ^y_{B,i} + σ^y_{S,i} σ^x_{B,i}).

## Architecture

### Current code path (single-Δ)

```
setup_problem(backend, ham_params, coupling_params, sim_params)
  → computes Δ = system gap (E₁ − E₀) if delta=nothing
  → builds H_SB(Δ) as MPO
  → stores in CoolingProblem.H_sys_bath
  → every cooling step reuses the same H_SB
```

### Multi-frequency code path (new)

```
run_cooling_multi_freq(problem_template, state, multi_freq_params, sim_params, ham_params)
  → for each step:
      1. pick Δ_r from schedule (round-robin or random)
      2. optionally pick t_m from Uniform(0, 2·te) if randomize_times=true
      3. rebuild H_SB(Δ_r) from OpSum  (~0.1s, negligible vs evolution)
      4. run normal cooling step: append bath → evolve(t_m) → sample bath
      5. measure
```

**Why rebuild each step?**  Building the H_SB MPO from `OpSum` costs ~0.1s.
A single TDVP evolution step for N = 50 costs ~10–60s. So the overhead of
rebuilding is < 1%. Pre-building R Hamiltonians is unnecessary complexity.

### Recommended TN method

**MPS + TDVP (MonteCarloWavefunction + ContinuousEvolution)**

- Bond dimension D ≤ 40 for a pure state ≪ D² for a density matrix
- No Trotter error (variational)
- Stochasticity handled by time-averaging the steady state (single trajectory,
  average last ~100 steps)
- This is already the faster method in the codebase

## Implementation Plan

### Step 1: Multi-frequency parameter type

New type in `src/parameter_types.jl`:

```julia
struct MultiFrequencyCouplingParameters <: CouplingParameters
    coupling::String                     # e.g. "XX"
    g::Float64                           # coupling strength
    steps::Int                           # total cooling steps
    te::Float64                          # mean evolution time
    delta_values::Vector{Float64}        # bath frequencies to cycle through
    randomize_times::Bool                # draw t_m ~ Uniform(0, 2·te)?
    schedule::Symbol                     # :round_robin or :random
end
```

### Step 2: Δ-value selection helpers

New file or section in `src/mode_analysis.jl` / new `src/multi_freq.jl`:

```julia
# Use DMRG excitations to find low-lying gaps
function compute_excitation_gaps(ham_params, backend; num_excitations=10)
    ...
    return gaps   # Vector{Float64}
end

# Uniform grid of Δ values from Δ_min to Δ_max
function uniform_delta_grid(delta_min, delta_max, R)
    return range(delta_min, delta_max, length=R) |> collect
end

# From the excitation spectrum
function spectral_delta_values(ham_params, backend; R=5)
    gaps = compute_excitation_gaps(ham_params, backend; num_excitations=2*R)
    # pick R representative gaps
    ...
end
```

### Step 3: Modified cooling loop

New function `run_cooling_multi_freq` in `src/cooling_evolution.jl`:

```julia
function run_cooling_multi_freq(problem, state, mf_params, sim_params, ham_params;
                                measure_modes=false)
    R = length(mf_params.delta_values)

    for step in 2:mf_params.steps+1
        # Pick frequency
        r = if mf_params.schedule == :round_robin
            mod1(step - 1, R)
        else  # :random
            rand(1:R)
        end
        delta_r = mf_params.delta_values[r]

        # Pick evolution time
        te_step = if mf_params.randomize_times
            rand() * 2 * mf_params.te   # Uniform(0, 2t)
        else
            mf_params.te
        end

        # Build H_SB for this step's Δ
        coupling_step = BasicCouplingParameters(
            mf_params.coupling, mf_params.g,
            mf_params.steps, te_step, delta_r)
        H_sb_step = construct_system_bath_hamiltonian(
            ham_params, problem.backend, problem.extra.sites, coupling_step)

        # Normal cooling step with this H_SB
        combined = prepare_combined_state(problem, state)
        evolved = evolve_state_with_H(combined, H_sb_step, te_step, sim_params)
        state = process_bath_and_update(problem, evolved, state, sim_params)
        perform_measurements!(measurements, step, problem, state, ham_params)
    end
    ...
end
```

### Step 4: Comparison script

`scripts/multi_freq_cooling.jl`: runs single-Δ vs multi-Δ for small N, produces
comparison plots of E/N vs step.

### Step 5: Production runs

Systematic runs for the paper:

| Run | N  | R  | Δ-values           | g    | te  | steps | D   |
|-----|----|----|---------------------|------|-----|-------|-----|
| A   | 20 | 1  | gap                 | 0.3  | 2.0 | 200   | 20  |
| B   | 20 | 5  | uniform grid        | 0.3  | 2.0 | 200   | 20  |
| C   | 20 | 5  | spectral            | 0.3  | 2.0 | 200   | 20  |
| D   | 20 | 5  | uniform + rand time | 0.3  | 2.0 | 200   | 20  |
| E   | 50 | 1  | gap                 | 0.3  | 2.0 | 500   | 40  |
| F   | 50 | 5  | uniform grid        | 0.3  | 2.0 | 500   | 40  |
| G   | 50 | 10 | uniform grid        | 0.3  | 2.0 | 500   | 40  |

### Current large-N status

The current `N=64` MCWF/MPS Trotter diagnostics show substantial transient
system-bath bond growth before the cooling trajectory approaches the ground
state.  The repository-level summary is
[`largeN_effective_bond_dimensions.md`](largeN_effective_bond_dimensions.md).
In particular, four-cycle fixed-detuning runs with `R = 1,2,5,10` already show
that `Dmax = 320` is not a converged cap, and some `Dmax = 640` schedules still
saturate the transient system-bath bond dimension.

These data should be used to design the next production campaign.  The table
above is therefore a target plan, not evidence that the listed bond dimensions
are already physically converged at large system size.

### Step 6: Analysis & figures for the paper

Key figures to produce:

1. **E/N vs step**: single-Δ vs multi-Δ (R = 1, 3, 5, 10) at fixed N
2. **Steady-state energy vs R**: analogous to Paper 1's Fig 8
3. **Steady-state energy vs N**: scaling study for single vs multi-Δ
4. **Effect of randomized times**: with vs without
5. **Coupling type comparison**: XX vs YY vs XY under multi-Δ
6. **Noise robustness**: multi-Δ cooling for pe = 0, 0.001, 0.01

## Timeline

1. **Implementation** (Steps 1–3): parameter type + loop modification + helpers
2. **Validation** (Step 4): small-N comparison, sanity checks
3. **Production** (Step 5): batch runs
4. **Paper** (Step 6): figures + text

## Open Questions

- What Δ range to use? The many-body spectrum is dense; we don't have clean
  mode energies like in the free-fermion case. Options:
  a. Uniform grid from the gap to the bandwidth
  b. Low-lying excitation energies from DMRG
  c. Heuristic: Δ_min = gap, Δ_max ≈ ||H_S||/N

- Does the interacting model have analogous "mode cooling"? The partial
  trace structure is different — there's no mode decomposition of the cooling
  map. This is a physics question worth investigating.

- Should we also compare with DSP (H_S turned off) for the interacting model?
