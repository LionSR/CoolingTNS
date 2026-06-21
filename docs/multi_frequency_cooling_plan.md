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
The purpose of the interacting-model follow-up is to test and study whether
these strategies carry over to **interacting (non-integrable) spin models**
using tensor-network simulations, where the Gaussian tricks do not apply.

## Model

Non-integrable Ising chain (open boundary conditions):

    H_S = J Σ σ^z_i σ^z_{i+1} + h_z Σ σ^z_i + h_x Σ σ^x_i

with J = 1, h_x = −1.05, h_z = 0.5. This is the same model used throughout
the CoolingAlgTN paper.

Bath: N independent spin-1/2 sites with H_B = (Δ/2) Σ bath_op_i. The bath
field is the Pauli operator returned by `get_bath_operator(coupling)`. For a
one-Pauli bath-side set, the historical convention uses a Z field for bath-side
X or Y and an X field for bath-side Z. For mixed symmetric labels, the
bath-side set contains two Pauli operators, and the field is the unique absent
Pauli: `XY`/`YX` use `Z`, `YZ`/`ZY` use `X`, and `XZ`/`ZX` use `Y`. This
guarantees noncommutation with every bath-side term in the system-bath
coupling. The bath state prepared by ED, MPS, and MPO routines is the
corresponding eigenvalue -1 state selected by `bath_ground_state_amplitudes`.

System-bath coupling labels use one local product for identical operators and
the symmetric Hermitian convention for mixed operators. Thus `XX` denotes
V_SB = g Σ σ^x_{S,i} σ^x_{B,i}, while `XY` denotes
V_SB = g Σ (σ^x_{S,i} σ^y_{B,i} + σ^y_{S,i} σ^x_{B,i}).

## Architecture

### Current setup-gap path (single-Δ)

```
setup_problem(backend, ham_params, coupling_params, sim_params)
  → computes the generic setup reference Δ = E₁ − E₀ if delta=nothing
  → builds H_SB(Δ) as MPO
  → stores in CoolingProblem.H_sys_bath
  → every cooling step reuses the same H_SB
```

This generic setup-gap convention remains the default for ordinary ED/TN
single-detuning runs. It is not, however, the automatic convention for the
mode-resolved integrable-Ising large-`N` diagnostics. In that case the
validation driver first chooses a detuning reference through
`campaign_base_detuning_reference`: for a parity-preserving code-basis coupling,
with `--measure-modes` and no explicit detuning interval, the reference is the
many-body generic pair scale
`2 min_{sin(phi_k) != 0} epsilon_k` on the deterministic parity-`+1` Fourier
grid selected before any trajectory-specific state parity is measured. This
agrees with `MapToSpin.tex` and `CoolingAlgTN.tex`. If later mode metadata
records a different `mode_gF`, that metadata remains the source of truth for
the stored observables; the detuning reference is setup-grid provenance, not a
reconstruction from `mode_ek_values`. The stored `mode_ek_values` remain
positive single-quasiparticle gaps used to label modes and resonance plots;
they are not by themselves the automatic bath-detuning reference for the
parity-preserving Ising cooling channel. If the reference grid contains special
modes, or if the coupling is not covered by this parity-preserving rule, the
driver requires an explicit `--delta-min/--delta-max` interval for
mode-resolved runs.

### Multi-frequency code path (new)

```
run_cooling_multi_freq(problem_template, state, multi_freq_params, sim_params, ham_params)
  → for each step:
      1. call multi_frequency_cycle_choice to choose Δ_r and t_m
         from the schedule (round-robin or random) and the randomized-time flag
      2. rebuild H_SB(Δ_r) from OpSum  (~0.1s, negligible vs evolution)
      3. run normal cooling step: append bath → evolve(t_m) → sample bath
      4. measure
```

**Why rebuild each step?**  Building the H_SB MPO from `OpSum` costs ~0.1s.
A single TDVP evolution step for N = 50 costs ~10–60s. So the overhead of
rebuilding is < 1%. Pre-building R Hamiltonians is unnecessary complexity.

### Recommended TN method

**MPS + TDVP (MonteCarloWavefunction + ContinuousEvolution)**

- The pure-state MPS trajectory representation is still the preferred large-`N`
  tensor-network route, because it avoids representing the density matrix as an
  MPO.  The numerical bond cap is not fixed by this preference: current `N=64`
  evidence shows that `D = 40` is only a pilot cap, and any production claim must
  choose `Dmax` from a convergence ladder with retained-state, transient
  system-bath, and TDVP sweep-level diagnostics.
- No Trotter error from the continuous-time TDVP integrator.  The remaining
  algorithmic errors are the usual TDVP projection and truncation errors; the
  current large-`N` HDF5 campaign records bond-cap and saturation diagnostics,
  not measured truncation-error histories.
- A single MCWF trajectory and a late-time average are diagnostic quantities
  only.  A numerical steady-state estimate requires independent trajectories,
  stability under the averaging window, and convergence in the retained and
  transient bond dimensions.
- The current timing evidence favors process-level MCWF+TDVP diagnostic scans
  with one Julia thread and one BLAS thread per independent job.  Production
  settings must still be benchmarked at the target bond cap.
- The large-`N` validation driver therefore defaults to `--methods mcwf`.
  Density-matrix/MPO runs remain available through explicit `--methods mpo` or
  `--methods mpo,mcwf` invocations, primarily for small-system channel
  validation against the MPS trajectory implementation.

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
    schedule::Symbol                     # :round_robin, :descending, or :random
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

For interacting models the grid endpoints are physical hypotheses about the
many-body spectrum. For integrable-Ising mode diagnostics they must also be
consistent with the mode-observable sector: the automatic endpoint source is
`ising_mode_detuning_reference`, while an explicit interval is recorded as such
and is not retroactively interpreted as the minimum of `mode_ek_values`.

### Step 3: Modified cooling loop

New function `run_cooling_multi_freq` in `src/cooling_evolution.jl`:

```julia
function run_cooling_multi_freq(problem, state, mf_params, sim_params, ham_params;
                                measure_modes=false)
    for step in 2:mf_params.steps+1
        choice = multi_frequency_cycle_choice(mf_params, step - 1)
        delta_r = choice.delta
        te_step = choice.te

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

The helper `multi_frequency_cycle_sequence` returns the complete planned
detuning/time sequence with the same `delta_list` and `te_list` convention used
in HDF5 output: the first entry is `NaN` because it corresponds to the initial
measurement before any cooling cycle.  This keeps the physical schedule and the
recorded diagnostic arrays on a single convention.  For random schedules or
randomized times, the helper samples from its supplied RNG; it matches a run
when `run_cooling_multi_freq` receives an RNG in the same state.

The deterministic schedules are:

- `:round_robin`, which cycles through the detuning grid in increasing order,
  then repeats.
- `:descending`, which cycles through the same fixed grid in decreasing order,
  then repeats.  This gives a high-detuning-first probe without introducing
  random schedule noise.

### Step 4: Comparison script

`scripts/multi_freq_cooling.jl`: runs single-Δ vs multi-Δ for small N, produces
comparison plots of E/N vs step.

### Step 5: Production-run design

Initial pre-validation target table, retained here as historical planning
values:

| Run | N  | R  | Δ-values           | g    | te  | steps | D   |
|-----|----|----|---------------------|------|-----|-------|-----|
| A   | 20 | 1  | gap                 | 0.3  | 2.0 | 200   | 20  |
| B   | 20 | 5  | uniform grid        | 0.3  | 2.0 | 200   | 20  |
| C   | 20 | 5  | spectral            | 0.3  | 2.0 | 200   | 20  |
| D   | 20 | 5  | uniform + rand time | 0.3  | 2.0 | 200   | 20  |
| E   | 50 | 1  | gap                 | 0.3  | 2.0 | 500   | 40  |
| F   | 50 | 5  | uniform grid        | 0.3  | 2.0 | 500   | 40  |
| G   | 50 | 10 | uniform grid        | 0.3  | 2.0 | 500   | 40  |

These rows should not be read as current production recommendations.  In view
of the `N=64` evidence below, `D = 20` and `D = 40` are pilot caps unless a
separate bond-dimension ladder shows that they are converged for the observable
being reported.  Paper-level data should record the stop-on-cap metadata, the
TDVP sweep-level bond trace, and enough independent trajectories to distinguish
tail fluctuations from ensemble fluctuations.

### Current large-N status

The current `N=64` MCWF/MPS Trotter diagnostics show substantial transient
system-bath bond growth while the short trajectories remain far above the
ground-state energy.  The repository-level summary is
[`largeN_effective_bond_dimensions.md`](largeN_effective_bond_dimensions.md).
In particular, four-cycle fixed-detuning runs with `R = 1,2,5,10` already show
that `Dmax = 320` is not a converged cap, and some `Dmax = 640` schedules still
saturate the transient system-bath bond dimension.

The large-`N` validation driver can now select the evolution branch explicitly.
The Trotter results above should not be conflated with MCWF+TDVP runs launched
with `--methods mcwf --evolution-method continuous`.  The first completed
`N=64`, two-cycle TDVP calibration at `Dmax = 96` resolves the early transient
bond dimensions for `R = 1,2,5,10` (`Dsb_eff = 50,54,51,51`) but still has
positive final energy densities.  These first two cycles are only a runtime
and early-bond calibration; they are not physically meaningful evidence for
cooling.  TDVP is therefore accessible, but the physical question remains the
long-cycle fixed point, not the first two cycles.
The next completed `R = 5`, five-cycle TDVP probe at `Dmax = 96` lowers the
energy density to `E/N = 0.66394232` after the early transient, but it reaches
the transient system-bath cap in cycle 3 and the retained system cap in cycle
4.  Thus it is evidence of high-bond TDVP cooling dynamics, not a converged
large-`N` benchmark.
Future long TDVP runs should enable `--tdvp-sweep-progress` so interrupted runs
retain the inner TDVP sweep-level bond-dimension trace in the HDF5 output.
Adding `--progress-csv` also preserves a textual per-observer-event trace for
live monitoring and partial-run inspection.
If the purpose of a run is only to locate the first cap event, the validation
driver can also be run with `--stop-on-bond-cap`.  This stops after the first
completed cycle whose retained system state or transient system-bath state
reaches the method-specific cap; when `--tdvp-sweep-progress` is enabled, the
transient test includes the inner TDVP sweep states recorded during that cycle.
The driver writes the completed prefix of each time series and records
`requested_steps`, `completed_steps`, and `stop_reasons` in the HDF5 group.  It
also stores the per-cycle `tdvp_sweep_max_bond` trace and
`tdvp_sweep_saturation_cycle` metadata, so the sweep-level cap source is
recoverable from the HDF5 file without consulting the progress CSV.  This option
is restricted to single-trajectory diagnostic runs; ensemble members whose cap
events may occur at different cycles should be launched as independent jobs.
When no explicit `--output` path is supplied, the generated HDF5 filename
receives a `_stopcap` suffix so these partial diagnostic outputs do not
overwrite full benchmark files with the same physical parameters.
When several `R` values or `Dmax` values are to be run on a many-core machine,
the validation driver should first be invoked with `--print-parallel-plan`.
This prints one independent command for each `(N, method, R, Dmax)` tuple and
assigns distinct HDF5 and progress CSV paths, so process-level parallelism can
be used without concurrent writes to the same output file.  The planning mode
also accepts `--plan-julia-threads` and `--plan-blas-threads`, which prefix the
printed commands with `JULIA_NUM_THREADS` and BLAS thread environment variables.
No internal scheduler is recommended at this stage.  The current reproducible
execution convention is to let an external shell, job array, or process manager
launch the printed commands, because the driver can then keep output ownership,
progress ordering, and deterministic seed assignment local to each independent
Julia process.

A first runtime-only calibration of this mechanism was run on 2026-06-19.  This
calibration used `N=64`, MCWF+TDVP, `R=2,5`, two cooling cycles, `Dmax=32`,
`cutoff=10^{-6}`, and the fixed detuning interval
`[0.5051167496264384, 3.0307004977586303]`.  It is not physical cooling
evidence: both trajectories hit the `D=32` cap by cycle 2, so the results are
only timing and orchestration diagnostics.

| execution mode | Julia threads | BLAS threads | jobs | wall time | traj cycles/hour | user time | system time | interpretation |
|---|---:|---:|---|---:|---:|---:|---:|---|
| serial, one driver process | 1 | 1 | `R=2` then `R=5` | 287.70 s | 50.05 | 280.23 s | 3.53 s | baseline |
| serial, one driver process | 1 | 16 | `R=2` then `R=5` | 284.34 s | 50.64 | 954.09 s | 503.05 s | no useful wall-time gain; much larger CPU use |
| two independent processes | 1 | 1 | `R=2` and `R=5` concurrently | 179.28 s | 80.32 | 172.59 s and 170.32 s | 2.10 s and 2.07 s | throughput speedup about 1.60 relative to the serial BLAS=1 baseline |

The practical recommendation from this small calibration is to start large-`N`
throughput scans with one Julia thread and one BLAS thread per independent
process, and to vary the number of independent processes externally rather than
adding an internal driver scheduler.  BLAS threading should be re-tested at
larger caps, but for this TDVP calibration it only increased CPU consumption.
The throughput column counts four completed trajectory-cycles: two completed
cycles for the `R=2` job and two completed
cycles for the `R=5` job, divided by the externally measured wall time for the
row.  It is a runtime throughput diagnostic only, not a cooling-performance
metric.  The benchmark artifacts are stored under
`/tmp/coolingtns_parallel_bench_serial_blas1_20260619`,
`/tmp/coolingtns_parallel_bench_parallel_blas1_R2_20260619`,
`/tmp/coolingtns_parallel_bench_parallel_blas1_R5_20260619`, and
`/tmp/coolingtns_parallel_bench_serial_blas16_20260619`.
For later timing tables, the HDF5 summary script reports
`traj cycles/hour = 3600 sum(completed_steps) / elapsed_total`.  This is the
single-run throughput from the stored HDF5 provenance.  For externally parallel
job groups, the group wall time should still be measured by the launcher, or
approximated by the maximum elapsed time among simultaneously launched
single-job HDF5 files; the per-row throughput column remains useful for
detecting slow jobs and comparing thread settings without hand recomputation.

These data should be used to design the next production campaign.  The
historical target table and the timing data above are therefore planning
inputs, not evidence that the listed bond dimensions are already physically
converged at large system size.

### Step 6: Analysis & figures for the paper

Key figures to produce:

1. **E/N vs step**: single-Δ vs multi-Δ (R = 1, 3, 5, 10) at fixed N
2. **Late-time or best-prefix energy vs R**: report this first as a
   finite-window diagnostic; promote it to a steady-state energy only after
   trajectory and bond-dimension convergence have been demonstrated
3. **Late-time or best-prefix energy vs N**: report the single-Δ versus
   multi-Δ comparison first as finite-window scaling; promote it to
   steady-state scaling only after trajectory and bond-dimension convergence
   have been demonstrated
4. **Effect of randomized times**: with vs without
5. **Coupling type comparison**: XX vs YY vs XY under multi-Δ
6. **Noise-response diagnostics**: compare finite-window multi-Δ cooling
   curves for pe = 0, 0.001, 0.01; promote this to a robustness claim only
   after the same trajectory and bond-dimension convergence checks

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
