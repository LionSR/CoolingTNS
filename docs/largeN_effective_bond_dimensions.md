# Large-N Effective Bond Dimensions

This note records the current tensor-network bond-dimension evidence for the
large-system multi-frequency cooling campaign.  It is a validation artifact for
issue #146, not a claim of converged ground-state cooling.

The current diagnostic target is the nonintegrable Ising chain

```math
H_S = J \sum_i Z_i Z_{i+1} + h_x \sum_i X_i + h_z \sum_i Z_i,
```

with open boundary conditions, `J = 1`, `h_x = -1.05`, and `h_z = 0.5`.  The
reported large-system numbers use the tensor-network Monte Carlo wavefunction
path, `N = 64`, `g = 0.3`, `tau = 0.2`, one trajectory, and the fixed detuning interval
`[0.5051167496264384, 3.0307004977586303]`.  The same detuning interval is used
for every value of the number of frequencies `R`, so changes in the bond cap do
not also change the physical protocol through a different gap estimate.  The
interval is the DMRG gap estimate
`Delta_min = 0.5051167496264384` together with
`Delta_max = 6 Delta_min`, matching the driver's default `delta_max_factor`
heuristic but holding the numerical interval fixed across the Dmax ladder.
Accordingly, the validation driver defaults to the MCWF/MPS method for
large-system campaigns.  The MPO density-matrix method remains an explicit
small-system validation option through `--methods mpo` or
`--methods mpo,mcwf`; it is not the present production route for the
`N = 64--100` bond-dimension studies.

The validation driver records the evolution branch in each HDF5 file.  The
four-cycle table below is a Trotter diagnostic (`--evolution-method trotter`,
`cutoff = 1e-7`).  The TDVP tables are MCWF+TDVP diagnostics
(`--evolution-method continuous`) with their stated cutoffs and bond caps.
These two protocols should not be conflated.

## Definitions

The code source of truth for these quantities is
`scripts/validation/run_largeN_multifrequency_tn_scaling.jl`, summarized by
`scripts/validation/summarize_largeN_bond_dimensions.jl`.
Before a new local `.h5` file is used as reader-facing large-`N` evidence, run
`scripts/validation/check_largeN_artifact_provenance.jl` on the artifact.  This
checker verifies that the HDF5 basename, not merely a machine-specific local
path, appears as a filename token in the provenance documents.  Thus a
`.worktree` path may move or disappear, while the recorded HDF5 basename remains
the stable audit handle for the evidence table and notes.
Some historical filenames cited below predate the current default naming
convention, which includes the canonical evolution-method token (`trotter` or
`continuous`), the bath coupling `g`, the evolution time `te`,
mode-measurement suffixes, and suffixes for non-default detuning schedules and
randomized evolution times.  Generated
per-job progress CSV filenames in a parallel plan reuse the same HDF5 protocol
stem, with the user-supplied CSV stem kept as a prefix.  Plans with
`--te-values` generate one child command per requested evolution time and rely
on the existing `te` token for distinct HDF5/progress-CSV stems; the axis
requires an explicit fixed detuning interval so changing `te` does not also
change the bath-frequency protocol.  Plans with `--trajectory-values` also add
a `_trajk` token to both HDF5 and progress-CSV stems and store the physical
MCWF trajectory labels in the HDF5 `trajectory_indices` dataset.  The HDF5
metadata is the authoritative protocol record.  The summary table includes the
stored `te` column, so a later time-ladder comparison does not have to infer
the bath-evolution time from a filename.  It also includes a `time protocol`
column, which distinguishes fixed cycle times from randomized cycle times using
the stored `randomize_times` metadata, and an `init` column, which records the
stored initial-state protocol (`product`, `ground`, `identity`, or
`theta=<value>`).  When those split trajectory-axis files are summarized
together,
`summarize_largeN_bond_dimensions.jl --combine-trajectories` groups compatible
rows by the physical protocol, including the coupling strength `g`, the
initial state, `te`, and whether the cycle times are fixed or randomized,
rejects duplicate trajectory labels, and reports one ensemble-level row.  Thus
different-coupling scans, product-state cooling files, ground-state controls,
and explicit theta-state files are not combined as if they were the same
physical experiment, and a fixed-time trajectory file and a randomized-time
trajectory file with the same mean `te` are likewise kept separate.  For
stop-on-cap files with unequal completed prefixes, the energy
columns in that combined row are statistics of the individual trajectory
summaries, not a reconstructed cycle-aligned ensemble time series.  If some
member files do not contain detuning histories, the combined `visited
detunings` entry records the known counts and an `unknownxq/M` term, rather
than attributing the observed histories to the whole ensemble.

The summary script also reports the stored `completed_steps`,
`requested_steps`, `elapsed_seconds`, and `stop_reasons` fields.  The
`completed/requested` column is therefore the HDF5 record of how far a
stop-on-cap trajectory actually ran, not a value transcribed from the progress
log by hand.  The `completed/requested periods` column divides those cycle
counts by `R` for deterministic detuning schedules (`round_robin` and
`descending`), so an `R = 10` row with five completed cycles is shown as only
`0.50` completed schedule periods.  Random schedules and legacy files without
stored schedule metadata are reported as `n/a` in this period column.  The
`visited detunings` column counts distinct stored detuning values that actually
appear in the completed cycle prefix, excluding the initial `NaN` measurement
row; for stop-on-cap runs it is therefore the realized detuning coverage before
the cap, not the intended full grid.  The `detuning coverage` column makes the
same distinction as a status label: deterministic `round_robin` and
`descending` runs report `full_grid_observed` only after every trajectory has
completed at least one full detuning-grid period.  Prefixes shorter than one
period are marked as `requested_partial_grid` or `stopped_partial_grid`;
single-detuning runs report `single_detuning`; random schedules report this
label as `n/a`.  The interrupted-run progress CSV summarizer reports the same
basic prefix evidence as `visited detunings` and `detuning coverage`, computed
from completed `updated` rows only; because the CSV does not store the
originally requested number of cycles, a nonzero prefix shorter than one full
grid is labelled `partial_grid_observed`, while completed rows with no finite
detuning values are labelled `missing_detuning_values`.  These CSV labels are
therefore observation-based for the finite detunings present in the progress
file; unlike the HDF5 summary, they are not gated by stored schedule metadata.
Progress CSV files written before the coupling-strength scan axis do not
contain a `g` column; the progress summarizer displays those entries as
`legacy_missing` rather than leaving a blank coupling-strength cell.  If a
file contains the `g` column but an individual field is empty, the summarizer
uses the neutral label `missing`.
The `elapsed_total` column sums
per-trajectory elapsed times, matching the sequential large-N campaign driver,
and `traj cycles/hour` is computed as
$$
  3600\,\frac{\sum_{\rm trajectories}\texttt{completed\_steps}}
              {\texttt{elapsed\_total}}.
$$
It is therefore a throughput diagnostic, not a cooling-performance metric.
Partial stop reasons are reported with counts such as `bond_capx1/2`.  Use
`scripts/validation/summarize_largeN_bond_dimensions.jl --compact` for the
short table format used in the notes.

The nominal parameter `Dmax` is not always the actual Trotter truncation cap.
The method-specific cap is

```math
D_{\rm cap} = \mathrm{tn\_method\_maxdim}(\mathrm{method}, D_{\max}).
```

For MCWF/MPS, `Dcap = Dmax`.  For MPO density-matrix Trotter evolution,
`Dcap = 4 Dmax`.

The summary table reports three effective bond-dimension labels.  They refer to
different states in the same cooling cycle and should not be interchanged.

- `Dsys_eff` is the effective bond dimension of the retained system state after
  bath measurement.  It is computed from the largest final link dimension of
  the `N`-site system MPS or MPO over trajectories.
- `Dsb_eff` is the effective bond dimension of the evolved enlarged
  system-bath state before bath measurement.  It is computed from the largest
  link dimension of the `2N`-site evolved MPS or MPO over all recorded cooling
  cycles and trajectories.
- `Dtdvp_sweep_eff` is the effective bond dimension inferred from the largest
  system-bath state seen by the inner TDVP sweep observer during MCWF+TDVP
  evolution.  The HDF5 source fields are `tdvp_sweep_max_bond` and
  `tdvp_sweep_saturation_cycle`.  Legacy files and runs without this diagnostic
  are summarized as `n/a`, not as a measured zero-dimensional state.  This is an
  inner-sweep diagnostic, not an additional retained physical state.

For Trotter cooling, `Dsb_eff` is the more stringent quantity, because the
algorithm must represent the enlarged system-bath state before the bath is
measured and discarded.  For MCWF+TDVP cooling, `Dtdvp_sweep_eff` is the
corresponding sweep-level cap diagnostic when `--tdvp-sweep-progress` has been
enabled.

The reported value is conservative.  If the run reaches `Dcap`, then the
observed maximum is only a lower bound on the bond dimension required by the
untruncated trajectory.  In that case the summary script writes a label such as
`>=640`.  Otherwise it writes the largest observed link dimension.

The summary script also reports a machine-readable `bond_status` column.  The
legal labels are centralized in
`scripts/validation/largeN_scaling_helpers.jl`, and the HDF5 and progress-CSV
summaries validate against that same vocabulary.  The helper also names each
legal label explicitly, so tests of generated rows and readers need not repeat
the wire-format strings.  The status is only a
bond-dimension diagnostic:

- `no_cap_hit`: neither the retained system state nor the evolved
  system-bath state nor the recorded TDVP sweep history reached the cap during
  the recorded run.
- `not_converged_system_cap`: the retained system state reached the cap.
- `not_converged_evolved_cap`: the evolved system-bath state reached the
  cap.
- `not_converged_tdvp_sweep_cap`: the inner TDVP sweep history reached the cap.

Combined labels are formed by joining the cap-hit sources with `_and_`; for
example, `not_converged_system_and_evolved_and_tdvp_sweep_cap` means that all
three recorded histories reached the cap.

A `no_cap_hit` entry does not by itself imply ground-state cooling or
trajectory convergence; it only means that the imposed bond cap was not reached
in the recorded run.

For long-cycle diagnostics, the summary script reports the initial row together
with three later energy readouts.  `initial E/N`, `initial relE`, and
`initial overlap` are read from the stored cycle-0 energy and ground-state
overlap histories; they make `--init-state ground` controls auditable from the
same table as the bond diagnostics.  `final E/N` is the last recorded energy
density, `best E/N` is the lowest recorded energy density, and `tail E/N` is the
average over the last ten recorded rows, or over the whole trace if fewer than
ten rows are present.  `best relE` is the smallest recorded relative energy;
this is kept as a separate column because a severely truncated trajectory could
in principle undershoot the ground-state reference in raw energy.  These fields
separate initial-state provenance, monotone cooling, transient low-energy
excursions, and late-time plateaus without assuming that a finite trajectory has
reached a fixed point.

The current HDF5 summary table therefore contains the sweep-specific columns
`Dtdvp_sweep_eff`, `peak tdvp sweep max`, and `tdvp sweep sat` in addition to
the retained-system and evolved system-bath columns.

The current large-`N` campaign files do not store measured per-bond
truncation-error time series.  The evidence below should therefore be read as
bond-cap and truncation-pressure diagnostics: it records which states reached
the imposed cap and when, not the discarded Schmidt weight at each truncation.
Measured truncation-error histories would be an additional diagnostic beyond
the cap/saturation metadata described here.  New campaign files record this
explicitly as `truncation_error_history_status = not_recorded`; the summary
script reports older files without that field as `legacy_missing`.  A future
file that stores a nonempty `truncation_errors` dataset without the explicit
status is reported as `measured`; an empty dataset is reported as `empty`
because it is not a measured discarded-weight history.

## Mode-Resolved Integrable-Ising Campaigns

The Bogoliubov mode observables `mode_hk` and occupations
`mode_nk = (mode_hk + 1)/2` are not defined for the default nonintegrable
Hamiltonian above.  They are defined for the translation-invariant
transverse-field Ising model with even `N` and periodic or antiperiodic spin
boundary conditions, using the parity-aware fermionic boundary-condition
convention described in `Notes/NotesED/MapToSpin.tex`.

These Bogoliubov occupations are distinct from the Jordan-Wigner Fourier
occupations \(\langle \tilde a_k^\dagger \tilde a_k\rangle\), which are stored
under `momentum_dist` when that separate observable is requested.  The mode
energy plots and cooling diagnostics use `mode_hk`, `mode_nk`, and
`mode_ek_values`; they should not be interpreted as the Fourier occupation
dataset.

The large-N driver therefore exposes mode measurements only through an explicit
integrable-Ising command.  For example,

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --model ising --bc periodic --Ns 64 --R-values 1,2,5,10 \
  --methods mcwf --evolution-method continuous --steps 40 --Dmax 80 \
  --h -1.05 --init-state theta --theta 0.0 --measure-modes \
  --delta-min 0.5051167496264384 --delta-max 3.0307004977586303 \
  --progress-csv .worktree/mode_largeN_N64_ising_periodic/tdvp_progress_modes.csv \
  --tdvp-sweep-progress
```

When `--measure-modes` is supplied, the driver fails early unless the requested
campaign is an even-size periodic or antiperiodic integrable Ising run using the
MCWF/TDVP tensor-network path.  The HDF5 output then stores the ensemble means
`mode_hk` and the derived Bogoliubov occupation `mode_nk`, the
trajectory-resolved arrays
`mode_hk_trajectories` and `mode_nk_trajectories`, the common `mode_k_indices`,
the positive quasiparticle gaps `mode_ek_values`, and the parity-selected
fermionic boundary condition metadata.  This keeps the occupation-number
diagnostics tied to the same convention as the ED and TN observable tests.
For practical large-`N` runs, the progress CSV should be enabled from the
start: a mode-resolved TDVP cycle can be much slower than the preceding cycle,
and without the CSV an interrupted trajectory may leave only an incomplete HDF5
file rather than a recoverable energy and bond-dimension trace.

The large-N HDF5 summary also audits this convention directly.  When a run
contains `mode_hk` and `mode_nk`,
`scripts/validation/summarize_largeN_bond_dimensions.jl` first verifies
`mode_nk = mode_occupation_from_hk(mode_hk)`, with paired `NaN` entries accepted
only for deliberately unmeasured strided rows.  It also verifies
`mode_ek_values = mode_energies_Jh(mode_k_indices,J,h,N)`, so the stored
positive gaps are tied to the same mode grid and Hamiltonian parameters.  It
then uses the library routine `ising_energy_from_mode_hk` to reconstruct the
integrable-Ising energy on the measured mode rows.  This routine is the source
of truth for the energy coefficient: it evaluates
```math
E_{\mathrm{modes}}(t)=\frac{\Lambda}{2}
\sum_{k\in\mathrm{grid}(g_F)}\operatorname{coeff}_k\,h_k(t),
```
with the signed special-mode convention from `Notes/NotesED/MapToSpin.tex`.
The stored `mode_ek_values` are positive quasiparticle gaps for resonance
labels and plots; they are not, by themselves, the signed reconstruction
coefficients on grids that contain special modes.  The full summary reports
the selected `mode gF`, the `mode source`, the measured-row fraction, the final
mode-reconstructed energy per spin on the last measured mode row, and the last
measured and maximum absolute discrepancy per spin from the stored direct
energy.  Runs without mode data are reported as `n/a` in these columns.  A small
discrepancy is the expected target for a parity-definite trajectory; a
reference-grid entry with
`mode_gF_source = "reference"` is a diagnostic of a chosen sector and should
not be read as an exact energy decomposition for a mixed-parity state.
If any mode-observable dataset is present, the summary requires the complete
mode-observable payload (`mode_hk`, `mode_nk`, `mode_k_indices`,
`mode_ek_values`, `mode_measurement_cycles`, `mode_gF`, and
`mode_gF_source`).  A partial payload is rejected rather than summarized as
`n/a`, because it is neither a valid absence of mode data nor a complete
occupation-number diagnostic.
The mode-energy plotting path applies the same positive-gap check: if
`mode_ek_values` are present but do not match the stored `mode_k_indices`,
`N`, `J`, and `h`, the file is rejected rather than repaired by recomputing
coefficients from the root metadata.

The full TN mode measurement evaluates the split-string correlator formula for
all Fourier modes, and is much more expensive than the scalar energy and bond
diagnostics.  The MPS path constructs these correlators by sweeping cached
environments through the Jordan-Wigner string intervals; the MPO density-matrix
path still contracts the corresponding Pauli-string MPOs directly.  Long
large-`N` runs may therefore use `--mode-measurement-stride s` together with
`--measure-modes`.  In that case
`mode_hk` and `mode_nk` retain the ordinary step-by-mode shape, but only cycles
`0, s, 2s, ...` and the requested final cycle are evaluated.  If an early stop
occurs on an off-stride cycle, the completed final cycle is also evaluated
before truncation.  Unmeasured rows are stored as `NaN`, and the measured
cooling cycles are listed in `mode_measurement_cycles`.  The library routine
`mode_measurement_cycle_rows` is the shared source of truth for the cycle-list
contract: it requires the measured cycles to be nonempty, sorted, unique, and
in range, and maps them to the one-based rows used by Julia arrays.  The
larger validator `validate_mode_measurement_rows` also checks the
`mode_nk = mode_occupation_from_hk(mode_hk)` relation and rejects non-finite
measured `mode_hk`, `mode_nk`, or energy rows.  The large-N writer, summary
script, and plotting utilities all use these conventions so that deliberately
unmeasured `NaN` rows are distinguished from malformed or non-finite measured
rows.

### N=64 Stopped Final Mode-Row Verification

After replacing the MPS split-string contractions by cached Jordan-Wigner
interval sweeps and then fusing the four MPS correlator sweeps, the stopped-row
contract was checked on 2026-06-28 with the full `N = 64`, `R = 1,2,5,10`,
`Dmax = 80` MCWF/continuous-TDVP mode-resolved run.  This is the same physical
protocol as the earlier `R = 2` stopped-row check that previously reached the
cycle-2 update but spent more than 20 minutes in the final mode measurement.
That earlier standalone timing check wrote

```text
.worktree/N64_R2_stopcap_after_correlator_sweep/largeN_multifrequency_tn_N64_R2_mcwf_continuous_ising_bcperiodic_stopcap_modestride5_inittheta_theta0_steps40_Dmax80_g0.3_te2_tau0.2_seed20260617.h5
```

The all-frequency post-fusion verification command was:

```bash
julia --project=. --startup-file=no \
  scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --model ising --bc periodic --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 40 --Dmax 80 --h -1.05 \
  --init-state theta --theta 0.0 \
  --measure-modes --mode-measurement-stride 5 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --stop-on-bond-cap --tdvp-sweep-progress \
  --progress-csv .worktree/post_fused_mode_N64_R1-2-5-10_D80_20260628/progress.csv \
  --outdir .worktree/post_fused_mode_N64_R1-2-5-10_D80_20260628
```

The run stopped after two completed cycles for every frequency count and wrote

```text
.worktree/post_fused_mode_N64_R1-2-5-10_D80_20260628/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_continuous_ising_bcperiodic_stopcap_modestride5_inittheta_theta0_steps40_Dmax80_g0.3_te2_tau0.2_seed20260617.h5
```

The compact HDF5 summary is:

| R | completed/requested | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | elapsed | mode rows | mode max abs dE/N |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 2/40 | single_detuning | -1.01728376 | -1.05000000 | >=80 | >=80 | >=80 | 813.8 s | 2/3 | 1.62e-10 |
| 2 | 2/40 | full_grid_observed | -1.08328139 | -1.08328139 | >=80 | >=80 | >=80 | 833.2 s | 2/3 | 5.38e-11 |
| 5 | 2/40 | stopped_partial_grid | -1.03401120 | -1.05000000 | >=80 | >=80 | >=80 | 798.6 s | 2/3 | 2.76e-11 |
| 10 | 2/40 | stopped_partial_grid | -1.04986697 | -1.05000000 | >=80 | >=80 | >=80 | 908.5 s | 2/3 | 5.64e-10 |

All four raw mode payloads satisfy the strided stopped-row contract:

| R | `mode_measurement_cycles` | measured rows | finite measured `mode_hk` | finite measured `mode_nk` | skipped row finite count | measured `mode_nk` range | final mode E/N |
|---:|---|---|---|---|---:|---|---:|
| 1 | `[0, 2]` | `[1, 3]` | true | true | 0 | `[1.4333e-4, 0.4288]` | -1.0172837582 |
| 2 | `[0, 2]` | `[1, 3]` | true | true | 0 | `[1.4333e-4, 0.3468]` | -1.0832813940 |
| 5 | `[0, 2]` | `[1, 3]` | true | true | 0 | `[1.4333e-4, 0.4266]` | -1.0340112040 |
| 10 | `[0, 2]` | `[1, 3]` | true | true | 0 | `[1.4333e-4, 0.3807]` | -1.0498669711 |

Thus the final off-stride stopped cycle is now present as a finite measured
mode row for every frequency count, while the skipped cycle remains explicitly
unmeasured.  The mode reconstruction uses `mode_gF = -1` with
`mode_gF_source = "state"` in every row, and reconstructs the direct Ising
energy on the measured rows to at worst `5.64e-10` per spin.

The physical interpretation is still negative.  `R = 2` gives the lowest
two-cycle stopped prefix in this run, but all four schedules hit the system,
transient system-bath, and TDVP-sweep effective caps by cycle 2.  The data
therefore verify the large-`N` mode-observable convention under realistic
stopped-row load; they do not demonstrate converged cooling or approach to the
ground state.

For mode-energy reconstruction checks, the simulated state should have a
definite Ising parity so that the fermionic momentum grid is selected by the
state itself.  The tested large-N command above therefore uses
`--init-state theta --theta 0.0`.  The default `--init-state product` is a
mixed-parity state for the periodic Ising chain in the code basis; in that case
the HDF5 metadata records `mode_gF_source = "reference"`, and the mode arrays
should be read as a fixed-reference diagnostic rather than as a state-sector
energy decomposition.

### N=64 te=0.5 Stopped Mode-Row Follow-Up

To check whether the cycle-2 stop in the preceding run was only a consequence
of the large bath-evolution time, the same parity-definite periodic-Ising mode
campaign was repeated on 2026-06-29 with `te = 0.5` and a shorter requested
window of 12 cooling cycles:

```bash
julia --project=. --startup-file=no \
  scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --model ising --bc periodic --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 80 \
  --J 1.0 --h -1.05 --g 0.3 --tau 0.2 --seed 20260617 \
  --init-state theta --theta 0.0 \
  --measure-modes --mode-measurement-stride 5 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --te 0.5 \
  --stop-on-bond-cap --tdvp-sweep-progress \
  --progress-csv .worktree/mode_te05_N64_R1-2-5-10_D80_20260629/progress.csv \
  --outdir .worktree/mode_te05_N64_R1-2-5-10_D80_20260629
```

The run wrote

```text
.worktree/mode_te05_N64_R1-2-5-10_D80_20260629/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_continuous_ising_bcperiodic_stopcap_modestride5_inittheta_theta0_steps12_Dmax80_g0.3_te0.5_tau0.2_seed20260617.h5
```

Reducing the bath-evolution time from `te = 2.0` to `te = 0.5` delays the first
stopped prefix from cycle 2 to cycle 4, but all four schedules still stop after
four completed cycles.  None of the four final rows improves on the initial
`E/N = -1.05000000`; the best row in every trace is the initial state.  The
compact HDF5 summary is:

| R | completed/requested | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | elapsed | mode rows | mode max abs dE/N |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 4/12 | single_detuning | -0.97121423 | -1.05000000 | >=80 | >=80 | >=80 | 301.3 s | 2/5 | 1.78e-15 |
| 2 | 4/12 | full_grid_observed | -1.03529914 | -1.05000000 | >=80 | >=80 | >=80 | 264.7 s | 2/5 | 3.55e-15 |
| 5 | 4/12 | stopped_partial_grid | -1.00788552 | -1.05000000 | >=80 | >=80 | >=80 | 283.5 s | 2/5 | 1.78e-15 |
| 10 | 4/12 | stopped_partial_grid | -0.95945491 | -1.05000000 | >=80 | >=80 | >=80 | 276.5 s | 2/5 | 2.66e-15 |

All four raw mode payloads again satisfy the strided stopped-row contract:

| R | `mode_measurement_cycles` | measured rows | finite measured `mode_hk` | finite measured `mode_nk` | skipped row finite count | measured `mode_nk` range | final mode E/N |
|---:|---|---|---|---|---:|---|---:|
| 1 | `[0, 4]` | `[1, 5]` | true | true | 0 | `[1.4333e-4, 0.3809]` | -0.9712142252 |
| 2 | `[0, 4]` | `[1, 5]` | true | true | 0 | `[1.4333e-4, 0.3637]` | -1.0352991436 |
| 5 | `[0, 4]` | `[1, 5]` | true | true | 0 | `[1.4333e-4, 0.3682]` | -1.0078855244 |
| 10 | `[0, 4]` | `[1, 5]` | true | true | 0 | `[1.4333e-4, 0.3792]` | -0.9594549148 |

The mode reconstruction uses `mode_gF = -1` with
`mode_gF_source = "state"` in every row, and reconstructs the direct Ising
energy on the measured rows to at worst `3.55e-15` per spin.  Among the final
cycle-4 rows, `R = 2` gives the lowest energy, but it is still above the
initial state and all retained-system, transient system-bath, and TDVP-sweep
effective dimensions have reached the imposed `Dmax = 80` cap.  Thus the
shorter-time follow-up is a useful stopped-row and bond-pressure diagnostic;
it is not evidence of a converged cooling trajectory.

### Dmax=16 Product-State Mode Probe

After adding the progress-CSV recommendation for mode-resolved continuous TDVP
runs, a bounded `N = 64` periodic-Ising probe was run on 2026-06-27 from the
default product state:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 5 --Dmax 16 \
  --cutoff 1e-7 --tau 0.2 --model ising --bc periodic --h -0.75 \
  --measure-modes --mode-measurement-stride 1 \
  --schedule descending --stop-on-bond-cap --te 0.5 \
  --progress-csv .worktree/mode_probe_N64_ising_periodic_D16_te0.5_20260627/tdvp_progress_modes.csv \
  --tdvp-sweep-progress \
  --outdir .worktree/mode_probe_N64_ising_periodic_D16_te0.5_20260627 \
  --verbose
```

This 2026-06-27 mode-probe file uses a legacy stem from before the default
output name added the `_g<value>` token.  The HDF5 metadata still stores the
default coupling `g = 0.3`.  The run wrote

```text
.worktree/mode_probe_N64_ising_periodic_D16_te0.5_20260627/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_continuous_ising_bcperiodic_stopcap_scheddesc_modes_steps5_Dmax16_te0.5_tau0.2_seed20260617.h5
```

The run used the analytic periodic-Ising pair reference
`detuning_reference_gap_source = "ising_mode_pair_reference"` with
`detuning_reference_gap = 1.01435154`.  The direct ground-state reference stored
in the file is `E0/N = -1.1463581992`.  The HDF5 and progress-CSV summaries
agree:

| R | completed/requested | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | mode source | mode max abs dE/N |
|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 3/5 | single_detuning | 0.82770229 | 0.82770229 | 13 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | reference | 0.016 |
| 2 | 3/5 | full_grid_observed | 0.82329220 | 0.82329220 | 12 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | reference | 0.016 |
| 5 | 3/5 | stopped_partial_grid | 0.89088421 | 0.89088421 | 13 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | reference | 0.016 |
| 10 | 3/5 | requested_partial_grid | 0.74932134 | 0.74932134 | 13 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | reference | 0.016 |

The progress CSV records that all four trajectories first reach the evolved
system-bath and TDVP-sweep caps at cycle 3.  The final-cycle ordering is
`R = 10 < R = 2 < R = 1 < R = 5`, where lower energy is better, but every row
is still far above the ground-state reference and cap-limited.  Thus this is an
occupation-diagnostic and logging check, not evidence of converged cooling.

The stored occupations are physically bounded in this run: `mode_nk` is finite
and lies in `[0.71657563, 0.99207860]` across all measured rows.  The largest
mode-energy discrepancy, `0.015625 = 1/64` per site, occurs on the initial
product-state row.  This is consistent with `mode_gF_source = "reference"`: the
product state is not in a definite fermion-parity sector, so the mode
reconstruction is a fixed-reference diagnostic rather than an exact sector
energy decomposition.  On the capped final rows the discrepancy is about
`0.001` for `R = 1` and about `0.010` for `R = 2,5,10`.

### Dmax=16 Theta-State Mode Control

A matched parity-definite control was then run with the same `N = 64`,
`h = -0.75`, `te = 0.5`, `Dmax = 16`, and analytic periodic-Ising detuning
reference, but with the initial state changed to `theta=0`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 5 --Dmax 16 \
  --cutoff 1e-7 --tau 0.2 --model ising --bc periodic --h -0.75 \
  --measure-modes --mode-measurement-stride 1 \
  --schedule descending --init-state theta --theta 0.0 \
  --stop-on-bond-cap --te 0.5 \
  --progress-csv .worktree/mode_probe_N64_ising_periodic_theta_D16_te0.5_20260627/tdvp_progress_modes.csv \
  --tdvp-sweep-progress \
  --outdir .worktree/mode_probe_N64_ising_periodic_theta_D16_te0.5_20260627 \
  --verbose
```

This matched 2026-06-27 mode-probe file uses the same legacy pre-`_g<value>`
stem convention.  The HDF5 metadata still stores the default coupling
`g = 0.3`.  The run wrote

```text
.worktree/mode_probe_N64_ising_periodic_theta_D16_te0.5_20260627/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_continuous_ising_bcperiodic_stopcap_scheddesc_modes_inittheta_theta0_steps5_Dmax16_te0.5_tau0.2_seed20260617.h5
```

The stored direct ground-state reference is again
`E0/N = -1.1463581992`, and the automatic detuning reference is
`detuning_reference_gap_source = "ising_mode_pair_reference"` with
`detuning_reference_gap = 1.01435154`.  The compact HDF5 summary is:

| R | completed/requested | detuning coverage | initial E/N | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | mode source | mode max abs dE/N |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 2/5 | single_detuning | -0.75000000 | -0.71810720 | -0.75000000 | 15 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | state | 1.30e-14 |
| 2 | 2/5 | full_grid_observed | -0.75000000 | -0.76352020 | -0.76352020 | 14 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | state | 1.30e-14 |
| 5 | 2/5 | stopped_partial_grid | -0.75000000 | -0.76214886 | -0.76214886 | 14 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | state | 1.30e-14 |
| 10 | 2/5 | requested_partial_grid | -0.75000000 | -0.76569923 | -0.76569923 | 14 | >=16 | >=16 | not_converged_evolved_and_tdvp_sweep_cap | state | 1.30e-14 |

The progress CSV records the following completed-cycle energies:

| R | cycle 1 E/N | cycle 2 E/N | system max bond at stop | evolved max bond at stop |
|---:|---:|---:|---:|---:|
| 1 | -0.74080740 | -0.71810720 | 15 | 16 |
| 2 | -0.76037236 | -0.76352020 | 14 | 16 |
| 5 | -0.75647516 | -0.76214886 | 14 | 16 |
| 10 | -0.76037302 | -0.76569923 | 14 | 16 |

This control separates the notation and parity-sector question from the cooling
question.  Every row records `mode_gF = -1` and
`mode_gF_source = "state"`, the measured mode rows are exactly
`0, 1, 2`, and the direct energy agrees with the mode reconstruction to at most
`1.30e-14` per site.  The stored occupations are finite and physically bounded:
`mode_nk` lies in `[0.00019670, 0.99055075]` across all measured rows.
Thus the `1/64` initial-row offset seen in the product-state probe is not a
formula error in the mode reconstruction; it is the expected consequence of
using a reference fermion sector for a mixed-parity product state.

The physics conclusion remains negative.  The `R = 10` short prefix is the
lowest-energy row in this matched theta-state control, but it is still far from
the finite-chain ground-state reference and the trajectory is already
evolved-state and TDVP-sweep cap-limited by cycle 2.  This run is therefore a
parity-sector and mode-observable consistency check, not a scalable cooling
result.

### First Parity-Definite N=64 Mode Scan

After exposing the initial-state controls, the first nontrivial large-chain
mode scan used the same parity-definite state for all four frequency counts:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --model ising --bc periodic --Ns 64 --R-values 1,2,5,10 \
  --methods mcwf --evolution-method continuous --steps 10 --Dmax 32 \
  --cutoff 1e-6 --tau 0.2 --te 1.0 --M-mcwf 1 --h -1.05 \
  --measure-modes --init-state theta --theta 0.0 \
  --delta-min 0.5 --delta-max 3.0 \
  --outdir .worktree/mode_largeN_theta_scan_20260619 \
  --progress-csv .worktree/mode_largeN_theta_scan_20260619/tdvp_progress_N64_ising_periodic_Dmax32_te1_modes_theta0.csv \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

This scan predated the analytic mode-detuning reference introduced in PR #232,
so it used the explicit positive detuning interval above instead of the then
negative automatic TN excited-state estimate.  The mode convention itself was
correct: every HDF5 group recorded `mode_gF_source = "state"`, and the direct
Ising energy agreed with the mode reconstruction

```math
E_{\mathrm{modes}}(t)=\frac{\Lambda}{2}
\sum_{k\in\mathrm{grid}(g_F)}\operatorname{coeff}_k h_k(t)
```

to numerical precision.  For this parity-definite product-state scan the
selected grid was half-integer (`mode_gF = -1`), so it contained no special
modes and the formula reduces to
\(\frac12\sum_k \varepsilon_k h_k(t)\), where \(\varepsilon_k\) denotes the
code-unit positive gap stored in HDF5 as `mode_ek_values`.  On integer grids
with special modes, however, the signed coefficient in the library routine must
be used rather than replacing it by the positive gap.

| R | completed cycles | final E/N | best E/N | relE | Dsys | Devolved | Dtdvp sweep | mode gF source | max \(|E-E_{\mathrm{modes}}|\) |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 10/10 | -1.050000000001 | -1.050000000001 | 0.19542960 | 1 | 1 | 1 | state | 1.208e-12 |
| 2 | 10/10 | -1.050000000002 | -1.050000000002 | 0.19542960 | 1 | 1 | 1 | state | 1.208e-12 |
| 5 | 10/10 | -1.050000000002 | -1.050000000002 | 0.19542960 | 1 | 1 | 1 | state | 1.208e-12 |
| 10 | 10/10 | -1.050000000002 | -1.050000000002 | 0.19542960 | 1 | 1 | 1 | state | 1.208e-12 |

The observed energy density has a simple check: `theta=0` is the `|+>`
product state, so the initial `ZZ` contribution vanishes and the field term
gives `E/N = h = -1.05`.

This scan verifies the large-N mode-observable convention, but it does not show
cooling.  The trajectory stays at bond dimension one and at
`E/N = -1.05` for every tested value `R = 1, 2, 5, 10`.  Thus increasing the
number of detunings alone does not produce cooling from this parity-definite
theta state under the present `XX` coupling and round-robin schedule.  The next
physics check should isolate whether this flat trajectory is imposed by a
symmetry of the state-coupling pair, by the chosen detuning/time scale, or by a
model-basis convention.

### Automatic Analytic Detuning Check

The current mode-resolved driver no longer uses the generic TN excited-state
DMRG estimate as the automatic detuning reference.  For the default periodic
`XX` integrable-Ising mode campaign, the system-side coupling is local `X`,
which preserves the code parity \(P_x\).  The automatic reference is therefore
the lowest generic two-quasiparticle energy,
\(2\min_{\sin\phi_k\ne 0} \epsilon_k\), on the deterministic `parity=+1`
setup grid selected before any trajectory-specific state parity is measured.
The measured `mode_gF` and `mode_gF_source` metadata remain the source of truth
for the stored `mode_hk` rows; they are observable provenance, not a
retroactive definition of the detuning reference.  On reference grids
containing special modes, such as the default antiperiodic reference sector,
the driver requires an explicit detuning interval.  A short `N = 64` periodic
check without `--delta-min` or `--delta-max` used

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --model ising --bc periodic --Ns 64 --R-values 1 \
  --methods mcwf --evolution-method continuous --steps 2 --Dmax 32 \
  --cutoff 1e-6 --tau 0.2 --te 0.1 --M-mcwf 1 --h -1.05 \
  --measure-modes --init-state theta --theta 0.0 \
  --tdvp-sweep-progress \
  --outdir .worktree/mode_pair_reference_20260620 \
  --output .worktree/mode_pair_reference_20260620/N64_mode_pair_R1_steps2_Dmax32.h5
```

The run selected the analytic reference
`detuning_reference_gap_source = "ising_mode_pair_reference"` with

```text
gap = detuning_delta_min = 0.283677054952,
detuning_delta_max = 1.702062329712,
mode_ek_values range = [0.141838527476, 4.098765891354].
```

The HDF5 output kept the state-selected half-integer grid
(`mode_gF = -1`, `mode_gF_source = "state"`) and stored mode arrays of shape
`3 x 64`, namely the initial row plus two completed cooling cycles.  The direct
energy and the mode reconstruction agreed with

```text
max_t |E(t) - E_modes(t)| = 1.18e-12.
```

The short-run summary is:

| R | completed cycles | detuning protocol | final E/N | best E/N | relE | Dsys | Devolved | Dtdvp sweep | mode gF source | max \(|E-E_{\mathrm{modes}}|\) |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 2/2 | gap-scaled analytic pair reference | -1.04991405 | -1.05000000 | 0.19550 | 5 | 6 | 6 | state | 1.18e-12 |

This is a convention and execution check, not a cooling claim.  It verifies
that the formerly failing automatic-detuning path now reaches the actual TDVP
evolution and records physically interpretable mode data at `N = 64`.

### Ten-Cycle Strided Mode Scan With Pair Detuning

The two-cycle check above is too short to assess cooling.  A longer
mode-resolved scan was therefore run over `R = 1, 2, 5, 10`, still using the
automatic periodic Ising pair reference.  Since the TN mode observable evaluates
all split-string correlators, the run used `--mode-measurement-stride 5`: mode
rows were measured only at cycles `0`, `5`, and `10`, while energy and bond
diagnostics were still recorded every cycle.

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --model ising --bc periodic --Ns 64 --R-values 1,2,5,10 \
  --methods mcwf --evolution-method continuous --steps 10 --Dmax 32 \
  --cutoff 1e-6 --tau 0.2 --te 0.1 --M-mcwf 1 --h -1.05 \
  --measure-modes --mode-measurement-stride 5 \
  --init-state theta --theta 0.0 --tdvp-sweep-progress --stop-on-bond-cap \
  --outdir .worktree/mode_pair_stride_20260620 \
  --output .worktree/mode_pair_stride_20260620/N64_mode_pair_R1-2-5-10_steps10_Dmax32_te0.1_stride5.h5
```

The output records `mode_measurement_cycles = [0, 5, 10]` for every `R`.
Unmeasured mode rows are intentionally stored as `NaN`.  On the measured rows,
the direct energy and the mode reconstruction agree to
`max |E-E_modes| = 1.18e-12` for all four frequency counts.

| R | completed cycles | final E/N | best E/N | relE | Dsys_eff | Dsb_eff | Dtdvp sweep eff | bond status | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 1 | 10/10 | -1.04974606 | -1.05000000 | 0.19562 | 17 | 22 | 22 | no cap | 399.2 s |
| 2 | 10/10 | -1.04976505 | -1.05000000 | 0.19561 | 17 | 22 | 22 | no cap | 330.7 s |
| 5 | 10/10 | -1.01930043 | -1.05000000 | 0.21895 | 17 | 21 | 21 | no cap | 313.4 s |
| 10 | 10/10 | -1.01557750 | -1.05000000 | 0.22181 | 17 | 22 | 22 | no cap | 319.3 s |

This scan is more informative than the two-cycle check and is still not a
cooling success.  For `R = 1` and `R = 2`, the energy remains essentially near
the initial theta-state value and drifts slightly upward.  For `R = 5` and
`R = 10`, the single trajectory heats substantially.  None of these runs comes
close to the finite-chain DMRG reference `E0/N = -1.3050442852`.

The main algorithmic conclusion is that, after the detuning convention is
corrected, the present `XX` round-robin pair-detuning schedule does not cool
this `N = 64` theta state toward the ground state on a ten-cycle prefix.  The
main computational conclusion is that full TN mode snapshots are expensive once
the MPS bond dimension grows; striding makes diagnostic runs possible, but a
production mode-occupation calculation will need either optimized correlator
contractions, a coarser observable schedule, or both.

## Reproduction commands

The four-cycle Dmax ladder was generated with

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf --steps 4 \
  --Dmax-values 320,640 --cutoff 1e-7 --tau 0.2 --M-mcwf 1 \
  --delta-min 0.5051167496264384 --delta-max 3.0307004977586303 \
  --outdir /tmp/coolingtns_largeN_dmax_ladder_steps4_20260618 --verbose
```

The saturated `R = 2` and `R = 10` schedules were then extended to `Dmax = 960`
with

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 2,10 --methods mcwf --steps 4 \
  --Dmax-values 960 --cutoff 1e-7 --tau 0.2 --M-mcwf 1 \
  --delta-min 0.5051167496264384 --delta-max 3.0307004977586303 \
  --outdir /tmp/coolingtns_largeN_dmax960_R2_R10_steps4_20260618 \
  --verbose
```

The HDF5 files were summarized with

```bash
julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl \
  /tmp/coolingtns_largeN_dmax_ladder_steps4_20260618/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_steps4_Dmax320_tau0.2_seed20260617.h5 \
  /tmp/coolingtns_largeN_dmax_ladder_steps4_20260618/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_steps4_Dmax640_tau0.2_seed20260617.h5 \
  /tmp/coolingtns_largeN_dmax960_R2_R10_steps4_20260618/largeN_multifrequency_tn_N64_R2-10_mcwf_steps4_Dmax960_tau0.2_seed20260617.h5
```

The two-cycle TDVP calibration was generated with

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,5,10 --methods mcwf --evolution-method continuous \
  --steps 2 --Dmax 96 --cutoff 1e-6 --tau 0.2 --M-mcwf 1 \
  --delta-min 0.5051167496264384 --delta-max 3.0307004977586303 \
  --outdir /tmp/coolingtns_tdvp_N64_R1-5-10_D96_steps2_20260618 \
  --progress-csv /tmp/coolingtns_tdvp_N64_R1-5-10_D96_steps2_20260618/progress.csv \
  --verbose
```

and similarly for `R = 2` in
`/tmp/coolingtns_tdvp_N64_R2_D96_steps2_20260618`.  The progress CSV records
the `initial`, `prepared`, `evolved`, and `updated` observer stages, so an
interrupted long TDVP run still leaves the pre-evolution system-bath bond
dimension, the post-evolution system-bath bond dimension, and the measured
per-cycle energy trace.

For long TDVP runs, add `--tdvp-sweep-progress` to the same command to append
`tdvp_sweep` rows to the progress CSV after each ITensorMPS TDVP sweep/substep.
These rows record the sweep index, the physical TDVP time reached inside the
current cooling step, and the current TDVP sweep bond dimensions.  Use
`--tdvp-outputlevel 1` when the ITensorMPS textual sweep summary is also useful.

## Current N=64 evidence

The strongest current four-cycle estimate is

| R | Dcap | Dsys_eff | Dsb_eff | bond_status | final E/N | relE | final sys max | peak evolved max | evolved sat |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 320 | 288 | >=320 | not_converged_evolved_cap | 1.53349398 | 2.15767 | 288 | 320 | 4 |
| 1 | 640 | 309 | 394 | no_cap_hit | 1.53349335 | 2.15767 | 309 | 394 | none |
| 2 | 320 | 318 | >=320 | not_converged_evolved_cap | 0.98416142 | 1.74297 | 318 | 320 | 4 |
| 2 | 640 | 588 | >=640 | not_converged_evolved_cap | 0.98420719 | 1.74300 | 588 | 640 | 4 |
| 2 | 960 | 637 | 862 | no_cap_hit | 0.98420691 | 1.74300 | 637 | 862 | none |
| 5 | 320 | 308 | >=320 | not_converged_evolved_cap | 1.04795663 | 1.79113 | 308 | 320 | 4 |
| 5 | 640 | 399 | 518 | no_cap_hit | 1.04794454 | 1.79112 | 399 | 518 | none |
| 10 | 320 | 310 | >=320 | not_converged_evolved_cap | 1.29587871 | 1.97829 | 310 | 320 | 4 |
| 10 | 640 | 488 | >=640 | not_converged_evolved_cap | 1.29572949 | 1.97818 | 488 | 640 | 4 |
| 10 | 960 | 489 | 737 | no_cap_hit | 1.29572864 | 1.97818 | 489 | 737 | none |

Thus `Dmax = 320` is not a converged cap by the fourth cooling cycle for any
of `R = 1,2,5,10`: the evolved system-bath state reaches the cap in all four
cases.  At `Dmax = 640`, the `R = 1` and `R = 5` trajectories are below cap
with observed evolved system-bath dimensions 394 and 518, respectively, while
`R = 2` and `R = 10` still reach the cap by the fourth cycle.  The focused
`Dmax = 960` follow-up resolves those two lower bounds for this four-cycle
diagnostic: `Dsb_eff = 862` for `R = 2` and `Dsb_eff = 737` for `R = 10`.

The strongest currently bounded effective bond dimensions (post-measurement
system `Dsys_eff` and evolved system-bath `Dsb_eff`) are:

```text
R =  1: Dsys_eff = 309, Dsb_eff = 394
R =  2: Dsys_eff = 637, Dsb_eff = 862
R =  5: Dsys_eff = 399, Dsb_eff = 518
R = 10: Dsys_eff = 489, Dsb_eff = 737
```

These values are far larger than the bond caps used in earlier exploratory
large-N curves.  In particular, `Dmax = 40`, `Dmax = 80`, and four-cycle
`Dmax = 320` runs should be read as bond-cap and truncation-pressure
diagnostics rather than as converged large-N cooling trajectories.  The
`Dmax = 960` follow-up also took production-scale time for the final high-bond
steps, so future ladders should write step-level checkpoints or be run as
scheduled jobs rather than as interactive smoke tests.

## Physical interpretation

The relative energies in the table are still of order one and lie far above
the DMRG ground-state reference `E0/N = -1.3246328892`.  Increasing the number
of detunings alone does not overcome the bond-dimension bottleneck in this
protocol by the fourth cycle.  The present data therefore support the following
limited conclusion:

```text
For N = 64, MCWF/MPS Trotter cooling with this fixed detuning interval already
requires Dsys of several hundred and Dsb up to 862 in the four-cycle schedules
tested here.  These runs diagnose entanglement growth and truncation pressure;
they do not yet establish scalable ground-state cooling.
```

A credible long-time `N = 64` to `N = 100` production calculation should use a
controlled Dmax ladder, keep the existing cap/saturation diagnostics, add
measured truncation-error histories when available, and either increase the
effective evolved system-bath bond cap or change the cooling protocol to
control the system-bath entanglement growth.
The large-`N` driver can now print paired MCWF/MPS Trotter and MCWF+TDVP
commands with `--evolution-method-values trotter,continuous`; this planning
axis requires an explicit fixed detuning interval, so the comparison keeps the
physical bath frequencies fixed while varying only the MPS evolution scheme.
When `--tdvp-sweep-progress` or a nonzero `--tdvp-outputlevel` is supplied for
such a paired plan, only the generated TDVP commands carry the TDVP-only
observer options.

## Dmax=64 Paired Trotter/TDVP Descending Comparison

After the paired-plan fix, the same `N = 64`, `Dmax = 64`,
fixed-detuning descending protocol was run for both MCWF/MPS evolution routes.
The two commands differed only in `--evolution-method` and in the TDVP-only
sweep observer option:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method trotter --steps 8 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --outdir .worktree/N64_mcwf_trotter_tdvp_compare_20260622 \
  --progress-csv .worktree/N64_mcwf_trotter_tdvp_compare_20260622/tdvp_progress_trotter_N64_R1-2-5-10_Dmax64_te1.0_descending.csv \
  --stop-on-bond-cap --verbose
```

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 8 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --outdir .worktree/N64_mcwf_trotter_tdvp_compare_20260622 \
  --progress-csv .worktree/N64_mcwf_trotter_tdvp_compare_20260622/tdvp_progress_continuous_N64_R1-2-5-10_Dmax64_te1.0_descending.csv \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

In the continuous command, `--tau 0.2` is part of the common recorded campaign
argument set.  The TDVP branch uses the total bath-evolution time `--te 1.0`;
it stores `tau` in the run metadata but does not use it as a Trotter time step.

The resulting HDF5 files were

```text
.worktree/N64_mcwf_trotter_tdvp_compare_20260622/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_stopcap_scheddesc_steps8_Dmax64_te1_tau0.2_seed20260617.h5
.worktree/N64_mcwf_trotter_tdvp_compare_20260622/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_continuous_stopcap_scheddesc_steps8_Dmax64_te1_tau0.2_seed20260617.h5
```

The compact bond summary is

| evolution | R | completed/requested | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| trotter | 1 | 5/8 | single_detuning | 1.40027789 | 1.38324273 | >=64 | >=64 | n/a | not_converged_system_and_evolved_cap | 35.4 s |
| trotter | 2 | 4/8 | full_grid_observed | 1.06961751 | 1.04681493 | 54 | >=64 | n/a | not_converged_evolved_cap | 8.3 s |
| trotter | 5 | 4/8 | stopped_partial_grid | 1.17618615 | 1.17618615 | 53 | >=64 | n/a | not_converged_evolved_cap | 7.4 s |
| trotter | 10 | 4/8 | requested_partial_grid | 0.92454989 | 0.92454989 | 53 | >=64 | n/a | not_converged_evolved_cap | 7.7 s |
| continuous | 1 | 5/8 | single_detuning | 1.39308425 | 1.37238762 | >=64 | >=64 | >=64 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 454.4 s |
| continuous | 2 | 4/8 | full_grid_observed | 1.03055274 | 1.00906275 | 56 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 202.2 s |
| continuous | 5 | 4/8 | stopped_partial_grid | 1.07479551 | 1.07479551 | 55 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 194.3 s |
| continuous | 10 | 4/8 | requested_partial_grid | 0.87319302 | 0.87319302 | 56 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 192.6 s |

Thus the corrected paired comparison preserves the earlier physical picture.
For this finite prefix and cap, TDVP gives lower energies than the Trotter-gate
route at every tested `R`, and `R = 10` is the best of the four frequency
counts for both evolution methods.  However, every run is cap-limited before
the requested eighth cooling cycle.  In particular, the evolved system-bath
effective bond dimension is only lower bounded by `64` for all `R`, and the
TDVP sweep observer also reaches `64` for every continuous run.  The best
observed energy density, `0.87319302`, is still far above the DMRG reference
`E0/N = -1.3246328892`.  In both HDF5 files above, this reference is the stored
dataset `N64/mcwf/E0 = -84.7765049115883`, so
`E0/N = -84.7765049115883/64 = -1.3246328892`; the same metadata stores the
finite-size gap `N64/mcwf/gap = 0.2744818345585003`.  These data are therefore
evidence about the relative finite-prefix behavior and bond-growth pressure of
the two MPS evolution routes, not evidence of scalable ground-state cooling.
The Trotter-TDVP energy difference also includes the Trotter route's splitting
error at `tau = 0.2`; a Trotter step-size ladder would be needed before
attributing that difference only to cooling efficiency or bond truncation.

## R=10 Trotter Step-Size Ladder at Dmax=64

To probe how sensitive the finite-prefix Trotter-vs-TDVP comparison above is
to Trotter discretization, the `N = 64`, `R = 10`, `Dmax = 64`,
fixed-detuning descending Trotter diagnostic was repeated with the same
trajectory seed and the same physical bath-evolution time `te = 1.0`, while
varying the nominal Trotter step size:

```bash
for TAU in 0.2 0.1 0.05; do
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
    --Ns 64 --R-values 10 --methods mcwf \
    --evolution-method trotter --steps 8 --Dmax 64 \
    --cutoff 1e-7 --tau "${TAU}" --model niising --bc open --te 1.0 \
    --delta-min 0.5051167496264384 \
    --delta-max 3.0307004977586303 \
    --schedule descending \
    --outdir .worktree/N64_trotter_tau_ladder_20260622 \
    --stop-on-bond-cap --verbose
done
```

The generated HDF5 files use the current explicit evolution-method stem:

```text
.worktree/N64_trotter_tau_ladder_20260622/largeN_multifrequency_tn_N64_R10_mcwf_trotter_stopcap_scheddesc_steps8_Dmax64_te1_tau0.2_seed20260617.h5
.worktree/N64_trotter_tau_ladder_20260622/largeN_multifrequency_tn_N64_R10_mcwf_trotter_stopcap_scheddesc_steps8_Dmax64_te1_tau0.1_seed20260617.h5
.worktree/N64_trotter_tau_ladder_20260622/largeN_multifrequency_tn_N64_R10_mcwf_trotter_stopcap_scheddesc_steps8_Dmax64_te1_tau0.05_seed20260617.h5
```

The compact bond summary is

| tau | completed/requested | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | bond_status | elapsed |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 0.20 | 4/8 | requested_partial_grid | 0.92454989 | 0.92454989 | 53 | >=64 | not_converged_evolved_cap | 28.2 s |
| 0.10 | 4/8 | requested_partial_grid | 0.87779382 | 0.87779382 | 51 | >=64 | not_converged_evolved_cap | 33.8 s |
| 0.05 | 5/8 | requested_partial_grid | 0.81458303 | 0.81458303 | >=64 | >=64 | not_converged_system_and_evolved_cap | 72.7 s |

The `tau = 0.2` physics row reproduces the earlier paired-comparison
`R = 10` Trotter row; its elapsed time differs because the ladder row is an
independent single-`R` job rather than one row inside the earlier multi-`R`
campaign.

The equal-cycle rows are consistent with a finite Trotter step-size
contribution to the `tau = 0.2` deficit, but this single-trajectory MCWF
comparison does not isolate that contribution from the changed jump-sampling
cadence discussed below.  At the same four completed cycles, reducing `tau`
from `0.2` to `0.1` lowers the Trotter energy density from `0.92454989` to
`0.87779382`, close to the four-cycle TDVP value `0.87319302` above.  Reducing
to `tau = 0.05` gives a lower value `0.81458303`, but it also reaches five
completed cycles before the stop-on-cap criterion fires, so it is not an
equal-cycle comparison with the TDVP row.

The physical conclusion is still negative: every Trotter step-size row is
bond-cap limited, and the best observed value in this ladder remains far above
the stored DMRG reference `E0/N = -1.3246328892`.  This ladder is therefore
evidence that both Trotter step size and MCWF sampling cadence affect the
finite-prefix comparison; it is not evidence of converged large-system cooling.
Because this is an MCWF diagnostic, changing `tau` also changes the
jump-sampling cadence, so the realized stochastic path is not held fixed across
the ladder; a clean
step-size-convergence measurement would require fixed realized jump times or an
ensemble average.

## Two-Cycle MCWF+TDVP Runtime Calibration

The first completed fixed-detuning TDVP calibration checks only two cooling
cycles.  This is not physically meaningful evidence for cooling or for a
fixed point.  It is useful only for verifying that the continuous-evolution
route is accessible and for estimating the first evolved system-bath bond
dimensions before attempting long traced runs.

| R | Dcap | Dsys_eff | Dsb_eff | bond_status | final E/N | relE | peak evolved mean | elapsed |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 96 | 36 | 50 | no_cap_hit | 1.45494183 | 2.09837 | 39.67 | 168.6 s |
| 2 | 96 | 39 | 54 | no_cap_hit | 1.00710734 | 1.76029 | 41.88 | 227.0 s |
| 5 | 96 | 36 | 51 | no_cap_hit | 1.50779337 | 2.13827 | 39.69 | 287.3 s |
| 10 | 96 | 35 | 51 | no_cap_hit | 1.38290684 | 2.04399 | 39.76 | 376.4 s |

Thus, for the first two TDVP cycles only, `Dmax = 96` is sufficient for all
four frequency counts tested here: the evolved system-bath dimensions are
about `50--54`, and the retained system-state dimensions are about `35--39`.
The energies, however, remain positive and far above the DMRG reference
`E0/N = -1.3246328892`.  The next meaningful TDVP test is therefore a longer
run with the progress CSV enabled, so that one can examine whether the expected
low-entanglement fixed point appears after the high-energy transient regime.

## Five-Cycle MCWF+TDVP Low-Cap Sweep Diagnostic

After adding sweep-level TDVP diagnostics, a deliberately low-cap five-cycle
run was performed for all four frequency counts:

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 5 --Dmax 32 \
  --cutoff 1e-6 --tau 0.2 --M-mcwf 1 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --outdir /tmp/coolingtns_tdvp_sweep_N64_R1-2-5-10_D32_steps5_20260618 \
  --progress-csv /tmp/coolingtns_tdvp_sweep_N64_R1-2-5-10_D32_steps5_20260618/progress.csv \
  --tdvp-sweep-progress --tdvp-outputlevel 1 --verbose
```

This run is a stress test, not a converged calculation.  Its purpose is to
check whether a five-cycle TDVP trace can be recorded at `N = 64` and to locate
the onset of the bond-dimension bottleneck.

Summarizing the retained progress CSV with the current convention gives

| R | Dcap | completed cycles | final E/N | relE | Dsys_eff | Dsb_eff | bond_status | system cap | evolved cap | tdvp sweep cap | first transient cap | max sweep dt | max sweep at | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---|---:|
| 1 | 32 | 5 | 1.44660215 | 2.09208 | >=32 | >=32 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 2 | 2 | 2:7 | 2:7 | 48.7 s | 2:8 | 714.6 s |
| 2 | 32 | 5 | 0.93477793 | 1.70569 | >=32 | >=32 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 2 | 2 | 2:7 | 2:7 | 110.4 s | 5:5 | 759.2 s |
| 5 | 32 | 5 | 0.71627699 | 1.54074 | >=32 | >=32 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 2 | 2 | 2:7 | 2:7 | 190.0 s | 4:7 | 852.7 s |
| 10 | 32 | 5 | 1.25928111 | 1.95066 | >=32 | >=32 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 2 | 2 | 2:7 | 2:7 | 173.9 s | 4:7 | 889.9 s |

The per-cycle energy trace from the progress CSV was

| R | cycle 1 | cycle 2 | cycle 3 | cycle 4 | cycle 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.48954935 | 1.45535389 | 1.50680281 | 1.49513020 | 1.44660215 |
| 2 | 1.35534537 | 1.00731535 | 0.98177652 | 0.95837994 | 0.93477793 |
| 5 | 1.50268855 | 1.50795671 | 1.39331737 | 1.05050253 | 0.71627699 |
| 10 | 1.43490779 | 1.38308413 | 1.28262081 | 1.29877448 | 1.25928111 |

The three cap sources agree on the onset cycle but not on the same object:
the retained system MPS and the evolved system-bath MPS first reach
`Dmax = 32` in cycle 2, while the TDVP sweep observer first reaches the cap in
cycle 2 sweep 7.  The `first transient cap` column reports the earliest
system-bath cap source among the evolved and TDVP-sweep observations.

The low-cap trajectory therefore does not support a physical cooling claim.
It does show that the sweep-progress rows are sufficient to diagnose both
rapid cap saturation and large sweep-to-sweep runtime variability.  A
meaningful next TDVP calculation should focus on `R = 2` and/or `R = 5` at a
larger cap, with the same sweep diagnostics enabled, and should stop early if
the cap is reached before a controlled third or fourth cycle is obtained.

## Five-Cycle MCWF+TDVP R=5, Dmax=96 Probe

The first longer fixed-detuning TDVP probe at the larger cap was then run for
`R = 5`:

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 5 --methods mcwf \
  --evolution-method continuous --steps 5 --Dmax 96 \
  --cutoff 1e-7 --tau 0.2 --M-mcwf 1 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --outdir /tmp/coolingtns_tdvp_sweep_N64_R5_D96_steps5_20260619 \
  --progress-csv /tmp/coolingtns_tdvp_sweep_N64_R5_D96_steps5_20260619/progress.csv \
  --tdvp-sweep-progress --tdvp-outputlevel 1 --verbose
```

Summarizing the progress CSV gives

| R | Dcap | completed cycles | final E/N | Dsys_eff | Dsb_eff | bond_status | system cap | evolved cap | tdvp sweep cap | first transient cap | max sweep dt | max sweep at |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---|
| 5 | 96 | 5 | 0.66394232 | >=96 | >=96 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 4 | 3 | 3:4 | 3:4 | 334.2 s | 5:4 |

The per-cycle energy and bond trace is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.50511675 | 1.50329932 | 11 | 16 | 47.2 s |
| 2 | 1.13651269 | 1.50854622 | 54 | 73 | 833.8 s |
| 3 | 1.76790862 | 1.39440026 | 95 | 96 | 2439.6 s |
| 4 | 2.39930456 | 0.99126917 | 96 | 96 | 3901.7 s |
| 5 | 3.03070050 | 0.66394232 | 96 | 96 | 5256.1 s |

This run is more informative than the two-cycle calibration: after the early
transient, the energy decreases for cycles 3 through 5.  It is nevertheless not
a converged TDVP benchmark.  The evolved system-bath MPS first reaches the
`Dmax = 96` cap during cycle 3, the TDVP sweep trace reaches it during cycle 3
sweep 4, and the retained system MPS reaches the cap in cycle 4.  Therefore the
reported effective bond dimensions are lower bounds,
`D_{\rm sys}^{\rm eff} >= 96` and `D_{\rm sb}^{\rm eff} >= 96`, rather than
resolved requirements.  The result should be read as evidence that the
physically relevant TDVP run has entered the high-bond regime, not as evidence
that `Dmax = 96` is sufficient for large-system cooling.

## Forty-Cycle MCWF+TDVP Stop-on-Cap Diagnostics

After adding the opt-in stop condition, the same fixed detuning interval was
used for a requested forty-cycle MCWF+TDVP diagnostic over all four frequency
counts:

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 40 --Dmax 96 \
  --cutoff 1e-6 --tau 0.2 --M-mcwf 1 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --outdir .worktree/largeN_stopcap_20260619 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

These runs were intended to answer whether the longer calculation can proceed
beyond the early transient, not to establish convergence.  All four `Dmax = 96`
jobs stopped after three completed cycles because the evolved system-bath state
reached the cap during cycle 3.  The `R = 2`, `Dmax = 128` follow-up stopped at
the same cycle.

The raw progress CSVs for this legacy campaign are not retained in the
repository.  The historical progress summary below predates #339, so the final
cap column is the first transient cap hit reported by the old summarizer, not
an evolved-stage-only cap column.  Current progress summaries report separate
`evolved cap`, `tdvp sweep cap`, and `first transient cap` columns.

| R | Dcap | completed/requested cycles | final E/N | Dsys_eff | Dsb_eff | legacy bond_status | legacy first transient cap | max sweep dt | elapsed |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 96 | 3/40 | 1.46044484 | 91 | >=96 | not_converged_evolved_cap | 3:7 | 131.5 s | 878.5 s |
| 2 | 96 | 3/40 | 0.98249764 | 95 | >=96 | not_converged_evolved_cap | 3:6 | 148.7 s | 1075.2 s |
| 5 | 96 | 3/40 | 1.39412230 | 90 | >=96 | not_converged_evolved_cap | 3:8 | 124.0 s | 710.0 s |
| 10 | 96 | 3/40 | 1.28228975 | 92 | >=96 | not_converged_evolved_cap | 3:7 | 126.0 s | 742.1 s |
| 2 | 128 | 3/40 | 0.98236229 | 120 | >=128 | not_converged_evolved_cap | 3:8 | 281.9 s | 1171.6 s |

The corresponding completed-cycle energy prefixes are

| R | Dcap | cycle 1 | cycle 2 | cycle 3 |
|---:|---:|---:|---:|---:|
| 1 | 96 | 1.48954935 | 1.45494183 | 1.46044484 |
| 2 | 96 | 1.35534537 | 1.00710734 | 0.98249764 |
| 5 | 96 | 1.50268855 | 1.50779337 | 1.39412230 |
| 10 | 96 | 1.43490779 | 1.38290684 | 1.28228975 |
| 2 | 128 | 1.35534537 | 1.00710734 | 0.98236229 |

Thus the stop-on-cap calculation is more useful than a two-cycle timing
calibration, but it is still a cap-location diagnostic.  Increasing `R = 2`
from `Dmax = 96` to `Dmax = 128` changes the third-cycle energy only in the
fourth decimal place and does not move the first cap event beyond cycle 3.  At
this fixed detuning interval and `te = 2.0`, modestly raising the cap does not
yet produce a physically controlled long-time cooling trajectory.

The HDF5 files for this particular stop-cap campaign predate the HDF5
persistence of TDVP sweep histories added later.  Consequently
`summarize_largeN_bond_dimensions.jl` reports `Dtdvp_sweep_eff = n/a` for these
legacy files.  The legacy progress rows above show cycle-3 `first transient
cap` events, with sweep indices when the first hit was observed by a TDVP sweep
row.  New stop-cap runs with `--tdvp-sweep-progress` store both
`tdvp_sweep_max_bond` and `tdvp_sweep_saturation_cycle` in HDF5.

## MCWF+TDVP Stop-on-Cap Scan at te=1.0

The previous stop-on-cap scan used `te = 2.0` and reached the evolved-state cap
after only three completed cycles.  To test whether the per-cycle TDVP evolution
time is driving the early entanglement growth, the same `N = 64`, `Dmax = 96`
diagnostic was repeated with `te = 1.0` for `R = 1,2,5,10`:

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 40 --Dmax 96 \
  --cutoff 1e-6 --tau 0.2 --te 1.0 --M-mcwf 1 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --outdir .worktree/largeN_te_scan_20260619 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The jobs were run as separate single-`R` processes with one Julia thread and
one BLAS thread.  The HDF5 summary is

| R | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 1 | 96 | 6/40 | 1.38944858 | 1.37249467 | 91 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 6 | 6 | 583.6 s |
| 2 | 96 | 5/40 | 1.04543594 | 1.04543594 | 75 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 5 | 5 | 354.3 s |
| 5 | 96 | 6/40 | 0.95391585 | 0.95391585 | 93 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 6 | 6 | 678.1 s |
| 10 | 96 | 6/40 | 1.06789254 | 1.06789254 | 95 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 6 | 6 | 814.4 s |

The completed-cycle energy prefixes are

| R | cycle 1 | cycle 2 | cycle 3 | cycle 4 | cycle 5 | cycle 6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.47283533 | 1.42021639 | 1.41664172 | 1.37249467 | 1.39299353 | 1.38944858 |
| 2 | 1.34971125 | 1.08774223 | 1.11219382 | 1.10108801 | 1.04543594 | n/a |
| 5 | 1.45742344 | 1.47791073 | 1.41073739 | 1.19093076 | 1.00541882 | 0.95391585 |
| 10 | 1.43660574 | 1.37874652 | 1.24211139 | 1.21842946 | 1.19862114 | 1.06789254 |

Thus lowering `te` from `2.0` to `1.0` materially delays the first cap event:
the `te = 2.0` scan stopped at cycle 3 for all four frequency counts, while
the `te = 1.0` scan reaches cycle 5 for `R = 2` and cycle 6 for
`R = 1,5,10`.  This is a useful protocol signal, not a converged cooling
calculation.  Every run still reaches the evolved and TDVP-sweep cap, and the
final energies remain far above the DMRG reference
`E0/N = -1.3246328892`.  Among these single-trajectory capped prefixes, `R = 5`
has the lowest final energy, followed by `R = 2` and `R = 10`, while `R = 1`
remains poor.

The next physical scan should therefore change the schedule before simply
raising `Dmax`: for example, test `te = 0.5` at `Dmax = 96`, or increase the
cap only after identifying a detuning/time schedule that postpones cap growth
without stalling the energy.

## Focused MCWF+TDVP R=2 and R=5 Probes at te=0.5

The `R = 5` schedule gave the lowest final energy in the `te = 1.0` scan.  A
single follow-up run was therefore made at the smaller per-cycle evolution time
`te = 0.5`.  The same `te = 0.5` probe was then repeated for `R = 2` to check
whether the observed energy stalling was specific to the five-frequency
schedule:

```bash
for R in 2 5; do
  julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
    --Ns 64 --R-values "$R" --methods mcwf \
    --evolution-method continuous --steps 40 --Dmax 96 \
    --cutoff 1e-6 --tau 0.2 --te 0.5 --M-mcwf 1 \
    --delta-min 0.5051167496264384 \
    --delta-max 3.0307004977586303 \
    --outdir .worktree/largeN_te_scan_20260619 \
    --tdvp-sweep-progress --stop-on-bond-cap --verbose
done
```

The HDF5 summary is

| R | te | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 2 | 0.5 | 96 | 11/40 | 1.10014440 | 1.10014440 | 86 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 11 | 11 | 619.7 s |
| 5 | 0.5 | 96 | 12/40 | 1.03468046 | 1.02937119 | 90 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 12 | 12 | 744.7 s |

The `R = 2` completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.50511675 | 1.32293350 | 2 | 3 | 29.0 s |
| 2 | 3.03070050 | 1.32029906 | 4 | 5 | 36.2 s |
| 3 | 0.50511675 | 1.32385375 | 6 | 8 | 44.5 s |
| 4 | 3.03070050 | 1.32156215 | 9 | 12 | 54.2 s |
| 5 | 0.50511675 | 1.30132150 | 14 | 17 | 67.9 s |
| 6 | 3.03070050 | 1.29845578 | 18 | 25 | 88.9 s |
| 7 | 0.50511675 | 1.30198653 | 29 | 35 | 118.1 s |
| 8 | 3.03070050 | 1.24827046 | 37 | 50 | 167.8 s |
| 9 | 0.50511675 | 1.25133930 | 55 | 68 | 248.2 s |
| 10 | 3.03070050 | 1.18081736 | 68 | 93 | 385.2 s |
| 11 | 0.50511675 | 1.10014440 | 86 | 96 | 619.7 s |

The `R = 5` completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.50511675 | 1.49437966 | 2 | 3 | 21.0 s |
| 2 | 1.13651269 | 1.49576087 | 4 | 5 | 27.3 s |
| 3 | 1.76790862 | 1.44714676 | 6 | 7 | 35.1 s |
| 4 | 2.39930456 | 1.38988194 | 8 | 10 | 44.2 s |
| 5 | 3.03070050 | 1.28612843 | 12 | 15 | 55.5 s |
| 6 | 0.50511675 | 1.28754107 | 17 | 21 | 71.7 s |
| 7 | 1.13651269 | 1.20105179 | 24 | 29 | 96.5 s |
| 8 | 1.76790862 | 1.15550268 | 32 | 40 | 137.0 s |
| 9 | 2.39930456 | 1.06446658 | 43 | 53 | 203.8 s |
| 10 | 3.03070050 | 1.02937119 | 55 | 69 | 308.1 s |
| 11 | 0.50511675 | 1.03303355 | 73 | 90 | 478.7 s |
| 12 | 1.13651269 | 1.03468046 | 90 | 96 | 744.7 s |

This confirms that reducing `te` can strongly delay the bond-cap event: for
`R = 2`, the cap moves from cycle 5 at `te = 1.0` to cycle 11 at `te = 0.5`,
and for `R = 5`, it moves from cycle 6 to cycle 12.  However, both smaller-`te`
runs have worse capped-prefix energies than their `te = 1.0` counterparts.
The result therefore argues against the simple rule "make `te` smaller" as a
scalable cooling strategy.  It points instead to a schedule problem: the
protocol should adapt the detunings, evolution times, or compression criteria
before a large-`Dmax` production run is interpreted as physical cooling.

## Random-Schedule MCWF+TDVP Probe at te=1.0

The code also supports a random detuning order using the same fixed detuning
set.  To test whether the round-robin ordering itself was responsible for the
early cap events, one-seed random-schedule probes were run for `R = 2` and
`R = 5` at `te = 1.0`:

```bash
for R in 2 5; do
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
    --Ns 64 --R-values "$R" --methods mcwf \
    --evolution-method continuous --steps 40 --Dmax 96 \
    --cutoff 1e-6 --tau 0.2 --te 1.0 --M-mcwf 1 \
    --delta-min 0.5051167496264384 \
    --delta-max 3.0307004977586303 \
    --schedule random \
    --outdir .worktree/largeN_schedule_scan_20260619 \
    --progress-csv ".worktree/largeN_schedule_scan_20260619/tdvp_progress_N64_mcwf_R${R}_Dmax96_te1.0_random.csv" \
    --tdvp-sweep-progress --stop-on-bond-cap --verbose
done
```

These files were generated before the driver default filename was updated to
include `te` and non-default schedules; the HDF5 metadata remains the source of
truth for the protocol.

The HDF5 summary is

| R | schedule | te | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 2 | random | 1.0 | 96 | 6/40 | 0.99951656 | 0.99951656 | >=96 | >=96 | >=96 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 6 | 6 | 6 | 1051.2 s |
| 5 | random | 1.0 | 96 | 5/40 | 1.12466000 | 1.12466000 | 78 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | none | 5 | 5 | 341.3 s |

The completed-cycle prefixes from the progress CSVs are

| R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.50511675 | 1.34970814 | 4 | 5 | 30.3 s |
| 2 | 2 | 3.03070050 | 1.09502553 | 9 | 12 | 47.1 s |
| 2 | 3 | 0.50511675 | 1.11903651 | 18 | 26 | 74.1 s |
| 2 | 4 | 3.03070050 | 1.10864310 | 38 | 49 | 144.4 s |
| 2 | 5 | 0.50511675 | 1.05144686 | 75 | 95 | 372.5 s |
| 2 | 6 | 0.50511675 | 0.99951656 | 96 | 96 | 1051.2 s |
| 5 | 1 | 1.13651269 | 1.44318374 | 4 | 5 | 30.4 s |
| 5 | 2 | 3.03070050 | 1.26075795 | 9 | 12 | 47.0 s |
| 5 | 3 | 0.50511675 | 1.18542501 | 19 | 26 | 74.2 s |
| 5 | 4 | 0.50511675 | 1.15334752 | 40 | 54 | 144.2 s |
| 5 | 5 | 0.50511675 | 1.12466000 | 78 | 96 | 341.3 s |

For `R = 2`, this particular random order gives a lower capped-prefix energy
than the round-robin `te = 1.0` run, `0.99951656` instead of `1.04543594`, but
only after the retained system state, evolved system-bath state, and TDVP
sweep history all reach the cap at cycle 6.  For `R = 5`, the random schedule
is worse than the round-robin `te = 1.0` run and reaches the evolved and TDVP
sweep caps one cycle earlier.  Thus random ordering is a real schedule
variable, but this one-seed diagnostic does not give evidence of a scalable
low-entanglement route to the ground state.

## Randomized-Time MCWF+TDVP Probe at Mean te=1.0

After exposing randomized evolution times in the large-N driver, the same
`N = 64`, `Dmax = 96` stop-on-cap diagnostic was repeated with round-robin
detuning order but randomized cycle times.  Each cycle draws
`t_m ~ Uniform(0, 2 te)` with mean `te = 1.0`, and the HDF5 output records the
realized `te_list`.

```bash
for R in 1 2 5 10; do
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
    --Ns 64 --R-values "$R" --methods mcwf \
    --evolution-method continuous --steps 40 --Dmax 96 \
    --cutoff 1e-6 --tau 0.2 --te 1.0 --randomize-times \
    --M-mcwf 1 \
    --delta-min 0.5051167496264384 \
    --delta-max 3.0307004977586303 \
    --outdir .worktree/largeN_randtime_scan_20260619 \
    --progress-csv ".worktree/largeN_randtime_scan_20260619/tdvp_progress_N64_mcwf_R${R}_Dmax96_te1.0_randtime.csv" \
    --tdvp-sweep-progress --stop-on-bond-cap --verbose
done
```

The HDF5 summary is

| R | mean te | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | 1.0 | 96 | 6/40 | 1.55558375 | 1.48133017 | 87 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | none | 6 | 6 | 858.6 s |
| 2 | 1.0 | 96 | 7/40 | 1.09926433 | 1.09097226 | >=96 | >=96 | >=96 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 7 | 7 | 7 | 1041.6 s |
| 5 | 1.0 | 96 | 7/40 | 1.35803471 | 1.35803471 | 93 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | none | 7 | 7 | 704.2 s |
| 10 | 1.0 | 96 | 8/40 | 1.07882431 | 1.07882431 | 91 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | none | 8 | 8 | 861.9 s |

The completed-cycle prefixes, including the realized cycle times, are

| R | cycle | delta | te | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.50511675 | 0.02811079 | 1.48437487 | 1 | 2 | 20.8 s |
| 1 | 2 | 0.50511675 | 1.58210982 | 1.50767726 | 8 | 8 | 38.3 s |
| 1 | 3 | 0.50511675 | 0.21089060 | 1.50784234 | 8 | 9 | 45.1 s |
| 1 | 4 | 0.50511675 | 1.51131337 | 1.48133017 | 24 | 31 | 97.1 s |
| 1 | 5 | 0.50511675 | 0.83104255 | 1.50550305 | 35 | 45 | 165.7 s |
| 1 | 6 | 0.50511675 | 1.85082244 | 1.55558375 | 87 | 96 | 858.6 s |
| 2 | 1 | 0.50511675 | 0.70976735 | 1.37357943 | 3 | 4 | 24.9 s |
| 2 | 2 | 3.03070050 | 1.35299857 | 1.10646940 | 9 | 13 | 47.4 s |
| 2 | 3 | 0.50511675 | 0.30863670 | 1.10717316 | 12 | 15 | 56.3 s |
| 2 | 4 | 3.03070050 | 1.33648073 | 1.09112697 | 33 | 45 | 126.3 s |
| 2 | 5 | 0.50511675 | 0.26346633 | 1.09135350 | 36 | 47 | 163.4 s |
| 2 | 6 | 3.03070050 | 0.31688886 | 1.09097226 | 49 | 60 | 215.4 s |
| 2 | 7 | 0.50511675 | 1.38166777 | 1.09926433 | 96 | 96 | 1041.6 s |
| 5 | 1 | 0.50511675 | 0.71014027 | 1.41861024 | 3 | 4 | 24.8 s |
| 5 | 2 | 1.13651269 | 1.89633000 | 1.42055762 | 16 | 20 | 64.2 s |
| 5 | 3 | 1.76790862 | 0.35000499 | 1.42019203 | 17 | 24 | 77.2 s |
| 5 | 4 | 2.39930456 | 0.11401318 | 1.42015600 | 17 | 24 | 83.7 s |
| 5 | 5 | 3.03070050 | 0.12515411 | 1.42011740 | 20 | 26 | 90.4 s |
| 5 | 6 | 0.50511675 | 0.48084740 | 1.42511952 | 31 | 39 | 121.2 s |
| 5 | 7 | 1.13651269 | 1.72149561 | 1.35803471 | 93 | 96 | 704.2 s |
| 10 | 1 | 0.50511675 | 0.57805193 | 1.42361164 | 3 | 4 | 23.0 s |
| 10 | 2 | 0.78573717 | 1.92354935 | 1.33488776 | 16 | 18 | 60.1 s |
| 10 | 3 | 1.06635758 | 0.00420173 | 1.33489373 | 16 | 16 | 66.1 s |
| 10 | 4 | 1.34697800 | 0.77368787 | 1.33666918 | 24 | 30 | 94.3 s |
| 10 | 5 | 1.62759842 | 0.33586292 | 1.33747991 | 28 | 34 | 114.7 s |
| 10 | 6 | 1.90821883 | 0.56199226 | 1.18038440 | 39 | 49 | 162.3 s |
| 10 | 7 | 2.18883925 | 0.77709643 | 1.17770501 | 59 | 76 | 298.4 s |
| 10 | 8 | 2.46945966 | 1.06039527 | 1.07882431 | 91 | 96 | 861.9 s |

Randomizing the cycle times delays the cap event for some schedules relative
to fixed `te = 1.0`, but it does not improve the best capped-prefix energy in
this one-seed scan.  The best randomized-time result is `R = 10` with
`E/N = 1.07882431`, still above the fixed-time `R = 5` stopped-prefix energy
`0.95391585`.  The randomized-time `R = 2` run also reaches the retained
system cap at cycle 7 and has worse final and best energies than the fixed-time
`R = 2` scan.  Thus time randomization is a meaningful protocol axis to record,
but this diagnostic does not support it as a solution to the large-N
bond-growth bottleneck.

## Post-Krylov-Expansion MCWF+TDVP N=64 Stop-on-Cap Probe

After PR #229 made Krylov-expanded two-site TDVP the default for MCWF cooling,
the `N = 64` fixed-detuning diagnostic was rerun.  This rerun supersedes the
flat pre-expansion TDVP traces as evidence about the corrected implementation,
but it does not supersede the older tables as historical bond-cap diagnostics.

The all-frequency low-cap scan was

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 8 --Dmax 32 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --progress-csv .worktree/postfix_tdvp_krylov_N64_20260619/tdvp_progress_N64_niising_open_mcwf_Dmax32_te1.0.csv \
  --outdir .worktree/postfix_tdvp_krylov_N64_20260619 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The higher-cap follow-up for the two best low-cap prefixes was

```bash
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 2,10 --methods mcwf \
  --evolution-method continuous --steps 8 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --progress-csv .worktree/postfix_tdvp_krylov_N64_20260619/tdvp_progress_N64_niising_open_mcwf_Dmax64_te1.0_R2_R10.csv \
  --outdir .worktree/postfix_tdvp_krylov_N64_20260619 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The common ED reference is `E0/N = -1.3246328892`, with finite-size gap
`0.27448183`.  The HDF5 summaries are

| Dcap | R | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 32 | 1 | 4/8 | 1.37255398 | 1.37255398 | >=32 | >=32 | >=32 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 4 | 4 | 4 | 294.0 s |
| 32 | 2 | 3/8 | 1.11215604 | 1.08753626 | 24 | >=32 | >=32 | not_converged_evolved_and_tdvp_sweep_cap | none | 3 | 3 | 111.4 s |
| 32 | 5 | 3/8 | 1.41126632 | 1.41126632 | 24 | >=32 | >=32 | not_converged_evolved_and_tdvp_sweep_cap | none | 3 | 3 | 73.5 s |
| 32 | 10 | 3/8 | 1.24038933 | 1.24038933 | 24 | >=32 | >=32 | not_converged_evolved_and_tdvp_sweep_cap | none | 3 | 3 | 79.9 s |
| 64 | 2 | 4/8 | 1.10143056 | 1.08753626 | 53 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | none | 4 | 4 | 443.2 s |
| 64 | 10 | 4/8 | 1.21672165 | 1.21672165 | 50 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | none | 4 | 4 | 633.1 s |

The completed-cycle prefixes are

| Dcap | R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 1 | 0.50511675 | 1.47291896 | 4 | 6 | 33.1 s |
| 32 | 1 | 2 | 0.50511675 | 1.42012553 | 11 | 15 | 69.9 s |
| 32 | 1 | 3 | 0.50511675 | 1.41672905 | 23 | 31 | 135.0 s |
| 32 | 1 | 4 | 0.50511675 | 1.37255398 | 32 | 32 | 294.0 s |
| 32 | 2 | 1 | 0.50511675 | 1.34957051 | 4 | 6 | 21.9 s |
| 32 | 2 | 2 | 3.03070050 | 1.08753626 | 11 | 15 | 51.1 s |
| 32 | 2 | 3 | 0.50511675 | 1.11215604 | 24 | 32 | 111.4 s |
| 32 | 5 | 1 | 0.50511675 | 1.45743454 | 4 | 6 | 21.3 s |
| 32 | 5 | 2 | 1.13651269 | 1.47827437 | 11 | 15 | 39.7 s |
| 32 | 5 | 3 | 1.76790862 | 1.41126632 | 24 | 32 | 73.5 s |
| 32 | 10 | 1 | 0.50511675 | 1.43565981 | 4 | 6 | 21.2 s |
| 32 | 10 | 2 | 0.78573717 | 1.37759077 | 11 | 15 | 39.9 s |
| 32 | 10 | 3 | 1.06635758 | 1.24038933 | 24 | 32 | 79.9 s |
| 64 | 2 | 1 | 0.50511675 | 1.34957051 | 4 | 6 | 22.2 s |
| 64 | 2 | 2 | 3.03070050 | 1.08753626 | 11 | 15 | 51.9 s |
| 64 | 2 | 3 | 0.50511675 | 1.11215599 | 25 | 33 | 121.1 s |
| 64 | 2 | 4 | 3.03070050 | 1.10143056 | 53 | 64 | 443.2 s |
| 64 | 10 | 1 | 0.50511675 | 1.43565981 | 4 | 6 | 20.9 s |
| 64 | 10 | 2 | 0.78573717 | 1.37759077 | 11 | 15 | 40.1 s |
| 64 | 10 | 3 | 1.06635758 | 1.24038933 | 24 | 32 | 81.1 s |
| 64 | 10 | 4 | 1.34697800 | 1.21672165 | 50 | 64 | 633.1 s |

Corrected TDVP dynamics is no longer flat: the Krylov-expanded two-site TDVP
run changes the energy already in the first few cycles.  The present
stop-on-cap data nevertheless remains far from ground-state cooling.  The best
observed capped prefix is the `R = 2` value `E/N = 1.08753626`, still far above
`E0/N = -1.3246328892`.  Raising the cap from `32` to `64` confirms that the
`Dmax = 32` data was cap-limited rather than converged.  For `R = 2` and
`R = 10`, the evolved system-bath MPS and the TDVP sweep observer both reach
bond dimension `64` by cycle 4, while the retained system state has already
grown to bond dimension `53` and `50`, respectively.

Thus the effective evolved system-bath bond dimension after the TDVP correction is at
least `64` by the fourth cooling cycle for these fixed `te = 1.0` schedules.
The next large-\(D\) calculation should therefore not be interpreted only as a
larger-bond rerun.  It should be paired with a schedule or adaptivity change
which can be tested against this post-expansion stop-on-cap baseline.

## Post-Krylov Descending-Detuning Probe

The post-expansion TDVP evidence above used the default round-robin order, so
each fixed grid was traversed from the lowest detuning upward.  To test whether
the early high-energy response is sensitive to this ordering, the driver now
also supports a deterministic descending schedule.  It uses the same fixed
detuning grid as round-robin, but visits the largest detuning first and then
steps downward before repeating.  This is a deterministic schedule probe, not a
random-schedule average.

The diagnostic command was

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 8 --Dmax 32 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_schedule_20260620/tdvp_progress_N64_niising_open_mcwf_Dmax32_te1.0_descending.csv \
  --outdir .worktree/descending_schedule_20260620 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The HDF5 summary is

| R | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | 32 | 4/8 | 1.37255398 | 1.37255398 | >=32 | >=32 | >=32 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 4 | 4 | 4 | 153.9 s |
| 2 | 32 | 3/8 | 1.00906275 | 1.00906275 | 24 | >=32 | >=32 | not_converged_evolved_and_tdvp_sweep_cap | none | 3 | 3 | 63.5 s |
| 5 | 32 | 3/8 | 1.21579225 | 1.21579225 | 24 | >=32 | >=32 | not_converged_evolved_and_tdvp_sweep_cap | none | 3 | 3 | 63.8 s |
| 10 | 32 | 3/8 | 0.95514749 | 0.95514749 | 24 | >=32 | >=32 | not_converged_evolved_and_tdvp_sweep_cap | none | 3 | 3 | 62.8 s |

The completed-cycle prefixes are

| R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.50511675 | 1.47291896 | 4 | 6 | 20.3 s |
| 1 | 2 | 0.50511675 | 1.42012553 | 11 | 15 | 49.2 s |
| 1 | 3 | 0.50511675 | 1.41672905 | 23 | 31 | 89.5 s |
| 1 | 4 | 0.50511675 | 1.37255398 | 32 | 32 | 153.9 s |
| 2 | 1 | 3.03070050 | 1.05928265 | 4 | 6 | 19.6 s |
| 2 | 2 | 0.50511675 | 1.01793111 | 11 | 15 | 40.4 s |
| 2 | 3 | 3.03070050 | 1.00906275 | 24 | 32 | 63.5 s |
| 5 | 1 | 3.03070050 | 1.38240655 | 4 | 6 | 19.3 s |
| 5 | 2 | 2.39930456 | 1.27834942 | 11 | 15 | 40.6 s |
| 5 | 3 | 1.76790862 | 1.21579225 | 24 | 32 | 63.8 s |
| 10 | 1 | 3.03070050 | 1.30592953 | 4 | 6 | 8.9 s |
| 10 | 2 | 2.75008008 | 1.13874047 | 11 | 15 | 25.5 s |
| 10 | 3 | 2.46945966 | 0.95514749 | 24 | 32 | 62.8 s |

The result confirms that detuning order is a real physical protocol variable.
For `R = 10`, descending order reaches `E/N = 0.95514749` before the cap, much
lower than the corresponding post-expansion ascending `Dcap = 32` prefix
`E/N = 1.24038933`.  For `R = 2`, the first high-detuning cycle also lowers the
energy faster than the ascending baseline.  However, every nontrivial
descending run still reaches the evolved-state and TDVP-sweep cap by the third
completed cycle, and the best prefix remains far above the ground-state
reference `E0/N = -1.3246328892`.  Thus descending order is a useful schedule
axis for later adaptive protocols, but it is not by itself evidence of
scalable ground-state cooling.

### Dmax=64 Descending R=10 Follow-up

The best low-cap descending prefix above was the `R = 10` run.  To test whether
that improvement survives a larger evolved system-bath bond cap, the same post-Krylov
TDVP diagnostic was repeated at `Dmax = 64` for `R = 10`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 8 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_schedule_dmax64_20260620/tdvp_progress_N64_niising_open_mcwf_R10_Dmax64_te1.0_descending.csv \
  --outdir .worktree/descending_schedule_dmax64_20260620 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The HDF5 summary is

| R | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 10 | 64 | 4/8 | 0.87319302 | 0.87319302 | 56 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | none | 4 | 4 | 211.4 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.30592953 | 4 | 6 | 22.2 s |
| 2 | 2.75008008 | 1.13874047 | 11 | 15 | 47.9 s |
| 3 | 2.46945966 | 0.95514768 | 24 | 33 | 84.3 s |
| 4 | 2.18883925 | 0.87319302 | 56 | 64 | 211.4 s |

Thus the descending schedule improvement is not only a `Dcap = 32` artifact:
raising the cap to `64` lets the same single trajectory complete one more
cooling cycle and lowers the best observed prefix from the corresponding
`Dcap = 32` value `0.95514749` to `0.87319302`.  The calculation is still not
converged.  The evolved system-bath state and the TDVP sweep observer both
reach the cap in cycle 4, so `Dsb_eff >= 64` and `Dtdvp_sweep_eff >= 64` are
lower bounds.  The energy
also remains far above the DMRG reference `E0/N = -1.3246328892`.  The result
supports descending high-to-low detuning as a useful protocol axis for future
adaptive scans, but it does not establish scalable ground-state cooling.

### Dmax=64 Descending All-R Reproducibility Scan

The `R = 10` follow-up above was then rerun as part of a single all-frequency
descending campaign, so that `R = 1, 2, 5, 10` use the same cap, stopping rule,
driver command, and fixed detuning interval:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 8 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_schedule_dmax64_all_20260620/tdvp_progress_N64_niising_open_mcwf_R1-2-5-10_Dmax64_te1.0_descending.csv \
  --outdir .worktree/descending_schedule_dmax64_all_20260620 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

Both the standalone `R = 10` command and this all-frequency command used the
driver default seed `20260617`, so the identical `R = 10` energy prefix is an
expected reproducibility check rather than a separate stochastic sample.  The
elapsed times are wall-clock measurements and should not be interpreted as
physics data; their difference reflects the execution context of the run.

The HDF5 summary is

| R | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | 64 | 5/8 | 1.39308425 | 1.37238762 | >=64 | >=64 | >=64 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 5 | 5 | 5 | 459.0 s |
| 2 | 64 | 4/8 | 1.03055274 | 1.00906275 | 56 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | none | 4 | 4 | 205.0 s |
| 5 | 64 | 4/8 | 1.07479551 | 1.07479551 | 55 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | none | 4 | 4 | 199.1 s |
| 10 | 64 | 4/8 | 0.87319302 | 0.87319302 | 56 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | none | 4 | 4 | 263.0 s |

The completed-cycle energy-density prefixes are

| R | completed-cycle E/N prefix |
|---:|---|
| 1 | `1.47291896, 1.42012553, 1.41672905, 1.37238762, 1.39308425` |
| 2 | `1.05928265, 1.01793111, 1.00906275, 1.03055274` |
| 5 | `1.38240655, 1.27834942, 1.21579249, 1.07479551` |
| 10 | `1.30592953, 1.13874047, 0.95514768, 0.87319302` |

The all-frequency scan confirms the qualitative conclusion from the standalone
`R = 10` run.  At `Dcap = 64`, descending `R = 10` gives the best capped
prefix among the tested frequency counts, but every run remains cap-limited.
The `R = 1` case reaches the cap in the retained system, the evolved
system-bath state, and the TDVP sweep history by the fifth completed cycle; the
other three cases reach the cap in the evolved system-bath state and TDVP sweep
history by the fourth completed cycle.  Thus the effective evolved system-bath
and TDVP-sweep bond dimensions are at least `64` across the whole descending
scan.  The best observed energy density,
`0.87319302`, is still far above the DMRG reference `E0/N = -1.3246328892`, so
this remains a bond-growth and schedule-order diagnostic rather than a
controlled cooling result.

### Dmax=128 Descending R=10 Follow-up

The best `Dcap = 64` descending prefix was then repeated for `R = 10` at
`Dmax = 128`, again with the fixed detuning interval, stop-on-cap rule, and
one pinned Julia/BLAS thread:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_schedule_dmax128_20260620/tdvp_progress_N64_niising_open_mcwf_R10_Dmax128_te1.0_descending.csv \
  --outdir .worktree/descending_schedule_dmax128_20260620 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The HDF5 file records the root seed `20260617`, the stored seed rule, and the
trajectory seed `[84360618]`.  Thus this is the same stochastic trajectory as
the earlier `R = 10` descending runs: the first three completed-cycle energies
match exactly, while the fourth agrees to six decimal places and then differs
because the larger cap reduces truncation in that cycle.

The HDF5 summary is

| R | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 10 | 128 | 5/12 | 0.84528025 | 0.84528025 | 116 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 5 | 5 | 803.4 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.30592953 | 4 | 6 | 31.1 s |
| 2 | 2.75008008 | 1.13874047 | 11 | 15 | 47.5 s |
| 3 | 2.46945966 | 0.95514768 | 24 | 33 | 84.0 s |
| 4 | 2.18883925 | 0.87319255 | 57 | 73 | 211.2 s |
| 5 | 1.90821883 | 0.84528025 | 116 | 128 | 803.4 s |

This confirms that the `Dcap = 64` cycle-four result was still truncation
limited: the same trajectory continues to cycle 5 when the cap is raised to
`128`.  The improvement, however, is modest compared with the additional bond
dimension and wall time.  The evolved system-bath state and TDVP sweep
observer both reach bond dimension `128` in cycle 5, while the retained system
state reaches bond dimension `116`.  Therefore the effective evolved
system-bath and TDVP-sweep dimensions are still only lower bounded by `128`,
and the best observed energy density `0.84528025` remains far above the DMRG
reference `E0/N = -1.3246328892`.  This is useful evidence for the required
bond scale of the descending schedule, not evidence of scalable cooling to the
ground state.

### Dmax=128 Descending Remaining Frequencies

The remaining frequency counts were then run at the same cap, fixed detuning
interval, schedule, stopping rule, and thread pinning.  The already-recorded
`R = 10` file above is reused for the all-frequency comparison rather than
recomputed.

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_schedule_dmax128_R1_R2_R5_20260620/tdvp_progress_N64_niising_open_mcwf_R1-2-5_Dmax128_te1.0_descending.csv \
  --outdir .worktree/descending_schedule_dmax128_R1_R2_R5_20260620 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The HDF5 output records the same root seed `20260617` and seed rule as the
`R = 10` run.  The new trajectory seeds are `[84270618, 84280618, 84310618]`
for `R = 1, 2, 5`; the previously recorded `R = 10` seed is `[84360618]`.
For `R = 1`, the descending grid has a single point, which the driver stores
as `delta_min`; this row is therefore a fixed-minimum-detuning reference rather
than a multi-detuning descending cycle.

Combining the new file with the `R = 10` file gives

| R | Dcap | completed/requested cycles | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | 128 | 6/12 | 1.38967820 | 1.37238762 | 126 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 6 | 6 | 1741.8 s |
| 2 | 128 | 5/12 | 0.88774576 | 0.88774576 | 116 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 5 | 5 | 933.0 s |
| 5 | 128 | 5/12 | 1.04315832 | 1.04315832 | 115 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 5 | 5 | 843.8 s |
| 10 | 128 | 5/12 | 0.84528025 | 0.84528025 | 116 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 5 | 5 | 803.4 s |

The completed-cycle prefixes for the new runs are

| R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.50511675 | 1.47291896 | 4 | 6 | 33.5 s |
| 1 | 2 | 0.50511675 | 1.42012553 | 11 | 15 | 51.5 s |
| 1 | 3 | 0.50511675 | 1.41672905 | 23 | 31 | 88.5 s |
| 1 | 4 | 0.50511675 | 1.37238762 | 48 | 62 | 203.7 s |
| 1 | 5 | 0.50511675 | 1.39314083 | 92 | 123 | 611.6 s |
| 1 | 6 | 0.50511675 | 1.38967820 | 126 | 128 | 1741.8 s |
| 2 | 1 | 3.03070050 | 1.05928265 | 4 | 6 | 11.4 s |
| 2 | 2 | 0.50511675 | 1.01793111 | 11 | 15 | 34.0 s |
| 2 | 3 | 3.03070050 | 1.00906275 | 25 | 33 | 92.4 s |
| 2 | 4 | 0.50511675 | 1.03055230 | 57 | 72 | 258.6 s |
| 2 | 5 | 3.03070050 | 0.88774576 | 116 | 128 | 933.0 s |
| 5 | 1 | 3.03070050 | 1.38240655 | 4 | 6 | 9.1 s |
| 5 | 2 | 2.39930456 | 1.27834942 | 11 | 15 | 25.4 s |
| 5 | 3 | 1.76790862 | 1.21579249 | 24 | 33 | 63.1 s |
| 5 | 4 | 1.13651269 | 1.07479517 | 55 | 70 | 193.2 s |
| 5 | 5 | 0.50511675 | 1.04315832 | 115 | 128 | 843.8 s |

At `Dcap = 128`, descending `R = 10` remains the best capped prefix among the
tested frequency counts.  The ordering of final energies is
`R = 10 < R = 2 < R = 5 < R = 1`, while all four runs reach the evolved
system-bath and TDVP-sweep caps before approaching the ground-state energy
density.  The low-frequency `R = 1` trajectory is especially unfavorable: it
reaches its best energy density at cycle 4 and then heats while the bond
dimension continues to grow.  Thus the larger cap strengthens the bond-growth
conclusion; it does not
turn the fixed descending schedule into a scalable ground-state cooling
protocol.

### Dmax=192 Descending R=10 Follow-up

The current best product-state schedule, `R = 10`, `g = 0.3`, `te = 1.0`, was
then repeated at the intermediate cap `Dmax = 192` to see whether the
`Dmax = 128` cycle-5 stop was the immediate bottleneck:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 192 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R10_g0.3_D192_te1_steps12_20260629/progress.csv \
  --outdir .worktree/gscan_N64_R10_g0.3_D192_te1_steps12_20260629 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R10_g0.3_D192_te1_steps12_20260629/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps12_Dmax192_g0.3_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R10_g0.3_D192_te1_steps12_20260629/progress.csv
```

The HDF5 file records the root seed `20260617`, the stored seed rule, and the
trajectory seed `[84360618]`, matching the `Dmax = 128`, `R = 10` trajectory
through the common prefix.

The compact HDF5 summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 10 | 0.3 | 192 | 6/12 | 0.60/1.20 | 6/10 | stopped_partial_grid | 0.74866556 | 0.74866556 | 189 | >=192 | >=192 | not_converged_evolved_and_tdvp_sweep_cap | 3111.3 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.30592953 | 4 | 6 | 30.3 s |
| 2 | 2.75008008 | 1.13874047 | 11 | 15 | 46.1 s |
| 3 | 2.46945966 | 0.95514768 | 24 | 33 | 81.6 s |
| 4 | 2.18883925 | 0.87319255 | 57 | 73 | 205.0 s |
| 5 | 1.90821883 | 0.84528144 | 118 | 155 | 767.4 s |
| 6 | 1.62759842 | 0.74866556 | 189 | 192 | 3111.3 s |

Raising the cap from `128` to `192` therefore buys one additional
descending-detuning step for this seed and schedule.  The cycle-5 energy is
essentially unchanged from the `Dmax = 128` prefix, but the transient
system-bath state would already have needed bond dimension `155` at that point.
The next completed step lowers the stopped prefix to `E/N = 0.74866556`, while
the retained system state reaches `Dsys_eff = 189` and the evolved system-bath
and TDVP-sweep diagnostics reach the `192` cap.  This is useful evidence that
the next effective transient dimension is at least `192`; it is still not a
controlled cooling result.  The run observes only `6/10` detunings and remains
far above the DMRG reference `E0/N = -1.3246328892`.

### Dmax=192 Descending Remaining Frequencies

The remaining frequency counts were then run at the same `Dmax = 192`,
`g = 0.3`, `te = 1.0`, fixed descending detuning interval, product initial
state, stop-on-cap rule, and one-thread execution convention.  The previously
recorded `R = 10` file above is reused for the matched comparison.  The new
files are

```text
.worktree/descending_te1_dmax192_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R1_mcwf_continuous_stopcap_scheddesc_steps12_Dmax192_g0.3_te1_tau0.2_seed20260617.h5
.worktree/descending_te1_dmax192_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R2_mcwf_continuous_stopcap_scheddesc_steps12_Dmax192_g0.3_te1_tau0.2_seed20260617.h5
.worktree/descending_te1_dmax192_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R5_mcwf_continuous_stopcap_scheddesc_steps12_Dmax192_g0.3_te1_tau0.2_seed20260617.h5
```

The `R = 1,2,5,10` trajectory seeds are
`[84270618, 84280618, 84310618, 84360618]`, respectively, following the stored
root seed rule from `20260617`.  As in the `Dmax = 128` comparison, the
`R = 1` row has only the single grid point `delta_min` and is therefore a
fixed-minimum-detuning reference.

The matched HDF5 summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 0.3 | 192 | 6/12 | 6.00/12.00 | 1/1 | single_detuning | 1.38968899 | 1.37238762 | 165 | >=192 | >=192 | not_converged_evolved_and_tdvp_sweep_cap | 1937.0 s | bond_cap |
| 2 | 0.3 | 192 | 6/12 | 3.00/6.00 | 2/2 | full_grid_observed | 0.85017412 | 0.85017412 | 189 | >=192 | >=192 | not_converged_evolved_and_tdvp_sweep_cap | 3657.7 s | bond_cap |
| 5 | 0.3 | 192 | 6/12 | 1.20/2.40 | 5/5 | full_grid_observed | 0.90017691 | 0.90017691 | 189 | >=192 | >=192 | not_converged_evolved_and_tdvp_sweep_cap | 3132.9 s | bond_cap |
| 10 | 0.3 | 192 | 6/12 | 0.60/1.20 | 6/10 | stopped_partial_grid | 0.74866556 | 0.74866556 | 189 | >=192 | >=192 | not_converged_evolved_and_tdvp_sweep_cap | 3111.3 s | bond_cap |

The new completed-cycle prefixes are

| R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.50511675 | 1.47291896 | 4 | 6 | 33.8 s |
| 1 | 2 | 0.50511675 | 1.42012553 | 11 | 15 | 51.6 s |
| 1 | 3 | 0.50511675 | 1.41672905 | 23 | 31 | 88.2 s |
| 1 | 4 | 0.50511675 | 1.37238762 | 48 | 62 | 201.3 s |
| 1 | 5 | 0.50511675 | 1.39314083 | 92 | 123 | 604.5 s |
| 1 | 6 | 0.50511675 | 1.38968899 | 165 | 192 | 1937.0 s |
| 2 | 1 | 3.03070050 | 1.05928265 | 4 | 6 | 31.7 s |
| 2 | 2 | 0.50511675 | 1.01793111 | 11 | 15 | 48.8 s |
| 2 | 3 | 3.03070050 | 1.00906275 | 25 | 33 | 86.1 s |
| 2 | 4 | 0.50511675 | 1.03055230 | 57 | 72 | 233.4 s |
| 2 | 5 | 3.03070050 | 0.88774113 | 118 | 156 | 922.4 s |
| 2 | 6 | 0.50511675 | 0.85017412 | 189 | 192 | 3657.7 s |
| 5 | 1 | 3.03070050 | 1.38240655 | 4 | 6 | 31.1 s |
| 5 | 2 | 2.39930456 | 1.27834942 | 11 | 15 | 47.4 s |
| 5 | 3 | 1.76790862 | 1.21579249 | 24 | 33 | 84.2 s |
| 5 | 4 | 1.13651269 | 1.07479517 | 55 | 70 | 213.2 s |
| 5 | 5 | 0.50511675 | 1.04315874 | 117 | 152 | 792.9 s |
| 5 | 6 | 3.03070050 | 0.90017691 | 189 | 192 | 3132.9 s |

The progress CSV saturation summary for the new rows is

| R | first transient cap | TDVP-sweep cap | max sweep at | max sweep elapsed |
|---:|---|---|---|---:|
| 1 | 6:5 | 6:5 | 6:5 | 430.2 s |
| 2 | 6:2 | 6:2 | 6:5 | 737.6 s |
| 5 | 6:2 | 6:2 | 6:5 | 708.5 s |

At this fixed cap and seed, the stopped-prefix ordering is
`R = 10 < R = 2 < R = 5 < R = 1`.  This is not a converged frequency
optimization: every row stops by the evolved system-bath and TDVP-sweep bond
caps, and the best stopped prefix, `E/N = 0.74866556`, is still far above
`E0/N = -1.3246328892`.  The useful conclusion is instead an effective-bond
one.  For `R = 2` and `R = 5`, the progress CSV first observes the transient
cap during cycle `6:2`, while the most expensive sweep occurs at cycle `6:5`;
finishing a saturated cycle therefore dominates the elapsed time.  The
transient dimension needed to continue these `te = 1.0` product-state
trajectories is already at least `192` after six completed cycles.

### Dmax=256 Descending R=10 Follow-up

The same `R = 10`, `g = 0.3`, `te = 1.0` trajectory was then repeated at
`Dmax = 256`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 256 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R10_g0.3_D256_te1_steps12_20260629/progress.csv \
  --outdir .worktree/gscan_N64_R10_g0.3_D256_te1_steps12_20260629 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R10_g0.3_D256_te1_steps12_20260629/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps12_Dmax256_g0.3_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R10_g0.3_D256_te1_steps12_20260629/progress.csv
```

It uses the same stored trajectory seed `[84360618]`.  The compact HDF5 summary
is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 10 | 0.3 | 256 | 6/12 | 0.60/1.20 | 6/10 | stopped_partial_grid | 0.74860471 | 0.74860471 | 232 | >=256 | >=256 | not_converged_evolved_and_tdvp_sweep_cap | 3760.2 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.30592953 | 4 | 6 | 30.1 s |
| 2 | 2.75008008 | 1.13874047 | 11 | 15 | 46.7 s |
| 3 | 2.46945966 | 0.95514768 | 24 | 33 | 81.3 s |
| 4 | 2.18883925 | 0.87319255 | 57 | 73 | 205.4 s |
| 5 | 1.90821883 | 0.84528144 | 118 | 155 | 783.0 s |
| 6 | 1.62759842 | 0.74860471 | 232 | 256 | 3760.2 s |

The `Dmax = 256` run still stops at cycle 6.  It changes the stopped-prefix
energy only from `0.74866556` to `0.74860471` relative to the `Dmax = 192`
run, while the retained system bond grows from `189` to `232` and the transient
system-bath and TDVP-sweep diagnostics again reach the cap.  Thus the sixth
cycle is becoming only weakly sensitive to this cap increase for this seed, but
continuing the trajectory would require effective transient bond dimension
above `256`.  This remains a bond-growth diagnostic, not scalable ground-state
cooling.

### Dmax=256 Descending Remaining Frequencies

The remaining frequency counts were then repeated at the same `Dmax = 256`,
`g = 0.3`, `te = 1.0`, fixed descending detuning interval, product initial
state, stop-on-cap rule, and one-thread execution convention.  The `R = 10`
file above is reused for the matched comparison.  The new files are

```text
.worktree/descending_te1_dmax256_R1_rerun_20260630/largeN_multifrequency_tn_N64_R1_mcwf_continuous_stopcap_scheddesc_steps12_Dmax256_g0.3_te1_tau0.2_seed20260617.h5
.worktree/descending_te1_dmax256_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R2_mcwf_continuous_stopcap_scheddesc_steps12_Dmax256_g0.3_te1_tau0.2_seed20260617.h5
.worktree/descending_te1_dmax256_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R5_mcwf_continuous_stopcap_scheddesc_steps12_Dmax256_g0.3_te1_tau0.2_seed20260617.h5
```

The `R = 1` artifact is a fresh rerun in its own output directory; an older
interrupted file from the matched `R = 1,2,5` directory is not used as
completed evidence.  The `R = 1,2,5,10` trajectory seeds are
`[84270618, 84280618, 84310618, 84360618]`, respectively.  As before, the
`R = 1` row is a fixed-minimum-detuning reference because its grid contains
only `delta_min`.

The matched HDF5 summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 0.3 | 256 | 7/12 | 7.00/12.00 | 1/1 | single_detuning | 1.35823962 | 1.35823962 | 242 | >=256 | >=256 | not_converged_evolved_and_tdvp_sweep_cap | 5170.7 s | bond_cap |
| 2 | 0.3 | 256 | 6/12 | 3.00/6.00 | 2/2 | full_grid_observed | 0.85012354 | 0.85012354 | 233 | >=256 | >=256 | not_converged_evolved_and_tdvp_sweep_cap | 4336.7 s | bond_cap |
| 5 | 0.3 | 256 | 6/12 | 1.20/2.40 | 5/5 | full_grid_observed | 0.90017025 | 0.90017025 | 231 | >=256 | >=256 | not_converged_evolved_and_tdvp_sweep_cap | 3452.5 s | bond_cap |
| 10 | 0.3 | 256 | 6/12 | 0.60/1.20 | 6/10 | stopped_partial_grid | 0.74860471 | 0.74860471 | 232 | >=256 | >=256 | not_converged_evolved_and_tdvp_sweep_cap | 3760.2 s | bond_cap |

The new completed-cycle prefixes are

| R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.50511675 | 1.47291896 | 4 | 6 | 30.6 s |
| 1 | 2 | 0.50511675 | 1.42012553 | 11 | 15 | 46.0 s |
| 1 | 3 | 0.50511675 | 1.41672905 | 23 | 31 | 78.3 s |
| 1 | 4 | 0.50511675 | 1.37238762 | 48 | 62 | 188.4 s |
| 1 | 5 | 0.50511675 | 1.39314083 | 92 | 123 | 577.2 s |
| 1 | 6 | 0.50511675 | 1.38968903 | 167 | 219 | 1851.5 s |
| 1 | 7 | 0.50511675 | 1.35823962 | 242 | 256 | 5170.8 s |
| 2 | 1 | 3.03070050 | 1.05928265 | 4 | 6 | 31.9 s |
| 2 | 2 | 0.50511675 | 1.01793111 | 11 | 15 | 49.8 s |
| 2 | 3 | 3.03070050 | 1.00906275 | 25 | 33 | 94.4 s |
| 2 | 4 | 0.50511675 | 1.03055230 | 57 | 72 | 230.0 s |
| 2 | 5 | 3.03070050 | 0.88774113 | 118 | 156 | 885.1 s |
| 2 | 6 | 0.50511675 | 0.85012354 | 233 | 256 | 4336.8 s |
| 5 | 1 | 3.03070050 | 1.38240655 | 4 | 6 | 30.5 s |
| 5 | 2 | 2.39930456 | 1.27834942 | 11 | 15 | 46.6 s |
| 5 | 3 | 1.76790862 | 1.21579249 | 24 | 33 | 82.2 s |
| 5 | 4 | 1.13651269 | 1.07479517 | 55 | 70 | 210.1 s |
| 5 | 5 | 0.50511675 | 1.04315874 | 117 | 152 | 810.1 s |
| 5 | 6 | 3.03070050 | 0.90017025 | 231 | 256 | 3452.4 s |

The progress CSV saturation summary for the new rows is

| R | first transient cap | TDVP-sweep cap | max sweep at | max sweep elapsed |
|---:|---|---|---|---:|
| 1 | 7:3 | 7:3 | 7:5 | 961.5 s |
| 2 | 6:4 | 6:4 | 6:5 | 1185.4 s |
| 5 | 6:4 | 6:4 | 6:5 | 901.4 s |

At this fixed cap and seed, the stopped-prefix ordering remains
`R = 10 < R = 2 < R = 5 < R = 1`.  This repeats the qualitative `Dmax = 192`
ordering, but it is still not a converged frequency optimization: every row
is stopped by the evolved system-bath and TDVP-sweep bond caps, and the best
stopped prefix, `E/N = 0.74860471`, remains far above
`E0/N = -1.3246328892`.  Increasing the cap from `192` to `256` changes the
`R = 2`, `R = 5`, and `R = 10` stopped-prefix energies by less than `10^{-4}`;
the `R = 2` and `R = 10` shifts are at the `10^{-5}` scale, and the `R = 5`
shift is at the `10^{-6}` scale.  The `R = 1` trajectory remains a poor
fixed-detuning reference even though it reaches a seventh completed cycle.
Continuing these `te = 1.0` product-state trajectories therefore requires
effective transient bond dimension above `256`; the present data are evidence
for bond growth, not scalable ground-state cooling.

### Dmax=320 Descending R=10 Follow-up

The same trajectory was next repeated at `Dmax = 320`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 320 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R10_g0.3_D320_te1_steps12_20260629/progress.csv \
  --outdir .worktree/gscan_N64_R10_g0.3_D320_te1_steps12_20260629 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R10_g0.3_D320_te1_steps12_20260629/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps12_Dmax320_g0.3_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R10_g0.3_D320_te1_steps12_20260629/progress.csv
```

It again uses the stored trajectory seed `[84360618]`.  The compact HDF5
summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 10 | 0.3 | 320 | 7/12 | 0.70/1.20 | 7/10 | stopped_partial_grid | 0.73765227 | 0.73765227 | 318 | >=320 | >=320 | not_converged_evolved_and_tdvp_sweep_cap | 14082.0 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.30592953 | 4 | 6 | 31.2 s |
| 2 | 2.75008008 | 1.13874047 | 11 | 15 | 47.0 s |
| 3 | 2.46945966 | 0.95514768 | 24 | 33 | 83.9 s |
| 4 | 2.18883925 | 0.87319255 | 57 | 73 | 210.1 s |
| 5 | 1.90821883 | 0.84528144 | 118 | 155 | 793.6 s |
| 6 | 1.62759842 | 0.74860558 | 238 | 318 | 3928.1 s |
| 7 | 1.34697800 | 0.73765227 | 318 | 320 | 14082.0 s |

Raising the cap from `256` to `320` allows one more descending-detuning step,
but the gain remains modest: the stopped-prefix energy changes from
`0.74860471` to `0.73765227`, while the elapsed time grows from `3760.2 s` to
`14082.0 s`.  At the common sixth cycle, the `Dmax = 320` trajectory gives
`E/N = 0.74860558`, so this comparison is between the two stopped prefixes
rather than between same-cycle values.  The progress CSV shows that the cycle-7
transient state already reaches the `320` cap on the first TDVP sweep, and the
final retained system state has `Dsys_eff = 318`.  Thus this run records an
effective transient bond dimension of at least `320` for the next detuning.  It
still observes only `7/10` detunings and remains far above the DMRG reference
`E0/N = -1.3246328892`, so it is evidence for bond-dimension growth rather
than scalable ground-state cooling.

### Dmax=128 Descending R=10 Probe at te=0.5

The best `Dmax = 128` descending prefix above used `R = 10` and `te = 1.0`.
To test whether a shorter per-cycle evolution time can delay the cap without
losing the schedule advantage, the same trajectory seed and fixed detuning
interval were rechecked with `te = 0.5` and a longer requested prefix:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 0.5 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_te05_dmax128_R10_recheck_20260629/progress.csv \
  --outdir .worktree/descending_te05_dmax128_R10_recheck_20260629 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The recheck wrote

```text
.worktree/descending_te05_dmax128_R10_recheck_20260629/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps20_Dmax128_g0.3_te0.5_tau0.2_seed20260617.h5
```

The HDF5 output again records trajectory seed `[84360618]`.  The HDF5 summary,
with the fixed `te` shown for comparison, is

| R | te | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop_reason |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 10 | 0.5 | 128 | 10/20 | 1.00/2.00 | 10/10 | full_grid_observed | 1.00873665 | 1.00873665 | 113 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | 807.8 s | bond_cap |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.36556055 | 3 | 4 | 25.1 s |
| 2 | 2.75008008 | 1.27175621 | 4 | 6 | 32.5 s |
| 3 | 2.46945966 | 1.18301786 | 8 | 9 | 40.9 s |
| 4 | 2.18883925 | 1.18241961 | 10 | 14 | 51.4 s |
| 5 | 1.90821883 | 1.18369439 | 16 | 21 | 65.8 s |
| 6 | 1.62759842 | 1.13684288 | 25 | 32 | 90.6 s |
| 7 | 1.34697800 | 1.11629039 | 34 | 48 | 140.6 s |
| 8 | 1.06635758 | 1.05446449 | 55 | 68 | 232.4 s |
| 9 | 0.78573717 | 1.04273197 | 73 | 101 | 423.5 s |
| 10 | 0.50511675 | 1.00873665 | 113 | 128 | 807.8 s |

Thus reducing `te` from `1.0` to `0.5` delays the evolved and TDVP-sweep cap
from cycle 5 to cycle 10 for this `R = 10`, `Dmax = 128` descending trajectory.
It does not improve the cooling prefix: the stopped energy density is
`1.00873665`, worse than the corresponding `te = 1.0` stopped value
`0.84528025`.  At this fixed descending schedule, smaller `te` is therefore a
cap-delay mechanism rather than a route to the ground state.

### Dmax=128 Descending Remaining Frequencies at te=0.5

The remaining frequency counts were then run at the same `Dmax = 128`,
`g = 0.3`, `te = 0.5`, fixed descending detuning interval, stopping rule, and
thread pinning.  The child commands were generated with planning mode,

```bash
julia --project=. --startup-file=no \
  scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 0.5 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_te05_dmax128_R1_R2_R5_20260630/progress.csv \
  --outdir .worktree/descending_te05_dmax128_R1_R2_R5_20260630 \
  --tdvp-sweep-progress --stop-on-bond-cap \
  --print-parallel-plan --plan-julia-threads 1 --plan-blas-threads 1
```

and the three one-thread child jobs wrote

```text
.worktree/descending_te05_dmax128_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R1_mcwf_continuous_stopcap_scheddesc_steps20_Dmax128_g0.3_te0.5_tau0.2_seed20260617.h5
.worktree/descending_te05_dmax128_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R2_mcwf_continuous_stopcap_scheddesc_steps20_Dmax128_g0.3_te0.5_tau0.2_seed20260617.h5
.worktree/descending_te05_dmax128_R1_R2_R5_20260630/largeN_multifrequency_tn_N64_R5_mcwf_continuous_stopcap_scheddesc_steps20_Dmax128_g0.3_te0.5_tau0.2_seed20260617.h5
```

Combining the new files with the already-recorded `R = 10` file gives the
matched `Dmax = 128`, `te = 0.5` comparison

| R | te | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop_reason |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 0.5 | 128 | 10/20 | 10.00/20.00 | 1/1 | single_detuning | 1.12013548 | 1.12013548 | 102 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | 757.7 s | bond_cap |
| 2 | 0.5 | 128 | 10/20 | 5.00/10.00 | 2/2 | full_grid_observed | 1.07026583 | 1.07026583 | 100 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | 834.5 s | bond_cap |
| 5 | 0.5 | 128 | 10/20 | 2.00/4.00 | 5/5 | full_grid_observed | 0.98089919 | 0.98089919 | 100 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | 761.4 s | bond_cap |
| 10 | 0.5 | 128 | 10/20 | 1.00/2.00 | 10/10 | full_grid_observed | 1.00873665 | 1.00873665 | 113 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | 807.8 s | bond_cap |

The completed-cycle energy-density prefixes for the new jobs are

| R | completed-cycle E/N prefix |
|---:|---|
| 1 | `1.40165476, 1.36503509, 1.32226325, 1.27186394, 1.27656917, 1.27846189, 1.23993121, 1.20065576, 1.15990439, 1.12013548` |
| 2 | `1.30289284, 1.30517300, 1.30289340, 1.30660230, 1.22081148, 1.22280886, 1.16004069, 1.11717079, 1.11503576, 1.07026583` |
| 5 | `1.42822826, 1.42714357, 1.38513415, 1.33261959, 1.26978109, 1.21606719, 1.12092091, 1.07529489, 0.99708597, 0.98089919` |

At this fixed `Dmax = 128` and `te = 0.5`, the best stopped prefix is now
`R = 5`, followed by `R = 10`, `R = 2`, and `R = 1`.  This does not change the
physical conclusion.  Every row reaches the evolved system-bath and TDVP-sweep
cap at cycle 10, and even the best stopped prefix,
`E/N = 0.98089919`, remains far above the DMRG reference
`E0/N = -1.3246328892`.  The shorter-time descending scan is therefore still a
bond-growth diagnostic, not evidence of scalable cooling to the ground state.

### Dmax=192 Descending R=10 Probe at te=0.5

The same `te = 0.5` trajectory was then repeated at `Dmax = 192`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 192 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 0.5 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_te05_dmax192_R10_20260629/progress.csv \
  --outdir .worktree/descending_te05_dmax192_R10_20260629 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/descending_te05_dmax192_R10_20260629/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps20_Dmax192_g0.3_te0.5_tau0.2_seed20260617.h5
.worktree/descending_te05_dmax192_R10_20260629/progress.csv
```

It again uses the stored trajectory seed `[84360618]`.  The compact HDF5
summary is

| R | te | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop_reason |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 10 | 0.5 | 192 | 11/20 | 1.10/2.00 | 10/10 | full_grid_observed | 1.00650854 | 1.00650854 | 154 | >=192 | >=192 | not_converged_evolved_and_tdvp_sweep_cap | 1712.8 s | bond_cap |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.36556055 | 3 | 4 | 24.9 s |
| 2 | 2.75008008 | 1.27175621 | 4 | 6 | 32.2 s |
| 3 | 2.46945966 | 1.18301786 | 8 | 9 | 40.5 s |
| 4 | 2.18883925 | 1.18241961 | 10 | 14 | 51.3 s |
| 5 | 1.90821883 | 1.18369439 | 16 | 21 | 65.8 s |
| 6 | 1.62759842 | 1.13684288 | 25 | 32 | 91.1 s |
| 7 | 1.34697800 | 1.11629039 | 34 | 48 | 139.7 s |
| 8 | 1.06635758 | 1.05446449 | 55 | 68 | 236.4 s |
| 9 | 0.78573717 | 1.04273212 | 73 | 101 | 433.6 s |
| 10 | 0.50511675 | 1.00873678 | 113 | 143 | 850.2 s |
| 11 | 3.03070050 | 1.00650854 | 154 | 192 | 1712.8 s |

Raising the cap from `128` to `192` at `te = 0.5` allows the trajectory to
complete the first full ten-detuning pass without saturation and to enter the
second pass.  The first-pass endpoint, `E/N = 1.00873678`, is essentially
unchanged from the `Dmax = 128` stopped value `1.00873665`.  The extra step at
the high detuning lowers the stopped prefix only to `E/N = 1.00650854` while
the evolved system-bath and TDVP-sweep diagnostics reach the `192` cap.  Thus
`te = 0.5` remains a cap-delay diagnostic for this schedule; it is still much
worse than the `te = 1.0` prefixes and remains far above
`E0/N = -1.3246328892`.

### Dmax=256 Descending R=10 Probe at te=0.5

The same `te = 0.5` trajectory was then repeated at `Dmax = 256`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 256 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 0.5 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_te05_dmax256_R10_20260629/progress.csv \
  --outdir .worktree/descending_te05_dmax256_R10_20260629 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/descending_te05_dmax256_R10_20260629/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps20_Dmax256_g0.3_te0.5_tau0.2_seed20260617.h5
.worktree/descending_te05_dmax256_R10_20260629/progress.csv
```

It again uses the stored trajectory seed `[84360618]`.  The compact HDF5
summary is

| R | te | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop_reason |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 10 | 0.5 | 256 | 12/20 | 1.20/2.00 | 10/10 | full_grid_observed | 1.00494536 | 1.00494536 | 221 | >=256 | >=256 | not_converged_evolved_and_tdvp_sweep_cap | 3703.6 s | bond_cap |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.36556055 | 3 | 4 | 24.8 s |
| 2 | 2.75008008 | 1.27175621 | 4 | 6 | 32.0 s |
| 3 | 2.46945966 | 1.18301786 | 8 | 9 | 40.1 s |
| 4 | 2.18883925 | 1.18241961 | 10 | 14 | 50.7 s |
| 5 | 1.90821883 | 1.18369439 | 16 | 21 | 65.6 s |
| 6 | 1.62759842 | 1.13684288 | 25 | 32 | 91.4 s |
| 7 | 1.34697800 | 1.11629039 | 34 | 48 | 142.1 s |
| 8 | 1.06635758 | 1.05446449 | 55 | 68 | 239.7 s |
| 9 | 0.78573717 | 1.04273212 | 73 | 101 | 439.0 s |
| 10 | 0.50511675 | 1.00873678 | 113 | 143 | 853.7 s |
| 11 | 3.03070050 | 1.00650866 | 155 | 204 | 1787.7 s |
| 12 | 2.75008008 | 1.00494536 | 221 | 256 | 3703.6 s |

Raising the cap from `192` to `256` at `te = 0.5` buys one additional
second-pass detuning.  Relative to the `Dmax = 192` stopped point, the
stopped-prefix energy changes only from `1.00650854` to `1.00494536`, while the
elapsed time grows from `1712.8 s` to `3703.6 s` and the retained system bond
grows to `Dsys_eff = 221`.  At the common cycle 11, the `Dmax = 256` run gives
`E/N = 1.00650866`, agreeing with the `Dmax = 192` stopped value to about
`1e-7`; the quoted improvement comes from the additional second-pass detuning.
The evolved system-bath and TDVP-sweep diagnostics again reach the cap, now at
cycle 12.  Thus the `te = 0.5` ladder continues to show delayed saturation with
weak energy improvement rather than scalable cooling toward the ground-state
reference.

### Dmax=320 Descending R=10 Probe at te=0.5

The same `te = 0.5` trajectory was then repeated at `Dmax = 320`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 320 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.3 --te 0.5 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/descending_te05_dmax320_R10_20260630/progress.csv \
  --outdir .worktree/descending_te05_dmax320_R10_20260630 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/descending_te05_dmax320_R10_20260630/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps20_Dmax320_g0.3_te0.5_tau0.2_seed20260617.h5
.worktree/descending_te05_dmax320_R10_20260630/progress.csv
```

It again uses the stored trajectory seed `[84360618]`.  The compact HDF5
summary is

| R | te | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop_reason |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 10 | 0.5 | 320 | 13/20 | 1.30/2.00 | 10/10 | full_grid_observed | 1.00365324 | 1.00365324 | 299 | >=320 | >=320 | not_converged_evolved_and_tdvp_sweep_cap | 8055.7 s | bond_cap |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.36556055 | 3 | 4 | 25.0 s |
| 2 | 2.75008008 | 1.27175621 | 4 | 6 | 32.4 s |
| 3 | 2.46945966 | 1.18301786 | 8 | 9 | 41.0 s |
| 4 | 2.18883925 | 1.18241961 | 10 | 14 | 51.4 s |
| 5 | 1.90821883 | 1.18369439 | 16 | 21 | 66.6 s |
| 6 | 1.62759842 | 1.13684288 | 25 | 32 | 92.9 s |
| 7 | 1.34697800 | 1.11629039 | 34 | 48 | 141.6 s |
| 8 | 1.06635758 | 1.05446449 | 55 | 68 | 234.8 s |
| 9 | 0.78573717 | 1.04273212 | 73 | 101 | 447.0 s |
| 10 | 0.50511675 | 1.00873678 | 113 | 143 | 863.0 s |
| 11 | 3.03070050 | 1.00650865 | 155 | 204 | 1814.0 s |
| 12 | 2.75008008 | 1.00494543 | 223 | 287 | 3889.6 s |
| 13 | 2.46945966 | 1.00365324 | 299 | 320 | 8055.6 s |

Raising the cap from `256` to `320` at `te = 0.5` buys one additional
second-pass detuning.  At the common cycle 12, the `Dmax = 320` run gives
`E/N = 1.00494543`, agreeing with the `Dmax = 256` stopped value
`1.00494536` at the scale of the cap-dependent numerical path difference.  The
additional step lowers the stopped prefix only to `E/N = 1.00365324`, while the
retained system bond grows to `Dsys_eff = 299` and the evolved system-bath and
TDVP-sweep diagnostics reach the `320` cap at cycle 13.  Thus the `te = 0.5`
cap ladder has the same qualitative interpretation through `Dmax = 320`:
larger caps postpone saturation and give only weak incremental energy
improvement, not scalable cooling toward the ground-state reference
`E0/N = -1.3246328892`.

### Weak-Coupling ED and N=64 Probe

The fixed descending channel also depends strongly on the coupling strength
`g`.  Before launching another large-`N` trajectory, an exact `N = 4`
density-matrix ED scan was run as a channel diagnostic.  The scan used the same
nonintegrable open-chain Hamiltonian, `R = 10`, ten descending cycles, and the
ED finite-size interval `[Delta, 6 Delta]` with
`Delta = 0.8484306542`.  It compared two initial states: the exact ground-state
density matrix and the product state.  The table reports the final ground-state
energy drift per site and the product-state energy reduction over the same
ten-cycle prefix:

| g | te | ground drift/N | ground overlap | product initial E/N | product final E/N | product best E/N | product cooling/N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.2 | 1.130e-03 | 0.99899815 | 1.25000000 | 1.24768857 | 1.24768857 | 2.311e-03 |
| 0.05 | 0.5 | 2.064e-03 | 0.99780366 | 1.25000000 | 1.23669081 | 1.23669081 | 1.331e-02 |
| 0.05 | 1.0 | 1.257e-03 | 0.99847658 | 1.25000000 | 1.20974816 | 1.20974816 | 4.025e-02 |
| 0.10 | 0.2 | 4.515e-03 | 0.99599826 | 1.25000000 | 1.24078037 | 1.24078037 | 9.220e-03 |
| 0.10 | 0.5 | 8.259e-03 | 0.99123828 | 1.25000000 | 1.19747893 | 1.19747893 | 5.252e-02 |
| 0.10 | 1.0 | 4.992e-03 | 0.99393667 | 1.25000000 | 1.09428205 | 1.09428205 | 1.557e-01 |
| 0.20 | 0.2 | 1.800e-02 | 0.98408397 | 1.25000000 | 1.21353489 | 1.21353489 | 3.647e-02 |
| 0.20 | 0.5 | 3.307e-02 | 0.96533187 | 1.25000000 | 1.05087550 | 1.05087550 | 1.991e-01 |
| 0.20 | 1.0 | 1.971e-02 | 0.97597631 | 1.25000000 | 0.70120498 | 0.70120498 | 5.488e-01 |
| 0.30 | 0.2 | 4.029e-02 | 0.96453115 | 1.25000000 | 1.16947300 | 1.16947300 | 8.053e-02 |
| 0.30 | 0.5 | 7.435e-02 | 0.92342255 | 1.25000000 | 0.83934986 | 0.83934986 | 4.107e-01 |
| 0.30 | 1.0 | 4.507e-02 | 0.94535349 | 1.25000000 | 0.22980770 | 0.22980770 | 1.020e+00 |

Thus the exact small-system channel already shows the expected tradeoff:
weaker coupling is closer to a ground-state fixed point, but it cools the
product state more slowly.  This is a channel-level diagnostic, not a
large-`N` convergence result.

The corresponding large-`N` diagnostic tested the more ground-preserving
choice `g = 0.1` at the same `N = 64`, `R = 10`, `Dmax = 64`, `te = 1.0`
descending setup used in the stop-on-cap scans:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g 0.1 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R10_g0.1_D64_te1_steps20_20260627/progress.csv \
  --outdir .worktree/gscan_N64_R10_g0.1_D64_te1_steps20_20260627 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R10_g0.1_D64_te1_steps20_20260627/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps20_Dmax64_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R10_g0.1_D64_te1_steps20_20260627/progress.csv
```

The HDF5 stem above is the historical output name from before the default
filename added the `_g<value>` token.  The file metadata stores `g = 0.1`, and
current reproductions of the same command write `_g0.1_` in the stem.

The compact HDF5 summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 10 | 0.1 | 64 | 5/20 | 0.50/2.00 | 5/10 | 1.33598052 | 1.33380878 | >=64 | >=64 | >=64 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 470.5 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.43159181 | 4 | 6 | 36.3 s |
| 2 | 2.75008008 | 1.43023698 | 10 | 13 | 54.6 s |
| 3 | 2.46945966 | 1.33380878 | 22 | 27 | 87.9 s |
| 4 | 2.18883925 | 1.33518504 | 46 | 55 | 190.8 s |
| 5 | 1.90821883 | 1.33598052 | 64 | 64 | 470.5 s |

The weaker coupling reduces early bond growth and gives a smoother first
three-cycle prefix, but it still reaches the retained-system, evolved
system-bath, and TDVP-sweep caps by the fifth completed cycle.  It also remains
far above the DMRG ground-state energy density `E0/N = -1.3246328892` and does
not complete even one full ten-frequency period.  Thus `g = 0.1` is useful as a
coupling-strength diagnostic, not as evidence for scalable large-`N`
ground-state cooling at `Dmax = 64`.

The intermediate value `g = 0.2` was then run at the same `N = 64`, `R = 10`,
`Dmax = 64`, `te = 1.0` descending setup.  The run used the current coupling
axis, so the HDF5 stem and progress CSV both record the coupling explicitly:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.2 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R10_g0.2_D64_te1_steps20_20260628/progress.csv \
  --outdir .worktree/gscan_N64_R10_g0.2_D64_te1_steps20_20260628 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R10_g0.2_D64_te1_steps20_20260628/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps20_Dmax64_g0.2_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R10_g0.2_D64_te1_steps20_20260628/progress.csv
```

The compact HDF5 summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 10 | 0.2 | 64 | 4/20 | 0.40/2.00 | 4/10 | stopped_partial_grid | 1.12969883 | 1.12680143 | 52 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 208.7 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.37983852 | 4 | 6 | 34.6 s |
| 2 | 2.75008008 | 1.27270149 | 10 | 14 | 51.7 s |
| 3 | 2.46945966 | 1.12680143 | 24 | 31 | 87.1 s |
| 4 | 2.18883925 | 1.12969883 | 52 | 64 | 208.7 s |

The same `R = 10`, `g = 0.2` trajectory was then repeated at `Dmax = 128`
to determine whether the intermediate coupling was limited primarily by the
`Dmax = 64` cap:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.2 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R10_g0.2_D128_te1_steps12_20260629/progress.csv \
  --outdir .worktree/gscan_N64_R10_g0.2_D128_te1_steps12_20260629 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R10_g0.2_D128_te1_steps12_20260629/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_steps12_Dmax128_g0.2_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R10_g0.2_D128_te1_steps12_20260629/progress.csv
```

The compact HDF5 summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 10 | 0.2 | 128 | 5/12 | 0.50/1.20 | 5/10 | stopped_partial_grid | 1.10035588 | 1.10035588 | 108 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | 655.9 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | 1.37983852 | 4 | 6 | 30.2 s |
| 2 | 2.75008008 | 1.27270149 | 10 | 14 | 45.7 s |
| 3 | 2.46945966 | 1.12680143 | 24 | 31 | 77.8 s |
| 4 | 2.18883925 | 1.12969897 | 53 | 66 | 185.9 s |
| 5 | 1.90821883 | 1.10035588 | 108 | 128 | 655.9 s |

Thus increasing the cap from `64` to `128` allows the `g = 0.2`, `R = 10`
trajectory to complete one additional descending-detuning step and lowers the
stopped prefix from `E/N = 1.12969883` to `E/N = 1.10035588`.  This is a
modest truncation-pressure improvement, not a scalable cooling result.  The run
still observes only `5/10` detunings, reaches the evolved system-bath and
TDVP-sweep caps at cycle 5, and remains far above the same DMRG reference
`E0/N = -1.3246328892`.  It also remains above the `g = 0.3`, `Dmax = 128`,
`R = 10` stopped prefix `E/N = 0.84528025`, so this point does not displace the
stronger-coupling trajectory as the best capped prefix in the present
descending schedule.

For the original `Dmax = 64` campaign, the remaining frequency counts were then
run at the same `g = 0.2` coupling and fixed detuning interval.  This command is
therefore part of the `Dmax = 64` frequency-count comparison, not a continuation
of the `Dmax = 128` diagnostic above.  It also uses the coupling axis
`--g-values 0.2`, so the whole `g = 0.2` frequency-count scan is described with
one reproduction convention; with one entry this is equivalent to the scalar
`--g 0.2` option.

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5 --methods mcwf \
  --evolution-method continuous --steps 20 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.2 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R1-2-5_g0.2_D64_te1_steps20_20260628/progress.csv \
  --outdir .worktree/gscan_N64_R1-2-5_g0.2_D64_te1_steps20_20260628 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R1-2-5_g0.2_D64_te1_steps20_20260628/largeN_multifrequency_tn_N64_R1-2-5_mcwf_continuous_stopcap_scheddesc_steps20_Dmax64_g0.2_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R1-2-5_g0.2_D64_te1_steps20_20260628/progress.csv
```

Combining this file with the `R = 10` run gives the following table.  The
`R = 10` row is repeated from the standalone summary above, now with the
detuning-coverage label included, because the comparison mixes full-grid and
partial-grid capped prefixes.

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.2 | 64 | 5/20 | 5.00/20.00 | 1/1 | single_detuning | 1.39267361 | 1.38151169 | >=64 | >=64 | >=64 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 445.2 s |
| 2 | 0.2 | 64 | 4/20 | 2.00/10.00 | 2/2 | full_grid_observed | 1.35357219 | 1.32596718 | 52 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 174.5 s |
| 5 | 0.2 | 64 | 5/20 | 1.00/4.00 | 5/5 | full_grid_observed | 1.33834379 | 1.33834379 | >=64 | >=64 | >=64 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 435.0 s |
| 10 | 0.2 | 64 | 4/20 | 0.40/2.00 | 4/10 | stopped_partial_grid | 1.12969883 | 1.12680143 | 52 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 208.7 s |

The completed-cycle prefixes for the new `R = 1,2,5` file are

| R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.50511675 | 1.44881502 | 4 | 6 | 32.4 s |
| 1 | 2 | 0.50511675 | 1.42277978 | 10 | 14 | 48.9 s |
| 1 | 3 | 0.50511675 | 1.40602728 | 22 | 29 | 82.2 s |
| 1 | 4 | 0.50511675 | 1.38151169 | 47 | 59 | 186.1 s |
| 1 | 5 | 0.50511675 | 1.39267361 | 64 | 64 | 445.2 s |
| 2 | 1 | 3.03070050 | 1.32596718 | 4 | 6 | 8.6 s |
| 2 | 2 | 0.50511675 | 1.33913318 | 10 | 13 | 24.6 s |
| 2 | 3 | 3.03070050 | 1.33766153 | 24 | 31 | 59.7 s |
| 2 | 4 | 0.50511675 | 1.35357219 | 52 | 64 | 174.5 s |
| 5 | 1 | 3.03070050 | 1.43370984 | 4 | 6 | 8.7 s |
| 5 | 2 | 2.39930456 | 1.43203380 | 10 | 14 | 24.6 s |
| 5 | 3 | 1.76790862 | 1.40744449 | 23 | 30 | 57.8 s |
| 5 | 4 | 1.13651269 | 1.37652541 | 51 | 62 | 168.0 s |
| 5 | 5 | 0.50511675 | 1.33834379 | 64 | 64 | 435.0 s |

At this fixed root seed, `R = 10` remains the best intermediate-coupling
prefix: its best recorded energy density is `1.12680143`, whereas the best
prefixes for `R = 1,2,5` are `1.38151169`, `1.32596718`, and `1.33834379`,
respectively.  In this convention `R = 1` is the single-detuning channel at
`delta_min`, while `R >= 2` sweeps the displayed interval; this follows from
the validation helper `largeN_delta_values`, which returns `[protocol.delta_min]`
for `R = 1` before constructing the multi-detuning range.  The price is not
avoided by using fewer frequencies.  The ranking is a best-before-cap
comparison, not an equal-coverage comparison: `R = 10` stops after observing
only `4/10` detunings, while `R = 1,2,5` observe their displayed grids.  All
four frequency counts reach at least one effective bond cap by cycle 4 or 5,
and none is close to the ground-state reference.

Thus, for the `R = 10` trajectory, increasing the coupling from `g = 0.1` to
`g = 0.2` makes the early cooling substantially stronger: the best recorded
prefix falls from `1.33380878` to `1.12680143`.  This improvement is bought by
faster system-bath bond growth.  The evolved system-bath and TDVP-sweep caps
appear on the fourth completed cycle, one cycle earlier than in the historical
`g = 0.1` trajectory.  The `g = 0.2` frequency-count scan is therefore an
intermediate-coupling diagnostic: it confirms that stronger coupling can cool
faster at the start of the descending `R = 10` schedule, but all recorded
prefixes remain far above the same DMRG ground-state reference
`E0/N = -1.3246328892`.  It is still not evidence for scalable large-`N`
ground-state cooling at `Dmax = 64`.
Moreover, the comparison between `g = 0.1` and `g = 0.2` is a fixed-seed,
single-trajectory MCWF comparison; ensemble averages are needed before the
size of this early-cooling gain can be assigned a stable coupling dependence.

The same diagnostic was then pushed to the weaker value `g = 0.05` for all
four frequency counts `R = 1,2,5,10`, keeping `N = 64`, `Dmax = 64`,
`te = 1.0`, the fixed descending detuning interval, and the same stop-on-cap
rule.  The reproduction command again uses the coupling-axis spelling
`--g-values 0.05`, matching the `g = 0.2` scan above; the earlier `g = 0.1`
command at the top of this section is deliberately left in its historical
scalar `--g 0.1` form to document the pre-`_g<value>` output name.

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. --startup-file=no scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5,10 --methods mcwf \
  --evolution-method continuous --steps 10 --Dmax 64 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open \
  --g-values 0.05 --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending \
  --progress-csv .worktree/gscan_N64_R1-2-5-10_g0.05_D64_te1_steps10_20260627/progress.csv \
  --outdir .worktree/gscan_N64_R1-2-5-10_g0.05_D64_te1_steps10_20260627 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/gscan_N64_R1-2-5-10_g0.05_D64_te1_steps10_20260627/largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_continuous_stopcap_scheddesc_steps10_Dmax64_g0.05_te1_tau0.2_seed20260617.h5
.worktree/gscan_N64_R1-2-5-10_g0.05_D64_te1_steps10_20260627/progress.csv
```

The compact HDF5 summary is

| R | g | Dcap | completed/requested cycles | completed/requested periods | visited detunings | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.05 | 64 | 5/10 | 5.00/10.00 | 1/1 | 1.48957680 | 1.48437500 | 61 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 373.5 s |
| 2 | 0.05 | 64 | 5/10 | 2.50/5.00 | 2/2 | 1.48653745 | 1.48437500 | 63 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 362.7 s |
| 5 | 0.05 | 64 | 5/10 | 1.00/2.00 | 5/5 | 1.48689280 | 1.48437500 | 61 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 349.1 s |
| 10 | 0.05 | 64 | 5/10 | 0.50/1.00 | 5/10 | 1.48521913 | 1.48422035 | 63 | >=64 | >=64 | not_converged_evolved_and_tdvp_sweep_cap | 360.8 s |

The initial product-state energy density in every row is `1.48437500`.  Thus
only `R = 10` reaches a best-prefix energy below the initial product-state
energy, and the improvement is only `1.55e-4` per site before the trajectory
heats to `1.48521913` at the cap.  The other three frequency counts do not cool
even transiently within the recorded prefix.  All four rows reach the evolved
system-bath and TDVP-sweep caps by the fifth completed cycle, with retained
system bond dimensions already in the range `61--63`.  Therefore lowering the
coupling from `g = 0.1` to `g = 0.05` makes the channel too weak to produce
scalable large-`N` cooling before the same effective bond-dimension bottleneck
is encountered.

## Near-Ground Initial-State Control

The large-`N` driver also supports a near-ground control through
`--init-state ground`.  This does not load a restarted trajectory from HDF5.
It reuses the system ground state already computed by `setup_problem`: the
DMRG MPS stored in the TN `CoolingProblem`, or the exact ground vector stored
in the ED `CoolingProblem`.  For density-matrix methods the initial system
state is the corresponding pure ground-state density operator.

This option is intended for the benchmark question raised by the capped
product-state runs: does the cooling channel and its system-bath evolution
still generate large bond dimensions when the system is already near the
ground-state reference?  A stable `--init-state ground` trajectory would be a
channel-preservation and bond-growth control; it would not show that the same
protocol can cool a generic product state to the ground state.  Conversely,
large bond growth or energy drift from this control would indicate that the
channel/evolution path itself is a bottleneck.

A representative command for the current best capped schedule is

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending --init-state ground \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

Generated HDF5 filenames include `_initground`, and the HDF5 metadata records
`init_state = "ground"`, so these controls cannot be confused with the default
product-state cooling runs.

### First N=64 R=10 Ground-State Control

The `R = 10`, `Dmax = 128`, `te = 1.0` descending schedule above was repeated
from the system ground state on 2026-06-23:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 10 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending --init-state ground \
  --progress-csv .worktree/ground_control_dmax128_R10_20260623/tdvp_progress_N64_niising_open_mcwf_R10_Dmax128_te1.0_descending_initground.csv \
  --outdir .worktree/ground_control_dmax128_R10_20260623 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/ground_control_dmax128_R10_20260623/largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_scheddesc_initground_steps12_Dmax128_te1_tau0.2_seed20260617.h5
.worktree/ground_control_dmax128_R10_20260623/tdvp_progress_N64_niising_open_mcwf_R10_Dmax128_te1.0_descending_initground.csv
```

The HDF5 summary is

| R | Dcap | completed/requested cycles | initial E/N | initial overlap | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 10 | 128 | 8/12 | -1.32463289 | 1.00000 | -1.25614747 | -1.32463289 | 121 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 8 | 8 | 1325.6 s |

The completed-cycle prefix is

| cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03070050 | -1.31529539 | 8 | 12 | 39.6 s |
| 2 | 2.75008008 | -1.30661058 | 11 | 16 | 57.9 s |
| 3 | 2.46945966 | -1.29378032 | 18 | 24 | 79.9 s |
| 4 | 2.18883925 | -1.29311923 | 29 | 38 | 117.1 s |
| 5 | 1.90821883 | -1.28910674 | 46 | 60 | 189.1 s |
| 6 | 1.62759842 | -1.27613501 | 68 | 90 | 340.2 s |
| 7 | 1.34697800 | -1.27011979 | 96 | 124 | 657.8 s |
| 8 | 1.06635758 | -1.25614747 | 121 | 128 | 1325.6 s |

The `initial E/N` and `initial overlap` columns verify that `--init-state ground`
starts from the DMRG reference state.  The control remains near the ground-state
energy while the default
product-state run is still at positive energy density, and it delays the
`Dmax = 128` evolved/TDVP cap from cycle 5 to cycle 8.  It nevertheless heats
monotonically after initialization and eventually reaches the evolved and TDVP
sweep caps.  Thus the product-state cap is not solely a near-ground TDVP
channel artifact, but the fixed descending channel is still not an exact
ground-state fixed point at this bond cap.

### Remaining N=64 Ground-State Frequency Controls

The remaining frequency counts were then repeated with the same cap, schedule,
detuning interval, stopping rule, and thread pinning:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
julia --project=. scripts/validation/run_largeN_multifrequency_tn_scaling.jl \
  --Ns 64 --R-values 1,2,5 --methods mcwf \
  --evolution-method continuous --steps 12 --Dmax 128 \
  --cutoff 1e-7 --tau 0.2 --model niising --bc open --te 1.0 \
  --delta-min 0.5051167496264384 \
  --delta-max 3.0307004977586303 \
  --schedule descending --init-state ground \
  --progress-csv .worktree/ground_control_dmax128_R1_R2_R5_20260623/tdvp_progress_N64_niising_open_mcwf_R1-2-5_Dmax128_te1.0_descending_initground.csv \
  --outdir .worktree/ground_control_dmax128_R1_R2_R5_20260623 \
  --tdvp-sweep-progress --stop-on-bond-cap --verbose
```

The run wrote

```text
.worktree/ground_control_dmax128_R1_R2_R5_20260623/largeN_multifrequency_tn_N64_R1-2-5_mcwf_continuous_stopcap_scheddesc_initground_steps12_Dmax128_te1_tau0.2_seed20260617.h5
.worktree/ground_control_dmax128_R1_R2_R5_20260623/tdvp_progress_N64_niising_open_mcwf_R1-2-5_Dmax128_te1.0_descending_initground.csv
```

The HDF5 summary is

| R | Dcap | completed/requested cycles | initial E/N | initial overlap | final E/N | best E/N | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | system sat | evolved sat | tdvp sweep sat | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | 128 | 8/12 | -1.32463289 | 1.00000 | -1.21970618 | -1.32463289 | 116 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 8 | 8 | 1775.2 s |
| 2 | 128 | 7/12 | -1.32463289 | 1.00000 | -1.24703609 | -1.32463289 | 108 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 7 | 7 | 899.0 s |
| 5 | 128 | 8/12 | -1.32463289 | 1.00000 | -1.25381271 | -1.32463289 | 106 | >=128 | >=128 | not_converged_evolved_and_tdvp_sweep_cap | none | 8 | 8 | 864.1 s |

The completed-cycle prefixes are

| R | cycle | delta | E/N | system max bond | evolved max bond | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.50511675 | -1.31411353 | 8 | 12 | 43.0 s |
| 1 | 2 | 0.50511675 | -1.29971241 | 11 | 16 | 62.8 s |
| 1 | 3 | 0.50511675 | -1.29269272 | 17 | 24 | 88.8 s |
| 1 | 4 | 0.50511675 | -1.27452490 | 26 | 36 | 127.7 s |
| 1 | 5 | 0.50511675 | -1.26397027 | 40 | 53 | 264.4 s |
| 1 | 6 | 0.50511675 | -1.25917164 | 58 | 78 | 415.4 s |
| 1 | 7 | 0.50511675 | -1.24743382 | 86 | 114 | 999.9 s |
| 1 | 8 | 0.50511675 | -1.21970618 | 116 | 128 | 1775.2 s |
| 2 | 1 | 3.03070050 | -1.31130003 | 8 | 12 | 25.5 s |
| 2 | 2 | 0.50511675 | -1.28628510 | 12 | 17 | 48.4 s |
| 2 | 3 | 3.03070050 | -1.28614583 | 17 | 24 | 76.4 s |
| 2 | 4 | 0.50511675 | -1.28516011 | 28 | 37 | 121.6 s |
| 2 | 5 | 3.03070050 | -1.27717497 | 44 | 57 | 210.9 s |
| 2 | 6 | 0.50511675 | -1.24822873 | 73 | 95 | 416.4 s |
| 2 | 7 | 3.03070050 | -1.24703609 | 108 | 128 | 899.0 s |
| 5 | 1 | 3.03070050 | -1.31866536 | 8 | 12 | 16.7 s |
| 5 | 2 | 2.39930456 | -1.31737635 | 11 | 16 | 34.9 s |
| 5 | 3 | 1.76790862 | -1.31066100 | 17 | 23 | 56.9 s |
| 5 | 4 | 1.13651269 | -1.29555827 | 25 | 34 | 89.1 s |
| 5 | 5 | 0.50511675 | -1.27252963 | 36 | 48 | 143.7 s |
| 5 | 6 | 3.03070050 | -1.26779560 | 50 | 66 | 241.9 s |
| 5 | 7 | 2.39930456 | -1.26295293 | 73 | 97 | 436.7 s |
| 5 | 8 | 1.76790862 | -1.25381271 | 106 | 128 | 864.1 s |

Combining these rows with the `R = 10` ground-state control gives the final
energy ordering `R = 10 < R = 5 < R = 2 < R = 1`, where lower is better.  All
four controls start at the same DMRG reference, all heat after initialization,
and all reach the evolved and TDVP sweep cap at `Dmax = 128` by cycle 7 or 8.
The near-ground controls therefore separate two effects: the product-state
trajectory is indeed harder before cycle 5, but the fixed descending channel
still creates enough entanglement and heating near the ground state to be a
scaling bottleneck at this cap.
