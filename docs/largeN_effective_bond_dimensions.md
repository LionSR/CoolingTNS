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

The validation driver records the evolution branch in each HDF5 file.  The
four-cycle table below is a Trotter diagnostic (`--evolution-method trotter`,
`cutoff = 1e-7`).  The later two-cycle table is an MCWF+TDVP calibration
(`--evolution-method continuous`, `cutoff = 1e-6`).  These two protocols should
not be conflated.

## Definitions

The code source of truth for these quantities is
`scripts/validation/run_largeN_multifrequency_tn_scaling.jl`, summarized by
`scripts/validation/summarize_largeN_bond_dimensions.jl`.

The nominal parameter `Dmax` is not always the actual Trotter truncation cap.
The method-specific cap is

```math
D_{\rm cap} = \mathrm{tn\_method\_maxdim}(\mathrm{method}, D_{\max}).
```

For MCWF/MPS, `Dcap = Dmax`.  For MPO density-matrix Trotter evolution,
`Dcap = 4 Dmax`.

The summary table reports two different bond dimensions.

- `Dsys_eff` is the effective bond dimension of the retained system state after
  bath measurement.  It is computed from the largest final link dimension of
  the `N`-site system MPS or MPO over trajectories.
- `Dsb_eff` is the effective bond dimension of the transient enlarged
  system-bath state before bath measurement.  It is computed from the largest
  link dimension of the `2N`-site evolved MPS or MPO over all recorded cooling
  cycles and trajectories.

For Trotter cooling, `Dsb_eff` is the more stringent quantity, because the
algorithm must represent the enlarged system-bath state before the bath is
measured and discarded.

The reported value is conservative.  If the run reaches `Dcap`, then the
observed maximum is only a lower bound on the bond dimension required by the
untruncated trajectory.  In that case the summary script writes a label such as
`>=640`.  Otherwise it writes the largest observed link dimension.

The summary script also reports a machine-readable `bond_status` column.  The
status is only a bond-dimension diagnostic:

- `no_cap_hit`: neither the retained system state nor the transient
  system-bath state reached the cap during the recorded run.
- `not_converged_system_cap`: the retained system state reached the cap.
- `not_converged_evolved_cap`: the transient system-bath state reached the
  cap.
- `not_converged_system_and_evolved_cap`: both histories reached the cap.

A `no_cap_hit` entry does not by itself imply ground-state cooling or
trajectory convergence; it only means that the imposed bond cap was not reached
in the recorded run.

For long-cycle diagnostics, the summary script reports three energy readouts.
`final E/N` is the last recorded energy density, `best E/N` is the lowest
recorded energy density, and `tail E/N` is the average over the last ten
recorded rows, or over the whole trace if fewer than ten rows are present.
`best relE` is the smallest recorded relative energy; this is kept as a
separate column because a severely truncated trajectory could in principle
undershoot the ground-state reference in raw energy.  These fields separate
monotone cooling, transient low-energy excursions, and late-time plateaus
without assuming that a finite trajectory has reached a fixed point.

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
dimension, the post-evolution transient bond dimension, and the measured
per-cycle energy trace.

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
of `R = 1,2,5,10`: the transient system-bath state reaches the cap in all four
cases.  At `Dmax = 640`, the `R = 1` and `R = 5` trajectories are below cap
with observed transient dimensions 394 and 518, respectively, while `R = 2`
and `R = 10` still reach the cap by the fourth cycle.  The focused
`Dmax = 960` follow-up resolves those two lower bounds for this four-cycle
diagnostic: `Dsb_eff = 862` for `R = 2` and `Dsb_eff = 737` for `R = 10`.

The strongest currently bounded effective bond dimensions (post-measurement
system `Dsys_eff` and transient system-bath `Dsb_eff`) are:

```text
R =  1: Dsys_eff = 309, Dsb_eff = 394
R =  2: Dsys_eff = 637, Dsb_eff = 862
R =  5: Dsys_eff = 399, Dsb_eff = 518
R = 10: Dsys_eff = 489, Dsb_eff = 737
```

These values are far larger than the bond caps used in earlier exploratory
large-N curves.  In particular, `Dmax = 40`, `Dmax = 80`, and four-cycle
`Dmax = 320` runs should be read as truncation diagnostics rather than as
converged large-N cooling trajectories.  The `Dmax = 960` follow-up also took
production-scale time for the final high-bond steps, so future ladders should
write step-level checkpoints or be run as scheduled jobs rather than as
interactive smoke tests.

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
controlled Dmax ladder, record truncation and saturation diagnostics, and either
increase the effective transient bond cap or change the cooling protocol to
control the system-bath entanglement growth.

## Two-Cycle MCWF+TDVP Runtime Calibration

The first completed fixed-detuning TDVP calibration checks only two cooling
cycles.  This is not physically meaningful evidence for cooling or for a
fixed point.  It is useful only for verifying that the continuous-evolution
route is accessible and for estimating the first transient bond dimensions
before attempting long traced runs.

| R | Dcap | Dsys_eff | Dsb_eff | bond_status | final E/N | relE | peak evolved mean | elapsed |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 96 | 36 | 50 | no_cap_hit | 1.45494183 | 2.09837 | 39.67 | 168.6 s |
| 2 | 96 | 39 | 54 | no_cap_hit | 1.00710734 | 1.76029 | 41.88 | 227.0 s |
| 5 | 96 | 36 | 51 | no_cap_hit | 1.50779337 | 2.13827 | 39.69 | 287.3 s |
| 10 | 96 | 35 | 51 | no_cap_hit | 1.38290684 | 2.04399 | 39.76 | 376.4 s |

Thus, for the first two TDVP cycles only, `Dmax = 96` is sufficient for all
four frequency counts tested here: the transient system-bath dimensions are
about `50--54`, and the retained system-state dimensions are about `35--39`.
The energies, however, remain positive and far above the DMRG reference
`E0/N = -1.3246328892`.  The next meaningful TDVP test is therefore a longer
run with the progress CSV enabled, so that one can examine whether the expected
low-entanglement fixed point appears after the high-energy transient regime.
