"""
    result_keys.jl

Canonical string keys for cooling result dictionaries.

These constants do not change the public output format.  They name the strings
used by `run_cooling`, HDF5 writers, plotting scripts, and tests, so changes to
the result schema can be made deliberately rather than by scattered literals.
`RESULT_KEYS` is the complete registry of keys owned by the cooling result schema.
"""

const RESULT_ENERGY = "E_list"
const RESULT_GROUND_STATE_OVERLAP = "GS_overlap_list"
const RESULT_PURITY = "purity_list"
const RESULT_BATH_MAGNETIZATION = "bath_mag_list"
const RESULT_BATH_SAMPLE_MAGNETIZATION = "nb_list"

const RESULT_MOMENTUM_DISTRIBUTION = "momentum_dist"
const RESULT_K_VALUES = "k_values"
const RESULT_MOMENTUM_GF = "momentum_gF"
const RESULT_MOMENTUM_GF_SOURCE = "momentum_gF_source"

const RESULT_MODE_GF = "mode_gF"
const RESULT_MODE_HK = "mode_hk"
const RESULT_MODE_NK = "mode_nk"
const RESULT_MODE_K_INDICES = "mode_k_indices"
const RESULT_MODE_ENERGIES = "mode_ek_values"

const RESULT_DELTA_LIST = "delta_list"
const RESULT_TE_LIST = "te_list"
const RESULT_DELTA_VALUES = "delta_values"
const RESULT_SCHEDULE = "schedule"
const RESULT_RANDOMIZE_TIMES = "randomize_times"

const RESULT_N_TRAJECTORIES = "n_trajectories"
const RESULT_ENERGY_TRAJECTORIES = "E_trajectories"
const RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES = "GS_trajectories"
const RESULT_ENERGY_STD = "E_std"
const RESULT_GROUND_STATE_OVERLAP_STD = "GS_std"

const RESULT_BOND_DIMS = "bond_dims"
const RESULT_TRUNCATION_ERRORS = "truncation_errors"
const RESULT_RENYI_ENTROPY = "renyi_entropy"
const RESULT_FINAL_STATE = "final_state"

const RESULT_KEYS = (
    RESULT_ENERGY,
    RESULT_GROUND_STATE_OVERLAP,
    RESULT_PURITY,
    RESULT_BATH_MAGNETIZATION,
    RESULT_BATH_SAMPLE_MAGNETIZATION,
    RESULT_MOMENTUM_DISTRIBUTION,
    RESULT_K_VALUES,
    RESULT_MOMENTUM_GF,
    RESULT_MOMENTUM_GF_SOURCE,
    RESULT_MODE_GF,
    RESULT_MODE_HK,
    RESULT_MODE_NK,
    RESULT_MODE_K_INDICES,
    RESULT_MODE_ENERGIES,
    RESULT_DELTA_LIST,
    RESULT_TE_LIST,
    RESULT_DELTA_VALUES,
    RESULT_SCHEDULE,
    RESULT_RANDOMIZE_TIMES,
    RESULT_N_TRAJECTORIES,
    RESULT_ENERGY_TRAJECTORIES,
    RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES,
    RESULT_ENERGY_STD,
    RESULT_GROUND_STATE_OVERLAP_STD,
    RESULT_BOND_DIMS,
    RESULT_TRUNCATION_ERRORS,
    RESULT_RENYI_ENTROPY,
    RESULT_FINAL_STATE,
)
