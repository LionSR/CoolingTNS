"""
    result_keys.jl

Canonical string keys for cooling result dictionaries.

These constants do not change the public output format.  They name the strings
used by `run_cooling`, HDF5 writers, plotting scripts, and tests, so changes to
the result schema can be made deliberately rather than by scattered literals.
"""

const RESULT_ENERGY = "E_list"
const RESULT_GROUND_STATE_OVERLAP = "GS_overlap_list"
const RESULT_PURITY = "purity_list"
const RESULT_BATH_MAGNETIZATION = "bath_mag_list"
const RESULT_BATH_SAMPLE_MAGNETIZATION = "nb_list"

const RESULT_MOMENTUM_DISTRIBUTION = "momentum_dist"
const RESULT_K_VALUES = "k_values"

const RESULT_MODE_GF = "mode_gF"
const RESULT_MODE_HK = "mode_hk"
const RESULT_MODE_K_INDICES = "mode_k_indices"
const RESULT_MODE_ENERGIES = "mode_ek_values"

const RESULT_DELTA_LIST = "delta_list"
const RESULT_TE_LIST = "te_list"
const RESULT_DELTA_VALUES = "delta_values"
const RESULT_SCHEDULE = "schedule"
const RESULT_RANDOMIZE_TIMES = "randomize_times"

const RESULT_N_TRAJECTORIES = "n_trajectories"
