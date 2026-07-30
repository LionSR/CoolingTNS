"""
    state_manipulation.jl

Dispatched functions for state manipulation operations like appending bath,
sampling, and tracing out subsystems.
"""

using ITensors
using ITensorMPS


# ============================================================================
# Utility Functions
# ============================================================================

"""Create projector from EDStateVector"""
projector(ψ::EDStateVector) = state_to_density_ed(ψ).data

"""Create MPO projector from MPS for TN backend"""
projector_mpo(ψ::MPS) = outer(ψ', ψ)
