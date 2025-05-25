"""
    ground_state_ed_clean.jl

Ground state computation for ED backend without Yao dependencies.
"""

include("ed_backend.jl")

# ============================================================================
# Ground State Computation for ED Backend
# ============================================================================

"""
    find_ground_state(H_sys::SparseMatrixCSC, backend::EDBackend)

Find ground state energy, state, and gap for ED backend.
"""
function find_ground_state(H_sys::SparseMatrixCSC, backend::EDBackend)
    E0, ψ0, gap = ground_state_ed(H_sys)
    return E0, ψ0, gap
end

"""
    find_ground_state(H_sys::Matrix, backend::EDBackend)

Find ground state for dense matrix (convert to sparse first).
"""
function find_ground_state(H_sys::Matrix, backend::EDBackend)
    return find_ground_state(sparse(H_sys), backend)
end