"""
    jordan_wigner_transform_complex(site::Int, N::Int) -> (a, a†)

Jordan-Wigner transformation that returns complex fermionic operators.
Returns `(a, a†)` where:
- `a` is the annihilation operator for site `site`
- `a†` is the creation operator for site `site`
"""
function jordan_wigner_transform_complex(site::Int, N::Int)
    # Jordan-Wigner string operator ∏_{j < site} Z_j
    string_op = I(2^N)
    for j in 1:(site - 1)
        string_op *= pauli_z(j, N)
    end

    X_i = pauli_x(site, N)
    Y_i = pauli_y(site, N)

    # Fermionic operators in terms of Pauli matrices:
    # a_i     = (X_i + i Y_i)/2 with the JW string
    # a†_i = (X_i - i Y_i)/2 with the JW string
    a = string_op * (X_i + im * Y_i) / 2
    a_dag = string_op * (X_i - im * Y_i) / 2

    return (a, a_dag)
end

# -----------------------------------------------------------------------------
# Momentum distribution (shared implementation)
# -----------------------------------------------------------------------------

function _allowed_k_indices(N::Int, bc::Symbol)
    if bc == :periodic
        return collect(-div(N, 2) + 1:div(N, 2))
    elseif bc == :antiperiodic
        return collect(-div(N - 1, 2):div(N - 1, 2))
    else
        error("Momentum distribution only defined for periodic/antiperiodic BC")
    end
end

_expect_amdag_an(ψ::EDStateVector, a_m_dag, a_n) = dot(ψ.data, a_m_dag * a_n * ψ.data)
_expect_amdag_an(ρ::EDDensityMatrix, a_m_dag, a_n) = tr(ρ.data * a_m_dag * a_n)

function _measure_momentum_distribution_ed_clean(state, ham_params)
    N = ham_params.N
    k_indices = _allowed_k_indices(N, ham_params.bc)

    n_k = zeros(Float64, length(k_indices))

    # Precompute all JW operators once (these are expensive 2^N × 2^N matrices)
    a1, a1_dag = jordan_wigner_transform_complex(1, N)
    a_ops = Vector{typeof(a1)}(undef, N)
    a_dag_ops = Vector{typeof(a1_dag)}(undef, N)
    a_ops[1] = a1
    a_dag_ops[1] = a1_dag
    for j in 2:N
        a_ops[j], a_dag_ops[j] = jordan_wigner_transform_complex(j, N)
    end

    for (ki, k) in enumerate(k_indices)
        nk = 0.0 + 0.0im

        # ⟨a†_k a_k⟩ where a_k = (1/√N) Σ_j exp(-2πikj/N) a_j
        for m in 1:N, n in 1:N
            phase = exp(2π * im * k * (m - n) / N) / N
            nk += phase * _expect_amdag_an(state, a_dag_ops[m], a_ops[n])
        end

        n_k[ki] = real(nk)
    end

    k_momentum = [2π * k / N for k in k_indices]
    return k_momentum, n_k
end

"""
    measure_momentum_distribution_ed_clean(state, ham_params) -> (k_values, n_k)

Compute the momentum distribution $n_k = \langle a_k^\dagger a_k \rangle$ using the
complex Jordan–Wigner mapping.

Supported boundary conditions: `:periodic`, `:antiperiodic`.
"""
measure_momentum_distribution_ed_clean(ψ::EDStateVector, ham_params) =
    _measure_momentum_distribution_ed_clean(ψ, ham_params)

measure_momentum_distribution_ed_clean(ρ::EDDensityMatrix, ham_params) =
    _measure_momentum_distribution_ed_clean(ρ, ham_params)
