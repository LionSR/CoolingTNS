"""
    jordan_wigner_transform_complex(site::Int, N::Int) -> (a, a†)

Jordan-Wigner transformation that returns complex fermionic operators.
Returns (a, a†) where:
- a is the annihilation operator for site i
- a† is the creation operator for site i
"""
function jordan_wigner_transform_complex(site::Int, N::Int)
    # Jordan-Wigner string operator
    string_op = I(2^N)
    for j in 1:(site-1)
        string_op *= pauli_z(j, N)
    end
    
    # Pauli operators for site i
    X_i = pauli_x(site, N)
    Y_i = pauli_y(site, N)
    
    # Fermionic operators in terms of Pauli matrices
    # a_i = (X_i + i*Y_i)/2 with Jordan-Wigner string
    # a†_i = (X_i - i*Y_i)/2 with Jordan-Wigner string
    a = string_op * (X_i + im * Y_i) / 2
    a_dag = string_op * (X_i - im * Y_i) / 2
    
    return (a, a_dag)
end

"""
    measure_momentum_distribution_ed_clean(ψ::EDStateVector, ham_params) -> (k_values, n_k)

Clean implementation using complex Jordan-Wigner operators.
"""
function measure_momentum_distribution_ed_clean(ψ::EDStateVector, ham_params)
    N = ham_params.N
    bc = ham_params.bc
    
    # Determine allowed k values based on boundary conditions
    if bc == :periodic
        k_indices = collect(-div(N,2)+1:div(N,2))
    elseif bc == :antiperiodic
        k_indices = collect(-div(N-1,2):div(N-1,2))
    else
        error("Momentum distribution only defined for periodic/antiperiodic BC")
    end
    
    n_k = zeros(Float64, length(k_indices))
    
    # For each momentum k
    for (ki, k) in enumerate(k_indices)
        # Compute ⟨a†_k a_k⟩ where a_k = (1/√N) Σ_j exp(-2πikj/N) a_j
        nk = 0.0
        
        for m in 1:N
            for n in 1:N
                # Get fermionic operators
                a_n, _ = jordan_wigner_transform_complex(n, N)
                _, a_m_dag = jordan_wigner_transform_complex(m, N)
                
                # Phase factors for Fourier transform
                phase = exp(2π * im * k * (m - n) / N) / N
                
                # Contribution to ⟨a†_k a_k⟩
                nk += phase * dot(ψ.data, a_m_dag * a_n * ψ.data)
            end
        end
        
        n_k[ki] = nk
    end
    
    # Convert k indices to momentum values
    k_momentum = [2π * k / N for k in k_indices]
    return k_momentum, n_k
end

# For density matrices
function measure_momentum_distribution_ed_clean(ρ::EDDensityMatrix, ham_params)
    N = ham_params.N
    bc = ham_params.bc
    
    # Determine allowed k values
    if bc == :periodic
        k_indices = collect(-div(N,2)+1:div(N,2))
    elseif bc == :antiperiodic
        k_indices = collect(-div(N-1,2):div(N-1,2))
    else
        error("Momentum distribution only defined for periodic/antiperiodic BC")
    end
    
    n_k = zeros(Float64, length(k_indices))
    
    for (ki, k) in enumerate(k_indices)
        nk = 0.0
        
        for m in 1:N
            for n in 1:N
                # Get fermionic operators
                a_n, _ = jordan_wigner_transform_complex(n, N)
                _, a_m_dag = jordan_wigner_transform_complex(m, N)
                
                # Phase factor
                phase = exp(2π * im * k * (m - n) / N) / N
                
                # Tr(ρ a†_m a_n) with phase
                nk += phase * tr(ρ.data * a_m_dag * a_n)
            end
        end
        
        n_k[ki] = nk
    end
    
    k_momentum = [2π * k / N for k in k_indices]
    return k_momentum, n_k
end