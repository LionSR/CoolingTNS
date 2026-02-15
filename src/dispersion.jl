"""
Analytical dispersion relations and k-space utilities for the transverse field Ising model.

These are pure-math functions with no plotting or I/O dependencies.
"""

"""
    generate_k_values(N::Int, bc::Symbol) -> Vector{Float64}

Generate k-values based on boundary conditions.
- Periodic BC: k = 2π n/N for n = 0, 1, ..., N-1
- Antiperiodic BC: k = π(2n+1)/N for n = 0, 1, ..., N-1
"""
function generate_k_values(N::Int, bc::Symbol)::Vector{Float64}
    if bc == :periodic
        return [2pi * n / N for n in 0:N-1]
    elseif bc == :antiperiodic
        return [pi * (2n + 1) / N for n in 0:N-1]
    else
        error("Unsupported boundary condition: $bc")
    end
end

"""
    compute_energy_dispersion(k_values, J::Real, h::Real) -> Vector{Float64}

Compute energy dispersion ε_k for the transverse field Ising model.
ε_k = -2√(J² + h² + 2Jh cos(k))
"""
function compute_energy_dispersion(k_values, J::Real, h::Real)::Vector{Float64}
    return [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_values]
end

"""
    compute_ground_state_occupation(k_values, J::Real, h::Real) -> Vector{Float64}

Compute ground state occupation n_k^(GS) for the transverse field Ising model.
n_k^(GS) = (1/2)(1 - (J cos(k) + h)/√(J² + h² + 2Jh cos(k)))
"""
function compute_ground_state_occupation(k_values, J::Real, h::Real)::Vector{Float64}
    return [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
end
