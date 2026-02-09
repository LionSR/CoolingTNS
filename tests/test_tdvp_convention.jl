"""
    test_tdvp_convention.jl

Determine the ITensorMPS TDVP time convention:
- Convention A: tdvp(H, t, ψ) computes exp(tH)ψ — need t = -im*te for Schrödinger
- Convention B: tdvp(H, t, ψ) computes exp(-iHt)ψ — need t = te (real) for Schrödinger

If convention B, then t = -im*te gives exp(-iH(-im*te)) = exp(-H*te) = imaginary time!
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^60)
println("TDVP Convention Test")
println("="^60)

# Create a simple 2-site Hamiltonian: H = σ_x ⊗ I + I ⊗ σ_x
N = 2
sites = siteinds("S=1/2", N)

# Use OpSum for the Hamiltonian
os = OpSum()
os += 1.0, "X", 1
os += 1.0, "X", 2
H_mpo = MPO(os, sites)

# Create initial state |↑↑⟩
ψ0 = MPS(sites, "Up")

# Exact Hamiltonian matrix
H_mat = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0]  # σ_x ⊗ I + I ⊗ σ_x

# Initial state |↑↑⟩ = |00⟩ = [1,0,0,0]
ψ0_vec = [1.0+0im, 0, 0, 0]

te = 0.5
tau = 0.01

# Exact real-time evolution: exp(-iHt)|ψ⟩
ψ_exact_realtime = exp(-im * te * H_mat) * ψ0_vec

# Exact imaginary-time evolution: exp(-Ht)|ψ⟩ (unnormalized)
ψ_exact_imagtime = exp(-te * H_mat) * ψ0_vec
ψ_exact_imagtime_norm = ψ_exact_imagtime / norm(ψ_exact_imagtime)

println("\nExact evolution of |↑↑⟩ under H = X₁ + X₂:")
println(@sprintf("  Real time (t=%.1f):  [%.6f, %.6f, %.6f, %.6f]", te, abs.(ψ_exact_realtime)...))
println(@sprintf("  Imag time (t=%.1f):  [%.6f, %.6f, %.6f, %.6f]", te, abs.(ψ_exact_imagtime_norm)...))

# Energy of exact states
E_exact_real = real(ψ_exact_realtime' * H_mat * ψ_exact_realtime / (ψ_exact_realtime' * ψ_exact_realtime))
E_exact_imag = real(ψ_exact_imagtime_norm' * H_mat * ψ_exact_imagtime_norm)
println(@sprintf("\n  Energy (real time):  %.6f", E_exact_real))
println(@sprintf("  Energy (imag time):  %.6f", E_exact_imag))
println("  Note: imaginary time should lower energy more than real time")

# Helper: MPS to vector
function mps_to_vector(ψ::MPS, sites)
    N = length(sites)
    dim = 2^N
    vec = zeros(ComplexF64, dim)
    for idx in 0:dim-1
        config = [((idx >> (k-1)) & 1) == 0 ? "Up" : "Dn" for k in 1:N]
        ψ_prod = MPS(sites, config)
        vec[idx+1] = inner(ψ_prod, ψ)
    end
    return vec
end

# === Test A: TDVP with t = -im*te (current code convention) ===
println("\n--- Test A: tdvp(H, -im*t, ψ) [current code] ---")
ψ_A = tdvp(H_mpo, -im * te, ψ0;
           time_step=-im * tau, nsite=2, reverse_step=false, normalize=false,
           maxdim=100, cutoff=1e-14, outputlevel=0)

ψ_A_vec = mps_to_vector(ψ_A, sites)
ψ_A_vec /= norm(ψ_A_vec)
E_A = real(inner(ψ_A', H_mpo, ψ_A) / inner(ψ_A, ψ_A))

println(@sprintf("  |ψ_A| = [%.6f, %.6f, %.6f, %.6f]", abs.(ψ_A_vec)...))
println(@sprintf("  Energy = %.6f", E_A))

overlap_real_A = abs(dot(ψ_exact_realtime / norm(ψ_exact_realtime), ψ_A_vec))
overlap_imag_A = abs(dot(ψ_exact_imagtime_norm, ψ_A_vec))
println(@sprintf("  Overlap with real-time: %.6f", overlap_real_A))
println(@sprintf("  Overlap with imag-time: %.6f", overlap_imag_A))

# === Test B: TDVP with t = te (real, Schrödinger convention) ===
println("\n--- Test B: tdvp(H, t, ψ) [real time parameter] ---")
ψ_B = tdvp(H_mpo, te, ψ0;
           time_step=tau, nsite=2, reverse_step=false, normalize=false,
           maxdim=100, cutoff=1e-14, outputlevel=0)

ψ_B_vec = mps_to_vector(ψ_B, sites)
ψ_B_vec /= norm(ψ_B_vec)
E_B = real(inner(ψ_B', H_mpo, ψ_B) / inner(ψ_B, ψ_B))

println(@sprintf("  |ψ_B| = [%.6f, %.6f, %.6f, %.6f]", abs.(ψ_B_vec)...))
println(@sprintf("  Energy = %.6f", E_B))

overlap_real_B = abs(dot(ψ_exact_realtime / norm(ψ_exact_realtime), ψ_B_vec))
overlap_imag_B = abs(dot(ψ_exact_imagtime_norm, ψ_B_vec))
println(@sprintf("  Overlap with real-time: %.6f", overlap_real_B))
println(@sprintf("  Overlap with imag-time: %.6f", overlap_imag_B))

# === Summary ===
println("\n" * "="^60)
println("SUMMARY:")
if overlap_real_A > 0.999
    println("  Convention A confirmed: tdvp(H, -im*t, ψ) = exp(-iHt)ψ [REAL TIME]")
    println("  Current code is CORRECT")
elseif overlap_imag_A > 0.999
    println("  Convention B detected: tdvp(H, -im*t, ψ) = exp(-Ht)ψ [IMAGINARY TIME]")
    println("  Current code is WRONG — doing imaginary time evolution!")
    println("  FIX: Use tdvp(H, t, ψ) with real t for Schrödinger evolution")
else
    println("  Neither convention matches perfectly")
    println("  overlap_real_A = $overlap_real_A, overlap_imag_A = $overlap_imag_A")
end
println("="^60)
