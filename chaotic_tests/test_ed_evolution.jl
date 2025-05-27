using LinearAlgebra

# Test ED evolution with a simple 2-qubit system
H = [
    2.0  0.0  0.0  1.0;
    0.0  0.0  1.0  0.0;
    0.0  1.0  0.0  0.0;
    1.0  0.0  0.0 -2.0
]

# Initial state |00⟩
ψ0 = [1.0, 0.0, 0.0, 0.0]

# Time evolution
t = 0.5

# Method 1: Full complex evolution
F = eigen(Symmetric(H))
U = F.vectors * Diagonal(exp.(-im * t * F.values)) * F.vectors'
ψ1_complex = U * ψ0
ψ1_real = real(ψ1_complex)
ψ1_imag = imag(ψ1_complex)

println("Method 1 (full complex):")
println("Real part: ", ψ1_real)
println("Imag part: ", ψ1_imag)
println("Norm of real part: ", norm(ψ1_real))
println("Energy: ", real(ψ1_complex' * H * ψ1_complex))
println()

# Method 2: What we're doing in ED backend (taking real part)
ψ2 = real(U * ψ0)
ψ2 = ψ2 / norm(ψ2)
println("Method 2 (normalized real part):")
println("State: ", ψ2)
println("Energy: ", ψ2' * H * ψ2)
println()

# The issue: taking only the real part loses quantum coherence!
println("The problem: Method 2 gives different energy than Method 1")