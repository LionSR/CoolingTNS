using HDF5
using PythonCall
plt = pyimport("matplotlib.pyplot")

# Read actual simulation data
filename = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

data = h5open(filename, "r") do file
    Dict(
        "momentum_dist" => read(file, "momentum_dist"),
        "k_values" => read(file, "k_values"),
        "N" => read(file, "N"),
        "J" => read(file, "J"),
        "h" => read(file, "h"),
        "delta" => read(file, "delta")
    )
end

# Fix data orientation
N = data["N"]
J = data["J"]
h = data["h"]
delta = data["delta"]
k_indices = data["k_values"]
momentum_dist = transpose(data["momentum_dist"])  # Now (n_k, steps)

# Convert k indices to actual momentum values
k_values = [2π * k / N for k in k_indices]

# Compute energy dispersion
epsilon_k = [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_values]

# Compute ground state values for comparison
n_k_gs = [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
e_k_gs = epsilon_k .* n_k_gs

# =========== Plot 1: n_k evolution during cooling ===========
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Plot at different cooling steps
steps_to_plot = [1, 6, 11, 16, 21]  # Initial, 25%, 50%, 75%, final
colors = plt.cm.viridis(range(0, 1, length=5))

for (idx, (step, color)) in enumerate(zip(steps_to_plot, colors))
    label = step == 1 ? "Initial" : "Step $(step-1)"
    ax1.plot(k_values/π, momentum_dist[:, step], "o-", 
            color=color, linewidth=2, markersize=6, label=label)
end

# Plot ground state for reference
ax1.plot(k_values/π, n_k_gs, "k--", linewidth=2.5, label="Ground state (T=0)")

# Add delta line
ax1.axvline(x=delta/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)

ax1.set_xlabel("k/π", fontsize=14)
ax1.set_ylabel("n_k", fontsize=14)
ax1.set_title("Momentum Distribution During Cooling\n(N=$N, J=$J, h=$h, BC=periodic)", fontsize=16)
ax1.grid(true, alpha=0.3)
ax1.legend(loc="best", fontsize=12)
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-0.1, 1.1)

plt.tight_layout()
fig1.savefig("Results/Figs/nk_cooling_evolution.pdf", dpi=300, bbox_inches="tight")
println("n_k cooling evolution saved to Results/Figs/nk_cooling_evolution.pdf")

# =========== Plot 2: e_k evolution during cooling ===========
fig2, ax2 = plt.subplots(figsize=(10, 6))

for (idx, (step, color)) in enumerate(zip(steps_to_plot, colors))
    e_k = epsilon_k .* momentum_dist[:, step]
    label = step == 1 ? "Initial" : "Step $(step-1)"
    ax2.plot(k_values/π, e_k, "o-", 
            color=color, linewidth=2, markersize=6, label=label)
end

# Plot ground state
ax2.plot(k_values/π, e_k_gs, "k--", linewidth=2.5, label="Ground state (T=0)")

# Add bare dispersion
ax2.plot(k_values/π, epsilon_k, ":", color="gray", linewidth=1.5, label="ε_k", alpha=0.7)

# Add delta line
ax2.axvline(x=delta/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)

ax2.set_xlabel("k/π", fontsize=14)
ax2.set_ylabel("e_k = ε_k n_k", fontsize=14)
ax2.set_title("Energy Distribution During Cooling\n(N=$N, J=$J, h=$h, BC=periodic)", fontsize=16)
ax2.grid(true, alpha=0.3)
ax2.legend(loc="best", fontsize=12)
ax2.set_xlim(-1.1, 1.1)

plt.tight_layout()
fig2.savefig("Results/Figs/ek_cooling_evolution.pdf", dpi=300, bbox_inches="tight")
println("e_k cooling evolution saved to Results/Figs/ek_cooling_evolution.pdf")

# Print diagnostics
println("\nDiagnostics:")
println("Initial n_k: ", round.(momentum_dist[:, 1], digits=3))
println("Final n_k:   ", round.(momentum_dist[:, 21], digits=3))
println("Ground state n_k: ", round.(n_k_gs, digits=3))
println("\nIs n_k changing? ", !(momentum_dist[:, 1] ≈ momentum_dist[:, 21]))