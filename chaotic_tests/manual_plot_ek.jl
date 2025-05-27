using CoolingTNS
using HDF5
using PythonCall

plt = pyimport("matplotlib.pyplot")

# Read existing data
filename = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

data = h5open(filename, "r") do file
    Dict(
        "momentum_dist" => read(file, "momentum_dist"),
        "k_values" => read(file, "k_values"),
        "N" => read(file, "N"),
        "J" => read(file, "J"),
        "h" => read(file, "h"),
        "delta" => read(file, "delta"),
        "bc" => read(file, "bc")
    )
end

# Fix the data orientation and k_values
N = data["N"]
k_indices = data["k_values"]  # These are currently integers
momentum_dist = transpose(data["momentum_dist"])  # Now (n_k, steps)

# Convert k indices to actual momentum values for periodic BC
# For N=6: k_indices = [-2, -1, 0, 1, 2, 3]
# Convert to k = 2πn/N
k_values = [2π * k / N for k in k_indices]

# Compute energy dispersion
J = data["J"]
h = data["h"]
epsilon_k = [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_values]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot e_k at different steps
steps_to_plot = [1, 6, 11, 16, 21]  # Initial, 25%, 50%, 75%, final
colors = plt.cm.viridis(range(0, 1, length=5))

for (idx, (step, color)) in enumerate(zip(steps_to_plot, colors))
    e_k = epsilon_k .* momentum_dist[:, step]
    label = step == 1 ? "Initial" : "Step $(step-1)"
    ax.plot(k_values/π, e_k, "o-", color=color, linewidth=2, markersize=6, label=label)
end

# Ground state
n_k_gs = [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
e_k_gs = epsilon_k .* n_k_gs
ax.plot(k_values/π, e_k_gs, "k--", linewidth=2.5, label="Ground state")

# Bare dispersion
ax.plot(k_values/π, epsilon_k, ":", color="gray", linewidth=1.5, label="ε_k", alpha=0.7)

# Delta line
ax.axvline(x=data["delta"]/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)

# Formatting
ax.set_xlabel("k/π", fontsize=14)
ax.set_ylabel("e_k = ε_k n_k", fontsize=14)
ax.set_title("Energy Distribution Evolution (N=$(N), J=$(J), h=$(h), BC=periodic)", fontsize=16)
ax.grid(true, alpha=0.3)
ax.legend(loc="best", fontsize=12)
ax.set_xlim(-1.1, 1.1)

plt.tight_layout()

# Save
fig.savefig("Results/Figs/ek_evolution_full_brillouin.pdf", dpi=300, bbox_inches="tight")
println("Full Brillouin zone plot saved to Results/Figs/ek_evolution_full_brillouin.pdf")