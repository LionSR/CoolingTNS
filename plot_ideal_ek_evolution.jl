using PythonCall
plt = pyimport("matplotlib.pyplot")

# Parameters
N = 6
J = 1.0
h = 2.0
k_indices = [-2, -1, 0, 1, 2, 3]
k_values = [2π * k / N for k in k_indices]

# Energy dispersion
epsilon_k = [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_values]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define different temperature states
temperatures = [Inf, 2.0, 1.0, 0.5, 0.1]  # β = 1/T
labels = ["Initial (T=∞)", "T=2.0", "T=1.0", "T=0.5", "Ground state"]
colors = plt.cm.viridis(range(0, 1, length=5))

for (T, label, color) in zip(temperatures, labels, colors)
    if T == Inf
        # Infinite temperature: all n_k = 0.5
        n_k = fill(0.5, length(k_values))
    elseif T == 0.1
        # Ground state approximation
        n_k = [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
    else
        # Thermal state: n_k = 1/(1 + exp(β*ε_k)) where ε_k is relative to chemical potential
        β = 1/T
        # For simplicity, using Fermi-Dirac with μ adjusted to give correct filling
        n_k = [1/(1 + exp(β * (epsilon_k[i] + 3.0))) for i in 1:length(k_values)]
    end
    
    e_k = epsilon_k .* n_k
    ax.plot(k_values/π, e_k, "o-", color=color, linewidth=2, markersize=6, label=label)
end

# Add bare dispersion
ax.plot(k_values/π, epsilon_k, ":", color="gray", linewidth=1.5, label="ε_k", alpha=0.7)

# Formatting
ax.set_xlabel("k/π", fontsize=14)
ax.set_ylabel("e_k = ε_k n_k", fontsize=14)
ax.set_title("Ideal Energy Distribution Evolution During Cooling\n(N=$N, J=$J, h=$h)", fontsize=16)
ax.grid(true, alpha=0.3)
ax.legend(loc="best", fontsize=12)
ax.set_xlim(-1.1, 1.1)

plt.tight_layout()

# Save
fig.savefig("Results/Figs/ideal_ek_evolution.pdf", dpi=300, bbox_inches="tight")
println("Ideal e_k evolution plot saved to Results/Figs/ideal_ek_evolution.pdf")

# Also show what n_k should look like
fig2, ax2 = plt.subplots(figsize=(10, 6))

for (T, label, color) in zip(temperatures, labels, colors)
    if T == Inf
        n_k = fill(0.5, length(k_values))
    elseif T == 0.1
        n_k = [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
    else
        β = 1/T
        n_k = [1/(1 + exp(β * (epsilon_k[i] + 3.0))) for i in 1:length(k_values)]
    end
    
    ax2.plot(k_values/π, n_k, "o-", color=color, linewidth=2, markersize=6, label=label)
end

ax2.set_xlabel("k/π", fontsize=14)
ax2.set_ylabel("n_k", fontsize=14)
ax2.set_title("Ideal Momentum Distribution Evolution\n(N=$N, J=$J, h=$h)", fontsize=16)
ax2.grid(true, alpha=0.3)
ax2.legend(loc="best", fontsize=12)
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-0.1, 1.1)

plt.tight_layout()

fig2.savefig("Results/Figs/ideal_nk_evolution.pdf", dpi=300, bbox_inches="tight")
println("Ideal n_k evolution plot saved to Results/Figs/ideal_nk_evolution.pdf")