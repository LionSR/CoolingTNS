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

# Define different temperature states
temperatures = [Inf, 2.0, 1.0, 0.5, 0.1]  # β = 1/T
labels = ["Initial (T=∞)", "T=2.0", "T=1.0", "T=0.5", "Ground state"]
colors = plt.cm.viridis(range(0, 1, length=5))

# =========== Plot 1: n_k evolution ===========
fig1, ax1 = plt.subplots(figsize=(10, 6))

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
    
    ax1.plot(k_values/π, n_k, "o-", color=color, linewidth=2, markersize=6, label=label)
end

ax1.set_xlabel("k/π", fontsize=14)
ax1.set_ylabel("n_k", fontsize=14)
ax1.set_title("Momentum Distribution Evolution\n(N=$N, J=$J, h=$h)", fontsize=16)
ax1.grid(true, alpha=0.3)
ax1.legend(loc="best", fontsize=12)
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-0.1, 1.1)

plt.tight_layout()
fig1.savefig("Results/Figs/nk_evolution_ideal.pdf", dpi=300, bbox_inches="tight")
println("n_k evolution plot saved to Results/Figs/nk_evolution_ideal.pdf")

# =========== Plot 2: e_k evolution ===========
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
    
    e_k = epsilon_k .* n_k
    ax2.plot(k_values/π, e_k, "o-", color=color, linewidth=2, markersize=6, label=label)
end

# Add bare dispersion for reference
ax2.plot(k_values/π, epsilon_k, ":", color="gray", linewidth=1.5, label="ε_k (bare)", alpha=0.7)

ax2.set_xlabel("k/π", fontsize=14)
ax2.set_ylabel("e_k = ε_k n_k", fontsize=14)
ax2.set_title("Energy Distribution Evolution\n(N=$N, J=$J, h=$h)", fontsize=16)
ax2.grid(true, alpha=0.3)
ax2.legend(loc="best", fontsize=12)
ax2.set_xlim(-1.1, 1.1)

plt.tight_layout()
fig2.savefig("Results/Figs/ek_evolution_ideal.pdf", dpi=300, bbox_inches="tight")
println("e_k evolution plot saved to Results/Figs/ek_evolution_ideal.pdf")