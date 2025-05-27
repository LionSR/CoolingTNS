using HDF5
using PythonCall
plt = pyimport("matplotlib.pyplot")

# Load ground state data
gs_data = h5open("gs_nk_data.h5", "r") do file
    Dict(
        "k_values" => read(file, "k_values"),
        "n_k_gs" => read(file, "n_k_gs"),
        "E0" => read(file, "E0"),
        "N" => read(file, "N"),
        "J" => read(file, "J"),
        "h" => read(file, "h")
    )
end

# Load simulation data (using the shorter simulation)
sim_file = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te0.5steps5_SimEDDM.h5"

sim_data = h5open(sim_file, "r") do file
    Dict(
        "momentum_dist" => read(file, "momentum_dist"),
        "k_values" => read(file, "k_values"),
        "N" => read(file, "N"),
        "J" => read(file, "J"),
        "h" => read(file, "h"),
        "delta" => read(file, "delta"),
        "E_list" => read(file, "E_list"),
        "e₀" => read(file, "e₀")
    )
end

# Verify parameters match
@assert gs_data["N"] == sim_data["N"] "N mismatch"
@assert gs_data["J"] ≈ sim_data["J"] "J mismatch"
@assert gs_data["h"] ≈ sim_data["h"] "h mismatch"

N = gs_data["N"]
J = gs_data["J"]
h = gs_data["h"]
delta = sim_data["delta"]

# Fix simulation data orientation
momentum_dist = transpose(sim_data["momentum_dist"])  # Now (n_k, steps)

# Use ground state k_values (should be the same)
k_values = gs_data["k_values"]
n_k_gs = gs_data["n_k_gs"]

# Compute energy dispersion
epsilon_k = [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_values]

# Check energies
println("Ground state energy comparison:")
println("  From numerical diagonalization: E0/N = ", gs_data["E0"]/N)
println("  Stored in simulation file: e₀/N = ", sim_data["e₀"]/N)

# Plot n_k evolution
fig1, ax1 = plt.subplots(figsize=(10, 6))
total_steps = size(momentum_dist, 2)
steps_to_plot = unique([1, div(total_steps,2), total_steps])
colors = ["blue", "green", "red"]

println("\nn_k evolution:")
for (idx, (step, color)) in enumerate(zip(steps_to_plot, colors))
    label = step == 1 ? "Initial" : "Step $(step-1)"
    println("  $label: ", round.(momentum_dist[:, step], digits=3))
    ax1.plot(k_values/π, momentum_dist[:, step], "o-", 
            color=color, linewidth=2, markersize=6, label=label)
end
println("  Numerical GS: ", round.(n_k_gs, digits=3))

# Plot numerical ground state
ax1.plot(k_values/π, n_k_gs, "k--", linewidth=2.5, label="Numerical GS (T=0)", marker="s", markersize=6)

ax1.axvline(x=delta/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)
ax1.set_xlabel("k/π", fontsize=14)
ax1.set_ylabel("n_k", fontsize=14)
ax1.set_title("Momentum Distribution with Numerical Ground State\n(N=$N, J=$J, h=$h)", fontsize=16)
ax1.grid(true, alpha=0.3)
ax1.legend(loc="best", fontsize=12)
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
plt.tight_layout()
fig1.savefig("Results/Figs/nk_numerical_gs.pdf", dpi=300, bbox_inches="tight")
println("\nn_k plot saved to Results/Figs/nk_numerical_gs.pdf")

# Plot e_k evolution
fig2, ax2 = plt.subplots(figsize=(10, 6))
for (idx, (step, color)) in enumerate(zip(steps_to_plot, colors))
    e_k = epsilon_k .* momentum_dist[:, step]
    label = step == 1 ? "Initial" : "Step $(step-1)"
    ax2.plot(k_values/π, e_k, "o-", 
            color=color, linewidth=2, markersize=6, label=label)
end

# Numerical ground state e_k
e_k_gs = epsilon_k .* n_k_gs
ax2.plot(k_values/π, e_k_gs, "k--", linewidth=2.5, label="Numerical GS (T=0)", marker="s", markersize=6)

ax2.plot(k_values/π, epsilon_k, ":", color="gray", linewidth=1.5, label="ε_k", alpha=0.7)
ax2.axvline(x=delta/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)
ax2.set_xlabel("k/π", fontsize=14)
ax2.set_ylabel("e_k = ε_k n_k", fontsize=14)
ax2.set_title("Energy Distribution with Numerical Ground State\n(N=$N, J=$J, h=$h)", fontsize=16)
ax2.grid(true, alpha=0.3)
ax2.legend(loc="best", fontsize=12)
ax2.set_xlim(-1.1, 1.1)
plt.tight_layout()
fig2.savefig("Results/Figs/ek_numerical_gs.pdf", dpi=300, bbox_inches="tight")
println("e_k plot saved to Results/Figs/ek_numerical_gs.pdf")