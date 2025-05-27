using CoolingTNS

# Parameters
N = 6
J = 1.0
h = 2.0
g = 0.3
te = 2.0
steps = 20

println("Running density matrix simulation with N=$N...")
run(`julia --project=. Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g $g --te $te --steps $steps --J $J --h $h`)

# Plot the results
dm_file = "Results/Cooling_HamIsingJ$(J)h$(h)bcperiodic_CouplingXXg$(g)te$(te)steps$(steps)_SimEDDM.h5"

if isfile(dm_file)
    println("\nGenerating n_k and e_k evolution plots...")
    
    # Update the plot script to use the DM file
    plot_script = """
    using HDF5
    using PythonCall
    plt = pyimport("matplotlib.pyplot")

    # Read simulation data
    filename = "$dm_file"
    
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

    # Compute ground state values
    n_k_gs = [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
    e_k_gs = epsilon_k .* n_k_gs

    # Plot n_k evolution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    steps_to_plot = [1, 6, 11, 16, 21]
    colors = plt.cm.viridis(range(0, 1, length=5))

    for (idx, (step, color)) in enumerate(zip(steps_to_plot, colors))
        label = step == 1 ? "Initial" : "Step \$(step-1)"
        ax1.plot(k_values/π, momentum_dist[:, step], "o-", 
                color=color, linewidth=2, markersize=6, label=label)
    end

    ax1.plot(k_values/π, n_k_gs, "k--", linewidth=2.5, label="Ground state (T=0)")
    ax1.axvline(x=delta/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)
    ax1.set_xlabel("k/π", fontsize=14)
    ax1.set_ylabel("n_k", fontsize=14)
    ax1.set_title("Momentum Distribution (DM, N=\$N, J=\$J, h=\$h)", fontsize=16)
    ax1.grid(true, alpha=0.3)
    ax1.legend(loc="best", fontsize=12)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    fig1.savefig("Results/Figs/nk_dm_evolution.pdf", dpi=300, bbox_inches="tight")
    println("n_k plot saved to Results/Figs/nk_dm_evolution.pdf")

    # Plot e_k evolution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for (idx, (step, color)) in enumerate(zip(steps_to_plot, colors))
        e_k = epsilon_k .* momentum_dist[:, step]
        label = step == 1 ? "Initial" : "Step \$(step-1)"
        ax2.plot(k_values/π, e_k, "o-", 
                color=color, linewidth=2, markersize=6, label=label)
    end

    ax2.plot(k_values/π, e_k_gs, "k--", linewidth=2.5, label="Ground state (T=0)")
    ax2.plot(k_values/π, epsilon_k, ":", color="gray", linewidth=1.5, label="ε_k", alpha=0.7)
    ax2.axvline(x=delta/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)
    ax2.set_xlabel("k/π", fontsize=14)
    ax2.set_ylabel("e_k = ε_k n_k", fontsize=14)
    ax2.set_title("Energy Distribution (DM, N=\$N, J=\$J, h=\$h)", fontsize=16)
    ax2.grid(true, alpha=0.3)
    ax2.legend(loc="best", fontsize=12)
    ax2.set_xlim(-1.1, 1.1)
    plt.tight_layout()
    fig2.savefig("Results/Figs/ek_dm_evolution.pdf", dpi=300, bbox_inches="tight")
    println("e_k plot saved to Results/Figs/ek_dm_evolution.pdf")

    # Print diagnostics
    println("\\nDiagnostics:")
    println("Initial n_k: ", round.(momentum_dist[:, 1], digits=3))
    println("Final n_k:   ", round.(momentum_dist[:, 21], digits=3))
    println("Ground state n_k: ", round.(n_k_gs, digits=3))
    """
    
    # Save and run the plot script
    open("plot_dm_results.jl", "w") do f
        write(f, plot_script)
    end
    
    run(`julia --project=. plot_dm_results.jl`)
else
    println("Error: DM file not found at $dm_file")
end