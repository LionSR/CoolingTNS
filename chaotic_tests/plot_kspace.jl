using CoolingTNS
using HDF5
using Plots
using Printf

# Read data file
filename = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

# Check if file exists
if !isfile(filename)
    error("File not found: $filename")
end

# Read data
data = h5open(filename, "r") do file
    Dict(
        "momentum_dist" => read(file, "momentum_dist"),
        "k_values" => read(file, "k_values"),
        "delta" => haskey(file, "delta") ? read(file, "delta") : nothing,
        "N" => read(file, "N"),
        "steps" => read(file, "steps")
    )
end

momentum_dist = data["momentum_dist"]
k_values = data["k_values"]
delta = data["delta"]
N = data["N"]
steps = data["steps"]

println("Data shape: momentum_dist = $(size(momentum_dist)), k_values = $(length(k_values))")
println("N = $N, steps = $steps, delta = $delta")

# Create line plot
p = plot(
    xlabel = "k/π",
    ylabel = "n_k",
    title = "Momentum Distribution Evolution (N=$N)",
    legend = :topright,
    size = (800, 600),
    dpi = 300
)

# Plot initial, middle, and final distributions
step_indices = [1, div(steps, 2), steps + 1]
labels = ["Initial", "Step $(div(steps, 2))", "Final"]
colors = [:blue, :green, :red]

for (idx, (step_idx, label, color)) in enumerate(zip(step_indices, labels, colors))
    if step_idx <= size(momentum_dist, 2)
        plot!(p, k_values/π, momentum_dist[:, step_idx], 
              label=label, color=color, linewidth=2, marker=:circle)
    end
end

# Add vertical line for delta if available
if !isnothing(delta) && delta != 0
    vline!(p, [delta/π], color=:black, linestyle=:dash, label="δ/π", linewidth=2)
end

# Save plot
savefig(p, "kspace_plot.pdf")
println("Plot saved as kspace_plot.pdf")

# Display plot
display(p)