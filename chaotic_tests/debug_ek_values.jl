using HDF5

# Read existing data
filename = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

data = h5open(filename, "r") do file
    Dict(
        "momentum_dist" => read(file, "momentum_dist"),
        "k_values" => read(file, "k_values"),
        "N" => read(file, "N"),
        "J" => read(file, "J"),
        "h" => read(file, "h"),
        "E_list" => read(file, "E_list")
    )
end

# Check the data
println("momentum_dist shape: ", size(data["momentum_dist"]))
println("k_values: ", data["k_values"])
println("N = ", data["N"])

# Check if momentum distribution is changing
momentum_dist = transpose(data["momentum_dist"])  # Now (n_k, steps)
println("\nMomentum distribution at different steps:")
println("Step 1: ", momentum_dist[:, 1])
println("Step 10: ", momentum_dist[:, 10])
println("Step 21: ", momentum_dist[:, 21])

# Check if they're all the same
println("\nAre all steps identical?")
println("Step 1 ≈ Step 10? ", momentum_dist[:, 1] ≈ momentum_dist[:, 10])
println("Step 1 ≈ Step 21? ", momentum_dist[:, 1] ≈ momentum_dist[:, 21])

# Check energy evolution
println("\nEnergy evolution:")
println("Initial E/N = ", data["E_list"][1] / data["N"])
println("Final E/N = ", data["E_list"][end] / data["N"])

# Compute e_k for first and last step
J = data["J"]
h = data["h"]
N = data["N"]
k_values = [2π * k / N for k in data["k_values"]]
epsilon_k = [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_values]

e_k_initial = epsilon_k .* momentum_dist[:, 1]
e_k_final = epsilon_k .* momentum_dist[:, 21]

println("\ne_k values:")
println("Initial: ", e_k_initial)
println("Final: ", e_k_final)
println("Are they the same? ", e_k_initial ≈ e_k_final)