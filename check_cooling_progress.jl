using HDF5

# Read existing data
filename = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

data = h5open(filename, "r") do file
    Dict(
        "E_list" => read(file, "E_list"),
        "GS_overlap_list" => read(file, "GS_overlap_list"),
        "bath_mag_list" => read(file, "bath_mag_list"),
        "e₀" => read(file, "e₀"),
        "N" => read(file, "N")
    )
end

N = data["N"]
E_list = data["E_list"]
GS_overlap = data["GS_overlap_list"]
bath_mag = data["bath_mag_list"]
e₀ = data["e₀"]

println("Cooling progress:")
println("================")
println("Ground state energy: e₀/N = ", e₀/N)
println("\nStep  | E/N      | ΔE from GS | GS overlap | Bath mag")
println("------|----------|------------|------------|----------")
for i in [1, 5, 10, 15, 21]
    E_per_site = E_list[i]/N
    ΔE = E_per_site - e₀/N
    println("$(lpad(i-1, 3))   | $(round(E_per_site, digits=4)) | $(round(ΔE, digits=4))    | $(round(GS_overlap[i], digits=4))   | $(round(bath_mag[i], digits=4))")
end

println("\nIs energy decreasing? ", E_list[1] > E_list[end])
println("Energy change: ", (E_list[end] - E_list[1])/N)

# Check if this is a Monte Carlo simulation
println("\n[Note: This is a Monte Carlo wavefunction simulation, not density matrix]")