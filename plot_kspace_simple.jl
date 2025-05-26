using CoolingTNS

# Generate k-space plots
filename = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

println("Generating k-space plot...")
CoolingTNS.plot_momentum_distribution(filename; save_fig=true)
println("K-space line plot saved.")

println("Generating k-space heatmap...")
CoolingTNS.plot_momentum_distribution_heatmap(filename; save_fig=true)
println("K-space heatmap saved.")