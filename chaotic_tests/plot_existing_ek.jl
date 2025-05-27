using CoolingTNS

# Use the existing Monte Carlo simulation result
mc_file = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

if isfile(mc_file)
    println("Plotting e_k evolution for existing MC simulation...")
    CoolingTNS.plot_ek_evolution(mc_file; save_fig=true)
else
    println("File not found: $mc_file")
end