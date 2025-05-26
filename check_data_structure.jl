using HDF5

filename = "Results/Cooling_HamIsingJ1.0h2.0_CouplingXXg0.3te2.0steps20_SimEDMC.h5"

h5open(filename, "r") do file
    println("Keys in file:")
    for key in keys(file)
        data = read(file, key)
        if isa(data, Array)
            println("  $key: $(typeof(data)), size = $(size(data))")
        else
            println("  $key: $(typeof(data)), value = $data")
        end
    end
end