using Pkg
Pkg.activate(".")  # Activate the current project

using ITensors
using PackageCompiler

# Precompile ITensors
PackageCompiler.create_sysimage(
    [:ITensors];
    sysimage_path="ITensors_sysimage.so",
    precompile_execution_file="precompile_itensors_execution.jl"
)

println("ITensors precompilation completed. Sysimage saved as 'ITensors_sysimage.so'")
