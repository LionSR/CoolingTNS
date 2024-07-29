using ITensors
using PackageCompiler

ITensors.compile()

println("ITensors precompilation completed. Sysimage saved as 'ITensors_sysimage.so'")
