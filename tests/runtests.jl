using Test
using CoolingTNS

# Set up test environment
@testset "CoolingTNS Tests" begin
    # Run individual test files
    include("test_hamiltonians.jl")
    include("test_initial_states.jl") 
    include("test_cooling_interface.jl")
end

# Summary
println("\nAll tests completed!")