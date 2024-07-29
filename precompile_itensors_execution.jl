using ITensors

# Add some common ITensors operations here
let
    N = 10
    sites = siteinds("S=1/2", N)
    psi = randomMPS(sites)
    H = randomMPO(sites)
    dmrg(H, psi; nsweeps=5)
end

println("Precompile execution completed")
