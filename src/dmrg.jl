using ITensors

function compute_energy_gap_and_ground_state(
    H_sys, 
    sites_sys; 
    excite_num=1, 
    proj_weight=15.0, 
    sweeps=Sweeps(30), 
    maxdim=[10, 10, 10, 20, 20], 
    cutoff=1E-8, 
    noise=1E-6
)
    # Set sweep parameters
    maxdim!(sweeps, maxdim...)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, noise)

    # DMRG calculations for ground and first excited states
    e₀, ϕ₀ = dmrg(H_sys, randomMPS(sites_sys), sweeps; outputlevel=false)
    e₁ = dmrg(H_sys, [ϕ₀], randomMPS(sites_sys), sweeps; outputlevel=false, weight=proj_weight)[1]
    
    # Energy gap calculation
    Δ = e₁ - e₀
    
    return Δ, e₀, ϕ₀
end

function compute_ground_state(
    H_sys, 
    sites_sys; 
    sweeps=Sweeps(30), 
    maxdim=[10, 10, 10, 20, 20], 
    cutoff=1E-8, 
    noise=1E-6
)
    # Set sweep parameters
    maxdim!(sweeps, maxdim...)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, noise)
    
    # DMRG calculation for ground state
    e₀, ϕ₀ = dmrg(H_sys, randomMPS(sites_sys), sweeps; outputlevel=false)
    
    return e₀, ϕ₀
end