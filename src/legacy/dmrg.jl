using ITensors
using ITensorMPS

PROJ_WEIGHT = 15.0
MAXDIM = [10, 10, 10, 20, 20]
SWEEPS = Sweeps(30)
CUTOFF = 1E-8
NOISE = 1E-6



function compute_energy_gap_and_ground_state(
    H_sys, 
    sites_sys; 
    excite_num=1, 
    proj_weight=PROJ_WEIGHT, 
    sweeps=SWEEPS, 
    maxdim=MAXDIM, 
    cutoff=CUTOFF, 
    noise=NOISE
)
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
    sweeps=SWEEPS, 
    maxdim=MAXDIM, 
    cutoff=CUTOFF, 
    noise=NOISE
)
    maxdim!(sweeps, maxdim...)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, noise)
    
    # DMRG calculation for ground state
    e₀, ϕ₀ = dmrg(H_sys, randomMPS(sites_sys), sweeps; outputlevel=false)
    
    return e₀, ϕ₀
end