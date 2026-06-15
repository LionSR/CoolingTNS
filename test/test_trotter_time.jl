using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

function _test_mps_vector(ψ::MPS, sites)
    dim = 2^length(sites)
    vec = zeros(ComplexF64, dim)
    for idx in 0:(dim - 1)
        config = [((idx >> (site - 1)) & 1) == 0 ? "Up" : "Dn" for site in eachindex(sites)]
        vec[idx + 1] = inner(MPS(sites, config), ψ)
    end
    return vec
end

function _x_gate(site, dt)
    return [exp(-1.0im * dt * op("X", site))]
end

@testset "TN Trotter requested-time slicing" begin
    @test CoolingTNS.trotter_time_slices(0.0, 0.2) == (0, 0.0)
    @test CoolingTNS.trotter_time_slices(0.19, 0.2) == (1, 0.19)
    @test CoolingTNS.trotter_time_slices(0.7, 0.3) == (3, 0.7 / 3)

    sites = siteinds("S=1/2", 1)
    ψ0 = MPS(sites, "Up")
    ρ0 = outer(ψ0', ψ0)
    ham_params = CoolingTNS.IsingParameters(1, 0.0, 0.0)

    for (t, tau) in [(0.19, 0.2), (0.7, 0.3)]
        sim_mps = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.TrotterEvolution();
            Dmax=20,
            cutoff=1e-14,
            tau=tau,
        )
        sim_mpo = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.TrotterEvolution();
            Dmax=20,
            cutoff=1e-14,
            tau=tau,
        )

        base_gates = _x_gate(sites[1], tau)
        step_gates = dt -> _x_gate(sites[1], dt)
        exact_state = apply(_x_gate(sites[1], t), ψ0; cutoff=1e-14, maxdim=20)

        ψ_evolved = CoolingTNS.evolve_state(
            ham_params, sim_mps, CoolingTNS.TNBackend(), nothing, ψ0, t, sites;
            gates=base_gates, step_gates=step_gates,
        )
        @test abs(abs(dot(_test_mps_vector(exact_state, sites), _test_mps_vector(ψ_evolved, sites))) - 1) < 1e-10

        ρ_evolved = CoolingTNS.evolve_state(
            ham_params, sim_mpo, CoolingTNS.TNBackend(), base_gates, ρ0, t, sites;
            step_gates=step_gates,
        )
        @test real(inner(exact_state', ρ_evolved, exact_state)) ≈ 1.0 atol=1e-10
    end
end
