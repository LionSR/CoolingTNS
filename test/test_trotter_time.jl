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

@testset "TN TDVP sweep observer diagnostics" begin
    sites = siteinds("S=1/2", 2)
    ψ0 = MPS(sites, ["Up", "Up"])
    ham_params = CoolingTNS.IsingParameters(2, 0.0, 0.0)
    sim_tdvp = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.MonteCarloWavefunction(),
        CoolingTNS.ContinuousEvolution();
        Dmax=8,
        cutoff=1e-12,
        tau=0.1,
    )
    os = OpSum()
    os += 1.0, "X", 1
    H = MPO(os, sites)

    sweeps = Int[]
    times = Float64[]
    maxdims = Int[]
    observer = CoolingTNS.tdvp_sweep_observer((; state, sweep, current_time, kwargs...) -> begin
        push!(sweeps, sweep)
        push!(times, -imag(current_time))
        push!(maxdims, maxlinkdim(state))
        return nothing
    end)

    ψ_evolved = CoolingTNS.evolve_state(
        ham_params, sim_tdvp, CoolingTNS.TNBackend(), H, ψ0, 0.2, sites;
        tdvp_sweep_observer! = observer,
    )

    @test length(ψ_evolved) == 2
    @test sweeps == [1, 2]
    @test times ≈ [0.1, 0.2] atol=1e-12
    @test all(>=(1), maxdims)
end

function _x_gate(site, dt)
    return [exp(-1.0im * dt * op("X", site))]
end

function _x_hamiltonian_ed()
    return ComplexF64[0 1; 1 0]
end

@testset "TN Trotter requested-time slicing" begin
    @test CoolingTNS.trotter_time_slices(0.0, 0.2) == (0, 0.0)
    @test CoolingTNS.trotter_time_slices(0.19, 0.2) == (1, 0.19)
    @test CoolingTNS.trotter_time_slices(0.7, 0.3) == (3, 0.7 / 3)
    @test_throws ArgumentError CoolingTNS.trotter_time_slices(-0.1, 0.2)
    @test_throws ArgumentError CoolingTNS.trotter_time_slices(0.1, 0.0)

    for (t, tau) in [(0.19, 0.2), (0.4, 0.2), (0.7, 0.3)]
        steps, dt = CoolingTNS.trotter_time_slices(t, tau)
        @test steps * dt ≈ t atol=1e-14
        @test dt <= tau + 1e-12
    end

    H_ed = _x_hamiltonian_ed()
    ψ_ed = CoolingTNS.EDStateVector(ComplexF64[1, 0], 1)
    ρ_ed = CoolingTNS.state_to_density_ed(ψ_ed)

    @test CoolingTNS.evolve_cooling_step_ed(H_ed, ψ_ed, 0.0, 0.2).data == ψ_ed.data
    @test CoolingTNS.evolve_cooling_step_ed(H_ed, ρ_ed, 0.0, 0.2).data == ρ_ed.data
    @test_throws ArgumentError CoolingTNS.evolve_cooling_step_ed(H_ed, ψ_ed, -0.1, 0.2)

    for (t, tau) in [(0.19, 0.2), (0.7, 0.3)]
        exact_ψ = CoolingTNS.evolve_ed(H_ed, ψ_ed, t)
        stepped_ψ = CoolingTNS.evolve_cooling_step_ed(H_ed, ψ_ed, t, tau)
        @test abs(abs(dot(exact_ψ.data, stepped_ψ.data)) - 1) < 1e-12

        exact_ρ = CoolingTNS.evolve_ed(H_ed, ρ_ed, t)
        stepped_ρ = CoolingTNS.evolve_cooling_step_ed(H_ed, ρ_ed, t, tau)
        @test norm(exact_ρ.data - stepped_ρ.data) < 1e-12
    end

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
