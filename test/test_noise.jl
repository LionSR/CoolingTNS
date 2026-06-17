using Test
using CoolingTNS
using LinearAlgebra

function _pauli_error_choices(q::Int, n_qubits::Int, p::Float64)
    dim = 2^n_qubits
    identity = Matrix{ComplexF64}(I, dim, dim)
    return (
        (1 - p, identity),
        (p / 3, Matrix{ComplexF64}(CoolingTNS.pauli_x(q, n_qubits))),
        (p / 3, Matrix{ComplexF64}(CoolingTNS.pauli_y(q, n_qubits))),
        (p / 3, Matrix{ComplexF64}(CoolingTNS.pauli_z(q, n_qubits))),
    )
end

function _enumerated_local_pauli_channel(ρ::CoolingTNS.EDDensityMatrix, p::Float64, qubits)
    dim = 2^ρ.n_qubits
    choices = [_pauli_error_choices(q, ρ.n_qubits, p) for q in qubits]
    averaged = zeros(ComplexF64, dim, dim)

    for error_history in Iterators.product(choices...)
        weight = 1.0
        op = Matrix{ComplexF64}(I, dim, dim)
        for (local_weight, local_op) in error_history
            weight *= local_weight
            op = local_op * op
        end
        averaged .+= weight .* (op * ρ.data * op')
    end

    return CoolingTNS.EDDensityMatrix(averaged, ρ.n_qubits)
end

@testset "Local Depolarizing Noise Convention" begin
    @testset "ED density matrices equal enumerated local Pauli averages" begin
        ψ = CoolingTNS.EDStateVector(
            ComplexF64[
                sqrt(0.2),
                0.3im,
                -0.4,
                sqrt(0.55),
            ],
            2,
        )
        ρ = CoolingTNS.state_to_density_ed(ψ)
        p = 0.27

        noisy = CoolingTNS.apply_depolarizing_ed(ρ, p, 1:ρ.n_qubits)
        enumerated = _enumerated_local_pauli_channel(ρ, p, 1:ρ.n_qubits)

        @test noisy.data ≈ enumerated.data atol=1e-12
        @test tr(noisy.data) ≈ 1.0 atol=1e-12
        @test ishermitian(noisy.data)

        dim = 2^ρ.n_qubits
        global_depolarized = (1 - p) * ρ.data + p * Matrix{ComplexF64}(I, dim, dim) / dim
        @test !isapprox(noisy.data, global_depolarized; atol=1e-8)
    end

    @testset "ED cooling apply_noise uses the same local channel" begin
        backend = CoolingTNS.EDBackend()
        ham_params = CoolingTNS.IsingParameters(1, 1.0, -2.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 0.2, nothing)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0,
        )
        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        ψ = CoolingTNS.EDStateVector(ComplexF64[1 / 2, im / 2, -1 / 2, 1 / 2], 2)
        ρ = CoolingTNS.state_to_density_ed(ψ)
        p = 0.41

        noisy_from_cooling = CoolingTNS.apply_noise(ρ, problem, p)
        noisy_direct = CoolingTNS.apply_depolarizing_ed(ρ, p, 1:ρ.n_qubits)

        @test noisy_from_cooling.data ≈ noisy_direct.data atol=1e-12
    end
end
