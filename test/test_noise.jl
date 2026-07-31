using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

@isdefined(test_mpo_to_matrix) || include("test_helpers.jl")

const NOISE_PAULI_X = ComplexF64[0 1; 1 0]
const NOISE_PAULI_Y = ComplexF64[0 -im; im 0]
const NOISE_PAULI_Z = ComplexF64[1 0; 0 -1]

"""RNG stub that replays a scripted draw sequence instead of sampling.

`rand(rng)` consumes the next `Float64` from `floats`; `rand(rng, 1:3)` consumes
the next index from `ints`. `n_floats`/`n_ints` expose how many draws the code
under test actually made, which is what pins the per-backend draw *count*.
"""
mutable struct ScriptedRNG <: AbstractRNG
    floats::Vector{Float64}
    ints::Vector{Int}
    n_floats::Int
    n_ints::Int
end

ScriptedRNG(floats::Vector{Float64}, ints::Vector{Int}=Int[]) =
    ScriptedRNG(floats, ints, 0, 0)

Base.rand(rng::ScriptedRNG, ::Type{Float64}) =
    (rng.n_floats += 1; rng.floats[rng.n_floats])

Base.rand(rng::ScriptedRNG, range::AbstractUnitRange{<:Integer}) =
    (rng.n_ints += 1; range[rng.ints[rng.n_ints]])

"""Dense 2x2 matrix of a single-site ITensor op, in the pinned MPO convention.

Wrapping the op in a one-site `MPO` reuses `test_mpo_to_matrix`, so rows are the
primed site index and columns the unprimed one. Transposing this would flip the
sign of the asserted σy.
"""
_single_site_op_matrix(name::String, s::Index) = test_mpo_to_matrix(MPO([op(name, s)]))

"""Single-site MPS with the given amplitudes, plus its dense state vector."""
function _single_site_mps(amplitudes::Vector{ComplexF64})
    sites = siteinds("S=1/2", 1)
    tensor = ITensor(ComplexF64, sites[1])
    tensor[sites[1] => 1] = amplitudes[1]
    tensor[sites[1] => 2] = amplitudes[2]
    return sites, MPS([tensor])
end

"""Dense state vector of a single-site MPS, in the `S=1/2` index order."""
_single_site_vector(ψ::MPS, s::Index) = ComplexF64[ψ[1][s => 1], ψ[1][s => 2]]

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

    @testset "TN noise op names are the Pauli matrices" begin
        s = siteinds("S=1/2", 1)[1]

        @test _single_site_op_matrix("σx", s) ≈ NOISE_PAULI_X atol=1e-14
        @test _single_site_op_matrix("σy", s) ≈ NOISE_PAULI_Y atol=1e-14
        @test _single_site_op_matrix("σz", s) ≈ NOISE_PAULI_Z atol=1e-14

        # The sign of the off-diagonal σy entries is the thing a transposed
        # row/column convention would silently flip.
        @test _single_site_op_matrix("σy", s)[1, 2] ≈ -im atol=1e-14
        @test _single_site_op_matrix("σy", s)[2, 1] ≈ +im atol=1e-14
    end

    @testset "TN depolarizing branch boundaries select the scripted Pauli" begin
        pe = 0.3
        first_boundary = pe / 3
        second_boundary = 2 * pe / 3
        amplitudes = ComplexF64[0.6, 0.8im]

        # One interior draw per branch, then each boundary probed from both
        # sides 1e-9 apart. `<` is strict everywhere, so a draw sitting exactly
        # on a boundary falls into the *next* branch. The two-sided probes are
        # what pin the thresholds themselves: an interior-only case list still
        # passes when `2 * pe / 3` is mutated to `pe / 2`.
        cases = (
            (0.05, NOISE_PAULI_X),
            (0.12, NOISE_PAULI_Y),
            (0.25, NOISE_PAULI_Z),
            (first_boundary - 1e-9, NOISE_PAULI_X),
            (first_boundary, NOISE_PAULI_Y),
            (second_boundary - 1e-9, NOISE_PAULI_Y),
            (second_boundary, NOISE_PAULI_Z),
            (pe - 1e-9, NOISE_PAULI_Z),
            (pe, ComplexF64[1 0; 0 1]),
            (0.7, ComplexF64[1 0; 0 1]),
        )

        for (draw, expected_pauli) in cases
            sites, ψ = _single_site_mps(amplitudes)
            rng = ScriptedRNG([draw])
            ψ_noisy = CoolingTNS.apply_depolarizing_noise(rng, ψ, sites, 1:1, pe)

            @test _single_site_vector(ψ_noisy, sites[1]) ≈
                expected_pauli * amplitudes atol=1e-14
            @test rng.n_floats == 1
        end
    end

    @testset "TN depolarizing draws once per position only" begin
        pe = 0.3
        sites = siteinds("S=1/2", 3)
        ψ = MPS(sites, "Up")

        # Two positions -> exactly two draws; the third site is untouched even
        # though its draw would have crossed the σx boundary.
        rng = ScriptedRNG([0.7, 0.05])
        ψ_noisy = CoolingTNS.apply_depolarizing_noise(rng, ψ, sites, [1, 3], pe)

        @test rng.n_floats == 2
        @test abs(inner(ψ_noisy, MPS(sites, ["Up", "Up", "Dn"]))) ≈ 1.0 atol=1e-12
    end

    @testset "ED depolarizing consumes one uniform draw plus one Pauli index" begin
        # ED samples the *real* representation Y_real = [0 -1; 1 0] rather than
        # the complex σy; the two differ by a global -i on a pure state and are
        # identical as a channel, so the ED assertion must use `pauli_y`.
        @test CoolingTNS.PAULI_OPERATORS ===
            (CoolingTNS.pauli_x, CoolingTNS.pauli_y, CoolingTNS.pauli_z)

        p = 0.3
        ψ = CoolingTNS.EDStateVector(ComplexF64[0.6, 0.8im], 1)

        for (pauli_index, builder) in enumerate(CoolingTNS.PAULI_OPERATORS)
            rng = ScriptedRNG([0.05], [pauli_index])
            noisy = CoolingTNS.apply_depolarizing_ed(ψ, p, [1], rng)

            expected = Matrix(builder(1, 1)) * ψ.data
            @test noisy.data ≈ normalize(expected) atol=1e-14
            @test rng.n_floats == 1
            @test rng.n_ints == 1
        end

        # A draw at or above `p` applies nothing and consumes no Pauli index.
        for draw in (p, 0.7)
            rng = ScriptedRNG([draw], Int[])
            noisy = CoolingTNS.apply_depolarizing_ed(ψ, p, [1], rng)

            @test noisy.data ≈ ψ.data atol=1e-14
            @test rng.n_floats == 1
            @test rng.n_ints == 0
        end
    end
end
