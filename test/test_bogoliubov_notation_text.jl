using Test

@testset "Bogoliubov angle and occupation notation in source comments" begin
    source_files = [
        joinpath(@__DIR__, "..", "src", "ed_backend.jl"),
        joinpath(@__DIR__, "..", "src", "ed_backend_complex_jw.jl"),
        joinpath(@__DIR__, "..", "src", "mode_analysis.jl"),
        joinpath(@__DIR__, "..", "src", "dispersion.jl"),
        joinpath(@__DIR__, "..", "src", "result_keys.jl"),
        joinpath(@__DIR__, "..", "src", "cooling_evolution.jl"),
        joinpath(@__DIR__, "..", "src", "cooling_evolution_ed_shared.jl"),
        joinpath(@__DIR__, "..", "src", "tn_mode_observables.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "PlotUtils.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_nk_evolution.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_dispersion_with_gs.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_mode_cooling.jl"),
    ]

    for file in source_files
        text = read(file, String)
        @test !occursin("sin²(φ_k)", text)
    end

    joined = join(read.(source_files, String), "\n")
    joined_flat = replace(joined, r"\s+" => " ")
    plotutils_text = read(joinpath(@__DIR__, "..", "scripts", "plotting", "PlotUtils.jl"), String)
    @test occursin("sin²(varphi_k)", joined)
    @test occursin("φ_k = 2πk/N", joined)
    @test occursin("n_k^{Bog}", joined)
    @test occursin("Bogoliubov occupation number ``n_k^{Bog}``", joined)
    @test occursin("Bogoliubov occupations, ⟨h_k⟩", joined)
    @test occursin("Σ_k ε_k (n_k^{OBC} - 1/2)", joined)
    @test occursin("Bogoliubov mode observable ``⟨h_k⟩``", joined)
    @test occursin("positive quasiparticle gaps used for resonance labels", joined_flat)
    @test occursin("measure_all_mode_observables", joined)
    @test occursin("Compatibility wrapper for [`measure_all_mode_observables`](@ref)", joined)
    @test occursin("k_indices, hk_values, εk_values = measure_all_mode_observables", joined)
    @test occursin("Historical constant name: the HDF5 dataset is `mode_ek_values` and stores", joined_flat)
    @test occursin("positive quasiparticle gaps ε_k for resonance labels", joined_flat)
    @test occursin("The historical name is retained for existing callers. New code should prefer `measure_all_mode_observables`", joined_flat)
    @test occursin("not signed energy-reconstruction coefficients", joined_flat)
    @test occursin("H_code = Λ · U H_notes U†", joined_flat)
    @test occursin("Λ = 2√(J²+h²)", joined_flat)
    @test occursin("code-basis state is first rotated to notes coordinates by ``U†``", joined_flat)
    @test occursin("JW operators above are evaluated as notes-basis Pauli strings", joined_flat)
    @test occursin("measure_raw_fourier_occupation_ed", joined)
    @test occursin("raw Fourier occupation ``tilde n_k``", joined_flat)
    @test occursin("compute_bdg_reference_occupation", joined)
    @test occursin("parity-unconstrained BdG reference value", joined_flat)
    @test occursin("mode-wise energy-minimizing BdG reference", joined_flat)
    @test occursin("RAW_FOURIER_BDG_REFERENCE_OCCUPATION_LABEL", joined)
    @test occursin("RAW_FOURIER_GS_OCCUPATION_LABEL = RAW_FOURIER_BDG_REFERENCE_OCCUPATION_LABEL", joined)
    @test occursin("compute_ground_state_occupation", plotutils_text)
    @test occursin("local real-space occupation is ``a_n^†a_n = (1 + σ^z_n)/2``", joined_flat)
    @test occursin("The returned ``tilde n_k`` is instead the Fourier correlator sum over ``⟨a_m^†a_n⟩``", joined_flat)
    @test occursin("Compatibility wrapper for [`measure_raw_fourier_occupation_ed`](@ref)", joined)
    @test occursin("New code should prefer `measure_raw_fourier_occupation_ed`", joined_flat)

    forbidden = [
        "(ε_k, n_k, ⟨h_k⟩)",
        "h_k = 2 n_k - 1",
        "h_k = 2n_k - 1",
        "Σ_k ε_k (n_k - 1/2)",
        "mode energies, occupations, ⟨h_k⟩",
        "Mode energy measurement: h_k",
        "mode energy observable",
        "Mode energy measurements ⟨h_k⟩",
        "does not affect the JW operators",
        "tilde n_k = " * "(1 + ⟨σ_z⟩)/2`` per mode",
        "RAW_FOURIER_BDG_" * "VACUUM_OCCUPATION_LABEL",
        "compute_bogoliubov_" * "vacuum_occupation",
        "equal to the Bogoliubov-vacuum expectation",
        "not necessarily the strict all-quasiparticle-empty",
        raw"\tilde n_k^" * raw"{\mathrm{GS}}",
        raw"\tilde n_k^" * raw"{\mathrm{BdG\,vac}}",
    ]
    @test all(phrase -> !occursin(phrase, joined), forbidden)
end
