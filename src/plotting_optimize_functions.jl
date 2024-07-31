using HDF5
using PythonCall
using LaTeXStrings

function safe_read_data(filename)
    try
        h5open(filename, "r") do file
            E_final_density = read(file, "Final energy density")
            GS_overlap_final = read(file, "Final ground state overlap")
            return E_final_density, GS_overlap_final
        end
    catch e
        @warn "Failed to read data from $filename: $e"
        return nothing, nothing
    end
end

function plotOptimize_energy_error_and_overlap_vs_N(ham_name, coupling_params, sim_params, search_params, N_values, e₀)
    plt = pyimport("matplotlib.pyplot")

    energies = Float64[]
    final_overlaps = Float64[]
    valid_N_values = Int[]

    search_name_part = create_search_name_part(search_params)

    for N in N_values
        filename = create_filename(ham_name, N, coupling_params, sim_params)
        filename = "Optimize$(filename)_$(search_name_part)"
        full_filename = "ResultsOpt/" * filename * ".h5"

        E_final_density, GS_overlap_final = safe_read_data(full_filename)
        if E_final_density !== nothing && GS_overlap_final !== nothing
            push!(energies, E_final_density)
            push!(final_overlaps, GS_overlap_final)
            push!(valid_N_values, N)
        end
    end

    if isempty(valid_N_values)
        @error "No valid data points found. Skipping plot generation."
        return
    end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(valid_N_values, energies, marker="o", linestyle="-", label="Energy")
    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density")
    ax1.axhline(y=e₀ / 100, xmin=0, xmax=1, linewidth=1.5, color="black", label=L"$E_0/N$")
    ax1.legend()

    ax2.plot(valid_N_values, final_overlaps, marker="o", linestyle="-", label="Final overlap")
    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    filename_saveto = create_filename(ham_name, valid_N_values[1], coupling_params, sim_params)
    filename_saveto = "Optimize$(filename_saveto)_$(search_name_part)_energy_and_overlap_vs_N.pdf"

    isdir("ResultsOpt/Figs") || mkpath("ResultsOpt/Figs")
    fig.savefig("ResultsOpt/Figs/" * filename_saveto, dpi=300)
end

function plotOptimize_energy_error_and_overlap_vs_N_pe_range(ham_name, coupling_params, sim_params, search_params, N_values, peInt_range, e₀)
    plt = pyimport("matplotlib.pyplot")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    search_name_part = "Search$(search_params["search_method"])trials$(search_params["num_trials"])"

    for peInt in peInt_range
        pe = peInt * 1e-3
        pe = round(pe, digits=4)
        sim_params["pe"] = pe
        sim_params["peInt"] = peInt

        energy_errors = Float64[]
        final_overlaps = Float64[]
        valid_N_values = Int[]

        for N in N_values
            filename = create_filename(ham_name, N, coupling_params, sim_params)
            filename = "Optimize$(filename)_$(search_name_part)"
            full_filename = "ResultsOpt/" * filename * ".h5"

            E_final_density, GS_overlap_final = safe_read_data(full_filename)
            if E_final_density !== nothing && GS_overlap_final !== nothing
                push!(energy_errors, abs(E_final_density - e₀ / 100))
                push!(final_overlaps, GS_overlap_final)
                push!(valid_N_values, N)
            end
        end

        if !isempty(valid_N_values)
            ax1.plot(valid_N_values, energy_errors, marker="o", linestyle="-", label="pe=$pe")
            ax2.plot(valid_N_values, final_overlaps, marker="o", linestyle="-", label="pe=$pe")
        else
            @warn "No valid data points found for pe=$pe. Skipping this pe value."
        end
    end

    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density error")
    ax1.legend()

    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    filename_saveto = create_filename(ham_name, N_values, coupling_params, sim_params)
    filename_saveto = "Optimize$(filename_saveto)_$(search_name_part)_energy_error_and_overlap_vs_N_multiple_pe.pdf"

    isdir("ResultsOpt/Figs") || mkpath("ResultsOpt/Figs")
    fig.savefig("ResultsOpt/Figs/" * filename_saveto, dpi=300)
end
