using HDF5
using PythonCall
using LaTeXStrings

function safe_read_data(filename)
    try
        h5open(filename, "r") do file
            e₀ = read(file, "e₀")
            E_final = read(file, "E_final")
            GS_overlap_final = read(file, "GS_overlap_final")
            return e₀, E_final, GS_overlap_final
        end
    catch e
        @warn "Failed to read data from $filename: $e"
        return nothing, nothing, nothing
    end
end

function plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename; moving_average=false)
    plt = pyimport("matplotlib.pyplot")

    steps = length(E_list) - 1

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    ax.plot(1:steps+1, E_list / N, alpha=0.75, marker="o", label="Cooling")
    if moving_average
        window_size = 10
        E_ma = [CoolingTNS.mean_last_window(E_list[1:i], window_size) for i in 1:length(E_list)]
        ax.plot(1:steps+1, E_ma / N, alpha=0.75, marker="o", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel(L"Energy density $E/N$")
    ax.axhline(y=e₀ / N, xmin=0, xmax=1, linewidth=1.5, color="black", label=L"$E_0/N$")
    ax.legend()

    ax = axs[1]
    ax.plot(1:steps+1, GS_overlap_list, marker="o", alpha=0.75, color="grey", label="Cooling")
    if moving_average
        GS_ma = [CoolingTNS.mean_last_window(GS_overlap_list[1:i], window_size) for i in 1:length(GS_overlap_list)]
        ax.plot(1:steps+1, GS_ma, marker="o", alpha=0.75, color="black", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel("Ground state overlap")
    ax.legend()

    isdir("Results") || mkdir("Results")
    fig.savefig("Results/$(filename).pdf", dpi=300)
end

function plot_energy_error_and_overlap_vs_N(ham_name, coupling_params, sim_params, N_values)
    plt = pyimport("matplotlib.pyplot")

    energy_errors = Float64[]
    final_overlaps = Float64[]
    valid_N_values = Int[]

    for N in N_values
        filename = create_filename(ham_name, N, coupling_params, sim_params)
        full_filename = "Results/" * filename * ".h5"

        e₀, E_final, GS_overlap_final = safe_read_data(full_filename)
        if e₀ !== nothing && E_final !== nothing && GS_overlap_final !== nothing
            push!(energy_errors, abs(E_final / N - e₀ / N))
            push!(final_overlaps, GS_overlap_final)
            push!(valid_N_values, N)
        end
    end

    if isempty(valid_N_values)
        @error "No valid data points found. Skipping plot generation."
        return
    end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(valid_N_values, energy_errors, marker="o", linestyle="-", label="Energy error")
    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density error")
    ax1.legend()

    ax2.plot(valid_N_values, final_overlaps, marker="o", linestyle="-", label="Final overlap")
    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    filename_saveto = create_filename(ham_name, valid_N_values[1], coupling_params, sim_params)
    filename_saveto = "$(filename_saveto)_energy_error_and_overlap_vs_N.pdf"

    isdir("Results/Figs") || mkpath("Results/Figs")
    fig.savefig("Results/Figs/" * filename_saveto, dpi=300)
end

function plot_energy_error_and_overlap_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, peInt_range)
    plt = pyimport("matplotlib.pyplot")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

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
            full_filename = "Results/" * filename * ".h5"

            e₀, E_final, GS_overlap_final = safe_read_data(full_filename)
            if e₀ !== nothing && E_final !== nothing && GS_overlap_final !== nothing
                push!(energy_errors, abs(E_final / N - e₀ / N))
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
    filename_saveto = "$(filename_saveto)_energy_error_and_overlap_vs_N_multiple_pe.pdf"

    isdir("Results/Figs") || mkpath("Results/Figs")
    fig.savefig("Results/Figs/" * filename_saveto, dpi=300)
end
