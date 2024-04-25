using HDF5
using PythonCall
using LaTeXStrings

function plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename; moving_average=false)
    plt = pyimport("matplotlib.pyplot")
    steps = length(E_list) - 1

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    ax.plot(1:steps+1, E_list / N, alpha=0.75, marker="o", label="Cooling")
    if moving_average
        window_size = 10
        E_ma = [mean(E_list[max(1, i - window_size + 1):i]) for i in 1:length(E_list)]
        ax.plot(1:steps+1, E_ma / N, alpha=0.75, marker="o", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel(L"Energy density $E/N$")
    ax.axhline(y=e₀ / N, xmin=0, xmax=1, linewidth=1.5, color="black", label=L"$E_0/N$")
    ax.legend()

    ax = axs[1]
    ax.plot(1:steps+1, GS_overlap_list, marker="o", alpha=0.75, color="grey", label="Cooling")
    if moving_average
        GS_ma = [mean(GS_overlap_list[max(1, i - window_size + 1):i]) for i in 1:length(GS_overlap_list)]
        ax.plot(1:steps+1, GS_ma, marker="o", alpha=0.75, color="black", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel("Ground state overlap")
    ax.legend()

    fig.savefig("Results/$(filename).pdf", dpi=300)
end


function plot_energy_error_and_overlap_vs_N(ham_name, coupling_params, sim_params, N_values)
    plt = pyimport("matplotlib.pyplot")

    energy_errors = Float64[]
    final_overlaps = Float64[]

    method = sim_params["method"]  # Ensure this key exists and correctly reflects the method used (MPS or MPO)
    coupling_name_part = "Coupling$(coupling_params["coupling"])g$(coupling_params["g"])te$(coupling_params["te"])steps$(coupling_params["steps"])"

    if method == "MPO"
        sim_name_part = "Sim$(method)tau$(sim_params["tau"])"
    else  # Assuming the other method is MPS
        sim_name_part = "Sim$(method)Dmax$(sim_params["Dmax"])"
    end
    sim_params["pe"] > 0 && (sim_name_part *= "pe$(sim_params["pe"])")

    for N in N_values
        ham_name_part = "Ham$(ham_name)Ns$(N)Nb$(N)"
        filename = "Cooling_$(ham_name_part)_$(coupling_name_part)_$(sim_name_part).h5"

        h5open("Results/" * filename, "r") do file
            e₀ = read(file, "e₀")
            E_final = read(file, "E_final")
            GS_overlap_final = read(file, "GS_overlap_final")
            push!(energy_errors, abs(E_final / N - e₀ / N))
            push!(final_overlaps, GS_overlap_final)
        end
    end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(N_values, energy_errors, marker="o", linestyle="-", label="Energy error")
    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density error")
    ax1.legend()

    ax2.plot(N_values, final_overlaps, marker="o", linestyle="-", label="Final overlap")
    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    ham_name_part = "Ham$(ham_name)"
    filename_saveto = "Cooling_$(ham_name_part)_$(coupling_name_part)_$(sim_name_part)_energy_error_and_overlap_vs_N.pdf"

    fig.savefig("Results/" * filename_saveto, dpi=300)
end


function plot_energy_error_and_overlap_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, peInt_range)
    plt = pyimport("matplotlib.pyplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for peInt in peInt_range
        pe = peInt * 1e-3
        pe = round(pe, digits=4)
        sim_params["pe"] = pe

        energy_errors = Float64[]
        final_overlaps = Float64[]

        method = sim_params["method"]
        coupling_name_part = "Coupling$(coupling_params["coupling"])g$(coupling_params["g"])te$(coupling_params["te"])steps$(coupling_params["steps"])"

        if method == "MPO"
            sim_name_part = "Sim$(method)tau$(sim_params["tau"])"
        else  # Assuming the other method is MPS
            sim_name_part = "Sim$(method)Dmax$(sim_params["Dmax"])"
        end
        if pe > 0
            sim_name_part *= "pe$(pe)"
        end

        for N in N_values
            ham_name_part = "Ham$(ham_name)Ns$(N)Nb$(N)"
            filename = "Cooling_$(ham_name_part)_$(coupling_name_part)_$(sim_name_part).h5"

            h5open("Results/" * filename, "r") do file
                e₀ = read(file, "e₀")
                E_final = read(file, "E_final")
                GS_overlap_final = read(file, "GS_overlap_final")
                push!(energy_errors, abs(E_final / N - e₀ / N))
                push!(final_overlaps, GS_overlap_final)
            end
        end

        ax1.plot(N_values, energy_errors, marker="o", linestyle="-", label="pe=$pe")
        ax2.plot(N_values, final_overlaps, marker="o", linestyle="-", label="pe=$pe")
    end

    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density error")
    ax1.legend()

    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    filename_saveto = "Cooling_Ham$(ham_name)_energy_error_and_overlap_vs_N_multiple_pe.pdf"
    fig.savefig("Results/" * filename_saveto, dpi=300)
end