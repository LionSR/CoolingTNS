# Uses shared utilities from plot_utils.jl (included by main module)

"""
    safe_read_data(filename) -> Tuple

Read cooling results from an HDF5 file.
Returns (e0, E_list, GS_overlap_list, Edensity_final) or (nothing, nothing, nothing, nothing) on failure.
"""
function safe_read_data(filename)
    if !isfile(filename)
        @warn "File not found: $filename"
        return nothing, nothing, nothing, nothing
    end

    try
        h5open(filename, "r") do file
            e0 = read(file, "e₀")
            E_list = read(file, "E_list")
            GS_overlap_list = read(file, "GS_overlap_list")
            Edensity_final = read(file, "Edensity_final")
            return e0, E_list, GS_overlap_list, Edensity_final
        end
    catch e
        msg = if isa(e, KeyError)
            "Missing key: $(e.key)"
        elseif isa(e, HDF5.HDF5Error)
            "HDF5 error: $(e.msg)"
        else
            "Error: $e"
        end
        @warn "Failed to read $filename: $msg"
        return nothing, nothing, nothing, nothing
    end
end

function plot_energy_and_overlap(E_list, GS_overlap_list, e0, N, filename; moving_average=false)
    plt = get_pyplot()

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
    ax.axhline(y=e0 / N, xmin=0, xmax=1, linewidth=1.5, color="black", label=L"$E_0/N$")
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

function plot_vs_N(ham_name::String, coupling_params::CouplingParameters, sim_params::UnifiedSimulationParameters,
                  backend::CoolingBackend, N_values::Vector{Int}; is_optimization=false)
    plt = get_pyplot()

    energy_densities = Float64[]
    final_overlaps = Float64[]
    valid_N_values = Int[]
    e0_values = Float64[]

    directory = is_optimization ? "ResultsOpt" : "Results"
    prefix = is_optimization ? "Optimize" : ""

    for N in N_values
        # Create proper HamiltonianParameters for filename generation
        # Infer problem type from ham_name
        if occursin("niIsing", ham_name)
            # Parse parameters from ham_name if possible, or use defaults
            ham_params = NiIsingParameters(N, 1.0, -1.05, 0.5)
        elseif occursin("Ising", ham_name)
            ham_params = IsingParameters(N, 1.0, 2.0)
        else
            ham_params = RydbergParameters(N, 1.0, 0.0, 1.0)
        end
        
        filename = create_filename(ham_params, coupling_params, sim_params, backend)
        filename = "$(prefix)$(filename)"
        if is_optimization
            search_name_part = create_search_name_part(sim_params)
            filename *= "_$(search_name_part)"
        end
        full_filename = "$(directory)/$(filename).h5"

        e0, E_final, GS_overlap_final, Edensity_final = safe_read_data(full_filename)
        if e0 !== nothing && E_final !== nothing && GS_overlap_final !== nothing && Edensity_final !== nothing
            push!(energy_densities, Edensity_final)
            push!(final_overlaps, GS_overlap_final)
            push!(valid_N_values, N)
            push!(e0_values, e0)
        end
    end

    if isempty(valid_N_values)
        @error "No valid data points found. Skipping plot generation."
        return
    end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(valid_N_values, energy_densities, marker="o", linestyle="-", label="Energy density")
    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density")
    ax1.plot(valid_N_values, e0_values ./ valid_N_values, linestyle="--", color="black", label=L"$E_0/N$")
    ax1.legend()

    ax2.plot(valid_N_values, final_overlaps, marker="o", linestyle="-", label="Final overlap")
    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    # Use first valid N for filename
    if occursin("niIsing", ham_name)
        ham_params = NiIsingParameters(valid_N_values[1], 1.0, -1.05, 0.5)
    elseif occursin("Ising", ham_name)
        ham_params = IsingParameters(valid_N_values[1], 1.0, 2.0)
    else
        ham_params = RydbergParameters(valid_N_values[1], 1.0, 0.0, 1.0)
    end
    backend = haskey(sim_params, "method") && sim_params["method"] == "ED" ? EDBackend() : TNBackend()
    
    filename_saveto = create_filename(ham_params, coupling_params, sim_params, backend)
    filename_saveto = "$(prefix)$(filename_saveto)_energy_density_and_overlap_vs_N.pdf"

    isdir("$(directory)/Figs") || mkpath("$(directory)/Figs")
    fig.savefig("$(directory)/Figs/$(filename_saveto)", dpi=300)
end

function plot_cooling_curve_noise(ham_name, N, coupling_params, sim_params, peInt_range; is_optimization=false)
    plt = get_pyplot()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    directory = is_optimization ? "ResultsOpt" : "Results"
    prefix = is_optimization ? "Optimize" : ""

    for peInt in peInt_range
        pe = peInt * 1e-3
        pe = round(pe, digits=4)
        sim_params["pe"] = pe
        sim_params["peInt"] = peInt

        # Create proper HamiltonianParameters
        if occursin("niIsing", ham_name)
            ham_params = NiIsingParameters(N, 1.0, -1.05, 0.5)
        elseif occursin("Ising", ham_name)
            ham_params = IsingParameters(N, 1.0, 2.0)
        else
            ham_params = RydbergParameters(N, 1.0, 0.0, 1.0)
        end
        backend = haskey(sim_params, "method") && sim_params["method"] == "ED" ? EDBackend() : TNBackend()
        
        filename = create_filename(ham_params, coupling_params, sim_params, backend)
        filename = "$(prefix)$(filename)"
        if is_optimization
            search_name_part = create_search_name_part(sim_params)
            filename *= "_$(search_name_part)"
        end
        full_filename = "$(directory)/$(filename).h5"

        e0, E_list, GS_overlap_list, _ = safe_read_data(full_filename)
        if e0 !== nothing && E_list !== nothing && GS_overlap_list !== nothing
            steps = length(E_list)
            ax1.plot(1:steps, E_list ./ N, label="pe=$pe")
            ax2.plot(1:steps, GS_overlap_list, label="pe=$pe")
        else
            @warn "No valid data found for pe=$pe. Skipping this pe value."
        end
    end

    ax1.set_xlabel("Cooling steps")
    ax1.set_ylabel("Energy density")
    ax1.legend()

    ax2.set_xlabel("Cooling steps")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    # Create HamiltonianParameters for filename
    if occursin("niIsing", ham_name)
        ham_params = NiIsingParameters(N, 1.0, -1.05, 0.5)
    elseif occursin("Ising", ham_name)
        ham_params = IsingParameters(N, 1.0, 2.0)
    else
        ham_params = RydbergParameters(N, 1.0, 0.0, 1.0)
    end
    backend = haskey(sim_params, "method") && sim_params["method"] == "ED" ? EDBackend() : TNBackend()
    
    filename_saveto = create_filename(ham_params, coupling_params, sim_params, backend)
    filename_saveto = "$(prefix)$(filename_saveto)_cooling_curve_noise.pdf"

    isdir("$(directory)/Figs") || mkpath("$(directory)/Figs")
    fig.savefig("$(directory)/Figs/$(filename_saveto)", dpi=300)
end

function plot_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, peInt_range; is_optimization=false)
    plt = get_pyplot()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    directory = is_optimization ? "ResultsOpt" : "Results"
    prefix = is_optimization ? "Optimize" : ""

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
            filename = "$(prefix)$(filename)"
            if is_optimization
                search_name_part = create_search_name_part(sim_params)
                filename *= "_$(search_name_part)"
            end
            full_filename = "$(directory)/$(filename).h5"

            e0, E_final, GS_overlap_final, Edensity_final = safe_read_data(full_filename)
            if e0 !== nothing && E_final !== nothing && GS_overlap_final !== nothing && Edensity_final !== nothing
                push!(energy_errors, abs(Edensity_final - e0 / N))
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

    filename_saveto = create_filename(ham_name, N_values[1], coupling_params, sim_params)
    filename_saveto = "$(prefix)$(filename_saveto)_energy_error_and_overlap_vs_N_multiple_pe.pdf"

    isdir("$(directory)/Figs") || mkpath("$(directory)/Figs")
    fig.savefig("$(directory)/Figs/$(filename_saveto)", dpi=300)
end


"""
    plot_momentum_distribution(filename; steps_to_plot=nothing, save_fig=true)

Plot the momentum distribution n_k vs k as a function of cooling steps.
Shows how population in different k modes changes during cooling.
Marks the resonant frequency delta with a vertical line.
"""
function plot_momentum_distribution(filename; steps_to_plot=nothing, save_fig=true)
    plt = get_pyplot()

    data = read_h5_data(filename)
    data === nothing && return

    if !haskey(data, "momentum_dist") || !haskey(data, "k_values")
        @warn "No k-space data found in file $filename"
        return
    end

    momentum_dist = data["momentum_dist"]
    k_values = data["k_values"]
    total_steps = size(momentum_dist, 1)
    step_indices = select_evolution_steps(total_steps; steps_to_plot=steps_to_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_evolution_colors(plt, length(step_indices))

    for (i, step) in enumerate(step_indices)
        if step <= total_steps
            n_k = momentum_dist[step, :]
            label = step == 1 ? "Initial" : "Step $step"
            ax.plot(k_values, n_k, "o-", color=colors[i], label=label, markersize=4)
        end
    end

    if haskey(data, "delta") && data["delta"] !== nothing
        ax2 = ax.twinx()
        ax2.axhline(y=data["delta"], color="red", linestyle="--", alpha=0.7, label="Bath freq delta")
        ax2.set_ylabel("Energy", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
    end

    ax.set_xlabel(L"Momentum $k$ (units of $2\pi/N$)")
    ax.set_ylabel(L"Occupation $n_k$")
    ax.set_title("Momentum Distribution Evolution")
    ax.legend()
    ax.grid(true, alpha=0.3)
    plt.tight_layout()

    if save_fig
        base_name = extract_filename_base(filename)
        save_figure(fig, dirname(filename), "momentum_dist_$(base_name).pdf")
    end

    plt.show()
end

"""
    plot_momentum_distribution_heatmap(filename; save_fig=true)

Plot the momentum distribution as a heatmap showing n_k vs (k, step).
This gives a comprehensive view of how all modes evolve during cooling.
"""
function plot_momentum_distribution_heatmap(filename; save_fig=true)
    plt = get_pyplot()

    data = read_h5_data(filename)
    data === nothing && return

    if !haskey(data, "momentum_dist") || !haskey(data, "k_values")
        @warn "No k-space data found in file $filename"
        return
    end

    momentum_dist = data["momentum_dist"]
    k_values = data["k_values"]
    total_steps = size(momentum_dist, 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(transpose(momentum_dist), aspect="auto", origin="lower",
                   extent=[1, total_steps, k_values[1], k_values[end]],
                   cmap="hot", interpolation="nearest")

    ax.set_xlabel("Cooling Step")
    ax.set_ylabel(L"Momentum $k$ (units of $2\pi/N$)")
    ax.set_title("Momentum Distribution Evolution Heatmap")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(L"Occupation $n_k$")

    if 0 in k_values
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.5, linewidth=1)
    end

    plt.tight_layout()

    if save_fig
        base_name = extract_filename_base(filename)
        save_figure(fig, dirname(filename), "momentum_dist_heatmap_$(base_name).pdf")
    end

    plt.show()
end
