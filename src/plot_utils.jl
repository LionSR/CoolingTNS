"""
Shared plotting utilities for k-space visualization.
Provides common helper functions to reduce duplication across plotting files.
"""

using HDF5
using PythonCall
using LaTeXStrings

# Lazy pyplot access - imported once per session
const _pyplot = Ref{Py}()

"""
    get_pyplot()

Get the matplotlib.pyplot module, caching the import for performance.
"""
function get_pyplot()
    if !isassigned(_pyplot)
        _pyplot[] = pyimport("matplotlib.pyplot")
    end
    return _pyplot[]
end

"""
    generate_k_values(N::Int, bc::Symbol) -> Vector{Float64}

Generate k-values based on boundary conditions.
- Periodic BC: k = 2pi*n/N for n = 0, 1, ..., N-1
- Antiperiodic BC: k = pi*(2n+1)/N for n = 0, 1, ..., N-1
"""
function generate_k_values(N::Int, bc::Symbol)::Vector{Float64}
    if bc == :periodic
        return [2pi * n / N for n in 0:N-1]
    elseif bc == :antiperiodic
        return [pi * (2n + 1) / N for n in 0:N-1]
    else
        error("Unsupported boundary condition: $bc")
    end
end

"""
    compute_energy_dispersion(k_values, J::Real, h::Real) -> Vector{Float64}

Compute energy dispersion epsilon_k for the transverse field Ising model.
epsilon_k = -2*sqrt(J^2 + h^2 + 2*J*h*cos(k))
"""
function compute_energy_dispersion(k_values, J::Real, h::Real)::Vector{Float64}
    return [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_values]
end

"""
    compute_ground_state_occupation(k_values, J::Real, h::Real) -> Vector{Float64}

Compute ground state occupation n_k^(GS) for the transverse field Ising model.
n_k^(GS) = (1/2)(1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k)))
"""
function compute_ground_state_occupation(k_values, J::Real, h::Real)::Vector{Float64}
    return [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
end

"""
    read_h5_data(filename::String) -> Union{Dict{String, Any}, Nothing}

Read all data from an HDF5 file into a dictionary.
Returns nothing if the file doesn't exist.
"""
function read_h5_data(filename::String)::Union{Dict{String, Any}, Nothing}
    if !isfile(filename)
        @warn "File not found: $filename"
        return nothing
    end

    data = Dict{String, Any}()
    h5open(filename, "r") do file
        for key in keys(file)
            data[key] = read(file, key)
        end
    end
    return data
end

"""
    select_evolution_steps(total_steps::Int; steps_to_plot=nothing) -> Vector{Int}

Select which steps to plot from an evolution.
Default: initial, 25%, 50%, 75%, and final steps.
"""
function select_evolution_steps(total_steps::Int; steps_to_plot=nothing)::Vector{Int}
    if steps_to_plot !== nothing
        return steps_to_plot
    end
    return unique([1,
                   div(total_steps, 4),
                   div(total_steps, 2),
                   div(3*total_steps, 4),
                   total_steps])
end

"""
    save_figure(fig, base_dir::String, fig_name::String)

Save a figure to the Figs subdirectory, creating it if needed.
"""
function save_figure(fig, base_dir::String, fig_name::String)
    fig_dir = joinpath(base_dir, "Figs")
    mkpath(fig_dir)
    fig_path = joinpath(fig_dir, fig_name)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    println("Figure saved to $fig_path")
end

"""
    extract_filename_base(filepath::String) -> String

Extract the base filename without extension from a file path.
"""
function extract_filename_base(filepath::String)::String
    base = splitpath(filepath)[end]
    return replace(base, ".h5" => "")
end

"""
    setup_kspace_axis(ax, bc::Symbol)

Configure common axis settings for k-space plots.
"""
function setup_kspace_axis(ax, bc::Symbol)
    ax.set_xlim(0, 2)
    ax.grid(true, alpha=0.3)
end

"""
    get_evolution_colors(n_steps::Int)

Generate a color array for evolution plots using viridis colormap.
"""
function get_evolution_colors(plt, n_steps::Int)
    return plt.cm.viridis(range(0, 1, length=n_steps))
end
