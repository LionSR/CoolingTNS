using PythonCall
using LaTeXStrings

function plot_energy_and_overlap(E_list, GS_overlap_list, steps, e₀, N, filename)
    plt = pyimport("matplotlib.pyplot")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    ax.plot(1:steps+1, E_list / N, alpha=0.75, marker="o", label="Cooling")
    ax.set_xlabel("Steps")
    ax.set_ylabel(L"Energy density $E/N$")
    ax.axhline(y=e₀ / N, xmin=0, xmax=1, linewidth=1.5, color="black", label=L"$E_0/N$")
    ax.legend()

    ax = axs[1]
    ax.plot(1:steps+1, GS_overlap_list, marker="o", alpha=0.75, color="grey")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Ground state overlap")

    fig.savefig("Results/$(filename).pdf", dpi=300)
end

