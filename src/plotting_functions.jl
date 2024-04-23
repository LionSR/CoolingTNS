using PythonCall
using LaTeXStrings

function plot_energy_and_overlap(E_list, GS_overlap_list, steps, e₀, N, filename; moving_average=false)
    plt = pyimport("matplotlib.pyplot")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    ax.plot(1:steps+1, E_list / N, alpha=0.75, marker="o", label="Cooling")
    if moving_average
        window_size = 10
        E_ma = [mean(E_list[max(1, i-window_size+1):i]) for i in 1:length(E_list)]
        ax.plot(1:steps+1, E_ma / N, alpha=0.75, marker="o", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel(L"Energy density $E/N$")
    ax.axhline(y=e₀ / N, xmin=0, xmax=1, linewidth=1.5, color="black", label=L"$E_0/N$")
    ax.legend()

    ax = axs[1]
    ax.plot(1:steps+1, GS_overlap_list, marker="o", alpha=0.75, color="grey", label="Cooling")
    if moving_average
        GS_ma = [mean(GS_overlap_list[max(1, i-window_size+1):i]) for i in 1:length(GS_overlap_list)]
        ax.plot(1:steps+1, GS_ma, marker="o", alpha=0.75, color="black", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel("Ground state overlap")
    ax.legend()

    fig.savefig("Results/$(filename).pdf", dpi=300)
end
