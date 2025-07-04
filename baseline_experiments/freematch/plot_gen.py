# === Plotting Function ===
import matplotlib.pyplot as plt

# Global font settings
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.frameon": False
})


def plot_f1_fnr(years, f1s, fnrs, save_path="f1_fnr_flexmatch_baseline_with_al.png"):
    # Convert to list if Series
    years = list(years)
    f1s = list(f1s)
    fnrs = list(fnrs)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("F1 Score", color="blue")
    ax1.plot(years, f1s, color="blue", label="F1 Score")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel("False Negative Rate (FNR)", color="red")
    ax2.plot(years, fnrs, color="red", label="FNR")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)

    # Set only year (4-digit) on x-axis, sampled to reduce overlap
    xtick_positions = []
    xtick_labels = []
    seen_years = set()
    for idx, ym in enumerate(years):
        year = ym.split("_")[0]
        if year not in seen_years:
            xtick_positions.append(idx)
            xtick_labels.append(year)
            seen_years.add(year)

    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels, rotation=0)

    # Add legend above plot to avoid overlapping x-label
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.show()
