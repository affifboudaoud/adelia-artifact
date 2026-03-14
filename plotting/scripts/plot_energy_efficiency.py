"""Energy efficiency plot: node-seconds ratio for FD vs AD.

Plots the energy cost ratio (N * T_FD(N)) / (P_AD * T_AD) vs number of FD
nodes.  Since all workloads run on identical GH200 GPUs under similar
GPU-compute load, the per-GPU power draw is approximately constant and
cancels in the ratio.  The node-seconds metric (N * wall-time) is the
standard HPC resource accounting unit and is proportional to total energy
consumed.

For distributed AD models (P_AD > 1), the denominator uses P_AD * T_AD to
account for the actual resource cost of AD.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

CSV_FILE = "resource_efficiency_all.csv"

NODE_COUNTS = [1, 2, 4, 8, 16, 32]

MODELS = {
    "gs_small": (r"GS ($d{=}3$)", "-"),
    "pst_small": (r"PST ($d{=}3$)", "-"),
    "gst_small": (r"GST-S ($d{=}4$)", "-"),
    "gst_medium": (r"GST-M ($d{=}4$)", "-"),
    "gst_large": (r"GST-L ($d{=}4$)", "-"),
    "gs_coreg2_small": (r"GS-C2 ($d{=}7$)", "-"),
    "gst_coreg2_small": (r"GST-C2 ($d{=}9$)", "-"),
    "gs_coreg3_small": (r"GS-C3 ($d{=}12$)", "-"),
    "gst_coreg3_small": (r"GST-C3 ($d{=}15$)", "-"),
    "sa1": (r"SA1 ($d{=}15$)", "--"),
    "ap1": (r"AP1 ($d{=}15$)", "--"),
    "wa1": (r"WA1 ($d{=}15$)", "--"),
}

FIGSIZE = (7, 4.5)
DPI = 300
MARKER_SIZE = 6
LINE_WIDTH = 1.5

MARKERS = ["s", "D", "^", "v", "p", "o", "P", "X", "*", "h", "d", ">"]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.95, len(MODELS)))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load the unified resource efficiency CSV.

    Returns
    -------
    pd.DataFrame
        Only rows with numeric (non-missing) gradient_time_mean_s.
    """
    path = os.path.join(DATA_DIR, CSV_FILE)
    df = pd.read_csv(path)
    df = df[df["gradient_time_mean_s"] != "missing"].copy()
    df["gradient_time_mean_s"] = df["gradient_time_mean_s"].astype(float)
    df["gradient_time_std_s"] = pd.to_numeric(
        df["gradient_time_std_s"], errors="coerce"
    )
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_energy_ratio(df):
    """Plot node-seconds ratio (N * T_FD) / (P_AD * T_AD) vs FD nodes."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_positions = list(range(len(NODE_COUNTS)))
    x_map = {n: i for n, i in zip(NODE_COUNTS, x_positions)}

    for i, (model, (label, lstyle)) in enumerate(MODELS.items()):
        mdf = df[df["model"] == model]
        ad_rows = mdf[mdf["method"] == "AD"]
        if ad_rows.empty:
            continue
        ad_time = ad_rows["gradient_time_mean_s"].values[0]
        ad_nodes = int(ad_rows["ad_min_nodes"].values[0])
        ad_node_seconds = ad_nodes * ad_time

        fd_rows = mdf[mdf["method"] == "FD"].sort_values("n_nodes")
        if fd_rows.empty:
            continue

        x_vals, ratios = [], []
        for _, row in fd_rows.iterrows():
            n = int(row["n_nodes"])
            if n not in x_map:
                continue
            fd_node_seconds = n * row["gradient_time_mean_s"]
            ratios.append(fd_node_seconds / ad_node_seconds)
            x_vals.append(x_map[n])

        if not ratios:
            continue

        disp_label = label
        if ad_nodes > 1:
            disp_label += f" (AD on {ad_nodes})"

        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i]

        ax.plot(
            x_vals,
            ratios,
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            linestyle=lstyle,
            color=color,
            label=disp_label,
            zorder=3,
        )

    # Break-even line
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=2)

    ax.set_yscale("log")

    ylims = ax.get_ylim()
    ax.axhspan(ylims[0], 1.0, alpha=0.06, color="green", zorder=0)
    ax.axhspan(1.0, ylims[1], alpha=0.06, color="red", zorder=0)

    ax.set_xlabel("Number of FD GPU nodes", fontsize=11)
    ax.set_ylabel(
        r"Energy ratio  $\frac{N \cdot T_{\mathrm{FD}}}{P_{\mathrm{AD}} "
        r"\cdot T_{\mathrm{AD}}}$",
        fontsize=11,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in NODE_COUNTS])
    ax.set_xlim(x_positions[0] - 0.3, x_positions[-1] + 0.3)

    ax.legend(
        fontsize=7,
        loc="upper left",
        ncol=2,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3, which="both", zorder=0)

    ax.text(
        0.98, 0.02, "FD cheaper",
        transform=ax.transAxes, fontsize=9, alpha=0.5,
        color="green", va="bottom", ha="right", style="italic",
    )
    ax.text(
        0.98, 0.98, "AD cheaper",
        transform=ax.transAxes, fontsize=9, alpha=0.5,
        color="red", va="top", ha="right", style="italic",
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def print_energy_table(df):
    """Print a LaTeX-ready table of node-seconds values."""
    print("\n% LaTeX table: energy (node-seconds) comparison")
    print(f"% {'Model':<10s} & d & P_AD & AD (node-s) & FD@1 (node-s) "
          "& FD@max (node-s) & Ratio")
    print()

    max_fd_n = max(NODE_COUNTS)

    for model, (label, _) in MODELS.items():
        mdf = df[df["model"] == model]
        ad_rows = mdf[mdf["method"] == "AD"]
        if ad_rows.empty:
            continue
        ad_time = ad_rows["gradient_time_mean_s"].values[0]
        ad_nodes = int(ad_rows["ad_min_nodes"].values[0])
        d = int(ad_rows["n_hyperparams"].values[0])
        display = ad_rows["display_name"].values[0]
        ad_ns = ad_nodes * ad_time

        fd_rows = mdf[mdf["method"] == "FD"].sort_values("n_nodes")
        if fd_rows.empty:
            fd1_ns_str = "--"
            fdmax_ns_str = "--"
            ratio_str = "--"
        else:
            fd1 = fd_rows[fd_rows["n_nodes"] == 1]
            fd1_ns_str = f"{fd1['gradient_time_mean_s'].values[0]:.3f}" if not fd1.empty else "--"

            fdmax = fd_rows[fd_rows["n_nodes"] == max_fd_n]
            if fdmax.empty:
                fdmax = fd_rows.iloc[-1:]
                actual_n = int(fdmax["n_nodes"].values[0])
            else:
                actual_n = max_fd_n

            fdmax_ns = actual_n * fdmax["gradient_time_mean_s"].values[0]
            fdmax_ns_str = f"{fdmax_ns:.3f}"
            ratio = fdmax_ns / ad_ns
            ratio_str = f"${ratio:.0f}\\times$"

        print(f"  {display:<10s} & {d:>2d} & {ad_nodes} & {ad_ns:.3f} & "
              f"{fd1_ns_str} & {fdmax_ns_str} & {ratio_str} \\\\")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_data()

    fig = plot_energy_ratio(data)

    out_pdf = os.path.join(OUTPUT_DIR, "energy_comparison.pdf")
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    out_png = os.path.join(OUTPUT_DIR, "energy_comparison.png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")

    print_energy_table(data)

    plt.close(fig)
