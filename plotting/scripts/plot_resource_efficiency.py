"""Resource efficiency plot: FD/AD gradient time ratio vs number of FD nodes.

Shows how many FD nodes are needed to match AD at its minimum feasible node
count.  For single-node AD models the baseline is 1 GPU; for distributed AD
models (AP1, SA1, WA1) the baseline is the minimum P_AD GPUs required.
The break-even line at y=1 indicates where FD matches AD speed.
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

# Node counts on the x-axis (evenly spaced positions)
NODE_COUNTS = [1, 2, 4, 8, 16, 32]

# Model display order and properties
# model_name: (display_label, line_style)
# Solid for single-node AD, dashed for distributed AD
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


def plot_ratio(df):
    """Plot FD/AD gradient time ratio vs number of FD nodes (log-scale y)."""
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

        fd_rows = mdf[mdf["method"] == "FD"].sort_values("n_nodes")
        if fd_rows.empty:
            continue

        x_vals, ratios, err_lo, err_hi = [], [], [], []
        for _, row in fd_rows.iterrows():
            n = int(row["n_nodes"])
            if n not in x_map:
                continue
            fd_time = row["gradient_time_mean_s"]
            fd_std = row["gradient_time_std_s"]
            ratio = fd_time / ad_time
            ratios.append(ratio)
            x_vals.append(x_map[n])

            if pd.notna(fd_std) and fd_std > 0:
                err_lo.append(fd_std / ad_time)
                err_hi.append(fd_std / ad_time)
            else:
                err_lo.append(0)
                err_hi.append(0)

        if not ratios:
            continue

        # Annotation for distributed AD models
        disp_label = label
        if ad_nodes > 1:
            disp_label += f" (AD on {ad_nodes})"

        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i]

        ax.errorbar(
            x_vals,
            ratios,
            yerr=[err_lo, err_hi],
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            linestyle=lstyle,
            color=color,
            label=disp_label,
            capsize=2,
            zorder=3,
        )

    # Break-even line
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=2)

    # Log scale
    ax.set_yscale("log")

    # Shaded regions
    ylims = ax.get_ylim()
    ax.axhspan(ylims[0], 1.0, alpha=0.06, color="green", zorder=0)
    ax.axhspan(1.0, ylims[1], alpha=0.06, color="red", zorder=0)

    ax.set_xlabel("Number of FD GPU nodes", fontsize=11)
    ax.set_ylabel(r"$T_{\mathrm{FD}}(N)\;/\;T_{\mathrm{AD}}$", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in NODE_COUNTS])
    ax.set_xlim(x_positions[0] - 0.3, x_positions[-1] + 0.3)

    ax.legend(
        fontsize=7,
        loc="upper right",
        ncol=2,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3, which="both", zorder=0)

    ax.text(
        0.02, 0.02, "FD faster",
        transform=ax.transAxes, fontsize=9, alpha=0.5,
        color="green", va="bottom", ha="left", style="italic",
    )
    ax.text(
        0.02, 0.98, "AD faster",
        transform=ax.transAxes, fontsize=9, alpha=0.5,
        color="red", va="top", ha="left", style="italic",
    )

    fig.tight_layout()
    return fig


def print_breakeven_table(df):
    """Print a summary table of wall-clock breakeven analysis."""
    print("\n% Breakeven summary")
    print(f"{'Model':<10s} {'d':>3s} {'P_AD':>4s} {'T_AD (s)':>10s} "
          f"{'Breakeven N':>12s} {'R_breakeven':>12s}")
    print("-" * 55)

    for model, (label, _) in MODELS.items():
        mdf = df[df["model"] == model]
        ad_rows = mdf[mdf["method"] == "AD"]
        if ad_rows.empty:
            continue
        ad_time = ad_rows["gradient_time_mean_s"].values[0]
        ad_nodes = int(ad_rows["ad_min_nodes"].values[0])
        d = int(ad_rows["n_hyperparams"].values[0])
        display = ad_rows["display_name"].values[0]

        fd_rows = mdf[mdf["method"] == "FD"].sort_values("n_nodes")
        breakeven_n = "--"
        r_break = "--"
        for _, row in fd_rows.iterrows():
            if row["gradient_time_mean_s"] <= ad_time:
                breakeven_n = str(int(row["n_nodes"]))
                r_break = f"{int(row['n_nodes']) / ad_nodes:.1f}"
                break

        print(f"{display:<10s} {d:>3d} {ad_nodes:>4d} {ad_time:>10.3f} "
              f"{breakeven_n:>12s} {r_break:>12s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_data()

    fig = plot_ratio(data)

    output_path = os.path.join(OUTPUT_DIR, "resource_efficiency_ratio.pdf")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")

    output_path_png = os.path.join(OUTPUT_DIR, "resource_efficiency_ratio.png")
    fig.savefig(output_path_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {output_path_png}")

    print_breakeven_table(data)

    plt.close(fig)
