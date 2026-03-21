"""Performance analysis: distributed stage breakdown.

Single-column stacked bar chart showing per-stage gradient time
for distributed models (WA1, AP1, WA2) on 4 nodes.

Usage:
    python plot_performance_analysis.py
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

BREAKDOWN_FILE = "distributed_breakdown.csv"
OUTPUT_FILENAME = "performance_analysis.pdf"

DIST_MODELS = ["WA1", "AP1", "WA2"]

STAGE_NAMES = {
    "chol_fwd_sub": "Chol + fwd sub",
    "bwd_si": "Bwd sub + SI",
    "grad_prior": "Prior grad",
    "grad_quad": "Quad grad",
}
STAGE_ORDER = ["chol_fwd_sub", "bwd_si", "grad_prior", "grad_quad"]
STAGE_COLORS = {
    "chol_fwd_sub": "#E8899A",
    "bwd_si": "#F2BCC5",
    "grad_prior": "#6B7B8D",
    "grad_quad": "#9AABB8",
}

FIGURE_WIDTH = 3.5
FIGURE_HEIGHT = 3.2
BAR_WIDTH = 0.5


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, BREAKDOWN_FILE))


def plot_analysis(breakdown):
    figure_style.apply()

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    x = np.arange(len(DIST_MODELS))

    bottoms = np.zeros(len(DIST_MODELS))
    for stage in STAGE_ORDER:
        times = []
        for model in DIST_MODELS:
            row = breakdown[(breakdown["model"] == model) &
                            (breakdown["stage"] == stage)]
            times.append(row["time_mean"].values[0] if not row.empty else 0)
        times = np.array(times)
        ax.bar(x, times, BAR_WIDTH, bottom=bottoms,
               color=STAGE_COLORS[stage], label=STAGE_NAMES[stage],
               zorder=3, edgecolor="white", linewidth=0.5)
        bottoms += times

    # Total error bars (95% CI): sum stage variances in quadrature
    total_cis = np.zeros(len(DIST_MODELS))
    for i, model in enumerate(DIST_MODELS):
        rows = breakdown[(breakdown["model"] == model) &
                         (breakdown["stage"].isin(STAGE_ORDER))]
        var_sum = (rows["time_std"] ** 2).sum()
        n_runs = rows["n_runs"].values[0] if not rows.empty else 5
        total_cis[i] = 1.96 * np.sqrt(var_sum) / np.sqrt(n_runs)
    ax.errorbar(x, bottoms, yerr=total_cis, fmt="none", ecolor="#2B2D42",
                capsize=5, elinewidth=1.5, capthick=1.5, zorder=5)

    # Percentage annotations for the two Python-loop stages combined
    for i, model in enumerate(DIST_MODELS):
        rows = breakdown[breakdown["model"] == model]
        total = rows["time_mean"].sum()
        loop_stages = rows[rows["stage"].isin(["chol_fwd_sub", "bwd_si"])]
        loop_pct = loop_stages["time_mean"].sum() / total * 100
        ax.annotate(
            f"{loop_pct:.0f}%",
            (x[i], bottoms[i] + total_cis[i]),
            textcoords="offset points", xytext=(0, 4),
            fontsize=8, ha="center", color="#2B2D42",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DIST_MODELS)
    ax.set_ylabel("Per-gradient time (s)")
    ax.legend(loc="upper left", framealpha=0.9,
              bbox_to_anchor=(0.0, 1.02))
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, bottoms.max() * 1.18)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=PAPER_FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    breakdown = load_data()
    fig = plot_analysis(breakdown)

    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")

    output_png = output_path.replace(".pdf", ".png")
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_png}")

    plt.close(fig)
