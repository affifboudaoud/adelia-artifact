"""Structure-preserving vs generic AD comparison plot.

Grouped bar chart: per-gradient time for AD-BTA, AD-Loop, AD-Loop-Ckpt,
AD-Dense, and FD on BTA spatio-temporal models. OOM entries marked with x.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import figure_style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

INPUT_FILE = "structure_comparison.csv"

MODEL_LABELS = {
    "gst_small": "GST-S",
    "gst_medium": "GST-M",
    "gst_large": "GST-L",
    "gst_coreg2_small": "GST-C2",
    "gst_coreg3_small": "GST-C3",
}

MODEL_ORDER = ["gst_small", "gst_medium", "gst_large", "gst_coreg2_small", "gst_coreg3_small"]

METHODS = ["AD-BTA", "AD-Loop", "AD-Loop-Ckpt", "AD-Dense", "FD"]
METHOD_COLORS = {
    "AD-BTA": "#E8899A",
    "AD-Loop": "#F2BCC5",
    "AD-Loop-Ckpt": "#F7D5DC",
    "AD-Dense": "#D4A843",
    "FD": "#6B7B8D",
}

FIGSIZE = (8, 4)
DPI = 300
BAR_WIDTH = 0.15
N_METHODS = len(METHODS)


def load_data():
    path = os.path.join(DATA_DIR, INPUT_FILE)
    return pd.read_csv(path)


def plot_comparison(df):
    figure_style.apply()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    models = [m for m in MODEL_ORDER if m in df["model"].values]
    x = np.arange(len(models))

    offsets = np.arange(N_METHODS) - (N_METHODS - 1) / 2.0

    for j, method in enumerate(METHODS):
        times = []
        oom_mask = []
        for model in models:
            row = df[(df["model"] == model) & (df["method"] == method)]
            if row.empty or row["time_mean"].values[0] < 0:
                times.append(0)
                oom_mask.append(True)
            else:
                times.append(row["time_mean"].values[0])
                oom_mask.append(False)

        positions = x + offsets[j] * BAR_WIDTH
        ax.bar(
            positions, times, BAR_WIDTH,
            color=METHOD_COLORS[method], label=method,
            zorder=3, edgecolor="white", linewidth=0.5,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Model")
    ax.set_ylabel("Per-gradient time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)

    # Mark OOM at correct y positions after log scale is set
    for j, method in enumerate(METHODS):
        for i, model in enumerate(models):
            row = df[(df["model"] == model) & (df["method"] == method)]
            if row.empty or row["time_mean"].values[0] < 0:
                pos = x[i] + offsets[j] * BAR_WIDTH
                ymin = ax.get_ylim()[0]
                ax.annotate(
                    "OOM", (pos, ymin * 2),
                    fontsize=6, ha="center", va="bottom",
                    color=METHOD_COLORS[method], fontweight="bold",
                    rotation=90,
                )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    fig = plot_comparison(df)

    out = os.path.join(OUTPUT_DIR, "structure_preserving_comparison.pdf")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out}")

    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")
