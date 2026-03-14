"""Breakdown analysis plot: forward vs backward pass time.

Stacked bar chart showing forward and backward pass times for AD,
with percentage labels for the backward/forward ratio.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

INPUT_FILE = "memory_breakdown.csv"

MODEL_LABELS = {
    "gst_small": "GST-S",
    "gst_medium": "GST-M",
    "gst_large": "GST-L",
    "gst_coreg2_small": "GST-C2",
    "gst_coreg3_small": "GST-C3",
}

MODEL_ORDER = ["gst_small", "gst_coreg2_small", "gst_coreg3_small", "gst_medium", "gst_large"]

FWD_COLOR = "#2166ac"
BWD_COLOR = "#b2182b"

FIGSIZE = (7, 4)
DPI = 300
BAR_WIDTH = 0.6


def load_data():
    path = os.path.join(DATA_DIR, INPUT_FILE)
    return pd.read_csv(path)


def plot_breakdown(df):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ad_data = df[df["method"] == "AD"]
    models = [m for m in MODEL_ORDER if m in ad_data["model"].values]
    x = np.arange(len(models))

    fwd_times = []
    bwd_times = []
    for model in models:
        row = ad_data[ad_data["model"] == model]
        if row.empty:
            fwd_times.append(0)
            bwd_times.append(0)
        else:
            fwd_times.append(max(row["forward_time"].values[0], 0))
            bwd_times.append(max(row["backward_time"].values[0], 0))

    fwd_times = np.array(fwd_times)
    bwd_times = np.array(bwd_times)

    ax.bar(
        x, fwd_times, BAR_WIDTH,
        color=FWD_COLOR, label="Forward",
        zorder=3, edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x, bwd_times, BAR_WIDTH, bottom=fwd_times,
        color=BWD_COLOR, label="Backward",
        zorder=3, edgecolor="white", linewidth=0.5,
    )

    # Annotate with backward/forward ratio
    for i in range(len(models)):
        total = fwd_times[i] + bwd_times[i]
        if total > 0 and fwd_times[i] > 0:
            ratio = bwd_times[i] / fwd_times[i]
            bwd_pct = 100 * bwd_times[i] / total
            ax.annotate(
                f"{ratio:.1f}x\n({bwd_pct:.0f}%)",
                (x[i], total),
                textcoords="offset points", xytext=(0, 5),
                fontsize=7, ha="center", color="#333333",
            )

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Per-gradient time (s)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    fig = plot_breakdown(df)

    out = os.path.join(OUTPUT_DIR, "breakdown_analysis.pdf")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out}")

    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")
