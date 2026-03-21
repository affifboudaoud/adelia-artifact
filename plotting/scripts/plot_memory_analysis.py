"""Memory analysis plot: AD vs FD peak GPU memory across models.

Grouped bar chart showing peak GPU memory for AD and FD,
with a dashed line at the GH200 96 GB limit.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figure_style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

INPUT_FILE = "memory_breakdown.csv"

MODEL_LABELS = {
    "gst_small": "GST-S\n(466)",
    "gst_medium": "GST-M\n(81k)",
    "gst_large": "GST-L\n(1M)",
    "gst_coreg2_small": "GST-C2\n(8.5k)",
    "gst_coreg3_small": "GST-C3\n(8.5k)",
}

MODEL_ORDER = ["gst_small", "gst_coreg2_small", "gst_coreg3_small", "gst_medium", "gst_large"]

METHOD_COLORS = {"AD": "#E8899A", "FD": "#6B7B8D"}

FIGSIZE = (7, 4)
DPI = 300
BAR_WIDTH = 0.35
GH200_MEM_GB = 96


def load_data():
    path = os.path.join(DATA_DIR, INPUT_FILE)
    return pd.read_csv(path)


def plot_memory(df):
    figure_style.apply()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    models = [m for m in MODEL_ORDER if m in df["model"].values]
    x = np.arange(len(models))

    for j, method in enumerate(["AD", "FD"]):
        mem_gb = []
        for model in models:
            row = df[(df["model"] == model) & (df["method"] == method)]
            if row.empty or row["peak_memory_bytes"].values[0] < 0:
                mem_gb.append(0)
            else:
                mem_gb.append(row["peak_memory_bytes"].values[0] / (1024**3))

        offset = (j - 0.5) * BAR_WIDTH
        bars = ax.bar(
            x + offset, mem_gb, BAR_WIDTH,
            color=METHOD_COLORS[method], label=method,
            zorder=3, edgecolor="white", linewidth=0.5,
        )

        # Annotate memory ratio
        if method == "AD":
            ad_mem = mem_gb
        else:
            for i in range(len(models)):
                if mem_gb[i] > 0 and ad_mem[i] > 0:
                    ratio = ad_mem[i] / mem_gb[i]
                    y_pos = max(ad_mem[i], mem_gb[i])
                    ax.annotate(
                        f"{ratio:.1f}x",
                        (x[i], y_pos),
                        textcoords="offset points", xytext=(0, 5),
                        fontsize=7, ha="center", color="#2B2D42",
                    )

    # GH200 memory limit
    ax.axhline(
        y=GH200_MEM_GB, color="red", linestyle="--",
        linewidth=1.0, alpha=0.7, zorder=2,
    )
    ax.text(
        len(models) - 0.5, GH200_MEM_GB + 1,
        f"GH200 limit ({GH200_MEM_GB} GB)",
        fontsize=8, color="red", alpha=0.7, ha="right",
    )

    ax.set_xlabel("Model (latent dimension)")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    fig = plot_memory(df)

    out = os.path.join(OUTPUT_DIR, "memory_analysis.pdf")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out}")

    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")
