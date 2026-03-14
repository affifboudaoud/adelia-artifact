#!/usr/bin/env python3
"""Generate single-node performance comparison figure (AD vs FD).

Produces a 2x2 grid of grouped bar charts:
  - Rows: GPU (A100, GH200)
  - Columns: Model size group (small, large)

Usage:
    python plot_speedup_comparison.py
    python plot_speedup_comparison.py --csv path/to/data.csv --output path/to/fig.pdf
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION — Edit these to customize the figure
# ============================================================

METHOD_NAMES = {
    "FD": "FD",
    "JAX": "DALIA-AD",
}

GPU_NAMES = {
    "A100": "A100",
    "GH200": "GH200",
}

GPUS = ["A100", "GH200"]

MODEL_DISPLAY_NAMES = {
    "gr": "GR",
    "gs_small": "GS",
    "gs_coreg2_small": "GS-C2",
    "gs_coreg3_small": "GS-C3",
    "gst_small": "GST-S",
    "gst_medium": "GST-M",
    "gst_large": "GST-L",
    "gst_coreg2_small": "GST-C2",
    "gst_coreg3_small": "GST-C3",
    "pst_small": "PST",
}

GROUPING_THRESHOLD_SECONDS = 20.0

SORT_BY = "N_hyper"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(
    SCRIPT_DIR, "..", "results", "speedup_optimization_time_no_jit.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "..", "writing", "698c58632d8bba3fe6c13a59", "figures"
)
DEFAULT_OUTPUT_FILENAME = "speedup_comparison.pdf"

COLORS = {"FD": "#4878CF", "JAX": "#D65F5F"}
HATCHES = {"FD": None, "JAX": "//"}
FIGURE_WIDTH = 7.16
FIGURE_HEIGHT = 4.5
BAR_WIDTH = 0.35
FONT_SIZE_TICKS = 8
FONT_SIZE_LABELS = 9
FONT_SIZE_TITLES = 9
FONT_SIZE_SPEEDUP = 7
EDGECOLOR = "black"
LINEWIDTH = 0.5
Y_HEADROOM = 1.15


def load_and_prepare_data(csv_path):
    """Read CSV, drop rows with no data for any GPU, parse speedup strings."""
    df = pd.read_csv(csv_path)

    fd_cols = [c for c in df.columns if "FD (s)" in c]
    df = df.dropna(subset=fd_cols, how="all")
    for col in fd_cols:
        df = df[~((df[col].notna()) & (df[col] == ""))]

    for col in df.columns:
        if "FD (s)" in col or "JAX (s)" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if "Speedup" in col:
            df[col] = df[col].astype(str).str.replace("x", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    return df


def split_into_groups(df, threshold):
    """Split models into small/large groups based on max FD time across GPUs."""
    fd_cols = [c for c in df.columns if "FD (s)" in c]
    df = df.copy()
    df["_max_fd"] = df[fd_cols].max(axis=1)

    small = df[df["_max_fd"] < threshold].sort_values(SORT_BY).reset_index(drop=True)
    large = df[df["_max_fd"] >= threshold].sort_values(SORT_BY).reset_index(drop=True)

    small = small.drop(columns=["_max_fd"])
    large = large.drop(columns=["_max_fd"])
    return small, large


def make_x_labels(df):
    """Generate x-axis labels as 'MODEL_NAME\n(d=N)'."""
    labels = []
    for _, row in df.iterrows():
        name = MODEL_DISPLAY_NAMES.get(row["Example"], row["Example"])
        n_hyper = int(row["N_hyper"])
        labels.append(f"{name}\n($d$={n_hyper})")
    return labels


def plot_single_panel(ax, df, gpu):
    """Plot a grouped bar chart for one GPU on the given axes."""
    fd_col = f"{gpu} FD (s)"
    jax_col = f"{gpu} JAX (s)"
    speedup_col = f"{gpu} Speedup"

    mask = df[fd_col].notna() & df[jax_col].notna()
    panel_df = df[mask].reset_index(drop=True)
    x = np.arange(len(panel_df))

    fd_vals = panel_df[fd_col].values.astype(float)
    jax_vals = panel_df[jax_col].values.astype(float)
    speedups = panel_df[speedup_col].values.astype(float)

    ax.bar(
        x - BAR_WIDTH / 2,
        fd_vals,
        BAR_WIDTH,
        label=METHOD_NAMES["FD"],
        color=COLORS["FD"],
        hatch=HATCHES["FD"],
        edgecolor=EDGECOLOR,
        linewidth=LINEWIDTH,
    )
    bars_jax = ax.bar(
        x + BAR_WIDTH / 2,
        jax_vals,
        BAR_WIDTH,
        label=METHOD_NAMES["JAX"],
        color=COLORS["JAX"],
        hatch=HATCHES["JAX"],
        edgecolor=EDGECOLOR,
        linewidth=LINEWIDTH,
    )

    for bar, spd in zip(bars_jax, speedups):
        ax.annotate(
            f"{spd:.1f}$\\times$",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_SPEEDUP,
            fontweight="bold",
        )

    labels = make_x_labels(panel_df)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS)

    ax.set_ylabel("Time (s)", fontsize=FONT_SIZE_LABELS)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    max_val = max(fd_vals.max(), jax_vals.max())
    ax.set_ylim(0, max_val * Y_HEADROOM)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input CSV path")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--output-name", default=DEFAULT_OUTPUT_FILENAME, help="Output filename"
    )
    args = parser.parse_args()

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = load_and_prepare_data(args.csv)
    small_df, large_df = split_into_groups(df, GROUPING_THRESHOLD_SECONDS)

    fig, axes = plt.subplots(
        len(GPUS),
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": [len(small_df), len(large_df)]},
    )

    panel_labels = iter("abcdefgh")
    for row_idx, gpu in enumerate(GPUS):
        for col_idx, (group_df, group_name) in enumerate(
            [(small_df, "Small Models"), (large_df, "Large Models")]
        ):
            ax = axes[row_idx, col_idx]
            plot_single_panel(ax, group_df, gpu)
            label = next(panel_labels)
            ax.set_title(
                f"({label}) {GPU_NAMES[gpu]} \u2014 {group_name}",
                fontsize=FONT_SIZE_TITLES,
                fontweight="bold",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=FONT_SIZE_LABELS,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
