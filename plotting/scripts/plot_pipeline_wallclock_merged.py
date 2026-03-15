"""Wall-clock figure: optimization + Hessian for AD vs FD.

Stacked bar chart with per-gradient and end-to-end speedup annotations.
No JIT time in the bars (JIT is a one-time cost, not part of the pipeline).

Usage:
    python plot_pipeline_wallclock_merged.py
"""

import argparse
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

CSV_FILE = "pipeline_wallclock.csv"
OUTPUT_FILENAME = "pipeline_wallclock_merged.pdf"

MODEL_ORDER = [
    "GST-S", "GST-C2", "GST-C3", "GST-M",
    "GST-L", "SA1", "AP1", "WA1", "WA2",
]
FAST_MODELS = ["GST-S", "GST-C2", "GST-C3", "GST-M"]
LARGE_MODELS = ["GST-L", "SA1", "AP1", "WA1", "WA2"]

COLORS = {
    "fd_optim": "#c4a265",
    "fd_hessian": "#c4a265",
    "ad_optim": "#5a9e91",
    "ad_hessian": "#5a9e91",
}
HATCH_HESSIAN = "///"

FIGURE_WIDTH = 7.16
FIGURE_HEIGHT = 2.8
BAR_WIDTH = 0.30
FONT_SIZE_TICKS = 9
FONT_SIZE_LABELS = 10
FONT_SIZE_SPEEDUP = 7.5
EDGECOLOR = "black"
LINEWIDTH = 0.5


def load_data():
    path = os.path.join(DATA_DIR, CSV_FILE)
    df = pd.read_csv(path)
    df["_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values("_order").reset_index(drop=True)
    df = df.drop(columns=["_order"])
    return df


def plot_panel(ax, df_panel, ylim_top, panel_title, time_unit="s"):
    n = len(df_panel)
    x = np.arange(n)

    scale = 3600.0 if time_unit == "h" else 1.0
    unit_label = "h" if time_unit == "h" else "s"

    for i, (_, row) in enumerate(df_panel.iterrows()):
        per_grad_speedup = row["t_per_grad_fd"] / row["t_per_grad_ad"]

        fd_optim = row["t_optim_fd"] / scale
        fd_hessian = row["t_hessian_fd"] / scale
        ad_optim = row["t_optim_ad"] / scale
        ad_hessian = row["t_hessian_ad"] / scale

        fd_total = (row["t_optim_fd"] + row["t_hessian_fd"]) / scale
        ad_total = (row["t_optim_ad"] + row["t_hessian_ad"]) / scale
        e2e_speedup = fd_total / ad_total if ad_total > 0 else 0

        # FD bar (right)
        ax.bar(
            x[i] + BAR_WIDTH / 2, fd_optim, BAR_WIDTH,
            color=COLORS["fd_optim"], edgecolor=EDGECOLOR, linewidth=LINEWIDTH,
            zorder=3,
        )
        ax.bar(
            x[i] + BAR_WIDTH / 2, fd_hessian, BAR_WIDTH,
            bottom=fd_optim,
            color=COLORS["fd_hessian"], edgecolor=EDGECOLOR, linewidth=LINEWIDTH,
            hatch=HATCH_HESSIAN, zorder=3,
        )

        # AD bar (left)
        ax.bar(
            x[i] - BAR_WIDTH / 2, ad_optim, BAR_WIDTH,
            color=COLORS["ad_optim"], edgecolor=EDGECOLOR, linewidth=LINEWIDTH,
            zorder=3,
        )
        ax.bar(
            x[i] - BAR_WIDTH / 2, ad_hessian, BAR_WIDTH,
            bottom=ad_optim,
            color=COLORS["ad_hessian"], edgecolor=EDGECOLOR, linewidth=LINEWIDTH,
            hatch=HATCH_HESSIAN, zorder=3,
        )

        # Annotation: (per-grad speedup, e2e speedup) above taller bar
        fd_top = fd_optim + fd_hessian
        ad_top = ad_optim + ad_hessian
        bar_top = max(fd_top, ad_top)
        ax.annotate(
            f"({per_grad_speedup:.1f}$\\times$, "
            f"{e2e_speedup:.1f}$\\times$)",
            xy=(x[i], bar_top),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=FONT_SIZE_SPEEDUP, fontweight="bold",
            annotation_clip=False,
        )

    labels = [row["model"] for _, row in df_panel.iterrows()]

    ax.set_ylim(0, ylim_top)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS)
    ax.set_ylabel(f"Wall-clock time ({unit_label})", fontsize=FONT_SIZE_LABELS)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(panel_title, fontsize=FONT_SIZE_LABELS)


def plot_pipeline(df):
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df_fast = df[df["model"].isin(FAST_MODELS)].reset_index(drop=True)
    df_large = df[df["model"].isin(LARGE_MODELS)].reset_index(drop=True)

    fig, (ax_fast, ax_large) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": [4, 5]},
    )

    plot_panel(ax_fast, df_fast, ylim_top=1800,
               panel_title="(a) Small/medium models")
    plot_panel(ax_large, df_large, ylim_top=18,
               panel_title="(b) Large models", time_unit="h")

    patches = [
        mpatches.Patch(color=COLORS["ad_optim"], label="Optimization (AD)"),
        mpatches.Patch(
            facecolor=COLORS["ad_hessian"], edgecolor=EDGECOLOR,
            linewidth=LINEWIDTH, hatch=HATCH_HESSIAN, label="Hessian (AD)",
        ),
        mpatches.Patch(color=COLORS["fd_optim"], label="Optimization (FD)"),
        mpatches.Patch(
            facecolor=COLORS["fd_hessian"], edgecolor=EDGECOLOR,
            linewidth=LINEWIDTH, hatch=HATCH_HESSIAN, label="Hessian (FD)",
        ),
    ]
    fig.legend(
        handles=patches,
        fontsize=FONT_SIZE_LABELS - 2,
        loc="lower center",
        ncol=4,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.5,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=PAPER_FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data()
    fig = plot_pipeline(df)

    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")

    output_png = output_path.replace(".pdf", ".png")
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_png}")

    plt.close(fig)
