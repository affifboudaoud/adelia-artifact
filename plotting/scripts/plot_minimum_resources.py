"""Minimum resources comparison: AD vs FD gradient time at minimum feasible nodes.

Two-panel grouped bar chart showing per-gradient time for AD and FD, each
running on the minimum number of GH200 nodes required.  Node counts are
annotated on bars.

Usage:
    python plot_minimum_resources.py
"""

import argparse
import os

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PAPER_FIG_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "writing", "698c58632d8bba3fe6c13a59", "figures"
)

CSV_FILE = "minimum_resources_comparison.csv"
OUTPUT_FILENAME = "minimum_resources.pdf"

BASE_MODELS = [
    "GST-S",
    "GST-C2",
    "GST-C3",
    "GST-M",
    "GST-L",
    "SA1",
    "AP1",
    "WA1",
    "WA2",
]

FAST_MODELS = ["GST-S", "GST-C2", "GST-C3", "GST-M"]
LARGE_MODELS = ["GST-L", "SA1", "AP1", "WA1", "WA2"]

COLORS = {
    "AD": "#5a9e91",
    "AD_overhead": "#8a8078",
    "FD": "#c4a265",
    "teval_tick": "#333333",
}

FIGURE_WIDTH = 7.16
FIGURE_HEIGHT = 3.5
BAR_WIDTH = 0.35
FONT_SIZE_TICKS = 9
FONT_SIZE_LABELS = 10
FONT_SIZE_SPEEDUP = 9
FONT_SIZE_NODES = 8
EDGECOLOR = "black"
LINEWIDTH = 0.5


def load_data():
    path = os.path.join(DATA_DIR, CSV_FILE)
    df = pd.read_csv(path)
    df = df[df["model"].isin(BASE_MODELS)].copy()

    df["_order"] = df["model"].map({m: i for i, m in enumerate(BASE_MODELS)})
    df = df.sort_values("_order").reset_index(drop=True)
    df = df.drop(columns=["_order"])
    return df


def make_label(row, c_ad=None):
    d = int(row["n_hyperparams"])
    base = f"{row['model']}\n($d$={d})"
    if c_ad is not None:
        if c_ad < 1.0:
            base += f"\n$c_{{\\mathrm{{AD}}}}$={c_ad:.2f}"
        else:
            base += f"\n$c_{{\\mathrm{{AD}}}}$={c_ad:.1f}"
    return base


def plot_panel(ax, df_panel, ylim_top, panel_title, skip_nodes=None):
    """Plot a single panel of the grouped bar chart.

    AD bars use two-tone encoding: blue up to t_eval (one FD evaluation cost),
    warm amber for overhead above t_eval (when c_AD > 1).
    FD bars have a tick mark at t_eval for visual reference.
    """
    skip_nodes = skip_nodes or set()
    n = len(df_panel)
    x = np.arange(n)

    models = df_panel["model"].values
    ad_times = df_panel["ad_gradient_time_mean"].values.astype(float)
    fd_times = df_panel["fd_gradient_time_mean"].values.astype(float)
    ad_std = df_panel["ad_gradient_time_std"].values.astype(float)
    fd_std = df_panel["fd_gradient_time_std"].values.astype(float)
    ad_n = df_panel["ad_n_samples"].values.astype(float)
    fd_n = df_panel["fd_n_samples"].values.astype(float)
    nodes = df_panel["min_nodes_ad"].values.astype(int)
    speedups = fd_times / ad_times

    d_vals = df_panel["n_hyperparams"].values.astype(float)
    t_eval = fd_times / (2 * d_vals + 1)
    c_ad = ad_times / t_eval
    labels = [
        make_label(row, c_ad=c_ad[i])
        for i, (_, row) in enumerate(df_panel.iterrows())
    ]

    ad_ci = 1.96 * ad_std / np.sqrt(ad_n)
    fd_ci = 1.96 * fd_std / np.sqrt(fd_n)

    # --- AD bars: two-tone when c_AD > 1 ---
    for i in range(n):
        ad_x = x[i] - BAR_WIDTH / 2
        if c_ad[i] <= 1.0:
            ax.bar(
                ad_x, ad_times[i], BAR_WIDTH,
                yerr=ad_ci[i], capsize=3,
                color=COLORS["AD"], edgecolor=EDGECOLOR, linewidth=LINEWIDTH,
                error_kw={"linewidth": 0.8, "zorder": 5},
                zorder=3,
            )
        else:
            ax.bar(
                ad_x, t_eval[i], BAR_WIDTH,
                color=COLORS["AD"], edgecolor=EDGECOLOR, linewidth=LINEWIDTH,
                zorder=3,
            )
            ax.bar(
                ad_x, ad_times[i] - t_eval[i], BAR_WIDTH,
                bottom=t_eval[i],
                yerr=ad_ci[i], capsize=3,
                color=COLORS["AD_overhead"], edgecolor=EDGECOLOR,
                linewidth=LINEWIDTH,
                error_kw={"linewidth": 0.8, "zorder": 5},
                zorder=3,
            )

    # --- FD bars ---
    ax.bar(
        x + BAR_WIDTH / 2,
        fd_times,
        BAR_WIDTH,
        yerr=fd_ci,
        capsize=3,
        color=COLORS["FD"],
        edgecolor=EDGECOLOR,
        linewidth=LINEWIDTH,
        error_kw={"linewidth": 0.8, "zorder": 5},
        zorder=3,
    )

    # --- t_eval tick marks on FD bars ---
    for i in range(n):
        fd_left = x[i]
        fd_right = x[i] + BAR_WIDTH
        ax.plot(
            [fd_left, fd_right], [t_eval[i], t_eval[i]],
            color=COLORS["teval_tick"], linewidth=1.2, zorder=4,
        )

    # Speedup annotations
    for i in range(n):
        ax.annotate(
            f"{speedups[i]:.1f}$\\times$",
            xy=(x[i] + BAR_WIDTH / 2, fd_times[i] + fd_ci[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_SPEEDUP,
            fontweight="bold",
        )

    # Node count annotations
    node_y = ylim_top * 0.02
    for i in range(n):
        m = models[i]
        if m in skip_nodes or nodes[i] == 1:
            continue
        ax.annotate(
            f"{nodes[i]}n",
            xy=(x[i], node_y),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_NODES,
            color="white",
            fontweight="bold",
        )

    ax.set_ylim(0, ylim_top)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS - 1)
    ax.set_ylabel("Per-gradient time (s)", fontsize=FONT_SIZE_LABELS)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(panel_title, fontsize=FONT_SIZE_LABELS)


def plot_minimum_resources(df):
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

    plot_panel(ax_fast, df_fast, ylim_top=3.5,
               panel_title="(a) Models with sub-second gradients")
    plot_panel(ax_large, df_large, ylim_top=750,
               panel_title="(b) Models with minute-scale gradients")

    ad_patch = mpatches.Patch(color=COLORS["AD"], label="AD (ours)")
    overhead_patch = mpatches.Patch(
        color=COLORS["AD_overhead"], label="AD overhead"
    )
    fd_patch = mpatches.Patch(color=COLORS["FD"], label="FD")
    teval_line = mlines.Line2D(
        [], [], color=COLORS["teval_tick"], linewidth=1.2,
        label="$t_{\\mathrm{eval}}$ (1 FD eval)",
    )
    ax_fast.legend(
        handles=[ad_patch, overhead_patch, fd_patch, teval_line],
        fontsize=FONT_SIZE_LABELS - 1,
        loc="upper left",
        framealpha=0.9,
        handlelength=1.2,
        handletextpad=0.4,
    )

    fig.tight_layout()
    return fig


def print_table(df):
    """Print a summary table for the paper."""
    print("\n% Minimum resources comparison table")
    print(f"{'Model':<8s} {'d':>3s} {'2d+1':>5s} {'P_AD':>4s} {'P_FD':>4s} "
          f"{'T_AD (s)':>10s} {'T_FD (s)':>10s} {'t_eval':>8s} {'c_AD':>6s} {'Speedup':>8s}")
    print("-" * 75)
    for _, row in df.iterrows():
        model = row["model"]
        d = int(row["n_hyperparams"])
        p_ad = int(row["min_nodes_ad"])
        ad_t = float(row["ad_gradient_time_mean"])

        fd_t = row["fd_gradient_time_mean"]
        if pd.isna(fd_t) or str(fd_t) == "infeasible":
            p_fd = "--"
            fd_str = "infeasible"
            t_eval_str = "--"
            c_ad_str = "--"
            spd = "--"
        else:
            fd_t = float(fd_t)
            p_fd = str(int(row["fd_n_nodes"]))
            fd_str = f"{fd_t:.3f}"
            t_eval = fd_t / (2 * d + 1)
            c_ad = ad_t / t_eval
            t_eval_str = f"{t_eval:.3f}"
            c_ad_str = f"{c_ad:.2f}"
            spd = f"{fd_t / ad_t:.1f}x"

        print(f"{model:<8s} {d:>3d} {2*d+1:>5d} {p_ad:>4d} {p_fd:>4s} "
              f"{ad_t:>10.3f} {fd_str:>10s} {t_eval_str:>8s} {c_ad_str:>6s} {spd:>8s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=PAPER_FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data()
    fig = plot_minimum_resources(df)

    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")

    output_png = output_path.replace(".pdf", ".png")
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_png}")

    print_table(df)
    plt.close(fig)
