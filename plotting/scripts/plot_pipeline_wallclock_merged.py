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
from matplotlib.legend_handler import HandlerBase
from matplotlib.path import Path as MplPath
from matplotlib.transforms import IdentityTransform
import numpy as np
import pandas as pd

import figure_style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

CSV_FILE = "pipeline_wallclock.csv"
OUTPUT_FILENAME = "pipeline_wallclock_merged.pdf"

MODEL_ORDER = [
    "GST-S", "GST-C2", "GST-M", "GST-C3",
    "GST-L", "WA1", "AP1", "WA2", "SA1",
]
FAST_MODELS = ["GST-S", "GST-C2", "GST-M", "GST-C3"]
LARGE_MODELS = ["GST-L", "WA1", "AP1", "WA2", "SA1"]

COLORS = {
    "fd_optim": "#6B7B8D",
    "fd_hessian": "#6B7B8D",
    "ad_optim": "#E8899A",
    "ad_hessian": "#E8899A",
}
HATCH_HESSIAN = "///"

FIGURE_WIDTH = 7.16
FIGURE_HEIGHT = 2.8
BAR_WIDTH = 0.30
FONT_SIZE_SPEEDUP = 7.5
EDGECOLOR = "black"
LINEWIDTH = 0.5


def _add_right_half_hatch(annotation):
    """Monkey-patch the bbox to draw hatching only on its right half."""
    patch = annotation.get_bbox_patch()
    _base_draw = patch.draw

    def _draw_split(renderer):
        _base_draw(renderer)
        tp = patch.get_path().transformed(patch.get_transform())
        bb = tp.get_extents()
        mid = (bb.x0 + bb.x1) * 0.5
        clip = MplPath(np.array([
            [mid, bb.y0 - 1], [bb.x1 + 1, bb.y0 - 1],
            [bb.x1 + 1, bb.y1 + 1], [mid, bb.y1 + 1],
            [mid, bb.y0 - 1],
        ]))
        orig = (patch.get_hatch(), patch.get_facecolor(), patch.get_linewidth())
        patch.set_hatch("///")
        patch.set_facecolor("none")
        patch.set_linewidth(0)
        patch.set_clip_path(clip, IdentityTransform())
        _base_draw(renderer)
        patch.set_hatch(orig[0])
        patch.set_facecolor(orig[1])
        patch.set_linewidth(orig[2])
        patch.set_clip_path(None)

    patch.draw = _draw_split


class _SplitPatchHandler(HandlerBase):
    """Legend handler: left-half solid, right-half hatched."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        left = mpatches.Rectangle(
            (xdescent, ydescent), width / 2, height,
            facecolor="#E8899A", alpha=0.35,
            edgecolor="#2B2D42", linewidth=0.4, transform=trans,
        )
        right = mpatches.Rectangle(
            (xdescent + width / 2, ydescent), width / 2, height,
            facecolor="#E8899A", alpha=0.35,
            edgecolor="#2B2D42", linewidth=0.4, hatch="///", transform=trans,
        )
        return [left, right]


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

        # 95% CI
        ad_ci = row.get("ci95_total_ad", 0) / scale if "ci95_total_ad" in row else 0
        fd_ci = row.get("ci95_total_fd", 0) / scale if "ci95_total_fd" in row else 0

        # FD bar (right)
        ax.bar(
            x[i] + BAR_WIDTH / 2, fd_optim, BAR_WIDTH,
            color=COLORS["fd_optim"], edgecolor="none",
            zorder=3,
        )
        ax.bar(
            x[i] + BAR_WIDTH / 2, fd_hessian, BAR_WIDTH,
            bottom=fd_optim,
            color=COLORS["fd_hessian"], edgecolor="#2B2D42",
            linewidth=0, hatch=HATCH_HESSIAN, zorder=3,
        )
        # Divider line between optimization and hessian
        ax.hlines(fd_optim, x[i], x[i] + BAR_WIDTH,
                  colors="black", linewidth=0.5, zorder=4)

        # AD bar (left)
        ax.bar(
            x[i] - BAR_WIDTH / 2, ad_optim, BAR_WIDTH,
            color=COLORS["ad_optim"], edgecolor="none",
            zorder=3,
        )
        ax.bar(
            x[i] - BAR_WIDTH / 2, ad_hessian, BAR_WIDTH,
            bottom=ad_optim,
            color=COLORS["ad_hessian"], edgecolor="#8B2040",
            linewidth=0, hatch=HATCH_HESSIAN, zorder=3,
        )
        # Divider line between optimization and hessian
        ax.hlines(ad_optim, x[i] - BAR_WIDTH, x[i],
                  colors="black", linewidth=0.5, zorder=4)

        # 95% CI error bars
        fd_top = fd_optim + fd_hessian
        ad_top = ad_optim + ad_hessian
        if fd_ci > 0:
            ax.errorbar(x[i] + BAR_WIDTH / 2, fd_top, yerr=fd_ci,
                        fmt="none", ecolor="#2B2D42", capsize=5, capthick=1.5,
                        linewidth=1.5, zorder=4)
        if ad_ci > 0:
            ax.errorbar(x[i] - BAR_WIDTH / 2, ad_top, yerr=ad_ci,
                        fmt="none", ecolor="#2B2D42", capsize=5, capthick=1.5,
                        linewidth=1.5, zorder=4)

        # Speedup badges matching bar-pair width
        bar_top = max(fd_top + fd_ci, ad_top + ad_ci)

        from matplotlib.offsetbox import AnnotationBbox, DrawingArea
        badge_w_pts = 27
        badge_h_pts = 14

        # Total wallclock speedup (bottom, half-solid / half-hatched)
        da_total = DrawingArea(badge_w_pts, badge_h_pts, 0, 0)
        da_total.add_artist(mpatches.Rectangle(
            (0, 0), badge_w_pts / 2, badge_h_pts,
            facecolor="#E8899A", alpha=0.35, edgecolor="none",
        ))
        da_total.add_artist(mpatches.Rectangle(
            (badge_w_pts / 2, 0), badge_w_pts / 2, badge_h_pts,
            facecolor="#E8899A", alpha=0.35, edgecolor="#8B2040",
            linewidth=0, hatch="///",
        ))
        da_total.add_artist(matplotlib.text.Text(
            badge_w_pts / 2, badge_h_pts / 2,
            f"{e2e_speedup:.1f}\u00d7",
            ha="center", va="center",
            fontsize=FONT_SIZE_SPEEDUP, fontweight="bold",
        ))
        ab_total = AnnotationBbox(
            da_total, (x[i], bar_top),
            xybox=(0, 9), boxcoords="offset points",
            frameon=False, annotation_clip=False, zorder=5,
        )
        ax.add_artist(ab_total)

        # Per-gradient speedup (top, solid background)
        da_grad = DrawingArea(badge_w_pts, badge_h_pts, 0, 0)
        da_grad.add_artist(mpatches.Rectangle(
            (0, 0), badge_w_pts, badge_h_pts,
            facecolor="#E8899A", alpha=0.35, edgecolor="none",
        ))
        da_grad.add_artist(matplotlib.text.Text(
            badge_w_pts / 2, badge_h_pts / 2,
            f"{per_grad_speedup:.1f}\u00d7",
            ha="center", va="center",
            fontsize=FONT_SIZE_SPEEDUP, fontweight="bold",
        ))
        ab_grad = AnnotationBbox(
            da_grad, (x[i], bar_top),
            xybox=(0, 25), boxcoords="offset points",
            frameon=False, annotation_clip=False, zorder=5,
        )
        ax.add_artist(ab_grad)

    labels = []
    for _, row in df_panel.iterrows():
        d = row["d"]
        t_eval = row["t_per_grad_fd"] / (2 * d + 1)
        c_ad = row["t_per_grad_ad"] / t_eval
        labels.append(f"{row['model']}\n$c_{{\\mathrm{{AD}}}}={c_ad:.1f}$")

    ax.set_ylim(0, ylim_top)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(f"Wall-clock time ({unit_label})")
    ax.set_title(panel_title)


def plot_pipeline(df):
    figure_style.apply()

    df_fast = df[df["model"].isin(FAST_MODELS)].reset_index(drop=True)
    df_large = df[df["model"].isin(LARGE_MODELS)].reset_index(drop=True)

    fig, (ax_fast, ax_large) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": [4, 5]},
    )

    plot_panel(ax_fast, df_fast, ylim_top=500,
               panel_title="(a) Small/medium models")
    plot_panel(ax_large, df_large, ylim_top=20,
               panel_title="(b) Large models", time_unit="h")

    total_saved_h = (
        df_large["t_total_fd"].sum() - df_large["t_total_ad"].sum()
    ) / 3600.0
    ax_large.text(
        0.02, 0.99,
        f"Total of $\\bf{{{total_saved_h:.0f}}}$ $\\bf{{hours}}$ saved across\nlarge models by using $\\bf{{ADELIA}}$",
        transform=ax_large.transAxes, fontsize=9,
        va="top", ha="left",
    )

    patches = [
        mpatches.Patch(color=COLORS["ad_optim"], label="Optimization (AD)"),
        mpatches.Patch(
            facecolor=COLORS["ad_hessian"], edgecolor="#8B2040",
            linewidth=0, hatch=HATCH_HESSIAN, label="Hessian (AD)",
        ),
        mpatches.Patch(color=COLORS["fd_optim"], label="Optimization (FD)"),
        mpatches.Patch(
            facecolor=COLORS["fd_hessian"], edgecolor="#2B2D42",
            linewidth=0, hatch=HATCH_HESSIAN, label="Hessian (FD)",
        ),
    ]
    ax_fast.legend(
        handles=patches,
        fontsize=7.5,
        loc="upper left",
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.tight_layout()
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
