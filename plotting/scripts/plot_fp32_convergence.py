#!/usr/bin/env python3
"""Generate FP32 vs FP64 convergence trajectory figure.

Produces a 1x3 panel figure showing optimization convergence for:
  (a) gs_small    — FP32 AD tracks FP64 closely
  (b) gst_small   — FP32 AD stops early, 8.8% objective difference
  (c) gst_medium  — FP32 AD false convergence, 1.9% objective difference

Usage:
    python plot_fp32_convergence.py
    python plot_fp32_convergence.py --output-dir path/to/figures/
"""

import argparse
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "..", "DALIA", "examples")
DEFAULT_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "..", "writing", "698c58632d8bba3fe6c13a59", "figures"
)

MODELS = [
    {
        "name": "gs_small",
        "label": "(a) GS ($d$=3)",
        "fp32": "A100_final_fp_20260128_152625_float32.txt",
        "fp64": "A100_final_fp_20260129_164127_float64.txt",
    },
    {
        "name": "gst_small",
        "label": "(b) GST-S ($d$=4)",
        "fp32": "A100_final_fp_20260128_152625_float32.txt",
        "fp64": "A100_final_fp_20260129_164127_float64.txt",
    },
    {
        "name": "gst_medium",
        "label": "(c) GST-M ($d$=4)",
        "fp32": "A100_final_fp_20260128_152625_float32.txt",
        "fp64": "A100_final_fp_20260129_164127_float64.txt",
    },
]

COLORS = {
    "AD FP64": "#4878CF",
    "AD FP32": "#D65F5F",
}
LINESTYLES = {
    "AD FP64": "-",
    "AD FP32": "-",
}
MARKERS = {
    "AD FP64": "o",
    "AD FP32": "^",
}

FIGURE_WIDTH = 7.16
FIGURE_HEIGHT = 2.2
FONT_SIZE_TICKS = 7
FONT_SIZE_LABELS = 8
FONT_SIZE_TITLES = 8
FONT_SIZE_LEGEND = 7
MARKER_SIZE = 3
LINEWIDTH = 1.0


ITER_PATTERN = re.compile(
    r"Iteration:\s*(\d+)\s*\(took:.*?\)\s*\|.*?Function Value:\s*([-\d.eE+nNaAiIfF]+)"
)


def parse_convergence(filepath):
    """Parse per-iteration objective values for FD and AD from an output file."""
    fd_iters, fd_vals = [], []
    ad_iters, ad_vals = [], []

    current_method = None
    with open(filepath, "r") as f:
        for line in f:
            if "RUNNING WITH FINITE DIFFERENCES" in line:
                current_method = "FD"
            elif "RUNNING WITH JAX AUTODIFF" in line:
                current_method = "AD"

            if current_method is None:
                continue

            m = ITER_PATTERN.search(line)
            if m:
                iteration = int(m.group(1))
                try:
                    val = float(m.group(2))
                except ValueError:
                    val = np.nan

                if current_method == "FD":
                    fd_iters.append(iteration)
                    fd_vals.append(val)
                elif current_method == "AD":
                    ad_iters.append(iteration)
                    ad_vals.append(val)

    return (
        np.array(fd_iters),
        np.array(fd_vals),
        np.array(ad_iters),
        np.array(ad_vals),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--output-name", default="convergence_fp32.pdf", help="Output filename"
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

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for ax, model in zip(axes, MODELS):
        outputs_dir = os.path.join(EXAMPLES_DIR, model["name"], "outputs")
        fp32_path = os.path.join(outputs_dir, model["fp32"])
        fp64_path = os.path.join(outputs_dir, model["fp64"])

        _, _, ad64_iters, ad64_vals = parse_convergence(fp64_path)
        _, _, ad32_iters, ad32_vals = parse_convergence(fp32_path)

        if len(ad64_iters) > 0:
            ax.plot(
                ad64_iters,
                ad64_vals,
                color=COLORS["AD FP64"],
                linestyle=LINESTYLES["AD FP64"],
                marker=MARKERS["AD FP64"],
                markersize=MARKER_SIZE,
                linewidth=LINEWIDTH,
                label="AD FP64",
                markevery=max(1, len(ad64_iters) // 10),
            )

        if len(ad32_iters) > 0:
            valid = ~np.isnan(ad32_vals)
            if valid.any():
                ax.plot(
                    ad32_iters[valid],
                    ad32_vals[valid],
                    color=COLORS["AD FP32"],
                    linestyle=LINESTYLES["AD FP32"],
                    marker=MARKERS["AD FP32"],
                    markersize=MARKER_SIZE,
                    linewidth=LINEWIDTH,
                    label="AD FP32",
                    markevery=max(1, int(valid.sum()) // 10),
                )

        ax.set_title(model["label"], fontsize=FONT_SIZE_TITLES, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=FONT_SIZE_LABELS)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Objective", fontsize=FONT_SIZE_LABELS)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=FONT_SIZE_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
