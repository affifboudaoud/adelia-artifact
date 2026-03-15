"""Convergence profile plot: AD vs FD success rate under perturbation.

Multi-panel figure showing the fraction of trials that converge to the
reference optimum (strict criterion) as a function of perturbation
magnitude delta. Auto-detects models in the CSV and produces one panel
per model.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
PAPER_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

AD_COLOR = "#2166ac"
FD_COLOR = "#b2182b"

DPI = 300

PANEL_ORDER = ["gst_coreg2_small", "gst_coreg3_small", "wa2_small"]
PANEL_LABELS = {
    "gst_coreg2_small": r"GST-C2 ($d{=}9$, synthetic)",
    "gst_coreg3_small": r"GST-C3 ($d{=}15$, synthetic)",
    "wa2_small": r"WA2-S ($d{=}15$, real-world)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot robustness convergence profile"
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(DATA_DIR, "robustness_wa2.csv"),
    )
    parser.add_argument("--output-name", type=str, default="robustness_profile")
    return parser.parse_args()


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "converged_strict" not in df.columns:
        raise ValueError("CSV missing converged_strict column")
    return df


def _wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for binomial proportion (returned as percentages)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return 100.0 * lo, 100.0 * p, 100.0 * hi


def compute_success_rates(df, model=None):
    """Compute success rate and Wilson 95% CI per (method, delta)."""
    if model is not None:
        df = df[df["model"] == model]
    rows = []
    for (method, delta), grp in df.groupby(["method", "delta"]):
        k = (grp["converged_strict"] == 1).sum()
        n = len(grp)
        lo, rate, hi = _wilson_ci(k, n)
        rows.append({
            "method": method,
            "delta": delta,
            "success_rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def _plot_single_panel(ax, rates_df, title=None, show_ylabel=True, show_legend=False):
    """Plot AD/FD lines with CI bands on a single axes."""
    for method, color, marker, ls, label in [
        ("AD", AD_COLOR, "o", "-", "AD"),
        ("FD", FD_COLOR, "^", "--", "FD"),
    ]:
        subset = rates_df[rates_df["method"] == method].sort_values("delta")
        ax.fill_between(
            subset["delta"], subset["ci_lo"], subset["ci_hi"],
            color=color, alpha=0.15, zorder=1,
        )
        ax.plot(
            subset["delta"], subset["success_rate"],
            color=color, marker=marker, markersize=4, linestyle=ls,
            linewidth=1.3, label=label, zorder=3,
        )

    ax.set_xscale("log")
    deltas = sorted(rates_df["delta"].unique())
    ax.set_xticks(deltas)
    ax.set_xticklabels([str(int(d)) if d == int(d) else str(d) for d in deltas])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel(r"Perturbation magnitude $\delta$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Success rate (%)", fontsize=8)
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    ax.grid(True, axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=8, pad=4)
    if show_legend:
        ax.legend(fontsize=7, framealpha=0.9, loc="lower left")
    ax.tick_params(labelsize=7)


def plot_profile_multi(df):
    """Multi-panel figure, one panel per model found in the data."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

    models_in_data = [m for m in PANEL_ORDER if m in df["model"].unique()]
    if not models_in_data:
        models_in_data = sorted(df["model"].unique())

    n_panels = len(models_in_data)

    if n_panels == 1:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        rates = compute_success_rates(df, models_in_data[0])
        title = PANEL_LABELS.get(models_in_data[0], models_in_data[0])
        _plot_single_panel(ax, rates, title=title, show_legend=True)
    else:
        fig, axes = plt.subplots(
            1, n_panels, figsize=(7.16, 2.5), sharey=True
        )
        for i, model in enumerate(models_in_data):
            rates = compute_success_rates(df, model)
            label_chr = chr(ord("a") + i)
            title = PANEL_LABELS.get(model, model)
            panel_title = f"({label_chr}) {title}"
            _plot_single_panel(
                axes[i], rates,
                title=panel_title,
                show_ylabel=(i == 0),
                show_legend=(i == 0),
            )

    fig.tight_layout()
    return fig


def plot_profile_single(df):
    """Single-panel figure (backward compat)."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    rates = compute_success_rates(df)
    _plot_single_panel(ax, rates, show_legend=True)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    df = load_data(args.input)

    n_models = df["model"].nunique()
    if n_models > 1:
        fig = plot_profile_multi(df)
    else:
        fig = plot_profile_single(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)

    for d in [OUTPUT_DIR, PAPER_DIR]:
        fig.savefig(
            os.path.join(d, f"{args.output_name}.pdf"),
            dpi=DPI, bbox_inches="tight",
        )
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{args.output_name}.png"),
        dpi=DPI, bbox_inches="tight",
    )

    print(f"Saved to {OUTPUT_DIR}/{args.output_name}.pdf")
    print(f"Saved to {PAPER_DIR}/{args.output_name}.pdf")


if __name__ == "__main__":
    main()
