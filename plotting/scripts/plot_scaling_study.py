"""Scaling study: AD speedup vs problem size.

Two-panel figure:
  (a) WA1 temporal scaling (nt = 2, 32, 64, 128, 256)
  (b) WA2 spatial scaling (ns = 72, 282, 1119)

Shows that AD speedup is consistent across problem sizes.

Usage:
    python plot_scaling_study.py
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
EXPERIMENT_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "experiments", "fig5_scaling_study"
)
PAPER_FIG_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "writing", "698c58632d8bba3fe6c13a59", "figures"
)

CSV_FILE = "minimum_resources_comparison.csv"
SCALING_CSV = os.path.join(EXPERIMENT_DIR, "results", "scaling_results.csv")
OUTPUT_FILENAME = "scaling_study.pdf"

WA1_MODELS = ["WA1-small", "WA1-nt32", "WA1-nt64", "WA1-nt128", "WA1-nt256"]
WA1_NT = [2, 32, 64, 128, 256]

WA2_MODELS = ["WA2-ns72", "WA2-ns282", "WA2-ns1119"]
WA2_NS = [72, 282, 1119]

COLOR_AD = "#5a9e91"
COLOR_FD = "#c4a265"

FIGURE_WIDTH = 7.16
FIGURE_HEIGHT = 2.8


def load_data(csv_path=None):
    """Load scaling study data, merging experiment results with baseline data.

    For models present in the experiment CSV, AD columns come from the
    experiment and FD columns are filled from the baseline when missing.
    This allows running AD-only experiments on multi-node while reusing
    existing FD measurements from single-node runs.
    """
    if csv_path is not None:
        return pd.read_csv(csv_path)

    baseline_path = os.path.join(DATA_DIR, CSV_FILE)
    df = pd.read_csv(baseline_path)

    if not os.path.exists(SCALING_CSV):
        return df

    df_exp = pd.read_csv(SCALING_CSV)
    ad_cols = ["ad_gradient_time_mean", "ad_gradient_time_std", "ad_n_samples"]
    fd_cols = ["fd_gradient_time_mean", "fd_gradient_time_std", "fd_n_samples",
               "fd_n_nodes"]

    for _, exp_row in df_exp.iterrows():
        model = exp_row["model"]
        base_match = df[df["model"] == model]

        if not base_match.empty:
            idx = base_match.index[0]
            # Update AD columns from experiment
            for col in ad_cols:
                if col in exp_row and exp_row[col] > 0:
                    df.loc[idx, col] = exp_row[col]
            # Fill FD from experiment if available, otherwise keep baseline
            for col in fd_cols:
                if col in exp_row and exp_row[col] > 0:
                    df.loc[idx, col] = exp_row[col]
            # Recompute derived columns
            ad_t = df.loc[idx, "ad_gradient_time_mean"]
            fd_t = df.loc[idx, "fd_gradient_time_mean"]
            if ad_t > 0 and fd_t > 0:
                df.loc[idx, "per_gradient_speedup"] = fd_t / ad_t
        else:
            # New model not in baseline — append as-is
            df = pd.concat([df, exp_row.to_frame().T], ignore_index=True)

    return df


def get_model_data(df, model_names):
    """Extract AD/FD times, 95% CIs, speedups, and c_AD for a list of model names."""
    ad_times, fd_times, speedups, c_ads = [], [], [], []
    ad_ci, fd_ci = [], []
    for name in model_names:
        row = df[df["model"] == name]
        if row.empty:
            for lst in (ad_times, fd_times, speedups, c_ads, ad_ci, fd_ci):
                lst.append(np.nan)
            continue
        row = row.iloc[0]
        ad_t = float(row["ad_gradient_time_mean"])
        fd_t = float(row["fd_gradient_time_mean"])
        d = int(row["n_hyperparams"])
        t_eval = fd_t / (2 * d + 1)
        ad_times.append(ad_t)
        fd_times.append(fd_t)
        speedups.append(fd_t / ad_t)
        c_ads.append(ad_t / t_eval)
        ad_n = float(row["ad_n_samples"])
        fd_n = float(row["fd_n_samples"])
        ad_ci.append(1.96 * float(row["ad_gradient_time_std"]) / np.sqrt(ad_n))
        fd_ci.append(1.96 * float(row["fd_gradient_time_std"]) / np.sqrt(fd_n))
    return (np.array(ad_times), np.array(fd_times), np.array(speedups),
            np.array(c_ads), np.array(ad_ci), np.array(fd_ci))


def plot_scaling(df):
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    w = 0.35

    # --- Panel (a): WA1 temporal scaling ---
    wa1_ad, wa1_fd, wa1_spd, wa1_cad, wa1_ad_ci, wa1_fd_ci = get_model_data(df, WA1_MODELS)
    x1 = np.arange(len(WA1_NT))

    ax1.bar(x1 - w / 2, wa1_ad, w, yerr=wa1_ad_ci, capsize=3,
            color=COLOR_AD, edgecolor="black", linewidth=0.5,
            error_kw={"linewidth": 0.8, "zorder": 5}, label="AD (ours)", zorder=3)
    ax1.bar(x1 + w / 2, wa1_fd, w, yerr=wa1_fd_ci, capsize=3,
            color=COLOR_FD, edgecolor="black", linewidth=0.5,
            error_kw={"linewidth": 0.8, "zorder": 5}, label="FD", zorder=3)

    for i, spd in enumerate(wa1_spd):
        if not np.isnan(spd):
            ax1.annotate(
                f"{spd:.1f}$\\times$",
                xy=(x1[i] + w / 2, wa1_fd[i] + wa1_fd_ci[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    wa1_labels = []
    for i, nt in enumerate(WA1_NT):
        if np.isnan(wa1_cad[i]):
            wa1_labels.append(str(nt))
        else:
            wa1_labels.append(f"{nt}\n$c_\\mathrm{{AD}}$={wa1_cad[i]:.1f}")

    ax1.set_ylim(0, max(wa1_fd) * 1.15)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(wa1_labels, fontsize=7)
    ax1.set_xlabel("Number of time steps ($n_t$)", fontsize=9)
    ax1.set_ylabel("Per-gradient time (s)", fontsize=9)
    ax1.set_title("(a) WA1: Temporal scaling", fontsize=9, fontweight="bold")
    ax1.legend().set_visible(False)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel (b): WA2 spatial scaling ---
    wa2_ad, wa2_fd, wa2_spd, wa2_cad, wa2_ad_ci, wa2_fd_ci = get_model_data(df, WA2_MODELS)
    x2 = np.arange(len(WA2_NS))

    ax2.bar(x2 - w / 2, wa2_ad, w, yerr=wa2_ad_ci, capsize=3,
            color=COLOR_AD, edgecolor="black", linewidth=0.5,
            error_kw={"linewidth": 0.8, "zorder": 5}, label="AD (ours)", zorder=3)
    ax2.bar(x2 + w / 2, wa2_fd, w, yerr=wa2_fd_ci, capsize=3,
            color=COLOR_FD, edgecolor="black", linewidth=0.5,
            error_kw={"linewidth": 0.8, "zorder": 5}, label="FD", zorder=3)

    for i, spd in enumerate(wa2_spd):
        if not np.isnan(spd):
            ax2.annotate(
                f"{spd:.1f}$\\times$",
                xy=(x2[i] + w / 2, wa2_fd[i] + wa2_fd_ci[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    wa2_labels = []
    for i, ns in enumerate(WA2_NS):
        if np.isnan(wa2_cad[i]):
            wa2_labels.append(str(ns))
        else:
            wa2_labels.append(f"{ns}\n$c_\\mathrm{{AD}}$={wa2_cad[i]:.1f}")

    ax2.set_ylim(0, max(wa2_fd) * 1.15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(wa2_labels, fontsize=7)
    ax2.set_xlabel("Spatial mesh nodes ($n_s$)", fontsize=9)
    ax2.set_ylabel("Per-gradient time (s)", fontsize=9)
    ax2.set_title("(b) WA2: Spatial scaling", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=PAPER_FIG_DIR)
    parser.add_argument("--csv", default=None,
                        help="Path to CSV file (default: merge experiment + baseline)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(csv_path=args.csv)
    fig = plot_scaling(df)

    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")

    output_png = output_path.replace(".pdf", ".png")
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_png}")

    plt.close(fig)
