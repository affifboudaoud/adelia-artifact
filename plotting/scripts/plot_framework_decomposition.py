"""Framework effect decomposition: observed vs algorithmic speedup.

Single-column dumbbell chart with colored background bands indicating the
framework ratio regime (r > 1, r ~ 1, r < 1) and r values on y-axis labels.

Usage:
    python plot_framework_decomposition.py
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

OUTPUT_FILENAME = "framework_decomposition.pdf"

COLOR_AD = "#E8899A"
COLOR_ALGO = "#D4A843"
COLOR_JAX_FASTER = "#F5DDE2"   # light pink for r > 1
COLOR_PARITY = "#EDECEB"       # warm light gray for r ~ 1
COLOR_CUPY_FASTER = "#DDE1E6"  # light slate for r < 1

FIGURE_WIDTH = 3.5
FIGURE_HEIGHT = 3.2

# r thresholds for the three regimes
R_HIGH = 1.3   # above this: JAX faster (r > 1)
R_LOW = 0.7    # below this: CuPy faster (r < 1)


def load_all_data():
    """Load and merge data from all required CSVs."""
    single_eval = pd.read_csv(os.path.join(DATA_DIR, "single_eval_comparison.csv"))
    memory = pd.read_csv(os.path.join(DATA_DIR, "memory_breakdown.csv"))

    # Build per-gradient speedup from fig6 raw results instead of the
    # old minimum_resources_comparison.csv.
    fig6_path = os.path.join(DATA_DIR, "fig6_raw_results.csv")
    minres = _build_minres_from_fig6(fig6_path)

    dist_path = os.path.join(DATA_DIR, "distributed_single_eval.csv")
    dist_eval = pd.read_csv(dist_path) if os.path.exists(dist_path) else pd.DataFrame()

    return single_eval, memory, minres, dist_eval


# Map from fig6 model names to display labels used by build_data().
_MODEL_LABEL_MAP = {
    "gst_small": "GST-S",
    "gst_medium": "GST-M",
    "gst_large": "GST-L",
    "gst_coreg2_small": "GST-C2",
    "gst_coreg3_small": "GST-C3",
    "sa1": "SA1",
    "ap1": "AP1",
    "wa1": "WA1",
    "wa2": "WA2",
}


def _build_minres_from_fig6(fig6_path):
    """Compute per-gradient speedup from fig6_raw_results.csv.

    For each model, takes the AD gradient time at min nodes and the FD
    gradient time at the same node count (minimum resources comparison).
    """
    raw = pd.read_csv(fig6_path)
    rows = []
    for model, label in _MODEL_LABEL_MAP.items():
        ad = raw[(raw["model"] == model) & (raw["method"] == "jax_autodiff")]
        if ad.empty:
            continue
        ad_row = ad.sort_values("n_nodes").iloc[0]
        ad_nodes = int(ad_row["n_nodes"])
        ad_time = float(ad_row["gradient_time_mean"])

        fd = raw[(raw["model"] == model) & (raw["method"] == "finite_diff")
                 & (raw["n_nodes"] == ad_nodes)]
        if fd.empty:
            continue
        fd_time = float(fd.iloc[0]["gradient_time_mean"])

        speedup = fd_time / ad_time if ad_time > 0 else 0
        rows.append({"model": label, "per_gradient_speedup": speedup})
    return pd.DataFrame(rows)


def build_data(single_eval, memory, minres, dist_eval):
    """Build observed vs algorithmic speedup + r value for all models."""
    rows = []

    model_map = {
        "gst_small": "GST-S",
        "gst_coreg2_small": "GST-C2",
        "gst_coreg3_small": "GST-C3",
        "gst_medium": "GST-M",
        "gst_large": "GST-L",
    }

    for m_key, m_label in model_map.items():
        ad_row = memory[(memory["model"] == m_key) & (memory["method"] == "AD")]
        if ad_row.empty:
            continue
        fwd = float(ad_row.iloc[0]["forward_time"])
        bwd = float(ad_row.iloc[0]["backward_time"])
        beta = bwd / fwd if fwd > 0 else 0

        se_row = single_eval[single_eval["model"] == m_key]
        if se_row.empty:
            continue
        d = int(se_row.iloc[0]["n_hyperparams"])
        r = float(se_row.iloc[0]["ratio_cupy_over_jax"])
        algo_speedup = (2 * d + 1) / (1 + beta)

        mr_row = minres[minres["model"] == m_label]
        if mr_row.empty:
            continue
        observed = float(mr_row.iloc[0]["per_gradient_speedup"])

        rows.append({
            "label": m_label,
            "algo_speedup": algo_speedup,
            "observed_speedup": observed,
            "r": r,
            "group": "single_node",
        })

    # SA1
    sa1_se = single_eval[single_eval["model"] == "sa1"]
    sa1_mr = minres[minres["model"] == "SA1"]
    if not sa1_se.empty and not sa1_mr.empty:
        r = float(sa1_se.iloc[0]["ratio_cupy_over_jax"])
        observed = float(sa1_mr.iloc[0]["per_gradient_speedup"])
        rows.append({
            "label": "SA1",
            "algo_speedup": observed / r,
            "observed_speedup": observed,
            "r": r,
            "group": "single_node",
        })

    # Distributed models
    dist_minres_map = {"ap1": "AP1", "wa1": "WA1", "wa2": "WA2"}
    if not dist_eval.empty:
        for _, row in dist_eval.iterrows():
            m = row["model"]
            if m not in dist_minres_map:
                continue
            r = float(row["ratio_cupy_over_jax"])
            mr_row = minres[minres["model"] == dist_minres_map[m]]
            if mr_row.empty:
                continue
            observed = float(mr_row.iloc[0]["per_gradient_speedup"])
            rows.append({
                "label": dist_minres_map[m],
                "algo_speedup": observed / r,
                "observed_speedup": observed,
                "r": r,
                "group": "distributed",
            })

    return pd.DataFrame(rows)


def plot_framework_decomposition(df):
    figure_style.apply()

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    if df.empty:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
        return fig

    y = np.arange(len(df))
    max_val = max(df["observed_speedup"].max(), df["algo_speedup"].max())
    xlim_right = max_val * 1.18

    # Background bands based on r value of each model
    # Track contiguous regime spans for right-side labels
    regime_spans = []  # (y_start, y_end, regime_key)
    for i in range(len(df)):
        r = df.iloc[i]["r"]
        if r > R_HIGH:
            color = COLOR_JAX_FASTER
            key = "jax"
        elif r < R_LOW:
            color = COLOR_CUPY_FASTER
            key = "cupy"
        else:
            color = COLOR_PARITY
            key = "parity"
        ax.axhspan(y[i] - 0.5, y[i] + 0.5, color=color, alpha=0.5, zorder=0)
        if regime_spans and regime_spans[-1][2] == key:
            regime_spans[-1] = (regime_spans[-1][0], y[i] + 0.5, key)
        else:
            regime_spans.append((y[i] - 0.5, y[i] + 0.5, key))

    # Right-side regime labels
    regime_labels = {
        "jax": "$r>1$\nJAX faster",
        "parity": "$r{\\approx}1$\nparity",
        "cupy": "$r<1$\nCuPy faster",
    }
    regime_colors = {
        "jax": COLOR_JAX_FASTER,
        "parity": "#6B7B8D",
        "cupy": COLOR_CUPY_FASTER,
    }
    for y_start, y_end, key in regime_spans:
        y_mid = (y_start + y_end) / 2
        if key == "cupy":
            y_mid = y_start + 0.5  # shift up to avoid legend overlap
        ax.text(
            xlim_right, y_mid, regime_labels[key],
            ha="right", va="center", fontsize=7,
            color="#2B2D42", style="italic",
        )

    # Connector lines
    for i in range(len(df)):
        obs = df.iloc[i]["observed_speedup"]
        algo = df.iloc[i]["algo_speedup"]
        ax.plot([algo, obs], [y[i], y[i]], color="#9AABB8",
                linewidth=1.5, zorder=2)

    # Plot dots -- use diamonds for distributed
    for grp, marker, ms in [("single_node", "o", 40),
                             ("distributed", "D", 40)]:
        mask = df["group"] == grp
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        kw = dict(marker=marker, s=ms, zorder=4, linewidths=0.8)
        label_algo = "Algorithmic ($r{=}1$)" if grp == "single_node" else None
        label_obs = "Observed" if grp == "single_node" else None
        ax.scatter(
            df.loc[mask, "algo_speedup"], y[idx],
            c=COLOR_ALGO, edgecolors=COLOR_ALGO, label=label_algo, **kw,
        )
        ax.scatter(
            df.loc[mask, "observed_speedup"], y[idx],
            c=COLOR_AD, edgecolors=COLOR_AD, label=label_obs, **kw,
        )

    # Annotations on dots
    for i in range(len(df)):
        obs = df.iloc[i]["observed_speedup"]
        algo = df.iloc[i]["algo_speedup"]
        right_val = max(obs, algo)
        left_val = min(obs, algo)
        rc = COLOR_AD if obs >= algo else COLOR_ALGO
        lc = COLOR_ALGO if obs >= algo else COLOR_AD

        ax.annotate(
            f"{right_val:.1f}$\\times$",
            xy=(right_val, y[i]), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=6.5, color=rc,
        )
        ax.annotate(
            f"{left_val:.1f}$\\times$",
            xy=(left_val, y[i]), xytext=(0, 6),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=6.5, color=lc,
        )

    # Y-axis: model name + r value
    ylabels = []
    for i in range(len(df)):
        label = df.iloc[i]["label"]
        r = df.iloc[i]["r"]
        ylabels.append(f"{label}  $r$={r:.1f}")

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, xlim_right)
    ax.set_ylim(len(df) - 0.5, -0.5)  # exact bounds so bands fill to axes
    ax.set_xlabel("Speedup over FD")
    ax.set_title("Observed vs. algorithmic speedup", fontweight="bold")

    # Legend for dots only
    ax.legend(loc="lower right", framealpha=0.9)

    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=PAPER_FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    single_eval, memory, minres, dist_eval = load_all_data()
    df = build_data(single_eval, memory, minres, dist_eval)

    fig = plot_framework_decomposition(df)

    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")

    output_png = output_path.replace(".pdf", ".png")
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_png}")

    plt.close(fig)
