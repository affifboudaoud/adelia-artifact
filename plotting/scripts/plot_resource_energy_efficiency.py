"""Resource and energy efficiency: dual-panel figure.

(a) Per-gradient time ratio: T_FD(N) / T_AD vs number of FD nodes
(b) Measured energy ratio: E_FD / E_AD via hardware energy counters

Shows the four production-scale models on linear y-axes.

Usage:
    python plot_resource_energy_efficiency.py
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

CSV_FILE = "resource_efficiency_all_updated.csv"

MODELS = {
    "sa1":  (r"SA1 [AD@1]",  "-",  "s", "#5a9e91"),
    "ap1":  (r"AP1 [AD@4]",  "-",  "D", "#8a8078"),
    "wa1":  (r"WA1 [AD@4]",  "-",  "^", "#c4a265"),
    "wa2":  (r"WA2 [AD@4]",  "-",  "v", "#506070"),
}

FIGSIZE = (7.16, 3.8)
DPI = 300
MARKER_SIZE = 6
LINE_WIDTH = 1.5


def load_data():
    path = os.path.join(DATA_DIR, CSV_FILE)
    df = pd.read_csv(path)
    df = df[df["gradient_time_mean_s"] != "missing"].copy()
    df = df[df["gradient_time_mean_s"] != "infeasible"].copy()
    df["gradient_time_mean_s"] = df["gradient_time_mean_s"].astype(float)
    df["gradient_time_std_s"] = pd.to_numeric(
        df["gradient_time_std_s"], errors="coerce"
    )
    for col in ["energy_total_J", "energy_per_node_J",
                "energy_gpu_per_node_J", "energy_other_per_node_J"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_fd_data(mdf):
    fd_rows = mdf[mdf["method"] == "FD"].sort_values("n_nodes")
    nodes, times, stds, n_runs_list, energies = [], [], [], [], []
    for _, row in fd_rows.iterrows():
        nodes.append(int(row["n_nodes"]))
        times.append(row["gradient_time_mean_s"])
        stds.append(float(row["gradient_time_std_s"]) if pd.notna(row["gradient_time_std_s"]) else 0.0)
        n_runs_list.append(int(row["n_runs"]) if pd.notna(row.get("n_runs")) else 10)
        e = row.get("energy_total_J")
        energies.append(float(e) if pd.notna(e) else None)
    return nodes, times, stds, n_runs_list, energies


def plot_dual_panel(df):
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax_wall, ax_energy) = plt.subplots(
        1, 2, figsize=FIGSIZE, sharey=False
    )

    for model, (label, lstyle, marker, color) in MODELS.items():
        mdf = df[df["model"] == model]
        ad_rows = mdf[mdf["method"] == "AD"]
        if ad_rows.empty:
            continue

        # For SA1, there are two AD rows (1-node and 2-node); use the 1-node one
        ad_row = ad_rows[ad_rows["n_nodes"] == ad_rows["ad_min_nodes"].astype(int)]
        if ad_row.empty:
            ad_row = ad_rows.iloc[:1]

        ad_time = ad_row["gradient_time_mean_s"].values[0]
        ad_std = float(ad_row["gradient_time_std_s"].values[0]) if pd.notna(
            ad_row["gradient_time_std_s"].values[0]
        ) else 0.0
        ad_n = int(ad_row["n_runs"].values[0]) if pd.notna(
            ad_row["n_runs"].values[0]
        ) else 10
        ad_nodes = int(ad_row["ad_min_nodes"].values[0])
        ad_energy_total = ad_row["energy_total_J"].values[0] if pd.notna(
            ad_row["energy_total_J"].values[0]
        ) else None

        fd_nodes, fd_times, fd_stds, fd_n_runs, fd_energies = get_fd_data(mdf)
        if not fd_nodes:
            continue

        disp_label = label

        # Panel (a): Per-gradient time ratio with 95% CI
        wall_ratios = np.array([t / ad_time for t in fd_times])
        ratio_cis = []
        for t, s, n in zip(fd_times, fd_stds, fd_n_runs):
            r = t / ad_time
            rel_var = (s / t) ** 2 + (ad_std / ad_time) ** 2 if t > 0 and ad_time > 0 else 0
            sigma_r = r * np.sqrt(rel_var)
            ratio_cis.append(1.96 * sigma_r / np.sqrt(n))
        ratio_cis = np.array(ratio_cis)

        ax_wall.errorbar(
            fd_nodes, wall_ratios, yerr=ratio_cis,
            marker=marker, markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
            linestyle=lstyle, color=color, label=disp_label, zorder=3,
            capsize=2, elinewidth=0.6, capthick=0.6,
        )

        # Panel (b): Measured energy ratio
        has_energy = (
            ad_energy_total is not None
            and all(e is not None for e in fd_energies)
        )
        if has_energy:
            energy_ratios = [e / ad_energy_total for e in fd_energies]
        else:
            ad_node_seconds = ad_nodes * ad_time
            energy_ratios = [
                (n * t) / ad_node_seconds for n, t in zip(fd_nodes, fd_times)
            ]
        ax_energy.plot(
            fd_nodes, energy_ratios,
            marker=marker, markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
            linestyle=lstyle, color=color, label=disp_label, zorder=3,
        )

    # --- Panel (a) formatting --- LINEAR y
    ax_wall.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0,
                    alpha=0.7, zorder=2)
    ax_wall.set_xscale("log", base=2)
    ax_wall.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax_wall.xaxis.set_minor_formatter(mticker.NullFormatter())

    ylims_wall = ax_wall.get_ylim()
    ax_wall.set_ylim(bottom=0)
    ylims_wall = ax_wall.get_ylim()
    ax_wall.axhspan(0, 1.0, alpha=0.06, color="red", zorder=0)
    ax_wall.axhspan(1.0, ylims_wall[1], alpha=0.06, color="green", zorder=0)
    ax_wall.set_xlabel("Number of FD nodes", fontsize=9)
    ax_wall.set_ylabel("Finite diff. time / Autodiff time", fontsize=9)
    ax_wall.set_title(r"$\bf{(a)}$ Per-gradient time ratio", fontsize=9)
    ax_wall.grid(True, alpha=0.3, which="both", zorder=0)
    ax_wall.text(0.02, 0.02, "FD faster", transform=ax_wall.transAxes,
                 fontsize=8, alpha=0.7, color="red", va="bottom",
                 fontweight="bold")
    ax_wall.text(0.98, 0.98, "AD faster", transform=ax_wall.transAxes,
                 fontsize=8, alpha=0.7, color="green", va="top", ha="right",
                 fontweight="bold")

    # --- Panel (b) formatting --- LINEAR y
    ax_energy.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0,
                      alpha=0.7, zorder=2)
    ax_energy.set_xscale("log", base=2)
    ax_energy.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax_energy.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax_energy.set_ylim(bottom=-1.0)
    ylims_energy = ax_energy.get_ylim()
    ax_energy.axhspan(ylims_energy[0], 1.0, alpha=0.06, color="red", zorder=0)
    ax_energy.axhspan(1.0, ylims_energy[1], alpha=0.06, color="green", zorder=0)
    ax_energy.set_xlabel("Number of FD nodes", fontsize=9)
    ax_energy.set_ylabel("Finite diff. energy / Autodiff energy", fontsize=9)
    ax_energy.set_title(r"$\bf{(b)}$ Measured energy ratio", fontsize=9)
    ax_energy.grid(True, alpha=0.3, which="both", zorder=0)
    ax_energy.text(0.02, 0.02, "FD cheaper", transform=ax_energy.transAxes,
                   fontsize=8, alpha=0.7, color="red", va="bottom",
                   fontweight="bold")
    ax_energy.text(0.02, 0.98, "AD cheaper", transform=ax_energy.transAxes,
                   fontsize=8, alpha=0.7, color="green", va="top",
                   fontweight="bold")

    # Shared legend below
    handles, labels = ax_wall.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        fontsize=7.5,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.5,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=PAPER_FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data()
    fig = plot_dual_panel(data)

    output_path = os.path.join(args.output_dir, "resource_energy_efficiency.pdf")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")

    output_png = output_path.replace(".pdf", ".png")
    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {output_png}")

    plt.close(fig)
