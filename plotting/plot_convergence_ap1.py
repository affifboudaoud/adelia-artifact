"""Plot AP1 convergence: AD vs FD.

Usage:
    python plot_convergence_ap1.py
"""
import os
import re
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import figure_style

figure_style.apply()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

AD_LOG = os.path.join(
    SCRIPT_DIR, "..", "experiments", "fig2_wallclock", "ap1",
    "outputs", "ap1_ad_3011462.log")
FD_LOG = os.path.join(
    SCRIPT_DIR, "..", "experiments", "fig2_wallclock", "ap1",
    "outputs", "ap1_fd_3011696.log")

OUTPUT_FILENAME = "convergence_ap1.pdf"

FIGURE_WIDTH = 3.5
FIGURE_HEIGHT = 3.2
LINEWIDTH = 0.5
EDGECOLOR = "black"


def parse_trajectory(path):
    f_vals, g_norms = [], []
    with open(path) as fh:
        for line in fh:
            if "comm_rank: 0 | Iteration:" in line:
                m = re.search(
                    r"Iteration:\s*(\d+).*Function Value:\s*([\d.e+-]+)"
                    r".*Norm\(Grad\):\s*\[\s*([\d.e+-]+)", line)
                if m:
                    f_vals.append(float(m.group(2)))
                    g_norms.append(float(m.group(3)))
    return np.array(f_vals), np.array(g_norms)


f_ad, g_ad = parse_trajectory(AD_LOG)
f_fd, g_fd = parse_trajectory(FD_LOG)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

# -- Panel (a): f vs iteration --
ax1.plot(range(1, len(f_ad) + 1), f_ad, "-",
         color=figure_style.AD_COLOR, linewidth=1.3, label="ADELIA")
ax1.plot(range(1, len(f_fd) + 1), f_fd, "o-",
         color=figure_style.FD_COLOR, linewidth=0.9,
         markersize=2.0, markeredgewidth=0.3, markeredgecolor=figure_style.SLATE_DARK,
         label="DALIA (FD)")
ax1.set_ylabel("Objective $f(\\theta)$")
ax1.set_xlim(0, 75)
ax1.set_ylim(138000, 147000)
from matplotlib.ticker import FuncFormatter
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
ax1.tick_params(labelbottom=False)
ax1.legend(framealpha=0.95, edgecolor="black", fancybox=False,
           borderpad=0.4, loc="upper right")

# -- Panel (b): ||grad|| vs iteration --
ax2.plot(range(1, len(g_ad) + 1), g_ad, "-",
         color=figure_style.AD_COLOR, linewidth=1.3, label="ADELIA")
ax2.plot(range(1, len(g_fd) + 1), g_fd, "o-",
         color=figure_style.FD_COLOR, linewidth=0.9,
         markersize=2.0, markeredgewidth=0.3, markeredgecolor=figure_style.SLATE_DARK,
         label="DALIA (FD)")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Gradient norm $\\|\\nabla f\\|$")
ax2.set_xlim(0, 75)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

fig.tight_layout(h_pad=0.4)


out_path = os.path.join(PAPER_FIG_DIR, OUTPUT_FILENAME)
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"AD: {len(f_ad)} iters, f={f_ad[-1]:.2f}, ||grad||={g_ad[-1]:.2f}")
print(f"FD: {len(f_fd)} iters, f={f_fd[-1]:.2f}, ||grad||={g_fd[-1]:.2f}")
print(f"Saved {out_path}")
