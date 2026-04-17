"""Plot convergence for AP1 and GST-T: AD vs FD. 2x2 grid."""
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
BASE_EXP = os.path.join(SCRIPT_DIR, "..", "experiments", "fig4_wallclock")
BASE_VAL = os.path.join(SCRIPT_DIR, "..", "validation")

OUTPUT_FILENAME = "convergence_ap1.pdf"
FIGURE_WIDTH = 3.5
FIGURE_HEIGHT = 2.2


def parse(path):
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


def parse_converge_core(path):
    """Parse converge_core.py output (different format)."""
    f_vals, g_norms = [], []
    with open(path) as fh:
        for line in fh:
            m = re.search(r'iter\s+\d+\s*\|\s*f=\s*([\d.e+-]+)\s*\|\s*\|\|grad\|\|=\s*([\d.e+-]+)', line)
            if m:
                f_vals.append(float(m.group(1)))
                g_norms.append(float(m.group(2)))
    return np.array(f_vals), np.array(g_norms)


# AP1
f_ad_ap1, g_ad_ap1 = parse(os.path.join(BASE_EXP, "ap1", "outputs", "ap1_ad_3011462.log"))
f_fd_ap1, g_fd_ap1 = parse(os.path.join(BASE_EXP, "ap1", "outputs", "ap1_fd_3011696.log"))

# GST-T
f_ad_gst, g_ad_gst = parse_converge_core(os.path.join(BASE_VAL, "gst_temperature", "outputs", "converge_3012136.log"))
f_fd_gst, g_fd_gst = parse(os.path.join(BASE_EXP, "gst_temperature", "outputs", "gst_t_fd_debug_3028192.log"))

models = [
    ("Temperature Monitoring (GST-T)", f_ad_gst, g_ad_gst, f_fd_gst, g_fd_gst),
    ("Air Pollution Monitoring (AP1)", f_ad_ap1, g_ad_ap1, f_fd_ap1, g_fd_ap1),
]

from matplotlib.ticker import FuncFormatter

def smart_fmt(x, _):
    if x == 0:
        return "0"
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x/1e6:.0f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{x:.0f}"

fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

for i, (name, f_ad, g_ad, f_fd, g_fd) in enumerate(models):
    ax_f = axes[0, i]
    ax_g = axes[1, i]

    ax_f.plot(range(1, len(f_ad) + 1), f_ad, "-",
              color=figure_style.AD_COLOR, linewidth=1.0)
    ax_f.plot(range(1, len(f_fd) + 1), f_fd, "o-",
              color=figure_style.FD_COLOR, linewidth=0.7,
              markersize=1.2, markeredgewidth=0.2, markeredgecolor=figure_style.SLATE_DARK)
    title_x = 0.25 if i == 0 else 0.45
    ax_f.set_title(name, fontsize=8, fontweight="bold", x=title_x)
    ax_f.yaxis.set_major_formatter(FuncFormatter(smart_fmt))
    ax_f.tick_params(labelsize=7.5)
    ax_f.tick_params(labelbottom=False)

    ax_g.plot(range(1, len(g_ad) + 1), g_ad, "-",
              color=figure_style.AD_COLOR, linewidth=1.0)
    ax_g.plot(range(1, len(g_fd) + 1), g_fd, "o-",
              color=figure_style.FD_COLOR, linewidth=0.7,
              markersize=1.2, markeredgewidth=0.2, markeredgecolor=figure_style.SLATE_DARK)
    ax_g.yaxis.set_major_formatter(FuncFormatter(smart_fmt))
    ax_g.tick_params(labelsize=7.5)
    if "AP1" in name:
        ax_f.set_xlim(0, 75)
        ax_g.set_xlim(0, 75)

axes[0, 0].set_ylabel("Objective $f(\\theta)$", fontsize=8, labelpad=2)
axes[1, 0].set_ylabel("Gradient norm $\\|\\nabla f\\|$", fontsize=8, y=0.35, labelpad=2)

fig.tight_layout(h_pad=0.3, w_pad=0.4, rect=[0, 0, 1, 0.97])

# x-axis labels back on axes
axes[1, 0].set_xlabel("Iteration", fontsize=8)
axes[1, 1].set_xlabel("Iteration", fontsize=8)

# Legend inside GST-T objective panel (top-left, has empty space at right)
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=figure_style.AD_COLOR, linewidth=1.0, label="ADELIA"),
    Line2D([0], [0], color=figure_style.FD_COLOR, linewidth=0.7, marker="o",
           markersize=2, label="DALIA (FD)"),
]
leg = axes[0, 0].legend(handles=handles, loc="upper right", fontsize=7.5,
           framealpha=0.95, edgecolor="black", fancybox=False,
           borderaxespad=0.3, borderpad=0.2)
leg.get_frame().set_linewidth(0.5)

# --- Annotations ---
# Bottom left (GST-T grad norm): same convergence quality
axes[1, 0].annotate("Same convergence\nfor small models",
                     xy=(0.55, 0.65), xycoords="axes fraction",
                     fontsize=8.5, ha="center", fontstyle="normal",
                     color=figure_style.TEXT_COLOR)

# Top right (AP1 objective): FD stalls
ax_ap1_f = axes[0, 1]
fd_last_iter = len(f_fd_ap1)
fd_last_f = f_fd_ap1[-1]
ax_ap1_f.annotate("FD stalls on\nlarge models",
                   xy=(fd_last_iter, fd_last_f),
                   xytext=(fd_last_iter + 12, fd_last_f - 2500),
                   fontsize=8.5, fontstyle="normal",
                   color=figure_style.TEXT_COLOR,
                   arrowprops=dict(arrowstyle="->", color=figure_style.SLATE_DARK,
                                   linewidth=0.7))

# Bottom right (AP1 grad norm): final grad norms
ax_ap1_g = axes[1, 1]
ad_final_g = g_ad_ap1[-1]
fd_final_g = g_fd_ap1[-1]

# Arrow to FD's last point
ax_ap1_g.annotate(f"FD: $\\|\\nabla f\\|$ = {fd_final_g/1e3:.0f}k",
                   xy=(len(g_fd_ap1), fd_final_g),
                   xytext=(len(g_fd_ap1) + 10, fd_final_g + 6000),
                   fontsize=7, fontstyle="normal",
                   color=figure_style.FD_COLOR, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=figure_style.FD_COLOR,
                                   linewidth=0.9))

# Arrow to AD's value near end of visible range
visible_iter = 70
ad_at_visible = g_ad_ap1[visible_iter - 1]
ax_ap1_g.annotate(f"AD: $\\|\\nabla f\\|$ = {ad_final_g:.1f}",
                   xy=(visible_iter, ad_at_visible),
                   xytext=(visible_iter - 40, ad_at_visible + 6000),
                   fontsize=7, fontstyle="normal",
                   color=figure_style.AD_COLOR, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=figure_style.AD_COLOR,
                                   linewidth=0.9))

out_path = os.path.join(PAPER_FIG_DIR, OUTPUT_FILENAME)
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)

for name, f_ad, g_ad, f_fd, g_fd in models:
    print(f"{name}: AD {len(f_ad)} iters ||g||={g_ad[-1]:.1f}, FD {len(f_fd)} iters ||g||={g_fd[-1]:.1f}")
print(f"Saved {out_path}")
