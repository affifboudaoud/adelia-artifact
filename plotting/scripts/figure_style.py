"""Shared figure style for all paper plots.

Ensures consistent fonts, sizes, and colors across all figures.
The font choice (STIX) matches the Times family used by IEEEtran.
"""

import matplotlib

RCPARAMS = {
    # Font: STIX matches Times (IEEEtran body text)
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    # Embed fonts as outlines for portability
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Axes background
    "axes.facecolor": "#F5F3F4",
    # Standardized sizes (pt)
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
}

# Palette
AD_COLOR = "#E8899A"
FD_COLOR = "#6B7B8D"
ACCENT_COLOR = "#D4A843"
TEXT_COLOR = "#2B2D42"
BG_COLOR = "#F5F3F4"
AD_LIGHT = "#F2BCC5"
AD_LIGHTER = "#F7D5DC"
SLATE_LIGHT = "#9AABB8"
GOLD_LIGHT = "#E8CFA0"
SLATE_DARK = "#4A5568"

# Annotation sizes
FONT_ANNOT = 7.5
FONT_ANNOT_SMALL = 7


def apply():
    """Apply the shared style to matplotlib."""
    matplotlib.rcParams.update(RCPARAMS)
