#!/bin/bash
# Generate all paper figures from reference CSV data in plotting/data/.
# Output: plotting/figures/*.pdf
#
# Paper-figure mapping (see plotting/README.md for the full table):
#   Table 2     -> scripts/plot_structure_comparison.py
#   Figure 3    -> plot_convergence_all.py
#   Figure 4    -> scripts/plot_pipeline_wallclock_merged.py
#   Figure 5    -> scripts/plot_scaling_study.py
#   Figure 6    -> scripts/plot_framework_decomposition.py
#   Figure 7    -> scripts/plot_resource_energy_efficiency.py
#   Figure 8    -> scripts/plot_performance_analysis.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIGURES_DIR="$SCRIPT_DIR/figures"
mkdir -p "$FIGURES_DIR"

echo "Generating figures in $FIGURES_DIR ..."

# Figure 3: convergence comparison (top-level script)
python "$SCRIPT_DIR/plot_convergence_all.py"

# Table 2 + Figures 4--8 (scripts/ directory)
cd "$SCRIPT_DIR/scripts"
python plot_structure_comparison.py
python plot_pipeline_wallclock_merged.py
python plot_scaling_study.py
python plot_framework_decomposition.py
python plot_resource_energy_efficiency.py
python plot_performance_analysis.py

echo "Done. Figures written to $FIGURES_DIR"
