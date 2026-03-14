#!/bin/bash
# Generate all paper figures from reference CSV data.
# Output: plotting/figures/*.pdf

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIGURES_DIR="$SCRIPT_DIR/figures"
mkdir -p "$FIGURES_DIR"

echo "Generating figures in $FIGURES_DIR ..."

cd "$SCRIPT_DIR/scripts"

python plot_structure_comparison.py
python plot_pipeline_wallclock.py
python plot_pipeline_wallclock_merged.py
python plot_framework_decomposition.py
python plot_scaling_study.py
python plot_resource_efficiency.py
python plot_resource_energy_efficiency.py
python plot_performance_analysis.py
python plot_breakdown_analysis.py
python plot_speedup_comparison.py
python plot_energy_efficiency.py
python plot_memory_analysis.py
python plot_minimum_resources.py
python plot_robustness_profile.py
python plot_fp32_convergence.py

echo "Done. Figures written to $FIGURES_DIR"
