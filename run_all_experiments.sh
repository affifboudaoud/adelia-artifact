#!/bin/bash
# Submit all artifact experiments to SLURM.
# Edit YOUR_ACCOUNT in each .sbatch file before running.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================================"
echo "ADELIA Artifact: Submitting all experiments"
echo "======================================================"
echo ""

# Gradient correctness validation (1 node, CPU only)
echo "Submitting gradient correctness validation..."
cd "$SCRIPT_DIR/validation"
sbatch validate_gradient_correctness.sbatch
echo ""

# Table 2: Structure comparison (1 node)
echo "Submitting Table 2..."
cd "$SCRIPT_DIR/experiments/tab2_structure_comparison"
sbatch run.sbatch
echo ""

# Figure 2: Wallclock for all 9 models
echo "Submitting Figure 2 (9 models)..."
for model in gst_small gst_medium gst_large gst_coreg2_small gst_coreg3_small sa1 ap1 wa1 wa2; do
    cd "$SCRIPT_DIR/experiments/fig2_wallclock/$model"
    sbatch run.sbatch
done
echo ""

# Figure 4: Framework decomposition
echo "Submitting Figure 4..."
cd "$SCRIPT_DIR/experiments/fig4_framework_decomposition"
sbatch run.sbatch
echo ""

# Figure 5: Scaling study
echo "Submitting Figure 5a (temporal scaling)..."
cd "$SCRIPT_DIR/experiments/fig5_scaling_study/wa1_temporal"
for f in run_[0-9]*.sbatch; do
    [ -f "$f" ] && sbatch "$f"
done

echo "Submitting Figure 5b (spatial scaling)..."
cd "$SCRIPT_DIR/experiments/fig5_scaling_study/wa2_spatial"
for f in run_ns*.sbatch; do
    [ -f "$f" ] && sbatch "$f"
done
echo ""

# Figure 6: Resource efficiency
echo "Submitting Figure 6 (FD scaling + AD baseline)..."
for model in sa1 ap1 wa1 wa2; do
    for f in "$SCRIPT_DIR/experiments/fig6_resource_efficiency/fd_scaling/$model"/*.sbatch; do
        [ -f "$f" ] && sbatch "$f"
    done
    for f in "$SCRIPT_DIR/experiments/fig6_resource_efficiency/ad_baseline/$model"/*.sbatch; do
        [ -f "$f" ] && sbatch "$f"
    done
done
echo ""

# Figure 7: Performance analysis
echo "Submitting Figure 7..."
cd "$SCRIPT_DIR/experiments/fig7_performance_analysis"
sbatch run.sbatch
echo ""

echo "All experiments submitted. Use 'squeue -u \$USER' to monitor."
