#!/bin/bash
source "$(dirname "$0")/../../../common/env_setup.sh"

OUTPUT_DIR="${ARTIFACT_DIR}/plotting/data"

cd "$(dirname "$0")"

srun python ../../../fig4_wallclock/gst_coreg3_small/run.py \
    --benchmark_mode \
    --benchmark_method jax_autodiff \
    --n_benchmark_runs 10 \
    --energy_monitor \
    --output_csv "${OUTPUT_DIR}/fig6_raw_results.csv" \
    --precision float64

echo "Job finished at $(date)"
