#!/bin/bash
source "$(dirname "$0")/../../../common/env_setup.sh"

OUTPUT_DIR="${ARTIFACT_DIR}/plotting/data"

cd "$(dirname "$0")"

srun python ../../../fig2_wallclock/wa2/run.py \
    --benchmark_mode \
    --benchmark_method jax_autodiff \
    --n_benchmark_runs 10 \
    --energy_monitor \
    --output_csv "${OUTPUT_DIR}/fig6_raw_results.csv" \
    --precision float64 \
    --distributed_method two_phase

echo "Job finished at $(date)"
