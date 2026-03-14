#!/bin/bash
# GST-S FD scaling (solver_min_p=1, nocuda).
source "$(dirname "$0")/../../../common/env_setup_nocuda.sh"

OUTPUT_DIR="${ARTIFACT_DIR}/plotting/data"
N_RUNS=${N_BENCHMARK_RUNS:-10}

cd "$(dirname "$0")"

srun python ../../../fig2_wallclock/gst_small/run.py \
    --benchmark_mode \
    --benchmark_method finite_diff \
    --n_benchmark_runs $N_RUNS \
    --energy_monitor \
    --output_csv "${OUTPUT_DIR}/fig6_raw_results.csv" \
    --precision float64

echo "Job finished at $(date)"
