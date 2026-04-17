#!/bin/bash
# AP1 FD scaling (solver_min_p=4, nocuda to avoid NCCL OOM/NaN at low node counts).
source "$(dirname "$0")/../../../common/env_setup_nocuda.sh"

OUTPUT_DIR="${ARTIFACT_DIR}/plotting/data"
N_RUNS=${N_BENCHMARK_RUNS:-10}

cd "$(dirname "$0")"

srun python ../../../fig4_wallclock/ap1/run.py \
    --benchmark_mode \
    --benchmark_method finite_diff \
    --n_benchmark_runs $N_RUNS \
    --solver_min_p 4 \
    --energy_monitor \
    --output_csv "${OUTPUT_DIR}/fig6_raw_results.csv" \
    --precision float64

echo "Job finished at $(date)"
