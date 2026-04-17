#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

cd "$(dirname "$0")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../results"
N_RUNS=${N_BENCHMARK_RUNS:-10}

srun python run_framework.py \
    --n_benchmark_runs $N_RUNS \
    --output_dir "$OUTPUT_DIR" \
    --precision float64

echo "Job finished at $(date)"
