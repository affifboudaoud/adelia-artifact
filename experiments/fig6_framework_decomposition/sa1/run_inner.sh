#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../results"
N_RUNS=${N_BENCHMARK_RUNS:-10}

echo "=== Framework comparison: sa1 ==="
cd "$(dirname "$0")/../../fig4_wallclock/sa1"
srun python run.py \
    --framework_comparison \
    --n_benchmark_runs $N_RUNS \
    --output_dir "$OUTPUT_DIR" \
    --precision float64

echo "Job finished at $(date)"
