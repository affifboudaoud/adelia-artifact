#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../results"
N_RUNS=${N_BENCHMARK_RUNS:-5}

echo "=== Framework comparison: wa2 (FD only, distributed) ==="
cd "$(dirname "$0")/../../fig4_wallclock/wa2"
srun python run.py \
    --framework_comparison \
    --framework_fd_only \
    --solver_min_p 4 \
    --n_benchmark_runs $N_RUNS \
    --output_dir "$OUTPUT_DIR" \
    --precision float64 \
    --distributed_method two_phase

echo "Job finished at $(date)"
