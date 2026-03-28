#!/bin/bash
source "$(dirname "$0")/../common/env_setup.sh"

cd "$(dirname "$0")"

srun python run_gst_t.py --benchmark_mode --n_benchmark_runs 20 --precision float64

echo "Job finished at $(date)"
