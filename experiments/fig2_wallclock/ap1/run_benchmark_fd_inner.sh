#!/bin/bash
source "$(dirname "$0")/../../common/env_setup_nocuda.sh"

cd "$(dirname "$0")"

srun python run.py --benchmark_mode --benchmark_method finite_diff --n_benchmark_runs 3 --precision float64 --solver_min_p 4

echo "Job finished at $(date)"
