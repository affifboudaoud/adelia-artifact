#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

cd "$(dirname "$0")"

srun python run.py --benchmark_mode --benchmark_method jax_autodiff --n_benchmark_runs 3 --precision float64

echo "Job finished at $(date)"
