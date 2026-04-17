#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

cd "$(dirname "$0")"

# FD-only: skip JAX warmup, just set CuPy backend
export SKIP_JAX_WARMUP=1

MAX_ITER=${MAX_ITER:-250}
srun python run_fd.py --max_iter $MAX_ITER --precision float64

echo "Job finished at $(date)"
