#!/bin/bash
source "$(dirname "$0")/../../common/env_setup_nocuda.sh"

cd "$(dirname "$0")"

MAX_ITER=${MAX_ITER:-250}
srun python run.py --max_iter $MAX_ITER --precision float64 --skip_jax --solver_min_p 4

echo "Job finished at $(date)"
