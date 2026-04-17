#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

cd "$(dirname "$0")"

MAX_ITER=${MAX_ITER:-250}
srun python run.py --max_iter $MAX_ITER --precision float64 --distributed_method two_phase

echo "Job finished at $(date)"
