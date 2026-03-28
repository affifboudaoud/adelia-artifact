#!/bin/bash
source "$(dirname "$0")/../../experiments/common/env_setup.sh"

cd "$(dirname "$0")"

MAX_ITER=${MAX_ITER:-200}
srun python converge.py --max_iter $MAX_ITER --precision float64

echo "Job finished at $(date)"
