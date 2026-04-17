#!/bin/bash
source "$(dirname "$0")/../../common/env_setup_nocuda.sh"

cd "$(dirname "$0")"

srun python run.py --max_iter 250 --precision float64 --skip_jax

echo "Job finished at $(date)"
