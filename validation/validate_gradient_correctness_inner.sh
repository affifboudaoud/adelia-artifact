#!/bin/bash
source "$(dirname "$0")/../experiments/common/env_setup.sh"

export JAX_PLATFORMS=cpu
export ARRAY_MODULE=numpy

cd "$(dirname "$0")"

python validate_gradient_correctness.py --precision float64

echo "Job finished at $(date)"
