#!/bin/bash
source "$(dirname "$0")/../../experiments/common/env_setup.sh"

export MPI_CUDA_AWARE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export USE_NCCL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$(dirname "$0")"

echo "=== Phase 1: CuPy/serinv FD (reference) ==="
srun python validate.py --precision float64 --phase 1

echo "=== Phase 2: JAX AD ==="
srun python validate.py --precision float64 --phase 2

echo "Validation for ap1 finished at $(date)"
