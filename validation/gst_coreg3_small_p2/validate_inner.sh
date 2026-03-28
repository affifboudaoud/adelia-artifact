#!/bin/bash
source "$(dirname "$0")/../../experiments/common/env_setup.sh"

export MPI_CUDA_AWARE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export USE_NCCL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$(dirname "$0")"

echo "=== Phase 1: CuPy/serinv FD (reference, P=2) ==="
srun python validate.py --precision float64 --phase 1 --solver_min_p 2

echo "=== Phase 2: JAX AD (P=2) ==="
srun python validate.py --precision float64 --phase 2

echo "Validation for gst_coreg3_small_p2 finished at $(date)"
