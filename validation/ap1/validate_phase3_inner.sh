#!/bin/bash
# Phase 3: JAX FD + JAX AD in one job
source "$(dirname "$0")/../../experiments/common/env_setup.sh"

export MPI_CUDA_AWARE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export USE_NCCL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$(dirname "$0")"
srun python validate.py --precision float64 --phase 3 --fd_eps 1e-3
