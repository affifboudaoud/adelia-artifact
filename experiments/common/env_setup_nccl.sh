#!/bin/bash
# Environment setup with NCCL and GPU-aware MPI enabled.
# Used by FD scaling experiments at higher node counts.
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"

export MPI_CUDA_AWARE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export USE_NCCL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
