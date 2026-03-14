#!/bin/bash
# Environment setup with GPU-aware MPI explicitly disabled.
# Used by FD baseline benchmarks to avoid NCCL OOM/NaN issues at low node counts.
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"

export MPI_CUDA_AWARE=0
export MPICH_GPU_SUPPORT_ENABLED=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
