#!/bin/bash
# Environment setup template for ADELIA artifact experiments.
# Adapt module names, conda paths, and NVIDIA_BASE for your cluster.

module load cuda
module load gcc

unset LD_PRELOAD

export CUDA_DIR=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export ARRAY_MODULE=cupy

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS='--xla_gpu_autotune_level=0'

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate serinv-env

# Add NVIDIA pip-package libraries to LD_LIBRARY_PATH (required for JAX GPU)
NVIDIA_BASE=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia
for d in $NVIDIA_BASE/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

# Set artifact root (override with ARTIFACT_DIR if needed)
export ARTIFACT_DIR="${ARTIFACT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$ARTIFACT_DIR/source/dalia:$ARTIFACT_DIR/source/serinv:$PYTHONPATH"
