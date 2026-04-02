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

# Activate conda environment
# Auto-detect conda installation path
if [ -n "$CONDA_EXE" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif command -v conda &> /dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
else
    CONDA_BASE="$HOME/miniconda3"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adelia-env

# Add NVIDIA pip-package libraries to LD_LIBRARY_PATH (required for JAX GPU)
NVIDIA_BASE=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia
for d in $NVIDIA_BASE/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

# Point XLA to libdevice.10.bc from nvidia-cuda-nvcc pip package
NVIDIA_NVCC_DIR=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$NVIDIA_NVCC_DIR"

# Set artifact root (override with ARTIFACT_DIR if needed)
export ARTIFACT_DIR="${ARTIFACT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$ARTIFACT_DIR/DALIA/src:$ARTIFACT_DIR/serinv/src:$PYTHONPATH"

# Data directory for production models (SA_1, WA_1, WA_2)
export ZENODO_DATA_DIR="${ZENODO_DATA_DIR:-$ARTIFACT_DIR/data}"
