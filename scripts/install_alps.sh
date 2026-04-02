#!/bin/bash
# Full installation script for ADELIA artifact on CSCS Alps (GH200 nodes).
#
# This script must be run on a compute node with GPU access, e.g.:
#   salloc -N1 --gpus-per-task=1 -A <account> -t 30 -p debug
#   uenv run --view=modules prgenv-gnu/24.11:v1 -- bash scripts/install_alps.sh
#
# Prerequisites:
#   - Miniconda installed at ~/miniconda3
#   - uenv with prgenv-gnu/24.11:v1 available
#
# What this script does:
#   1. Creates the adelia-env conda environment (Python 3.11)
#   2. Installs JAX, CuPy, and NVIDIA CUDA packages
#   3. Builds mpi4py from source against Cray MPICH
#   4. Builds mpi4jax with CUDA support
#   5. Clones and installs DALIA and serinv (if not already present)
#   6. Downloads input data (~1.5 GB)
#   7. Verifies the installation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="adelia-env"

echo "=== ADELIA Artifact Installation ==="
echo "Artifact directory: $ARTIFACT_DIR"

# ── Step 1: Conda environment ──────────────────────────────────────────────
echo ""
echo "[1/7] Creating conda environment: $ENV_NAME"
source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "$ENV_NAME"; then
    echo "  Environment $ENV_NAME already exists, skipping creation."
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"

# ── Step 2: Core Python packages ───────────────────────────────────────────
echo ""
echo "[2/7] Installing Python packages (JAX, CuPy, etc.)"
pip install numpy scipy matplotlib pandas tabulate pydantic gdown Cython 2>&1 | tail -1

pip install jax==0.8.1 jaxlib==0.8.1 2>&1 | tail -1
pip install jax-cuda12-pjrt==0.8.1 jax-cuda12-plugin==0.8.1 2>&1 | tail -1
pip install cupy-cuda12x==13.6.0 2>&1 | tail -1

# NVIDIA CUDA packages (cuDNN, cuBLAS, cuSolver, libdevice, etc.)
pip install \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cufft-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-nccl-cu12 nvidia-cuda-nvcc-cu12 2>&1 | tail -1

# ── Step 3: mpi4py (built from source against Cray MPICH) ─────────────────
echo ""
echo "[3/7] Building mpi4py from source (Cray MPICH)"
module load cray-mpich gcc cuda 2>/dev/null || true

if python -c "from mpi4py import MPI" 2>/dev/null; then
    echo "  mpi4py already installed and working, skipping."
else
    CC=$(which mpicc) pip install --no-cache-dir --no-binary mpi4py mpi4py 2>&1 | tail -1
fi

# ── Step 4: mpi4jax with CUDA support ─────────────────────────────────────
echo ""
echo "[4/7] Building mpi4jax with CUDA support"

NEED_MPI4JAX=true
if python -c "import mpi4jax" 2>/dev/null; then
    if ls "$CONDA_PREFIX"/lib/python3.11/site-packages/mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda* 2>/dev/null; then
        echo "  mpi4jax with CUDA already installed, skipping."
        NEED_MPI4JAX=false
    fi
fi

if [ "$NEED_MPI4JAX" = true ]; then
    export MPI4JAX_BUILD_MPICC=$(which mpicc)
    export CUDA_ROOT=$CUDA_HOME
    pip install --no-cache-dir --no-binary mpi4jax "mpi4jax==0.8.1.post2" 2>&1 | tail -1
fi

# ── Step 5: DALIA and serinv ───────────────────────────────────────────────
echo ""
echo "[5/7] Installing DALIA and serinv"

if [ -f "$ARTIFACT_DIR/DALIA/pyproject.toml" ]; then
    echo "  DALIA source found, installing..."
else
    echo "  Cloning DALIA (branch: cleaning)..."
    git clone -b cleaning https://github.com/affifboudaoud/DALIA.git "$ARTIFACT_DIR/DALIA"
fi
pip install -e "$ARTIFACT_DIR/DALIA/" 2>&1 | tail -1

if [ -f "$ARTIFACT_DIR/serinv/pyproject.toml" ]; then
    echo "  serinv source found, installing..."
else
    echo "  Cloning serinv (branch: serinv_jax)..."
    git clone -b serinv_jax https://github.com/affifboudaoud/serinv.git "$ARTIFACT_DIR/serinv"
fi
pip install -e "$ARTIFACT_DIR/serinv/" 2>&1 | tail -1

# ── Step 6: Download data ─────────────────────────────────────────────────
echo ""
echo "[6/7] Downloading input data"

if [ -d "$ARTIFACT_DIR/data" ]; then
    echo "  data/ directory already exists, skipping download."
else
    bash "$ARTIFACT_DIR/scripts/download_data.sh"
fi

# ── Step 7: Verify installation ───────────────────────────────────────────
echo ""
echo "[7/7] Verifying installation"

python -c "
import jax
import jax.numpy as jnp
print(f'  JAX {jax.__version__}, backend: {jax.default_backend()}, devices: {len(jax.devices())}')

# GPU warmup (mandatory before importing DALIA)
if jax.default_backend() == 'gpu':
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
    print('  JAX GPU warmup: OK')

import os
os.environ['ARRAY_MODULE'] = 'cupy'
import dalia
print('  DALIA import: OK')

import serinv
print('  serinv import: OK')

from mpi4py import MPI
print(f'  mpi4py: OK (rank {MPI.COMM_WORLD.Get_rank()})')

import mpi4jax
print(f'  mpi4jax {mpi4jax.__version__}: OK')
"

echo ""
echo "=== Installation complete ==="
echo "To run experiments:"
echo "  cd $ARTIFACT_DIR/experiments/<experiment_dir>"
echo "  sbatch run.sbatch"
