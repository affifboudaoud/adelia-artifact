# Installation Guide

## Prerequisites

- Python >= 3.11
- CUDA >= 12.x with compatible GPU (tested on NVIDIA H100/GH200)
- MPI implementation (tested with Cray MPICH 8.1.x)
- Conda or Miniconda

## Quick Install (CSCS Alps)

On Alps, a single script handles the full installation. Run it on a compute node:

```bash
salloc -N1 --gpus-per-task=1 -A <account> -t 30 -p debug
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash scripts/install_alps.sh
```

This creates the `adelia-env` conda environment, installs all dependencies
(JAX, CuPy, mpi4py, mpi4jax with CUDA, DALIA, serinv), downloads the input
data, and verifies the installation. See the script for details.

If you prefer to install step-by-step, or are on a different cluster, follow
the manual instructions below.

## Manual Installation

### Step 1: Create Conda Environment

```bash
conda create -n adelia-env python=3.11 -y
conda activate adelia-env
```

### Step 2: Install Core Dependencies

```bash
pip install numpy scipy matplotlib pandas tabulate pydantic gdown
```

### Step 3: Install JAX with CUDA Support

```bash
pip install jax==0.8.1 jaxlib==0.8.1
pip install jax-cuda12-pjrt==0.8.1 jax-cuda12-plugin==0.8.1
```

JAX also requires NVIDIA pip packages for GPU operations (cuDNN, cuBLAS, etc.)
and the `nvidia-cuda-nvcc-cu12` package for XLA's `libdevice.10.bc`:

```bash
pip install \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cufft-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-nccl-cu12 nvidia-cuda-nvcc-cu12
```

> **Note on aarch64 (GH200):** The `jax[cuda12]` pip extra may not resolve
> correctly on aarch64 because `jax-cuda12-plugin` versions don't always match.
> Install the CUDA plugin and NVIDIA packages explicitly as shown above.

### Step 4: Install CuPy

```bash
pip install cupy-cuda12x==13.6.0
```

### Step 5: Install mpi4py and mpi4jax (for distributed experiments)

Both packages **must be built from source** against your system's MPI library.
Pre-built pip wheels will not find `libmpi.so` at runtime.

mpi4jax requires CUDA support for GPU-to-GPU MPI communication. Build it on a
node with GPU access and `CUDA_HOME` set.

```bash
# On CSCS Alps — run inside uenv on a compute node with a GPU:
module load cray-mpich gcc cuda

# mpi4py: build from source against Cray MPICH
CC=$(which mpicc) pip install --no-cache-dir --no-binary mpi4py mpi4py

# mpi4jax: build from source with CUDA support
pip install Cython
export MPI4JAX_BUILD_MPICC=$(which mpicc)
export CUDA_ROOT=$CUDA_HOME
pip install --no-cache-dir --no-binary mpi4jax "mpi4jax==0.8.1.post2"
```

Verify that the CUDA bridge was built:
```bash
ls $CONDA_PREFIX/lib/python3.11/site-packages/mpi4jax/_src/xla_bridge/
# Should include: mpi_xla_bridge_cuda.cpython-311-aarch64-linux-gnu.so
```

### Step 6: Install DALIA and Serinv

**Option A: From the artifact submodules (requires git-lfs)**
```bash
cd adelia-artifact
git submodule update --init
pip install -e DALIA/
pip install -e serinv/
```

> **Note:** The DALIA submodule uses Git LFS for example data files.
> If `git-lfs` is not installed, `git submodule update --init` may fail.
> In that case, use Option B below.

**Option B: Clone directly (recommended if git-lfs is unavailable)**
```bash
git clone -b cleaning https://github.com/affifboudaoud/DALIA.git
git clone -b serinv_jax https://github.com/affifboudaoud/serinv.git
pip install -e DALIA/ -e serinv/
```

### Step 7: Download Input Data

```bash
pip install gdown  # if not already installed
./scripts/download_data.sh
```

This downloads ~1.5 GB of input datasets from Google Drive into `data/`.

### Step 8: Verify Installation

Run this on a compute node (needs GPU):

```bash
python -c "
import jax
import jax.numpy as jnp
print('JAX backend:', jax.default_backend())

# JAX GPU warmup (MUST be done before importing dalia)
if jax.default_backend() == 'gpu':
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
    print('JAX GPU warmup: OK')

import os
os.environ['ARRAY_MODULE'] = 'cupy'
import dalia
print('DALIA import: OK')

import serinv
print('serinv import: OK')

from mpi4py import MPI
print('mpi4py: OK')

import mpi4jax
print('mpi4jax: OK')
"
```

## Critical: Import Order

Due to a cuSolver/CuPy conflict, the following import order is mandatory:

1. `import jax` and perform GPU warmup (`jnp.linalg.cholesky(eye(2))`)
2. Set `os.environ["ARRAY_MODULE"] = "cupy"`
3. `from dalia...` (triggers CuPy initialization)

Reversing steps 1 and 3 causes `cusolverDnCreate` failures. All experiment
scripts in this artifact follow this order.

## CSCS Alps-Specific Notes

SBATCH scripts use `uenv` for module loading:
```bash
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
```

**Before submitting any job**, edit [`experiments/common/env_setup.sh`](experiments/common/env_setup.sh)
and set:

- `SBATCH_ACCOUNT` — your CSCS allocation (e.g. `g34`).
- `CONDA_ENV` — the conda environment created by `install_alps.sh`
  (default: `adelia-env`).

Both values are consumed by every experiment's `run_inner.sh` wrapper.

### Rebuilding the artifact description PDF

From the artifact root:
```bash
make                             # pdflatex (reviewer default)
# or, if pdflatex is unavailable:
tectonic artifact_description.tex
```

## Environment Export

A reference `environment.yml` is provided. To recreate:
```bash
conda env create -f environment.yml
```

Note: This captures package versions but not system-level dependencies (MPI,
CUDA drivers). Those must be provided by the cluster. For a complete setup on
Alps, use `scripts/install_alps.sh` instead.

## Troubleshooting

### `git submodule update --init` fails with "git-lfs: command not found"

The DALIA submodule uses Git LFS for example data files (`.npz`, `.npy`).
If git-lfs is not installed, the checkout will fail. Solutions:

- **Install git-lfs:** `conda install git-lfs && git lfs install`, then retry.
- **Skip submodules and clone directly:**
  ```bash
  git clone -b cleaning https://github.com/affifboudaoud/DALIA.git
  git clone -b serinv_jax https://github.com/affifboudaoud/serinv.git
  ```

### `libdevice not found at ./libdevice.10.bc`

XLA needs `libdevice.10.bc` from the CUDA NVCC toolkit. Install the pip package
and point XLA to it:

```bash
pip install nvidia-cuda-nvcc-cu12
```

Then set this environment variable (already included in `env_setup.sh`):
```bash
NVIDIA_NVCC_DIR=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$NVIDIA_NVCC_DIR"
```

### `CUDNN_STATUS_INTERNAL_ERROR` / `DNN library initialization failed`

JAX GPU operations need NVIDIA cuDNN and other CUDA runtime libraries from pip.
These are **not** installed automatically when using `pip install jax jaxlib`
without the `[cuda12]` extra (which may not resolve on aarch64):

```bash
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12 \
    nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-nccl-cu12 nvidia-cuda-nvcc-cu12
```

Also ensure the NVIDIA pip package libraries are on `LD_LIBRARY_PATH`
(handled by `env_setup.sh`):
```bash
NVIDIA_BASE=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia
for d in $NVIDIA_BASE/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done
```

### `RuntimeError: cannot load MPI library` / `libmpi.so: cannot open shared object file`

The pre-built pip wheel for mpi4py does not link against the cluster's MPI
library. You must build mpi4py **from source** against Cray MPICH (or your
system's MPI). Run this on a compute node with MPI modules loaded:

```bash
module load cray-mpich gcc
CC=$(which mpicc) pip install --no-cache-dir --no-binary mpi4py mpi4py
```

### `ImportError: The mpi4jax GPU extensions could not be imported`

mpi4jax was built without CUDA support. The `mpi_xla_bridge_cuda.so` file is
missing. Rebuild mpi4jax on a node with GPU access and `CUDA_HOME` set:

```bash
module load cray-mpich gcc cuda
export MPI4JAX_BUILD_MPICC=$(which mpicc)
export CUDA_ROOT=$CUDA_HOME
pip install Cython
pip install --no-cache-dir --no-binary mpi4jax "mpi4jax==0.8.1.post2"
```

Verify the CUDA bridge exists:
```bash
ls $CONDA_PREFIX/lib/python3.11/site-packages/mpi4jax/_src/xla_bridge/
# Should include: mpi_xla_bridge_cuda.*.so
```

### `cusolverDnCreate` failures on import

This is caused by importing DALIA/CuPy **before** JAX GPU warmup. See the
[Critical: Import Order](#critical-import-order) section above. All experiment
scripts handle this automatically.

### `OSError: [Errno 116] Stale file handle` (CuPy kernel cache)

This can occur on parallel filesystems (e.g., Lustre) when multiple MPI ranks
race to read CuPy's compiled kernel cache. It is typically transient —
resubmitting the job usually resolves it. If it persists, set per-rank cache
directories:

```bash
export CUPY_CACHE_DIR="/tmp/cupy_cache_${SLURM_PROCID}"
```
