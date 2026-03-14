# Installation Guide

## Prerequisites

- Python >= 3.11
- CUDA >= 12.x with compatible GPU (tested on NVIDIA H100/GH200)
- MPI implementation (tested with Cray MPICH 8.1.x)
- Conda or Miniconda

## Step 1: Create Conda Environment

```bash
conda create -n adelia-env python=3.11 -y
conda activate adelia-env
```

## Step 2: Install Core Dependencies

```bash
pip install numpy scipy matplotlib pydantic
```

## Step 3: Install JAX with CUDA Support

```bash
pip install jax[cuda12]==0.4.38 jaxlib==0.4.38
```

Verify JAX GPU backend:
```python
import jax
print(jax.devices())  # Should show CudaDevice
```

## Step 4: Install CuPy

```bash
pip install cupy-cuda12x==13.2.0
```

## Step 5: Install mpi4jax (for distributed experiments)

mpi4jax must be built against your system's MPI library:

```bash
# On CSCS Alps with Cray MPICH:
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash -c '
  module load cray-mpich gcc
  export MPI4JAX_BUILD_MPICC=$(which mpicc)
  pip install mpi4jax
'
```

For other systems:
```bash
export MPI4JAX_BUILD_MPICC=$(which mpicc)
pip install mpi4jax
```

## Step 6: Install DALIA and Serinv

**Option A: From the artifact source snapshot**
```bash
cd artifact/source/dalia
pip install -e .

cd artifact/source/serinv
pip install -e .
```

**Option B: From the git repositories (latest version)**
```bash
git clone https://github.com/affifboudaoud/DALIA.git
cd DALIA && pip install -e . && cd ..

git clone https://github.com/affifboudaoud/serinv.git
cd serinv && pip install -e . && cd ..
```

## Step 7: LD_LIBRARY_PATH Setup (NVIDIA pip packages)

JAX's CUDA backend requires NVIDIA pip-package shared libraries on `LD_LIBRARY_PATH`. Without this, JAX falls back to CPU.

```bash
NVIDIA_BASE=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia
for d in $NVIDIA_BASE/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done
```

This is included in `experiments/common/env_setup.sh`.

## Step 8: Verify Installation

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
print('Serinv import: OK')
"
```

## Critical: Import Order

Due to a cuSolver/CuPy conflict, the following import order is mandatory:

1. `import jax` and perform GPU warmup (`jnp.linalg.cholesky(eye(2))`)
2. Set `os.environ["ARRAY_MODULE"] = "cupy"`
3. `from dalia...` (triggers CuPy initialization)

Reversing steps 1 and 3 causes `cusolverDnCreate` failures. All experiment scripts in this artifact follow this order.

## CSCS Alps-Specific Notes

SBATCH scripts use `uenv` for module loading:
```bash
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
```

Edit `experiments/common/env_setup.sh` to match your cluster's module system, conda path, and SLURM account.

## Environment Export

A reference `environment.yml` is provided. To recreate:
```bash
conda env create -f environment.yml
```

Note: This captures package versions but not system-level dependencies (MPI, CUDA drivers). Those must be provided by the cluster.
