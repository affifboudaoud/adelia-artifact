#!/bin/bash
# Phase 1: CuPy/serinv FD reference (distributed solver, no CUDA-aware MPI)
source "$(dirname "$0")/../../experiments/common/env_setup_nocuda.sh"

cd "$(dirname "$0")"
srun python validate.py --precision float64 --phase 1 --solver_min_p 4
