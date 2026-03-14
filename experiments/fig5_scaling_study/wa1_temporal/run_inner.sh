#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

cd "$(dirname "$0")"

NT=${NT:-2}

# FD hangs on multi-node (CuPy cross-device issue), so only run both on 1 node
if [ -n "${BENCHMARK_METHOD}" ]; then
    METHOD="${BENCHMARK_METHOD}"
elif [ "${SLURM_NNODES:-1}" -eq 1 ]; then
    METHOD="both"
else
    METHOD="jax_autodiff"
fi

srun python run.py --benchmark_mode --n_benchmark_runs 10 --precision float64 --benchmark_method $METHOD --distributed_method two_phase --nt $NT

echo "Job finished at $(date)"
