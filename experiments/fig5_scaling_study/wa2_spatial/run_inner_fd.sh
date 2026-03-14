#!/bin/bash
source "$(dirname "$0")/../../common/env_setup_nocuda.sh"

cd "$(dirname "$0")"

export NS=${NS:-1119}

# Use distributed solver on multi-node, default on single-node
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    SOLVER_ARG="--solver_min_p ${SLURM_NNODES}"
else
    SOLVER_ARG=""
fi

srun python run.py --benchmark_mode --n_benchmark_runs 10 --precision float64 --benchmark_method finite_diff $SOLVER_ARG

echo "Job finished at $(date)"
