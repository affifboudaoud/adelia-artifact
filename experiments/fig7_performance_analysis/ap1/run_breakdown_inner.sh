#!/bin/bash
source "$(dirname "$0")/../../common/env_setup.sh"

cd "$(dirname "$0")/../../fig2_wallclock/ap1"

srun python run.py --breakdown --n_benchmark_runs 3 --precision float64 --distributed_method two_phase --breakdown_csv "${ARTIFACT_DIR}/plotting/data/distributed_breakdown.csv"

echo "Job finished at $(date)"
