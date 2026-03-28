#!/bin/bash
#SBATCH --job-name=dbg_objfn_p2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=g34
#SBATCH --partition=debug
#SBATCH --output=outputs/debug_run_%j.log
#SBATCH --error=outputs/debug_run_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/debug_run_inner.sh"
