#!/bin/bash
#SBATCH --job-name=dbg_prec_p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=g34
#SBATCH --output=outputs/debug_prec_p1_%j.log
#SBATCH --error=outputs/debug_prec_p1_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/debug_prec_p1_inner.sh"
