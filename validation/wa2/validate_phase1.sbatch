#!/bin/bash
#SBATCH --job-name=val_wa2_fd
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=outputs/validate_phase1_%j.log
#SBATCH --error=outputs/validate_phase1_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/validate_phase1_inner.sh"
