#!/bin/bash
#SBATCH --job-name=val_c3s_ad_p2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=g34
#SBATCH --output=outputs/validate_phase2_%j.log
#SBATCH --error=outputs/validate_phase2_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/validate_phase2_inner.sh"
