#!/bin/bash
#SBATCH --job-name=sa1_fd_conv
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --account=g34
#SBATCH --output=outputs/sa1_fd_conv_%j.log
#SBATCH --error=outputs/sa1_fd_conv_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/run_fd_conv_inner.sh"
