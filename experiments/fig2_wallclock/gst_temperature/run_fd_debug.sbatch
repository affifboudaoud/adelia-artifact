#!/bin/bash
#SBATCH --job-name=gst_t_fd_dbg
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --partition=debug
#SBATCH --output=outputs/gst_t_fd_debug_%j.log
#SBATCH --error=outputs/gst_t_fd_debug_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/run_inner_fd.sh"
