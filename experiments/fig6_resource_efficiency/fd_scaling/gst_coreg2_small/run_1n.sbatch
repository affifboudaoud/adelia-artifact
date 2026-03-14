#!/bin/bash
#SBATCH --job-name=fig6_gst_coreg2_small_fd_1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --account=lp16
#SBATCH --output=outputs/gst_coreg2_small_fd_1n_%j.log
#SBATCH --error=outputs/gst_coreg2_small_fd_1n_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
