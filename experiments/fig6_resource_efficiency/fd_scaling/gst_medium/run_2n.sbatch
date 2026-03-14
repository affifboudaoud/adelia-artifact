#!/bin/bash
#SBATCH --job-name=fig6_gst_medium_fd_2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --account=lp16
#SBATCH --output=outputs/gst_medium_fd_2n_%j.log
#SBATCH --error=outputs/gst_medium_fd_2n_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
