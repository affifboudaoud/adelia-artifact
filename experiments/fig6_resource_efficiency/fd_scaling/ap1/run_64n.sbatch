#!/bin/bash
#SBATCH --job-name=fig6_ap1_fd_64n
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=lp16
#SBATCH --output=outputs/ap1_fd_64n_%j.log
#SBATCH --error=outputs/ap1_fd_64n_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
