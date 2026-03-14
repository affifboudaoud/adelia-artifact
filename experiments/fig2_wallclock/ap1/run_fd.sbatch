#!/bin/bash
#SBATCH --job-name=adelia_ap1_fd
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/ap1_fd_%j.log
#SBATCH --error=outputs/ap1_fd_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_fd_inner.sh
