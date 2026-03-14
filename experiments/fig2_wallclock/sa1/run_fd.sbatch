#!/bin/bash
#SBATCH --job-name=adelia_sa1_fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/sa1_fd_%j.log
#SBATCH --error=outputs/sa1_fd_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_fd_inner.sh
