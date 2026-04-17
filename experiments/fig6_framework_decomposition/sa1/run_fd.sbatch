#!/bin/bash
#SBATCH --job-name=fig4_sa1_fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/fig4_sa1_fd_%j.log
#SBATCH --error=outputs/fig4_sa1_fd_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_fd_inner.sh
