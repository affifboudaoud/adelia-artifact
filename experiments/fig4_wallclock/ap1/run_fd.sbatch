#!/bin/bash
#SBATCH --job-name=adelia_ap1_fd
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --account=g34
#SBATCH --output=outputs/ap1_fd_%j.log
#SBATCH --error=outputs/ap1_fd_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_fd_inner.sh
