#!/bin/bash
#SBATCH --job-name=adelia_wa1_fd
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --account=g34
#SBATCH --output=outputs/wa1_fd_%j.log
#SBATCH --error=outputs/wa1_fd_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_fd_inner.sh
