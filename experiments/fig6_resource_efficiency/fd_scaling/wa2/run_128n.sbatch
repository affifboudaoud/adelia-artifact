#!/bin/bash
#SBATCH --job-name=fig6_wa2_fd_128n
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=outputs/wa2_fd_128n_%j.log
#SBATCH --error=outputs/wa2_fd_128n_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
