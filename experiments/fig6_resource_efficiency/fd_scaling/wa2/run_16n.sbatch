#!/bin/bash
#SBATCH --job-name=fig6_wa2_fd_16n
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa2_fd_16n_%j.log
#SBATCH --error=outputs/wa2_fd_16n_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
