#!/bin/bash
#SBATCH --job-name=wa1_nt256_fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa1_nt256_fd_%j.log
#SBATCH --error=outputs/wa1_nt256_fd_%j.err

export NT=256
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner_fd.sh
