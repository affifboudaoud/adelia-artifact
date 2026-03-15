#!/bin/bash
#SBATCH --job-name=adelia_wa2_ns72
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa2_ns72_%j.log
#SBATCH --error=outputs/wa2_ns72_%j.err

export NS=72
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
