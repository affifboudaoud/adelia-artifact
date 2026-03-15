#!/bin/bash
#SBATCH --job-name=adelia_wa2_ns282
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa2_ns282_%j.log
#SBATCH --error=outputs/wa2_ns282_%j.err

export NS=282
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
