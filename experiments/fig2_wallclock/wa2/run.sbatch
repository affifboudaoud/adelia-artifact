#!/bin/bash
#SBATCH --job-name=adelia_wa2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa2_%j.log
#SBATCH --error=outputs/wa2_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
