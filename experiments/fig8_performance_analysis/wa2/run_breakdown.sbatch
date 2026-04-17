#!/bin/bash
#SBATCH --job-name=wa2_breakdown
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa2_breakdown_%j.log
#SBATCH --error=outputs/wa2_breakdown_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_breakdown_inner.sh
