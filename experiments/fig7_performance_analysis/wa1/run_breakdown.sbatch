#!/bin/bash
#SBATCH --job-name=wa1_breakdown
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --account=g34
#SBATCH --output=outputs/wa1_breakdown_%j.log
#SBATCH --error=outputs/wa1_breakdown_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_breakdown_inner.sh
