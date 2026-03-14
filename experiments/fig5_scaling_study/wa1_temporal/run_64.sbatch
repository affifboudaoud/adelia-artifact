#!/bin/bash
#SBATCH --job-name=adelia_wa1_nt64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa1_nt64_%j.log
#SBATCH --error=outputs/wa1_nt64_%j.err

export NT=64
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
