#!/bin/bash
#SBATCH --job-name=adelia_wa1_nt128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa1_nt128_%j.log
#SBATCH --error=outputs/wa1_nt128_%j.err

export NT=128
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
