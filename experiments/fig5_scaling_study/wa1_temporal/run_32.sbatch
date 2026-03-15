#!/bin/bash
#SBATCH --job-name=adelia_wa1_nt32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa1_nt32_%j.log
#SBATCH --error=outputs/wa1_nt32_%j.err

export NT=32
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
