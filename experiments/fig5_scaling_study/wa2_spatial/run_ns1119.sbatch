#!/bin/bash
#SBATCH --job-name=adelia_wa2_ns1119
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa2_ns1119_%j.log
#SBATCH --error=outputs/wa2_ns1119_%j.err

export NS=1119
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
