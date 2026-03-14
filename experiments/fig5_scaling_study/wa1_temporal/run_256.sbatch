#!/bin/bash
#SBATCH --job-name=adelia_wa1_nt256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/wa1_nt256_%j.log
#SBATCH --error=outputs/wa1_nt256_%j.err

export NT=256
export BENCHMARK_METHOD=jax_autodiff
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
