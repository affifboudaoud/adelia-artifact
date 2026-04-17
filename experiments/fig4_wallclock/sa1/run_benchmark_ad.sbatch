#!/bin/bash
#SBATCH --job-name=sa1_bench_ad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/sa1_bench_ad_%j.log
#SBATCH --error=outputs/sa1_bench_ad_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_benchmark_ad_inner.sh
