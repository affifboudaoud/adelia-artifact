#!/bin/bash
#SBATCH --job-name=adelia_tab2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/tab2_%j.log
#SBATCH --error=outputs/tab2_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
