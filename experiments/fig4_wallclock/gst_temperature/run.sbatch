#!/bin/bash
#SBATCH --job-name=adelia_gst_temperature
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/gst_temperature_%j.log
#SBATCH --error=outputs/gst_temperature_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
