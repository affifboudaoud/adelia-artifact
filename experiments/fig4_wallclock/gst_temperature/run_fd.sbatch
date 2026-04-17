#!/bin/bash
#SBATCH --job-name=gst_temp_fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/gst_temperature_fd_%j.log
#SBATCH --error=outputs/gst_temperature_fd_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner_fd.sh
