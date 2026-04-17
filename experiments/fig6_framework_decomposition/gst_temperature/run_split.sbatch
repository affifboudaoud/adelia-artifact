#!/bin/bash
#SBATCH --job-name=fig4_gst_t
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=g34
#SBATCH --output=outputs/fig4_gst_temperature_split_%j.log
#SBATCH --error=outputs/fig4_gst_temperature_split_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner_split.sh
