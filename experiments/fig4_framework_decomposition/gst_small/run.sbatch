#!/bin/bash
#SBATCH --job-name=fig4_gst_small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=lp16
#SBATCH --output=outputs/fig4_gst_small_%j.log
#SBATCH --error=outputs/fig4_gst_small_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
