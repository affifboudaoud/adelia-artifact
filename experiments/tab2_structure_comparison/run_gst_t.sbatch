#!/bin/bash
#SBATCH --job-name=tab2_gst_t
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=g34
#SBATCH --output=outputs/tab2_gst_t_%j.log
#SBATCH --error=outputs/tab2_gst_t_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner_gst_t.sh
