#!/bin/bash
#SBATCH --job-name=adelia_gst_coreg3_small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=lp16
#SBATCH --output=outputs/gst_coreg3_small_%j.log
#SBATCH --error=outputs/gst_coreg3_small_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
