#!/bin/bash
#SBATCH --job-name=conv_gst_temperature
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --partition=debug
#SBATCH --output=outputs/converge_%j.log
#SBATCH --error=outputs/converge_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/converge_inner.sh"
