#!/bin/bash
#SBATCH --job-name=val_gst_medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=g34
#SBATCH --output=outputs/validate_%j.log
#SBATCH --error=outputs/validate_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "${SLURM_SUBMIT_DIR}/../validate_inner.sh" gst_medium
