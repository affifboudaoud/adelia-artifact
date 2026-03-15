#!/bin/bash
#SBATCH --job-name=adelia_validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=validate_%j.log
#SBATCH --error=validate_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash validate_quick.sh
