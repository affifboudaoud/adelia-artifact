#!/bin/bash
#SBATCH --job-name=validate_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=g34
#SBATCH --output=validate_all_%j.log
#SBATCH --error=validate_all_%j.err

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash "$SCRIPT_DIR/validate_all_inner.sh"
