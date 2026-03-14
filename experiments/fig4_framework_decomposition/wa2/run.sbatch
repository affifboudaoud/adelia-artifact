#!/bin/bash
#SBATCH --job-name=fig4_wa2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/fig4_wa2_%j.log
#SBATCH --error=outputs/fig4_wa2_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
