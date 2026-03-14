#!/bin/bash
#SBATCH --job-name=fig4_wa2_fd
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/fig4_wa2_fd_%j.log
#SBATCH --error=outputs/fig4_wa2_fd_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_fd_inner.sh
