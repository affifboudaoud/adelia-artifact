#!/bin/bash
#SBATCH --job-name=adelia_ad_wa1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --account=lp16
#SBATCH --output=outputs/ad_wa1_%j.log
#SBATCH --error=outputs/ad_wa1_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
