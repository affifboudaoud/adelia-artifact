#!/bin/bash
#SBATCH --job-name=adelia_sa1_ad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --account=lp16
#SBATCH --output=outputs/sa1_ad_%j.log
#SBATCH --error=outputs/sa1_ad_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_ad_inner.sh
