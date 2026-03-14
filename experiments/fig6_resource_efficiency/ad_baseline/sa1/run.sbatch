#!/bin/bash
#SBATCH --job-name=adelia_ad_sa1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=lp16
#SBATCH --output=outputs/ad_sa1_%j.log
#SBATCH --error=outputs/ad_sa1_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_inner.sh
