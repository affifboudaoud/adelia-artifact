#!/bin/bash
#SBATCH --job-name=adelia_ap1_ad
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=g34
#SBATCH --output=outputs/ap1_ad_%j.log
#SBATCH --error=outputs/ap1_ad_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash run_ad_inner.sh
