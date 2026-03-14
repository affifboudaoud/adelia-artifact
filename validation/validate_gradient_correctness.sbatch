#!/bin/bash
#SBATCH --job-name=adelia_grad_validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=outputs/grad_validate_%j.log
#SBATCH --error=outputs/grad_validate_%j.err

uenv run --view=modules prgenv-gnu/24.11:v1 -- bash validate_gradient_correctness_inner.sh
