#!/bin/bash
source "$(dirname "$0")/../../experiments/common/env_setup.sh"

cd "$(dirname "$0")"
srun python validate.py --precision float64 --phase 3
