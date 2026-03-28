#!/bin/bash
source "$(dirname "$0")/../experiments/common/env_setup.sh"

MODEL=${1:?"Usage: $0 <model_name>"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/$MODEL"
srun python validate.py --precision float64

echo "Validation for ${MODEL} finished at $(date)"
