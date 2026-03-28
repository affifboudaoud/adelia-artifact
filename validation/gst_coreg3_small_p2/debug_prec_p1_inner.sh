#!/bin/bash
source "$(dirname "$0")/../../experiments/common/env_setup.sh"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$(dirname "$0")"
python debug_prec_grad.py --p1
