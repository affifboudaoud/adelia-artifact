#!/bin/bash
# Quick validation: runs gst_small with both AD and FD on 1 node (~2 min).
# Verifies gradient error < 1e-5 and objective relative error < 1e-4.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export ARTIFACT_DIR

source "$ARTIFACT_DIR/experiments/common/env_setup.sh"

echo "======================================================"
echo "ADELIA Quick Validation"
echo "======================================================"
echo "Artifact directory: $ARTIFACT_DIR"
echo "Date: $(date)"
echo ""

cd "$ARTIFACT_DIR/experiments/fig2_wallclock/gst_small"

python run.py --max_iter 50 --precision float64

echo ""
echo "======================================================"
echo "Validation completed successfully."
echo "======================================================"
