#!/bin/bash
# Submit all 7 single-node validation jobs in parallel.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS="gst_small gst_medium gst_large gst_temperature gst_coreg2_small gst_coreg3_small sa1"

for model in $MODELS; do
    echo "Submitting $model ..."
    sbatch "$SCRIPT_DIR/$model/validate.sbatch"
done

echo "All validation jobs submitted."
