#!/bin/bash
source "$(dirname "$0")/../experiments/common/env_setup.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS="gst_small gst_coreg2_small gst_coreg3_small gst_medium gst_temperature gst_large sa1"

FAILURES=0
for model in $MODELS; do
    echo "============================================"
    echo "Validating: $model"
    echo "============================================"
    cd "$SCRIPT_DIR/$model"
    srun python validate.py --precision float64
    if [ $? -ne 0 ]; then
        echo "FAILED: $model"
        FAILURES=$((FAILURES + 1))
    fi
done

echo "============================================"
echo "Validation complete. Failures: $FAILURES / 7"
echo "============================================"

exit $FAILURES
