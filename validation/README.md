# Gradient Correctness Validation

Validates that the structure-preserving backward pass (AD-BTA) produces
gradients matching native JAX reverse-mode AD through `lax.scan` (AD-Loop).
This supports the claim in Section V that the two exact AD methods agree to
below 1e-7 in the worst case.

## Models Tested

| Model | Type | Dimensions (nt x ns) |
|-------|------|---------------------|
| GST-S | Univariate | 5 x 92 |
| GST-C2 | 2-variate coregional | 12 x 708 |
| GST-C3 | 3-variate coregional | 8 x 1062 |
| GST-M | Univariate | 100 x 812 |

## What It Measures

For each model, the script compares AD-BTA and AD-Loop on:
- Forward value (absolute and relative error)
- Gradient (max absolute and relative error across all hyperparameters)
- Latent variable x (max absolute error)
- JIT compilation time for both methods

## Resources

- **Nodes**: 1
- **Runtime**: ~1 hour
- **Environment**: CPU-only (`JAX_PLATFORMS=cpu`, `ARRAY_MODULE=numpy`)

## How to Run

```bash
# Quick validation (~2 min, runs one small model)
sbatch validate_quick.sbatch

# Full validation (all 4 models, ~1 hour)
sbatch validate_gradient_correctness.sbatch
```

Edit the `--account` field in the `.sbatch` files to match your allocation.

## Output

Results are written to `outputs/` as CSV with fields:
`model`, `dims`, `d`, `f_bta`, `f_scan`, `f_rel_err`, `grad_max_abs_err`,
`grad_max_rel_err`, `x_max_abs_err`, `jit_bta`, `jit_scan`, `timestamp`.

## Dependencies

None. This is a standalone validation experiment.
