# Gradient Validation

This directory contains gradient correctness checks for ADELIA. Each
per-model subdirectory submits a single-GPU SLURM job that loads the
benchmark model, computes the log-posterior gradient with ADELIA's
structured AD path, and compares it against two reference gradients:

1. **Finite differences** (FD): central differences with `h = 1e-3`.
   Truncation error is `O(h²)`, so AD-vs-FD agreement is expected to
   about `5e-4` to `6e-3` relative error depending on conditioning.
2. **JAX reverse-mode AD on a dense loop implementation** (AD-Loop):
   semantically identical gradient via different execution path.
   AD-vs-AD-Loop agreement is expected to `< 1.2e-7` relative error.
   Only available on models that fit a dense backward pass in GPU
   memory (GST-S, GST-C2, GST-C3, GST-M).

## Running

Single model:

```bash
cd validation/gst_small
sbatch validate.sbatch
```

All models (submits one job per model):

```bash
sbatch validation/validate_all.sbatch
```

Distributed (two-phase) variant:

```bash
cd validation/gst_coreg3_small_p2
sbatch validate.sbatch   # P = 2 MPI ranks
```

## Pass criteria

A validation run is considered passing if every check below holds:

| Check | Tolerance |
|---|---|
| Forward-value relative error (AD vs FD or AD vs AD-Loop) | `< 1e-6` |
| Gradient cosine similarity (AD vs reference) | `> 0.999` |
| Max component-wise relative error vs AD-Loop (when available) | `< 1.2e-7` |
| Max component-wise relative error vs FD | `< 1e-2` |
| Distributed two-phase vs AD-Loop on GST-C3 (`P=2`) | `< 6.8e-6` |

Each job's `outputs/` directory contains stdout with the numerical
comparison table and a PASS/FAIL summary line.

## Files

- `validate_all.sbatch`, `validate_all_inner.sh` — master submission
  wrappers that iterate over every per-model subdirectory.
- `validate_core.py` — shared gradient-comparison logic (imported by
  every per-model `validate.py`).
- `converge_core.py` — shared optimizer-convergence check used by the
  `converge.sbatch` scripts (L-BFGS on the full objective).
- `<model>/validate.sbatch` — per-model submission script.
- `<model>/validate.py` — per-model driver that builds the model and
  invokes `validate_core`.
