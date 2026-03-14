# Figure 2: Wallclock Timing

Runs end-to-end L-BFGS optimization using both JAX autodiff (AD) and finite
differences (FD) across all 9 models, measuring total wallclock time and
per-gradient cost. Reproduces **Figure 2** in the paper.

## Models

**Single-node univariate** (AD + FD in one job):

| Model | Type | nt x ns | Nodes | Time |
|-------|------|---------|-------|------|
| gst_small | Gaussian ST | 5 x 92 | 1 | ~5 min |
| gst_medium | Gaussian ST | 100 x 812 | 1 | ~30 min |
| gst_coreg2_small | Coregional (2v) | 12 x 708 | 1 | ~10 min |
| gst_coreg3_small | Coregional (3v) | 8 x 1062 | 1 | ~30 min |

**Single-node large** (AD and FD as separate jobs):

| Model | Type | nt x ns | Nodes | Time |
|-------|------|---------|-------|------|
| gst_large | Gaussian ST | 250 x 4002 | 1 | AD ~30 min, FD ~1h |

**Distributed coregional** (AD and FD as separate jobs):

| Model | Type | nt x ns | nv | Nodes | AD time | FD time |
|-------|------|---------|-----|-------|---------|---------|
| sa1 | Coregional (3v) | 192 x 1673 | 3 | 1 | ~2h | ~4h |
| wa1 | Coregional (3v) | 512 x 1247 | 3 | 4 | ~4h | ~8h |
| ap1 | Coregional (3v) | 48 x 4210 | 3 | 4 | ~6h | ~12h |
| wa2 | Coregional (3v) | 48 x 4485 | 3 | 4 | ~8h | ~12h |

## How to Run

**Small models** (AD + FD in one job):
```bash
cd gst_small && sbatch run.sbatch
cd gst_medium && sbatch run.sbatch
cd gst_coreg2_small && sbatch run.sbatch
cd gst_coreg3_small && sbatch run.sbatch
```

**gst_large** (must run AD and FD separately due to GPU memory):
```bash
cd gst_large
sbatch run.sbatch                                     # runs AD only by default
sbatch --wrap="... srun python run.py --skip_jax ..."  # FD only
```

**Distributed models** (SA1, WA1, AP1, WA2):
```bash
cd wa1
sbatch run_ad.sbatch    # JAX autodiff (4 nodes, two-phase distributed)
sbatch run_fd.sbatch    # Finite differences (4 nodes, serinv solver)
```

All sbatch files use `--account=lp16`. Edit if needed.

For longer runs (full convergence), increase max iterations:
```bash
sbatch --export=ALL,MAX_ITER=1000 --time=12:00:00 run_ad.sbatch
```

### Per-stage breakdown (for analysis)

WA1 has dedicated breakdown scripts. For AP1/WA2, use the breakdown sbatch:
```bash
cd wa1 && sbatch run_breakdown.sbatch
cd ap1 && sbatch run_breakdown.sbatch
cd wa2 && sbatch run_breakdown.sbatch
```

## Distributed AD Architecture

The 4-node AD runs use the **two-phase parallel algorithm**:

- **Communicator setup**: With `gradient_method=jax_autodiff`, all ranks collaborate
  on a single objective+gradient evaluation (F()=1, Q()=1, S()=4). This is configured
  automatically in `dalia.py` — no manual `solver_min_p` needed.

- **Partition balancing**: Root processes more blocks than non-root ranks (ratio 2.0)
  to compensate for the slower permuted BTA factorization on non-root.

- **Reduced system**: For small blocks (WA1, bs=3741), uses fused `lax.scan` on GPU.
  For large blocks (AP1/WA2, bs>12000), falls back to per-block JIT with CPU staging.

- **grad_prior**: For small blocks (WA1), uses two-phase parallel. For large blocks
  (AP1/WA2), falls back to sequential pipeline (GPU memory constraint).

## Key Flags

| Flag | Description |
|------|-------------|
| `--max_iter N` | L-BFGS iterations (default 250) |
| `--skip_fd` / `--skip_jax` | Run only AD or FD |
| `--breakdown` | Per-stage timing breakdown |
| `--precision float64` | Required for all runs |

## Output

Results are appended to `results/benchmark_results.csv` with fields:
`model`, `method`, `n_nodes`, `n_hyperparams`, `n_iterations`,
`per_gradient_mean`, `per_gradient_std`, `optimization_time`,
`hessian_time`, `marginals_time`, `wallclock_time`, `jit_time`,
`final_f`, `timestamp`.

Per-stage breakdowns print to stdout (in the `.log` files).

## Input Data

Each model loads from `../../data/{model}/`:
- `inputs_spatio_temporal/` -- spatial/temporal basis, SPDE parameters
- `inputs_regression/` -- regression design matrices
- `reference_outputs/` -- reference theta for initial conditions
