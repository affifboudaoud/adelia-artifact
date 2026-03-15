# Figure 2: Wallclock Timing

Compares end-to-end wall-clock time for JAX autodiff (AD) vs finite differences
(FD) across all 9 models. Reproduces **Figure 2** in the paper.

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

```bash
cd wa1 && sbatch run_breakdown.sbatch
cd ap1 && sbatch run_breakdown.sbatch
cd wa2 && sbatch run_breakdown.sbatch
```

## Generating the Figure

After running the benchmarks, generate the plotting CSV and the figure:

```bash
# 1. Generate pipeline_wallclock.csv from benchmark results
python generate_csv.py

# 2. Generate the figure
cd ../../plotting
python scripts/plot_pipeline_wallclock_merged.py
```

Output: `plotting/figures/pipeline_wallclock_merged.pdf`

## How the Wallclock Estimates Work

The figure reports estimated total wall-clock time (optimization + Hessian) for
both AD and FD. The key design choice: **we use the same number of L-BFGS
iterations for both methods** to ensure a fair comparison at equal computational
budget. This is conservative for two reasons:

1. **AD computes exact gradients**, so L-BFGS typically converges to a better
   solution in the same number of iterations. FD's approximate gradients can
   cause the line search to stall or the optimizer to converge to a worse point.

2. **FD often needs more iterations** than AD to reach the same solution quality,
   so using AD's iteration count underestimates FD's true cost.

### Formulas

The per-gradient time (`t_grad_AD`, `t_grad_FD`) is measured directly from
benchmark runs. The total wallclock is estimated as:

```
AD wallclock = n_iters × t_grad_AD  +  2 × d × t_grad_AD
               ├── optimization ──┘     ├── Hessian ──────┘

FD wallclock = n_iters × t_grad_FD  +  ceil(n_eval_hess / F) × t_eval_f
               ├── optimization ──┘     ├── Hessian ────────────────────┘
```

Where:
- `n_iters` = AD's iteration count (same for both)
- `d` = number of hyperparameters
- `t_eval_f` = single objective evaluation time = `t_grad_FD / ceil((2d+1) / F)`
- `n_eval_hess` = `1 + 2d + 2d(d-1)` (2nd-order finite differences of the objective)
- `F` = number of F()-level parallel ranks

The AD Hessian uses central differences of the analytical gradient (2d gradient
evaluations). The FD Hessian uses 2nd-order finite differences of the objective
function (O(d²) evaluations), parallelized across F() ranks.

JIT compilation time is excluded from the figure because it is a one-time cost
that is amortized over the optimization run.

The `generate_csv.py` script implements these formulas and reads per-gradient
times from `results/benchmark_results.csv`.

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
