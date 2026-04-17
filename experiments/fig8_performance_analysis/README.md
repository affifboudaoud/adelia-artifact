# Figure 7: Performance Analysis

Produces a per-stage timing breakdown of the distributed AD gradient
computation (two-phase algorithm) for the three largest models.
Reproduces **Figure 7** in the paper.

## Stages Measured

| Stage | Description |
|-------|-------------|
| `chol_fwd_sub` | Cholesky factorization + forward substitution |
| `bwd_si` | Backward pass + selected inversion |
| `grad_prior` | Prior gradient + log-determinant |
| `grad_quad` | Likelihood gradient (quadratic form) |
| `rs_aggregate` | Reduced system overhead (MPI allreduce, factorize, scatter) |

## Models

| Model | nt x ns (nv) | Nodes | Time limit |
|-------|-------------|-------|------------|
| WA1 | 512 x 1247 (3-variate) | 4 | 1 hour |
| AP1 | 48 x 4210 (3-variate) | 4 | 2 hours |
| WA2 | 48 x 4485 (3-variate) | 4 | 2 hours |

## How to Run

Each model runs as a separate job:
```bash
cd wa1 && sbatch run_breakdown.sbatch
cd ap1 && sbatch run_breakdown.sbatch
cd wa2 && sbatch run_breakdown.sbatch
```

Each inner script `cd`s into the corresponding `fig4_wallclock/{model}/`
directory and runs `run.py --breakdown`.

Edit `--account` in each `.sbatch` file to match your allocation.

**Important:** All three models append to the same CSV. Delete the output
CSV before a full re-run to avoid duplicate rows:
```bash
rm -f ../../plotting/data/distributed_breakdown.csv
```

## Output

Results are appended to `../../plotting/data/distributed_breakdown.csv` with
fields: `model`, `n_nodes`, `stage`, `time_mean`, `time_std`, `n_runs`.

## Reproducing Figure 7

Run all three models, then generate the figure:
```bash
cd ../../plotting
python scripts/plot_performance_analysis.py
```

## Dependencies

- Uses `fig4_wallclock/{model}/run.py` with the `--breakdown` flag
- Requires distributed model data in `../../data/{wa1,ap1,wa2}/`
