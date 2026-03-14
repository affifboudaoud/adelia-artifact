# Figure 4: Framework Decomposition

Decomposes the observed AD-vs-FD speedup into framework and algorithmic
components. Compares single forward-evaluation cost in JAX vs CuPy and
measures the backward/forward time ratio. Reproduces **Figure 4** in the
paper.

## What It Measures

- **Framework ratio** r = t_CuPy / t_JAX (single forward evaluation)
- **Backward/forward ratio** beta (time in backward pass vs forward pass)
- **Observed speedup** (from the Figure 6 resource-efficiency experiment)

## Models

**Single-node** (AD + FD run in one job):

| Model | Dimensions | Nodes | Time limit |
|-------|-----------|-------|------------|
| gst_small | 5 x 92 | 1 | 30 min |
| gst_medium | 100 x 812 | 1 | 30 min |
| gst_coreg2_small | 12 x 708 | 1 | 1 hour |
| gst_coreg3_small | 8 x 1062 | 1 | 1 hour |

**Single-node, large** (AD and FD run as separate jobs to avoid GPU OOM):

| Model | Dimensions | Nodes | Time limit |
|-------|-----------|-------|------------|
| gst_large | 250 x 4002 | 1 | 1 hour each |
| sa1 | 192 x 5019 | 1 | 2 hours each |

**Distributed** (AD and FD run as separate jobs, 4 nodes each):

| Model | Dimensions | Nodes | Time limit |
|-------|-----------|-------|------------|
| ap1 | 48 x 12630 | 4 | 2 hours each |
| wa1 | 512 x 3741 | 4 | 3 hours each |
| wa2 | 48 x 13455 | 4 | 3 hours each |

## How to Run

**Small single-node models** (AD + FD fit together on one GPU):
```bash
cd gst_small && sbatch run.sbatch
cd gst_medium && sbatch run.sbatch
cd gst_coreg2_small && sbatch run.sbatch
cd gst_coreg3_small && sbatch run.sbatch
```

**Large models** (separate AD and FD jobs):
```bash
cd gst_large
sbatch run_ad.sbatch
sbatch run_fd.sbatch

cd sa1
sbatch run_ad.sbatch
sbatch run_fd.sbatch
```

**Distributed models** (separate AD and FD, 4 nodes each):
```bash
cd ap1
sbatch run_ad.sbatch
sbatch run_fd.sbatch

cd wa1
sbatch run_ad.sbatch
sbatch run_fd.sbatch

cd wa2
sbatch run_ad.sbatch
sbatch run_fd.sbatch
```

FD jobs for distributed models use `--solver_min_p 4` so that all ranks
collaborate on a single distributed solver (otherwise the full BTA structure
exceeds single-GPU memory).

Edit `--account` in each `.sbatch` file to match your allocation.

## Key Flags

| Flag | Description |
|------|-------------|
| `--framework_comparison` | Run framework comparison instead of optimization |
| `--framework_ad_only` | Only create JAX AD instance (skip FD) |
| `--framework_fd_only` | Only create FD instance (skip JAX AD) |
| `--solver_min_p N` | Minimum ranks per solver (use 4 for distributed FD) |
| `--n_benchmark_runs N` | Number of timed runs (default: 10 single-node, 5 distributed) |

## Output

Results are appended to CSVs in `results/`:

| CSV | Content |
|-----|---------|
| `single_eval_comparison.csv` | JAX vs CuPy forward time, ratio r (single-node) |
| `distributed_single_eval.csv` | Same for distributed models |
| `memory_breakdown.csv` | Forward/backward time split and FD gradient time (single-node) |

For large/distributed models that run AD and FD separately, each job appends
its half of the data (JAX columns or CuPy columns). The rows must be merged
before plotting (combine the JAX forward time from the AD row with the CuPy
forward time from the FD row).

**Important:** Since results append to CSVs, delete the CSV files before a
full re-run to avoid duplicate rows.

## Reproducing Figure 4

Figure 4 aggregates data from **two** experiments:

1. **This experiment** -- framework ratio r and backward/forward ratio beta
2. **Figure 6** (`fig6_resource_efficiency/`) -- observed AD-vs-FD speedup
   (stored in `minimum_resources_comparison.csv`)

Run both experiments, then generate the figure:
```bash
cd ../../plotting
python scripts/plot_framework_decomposition.py
```

## Dependencies

- Uses `fig2_wallclock/{model}/run.py` as the underlying benchmark script
  (each inner script `cd`s into the corresponding fig2 directory)
- Requires `minimum_resources_comparison.csv` from `fig6_resource_efficiency/`
  for the observed-speedup data points in the final plot
