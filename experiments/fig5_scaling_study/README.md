# Figure 5: Scaling Study

Tests whether the AD speedup over FD remains consistent as problem size
increases. Reproduces **Figure 5** in the paper:
- **(a) Temporal scaling** (WA1): fixed ns=1247, varying nt = 2, 32, 64, 128, 256
- **(b) Spatial scaling** (WA2): fixed nt=48, varying ns = 72, 282, 1119

## Models

**Single-node** (both AD and FD run in one job, FD first then AD):

| Model | nt | ns | N (latent) | Nodes | Time limit |
|-------|----|----|------------|-------|------------|
| WA1-small | 2 | 1247 | 7,485 | 1 | 30 min |
| WA1-nt32 | 32 | 1247 | 119,715 | 1 | 30 min |
| WA1-nt64 | 64 | 1247 | 239,427 | 1 | 1 hour |
| WA1-nt128 | 128 | 1247 | 478,851 | 1 | 2 hours |
| WA2-ns72 | 48 | 72 | 10,371 | 1 | 30 min |
| WA2-ns282 | 48 | 282 | 40,611 | 1 | 1 hour |
| WA2-ns1119 | 48 | 1119 | 161,139 | 1 | 2 hours |

**Distributed** (AD and FD run as separate jobs):

| Model | nt | ns | N (latent) | Nodes | Time limit |
|-------|----|----|------------|-------|------------|
| WA1-nt256 | 256 | 1247 | 957,699 | 4 | 3 hours |

This model exceeds single-GPU memory. AD uses distributed JAX (two-phase).
FD uses `DistSerinvSolver` (`--solver_min_p 4`) with CUDA-aware MPI disabled.

## How to Run

**Single-node models** (one sbatch per model, runs both FD and AD):
```bash
cd wa1_temporal
sbatch run_2.sbatch
sbatch run_32.sbatch
sbatch run_64.sbatch
sbatch run_128.sbatch

cd ../wa2_spatial
sbatch run_ns72.sbatch
sbatch run_ns282.sbatch
sbatch run_ns1119.sbatch
```

**Distributed models** (separate AD and FD jobs per model):
```bash
cd wa1_temporal
sbatch run_256.sbatch        # AD
sbatch run_256_fd.sbatch     # FD
```

All jobs exit immediately after collecting 10 timed gradient measurements
(2 warmup + 10 timed). Edit `--account` in each `.sbatch` file to match
your allocation.

## Output

All jobs append to a shared `results/scaling_results.csv` in
`minimum_resources_comparison` format. When AD and FD run as separate
jobs, the second job merges its results into the existing row for
that model.

Key columns: `model`, `ad_gradient_time_mean`, `ad_gradient_time_std`,
`fd_gradient_time_mean`, `fd_gradient_time_std`, `per_gradient_speedup`.

The plotting script `plotting/scripts/plot_scaling_study.py` reads this
CSV automatically and merges with the baseline data.

## Directory Structure

```
fig5_scaling_study/
├── README.md
├── results/
│   └── scaling_results.csv      # Shared output (all models append here)
├── wa1_temporal/                 # Figure 5a
│   ├── run.py                   # Benchmark driver
│   ├── run_inner.sh             # AD (or both) inner script
│   ├── run_inner_fd.sh          # FD-only inner script (distributed)
│   ├── run_2.sbatch             # nt=2,   1 node
│   ├── run_32.sbatch            # nt=32,  1 node
│   ├── run_64.sbatch            # nt=64,  1 node
│   ├── run_128.sbatch           # nt=128, 1 node
│   ├── run_256.sbatch           # nt=256, 4 nodes, AD only
│   ├── run_256_fd.sbatch        # nt=256, 4 nodes, FD only
│   └── outputs/
└── wa2_spatial/                  # Figure 5b
    ├── run.py
    ├── run_inner.sh
    ├── run_inner_fd.sh
    ├── run_ns72.sbatch           # ns=72,   1 node
    ├── run_ns282.sbatch          # ns=282,  1 node
    ├── run_ns1119.sbatch         # ns=1119, 1 node
    └── outputs/
```

## Implementation Notes

- Single-node jobs run FD first, then free GPU memory (`cleanup_gpu_memory`),
  then run AD. This prevents the FD solver from starving the AD JIT compiler.
- `run_inner.sh` auto-detects `SLURM_NNODES`: runs `--benchmark_method both`
  on 1 node, `--benchmark_method jax_autodiff` on multi-node.
- `run_inner_fd.sh` uses `env_setup_nocuda.sh` (CuPy + CUDA-aware MPI
  disabled) and `--solver_min_p 4` for the distributed solver.

## Input Data

Each model loads from `../../data/scaling_wa1_temporal/` or
`../../data/scaling_wa2_spatial/`:
- `inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/` -- spatial/temporal basis, SPDE
  parameters, regression matrices, reference outputs

## Dependencies

None. Each scaling point is independent.
