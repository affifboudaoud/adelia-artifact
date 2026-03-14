# Figure 6: Resource Efficiency

Measures how gradient computation time scales with the number of nodes for
FD, and compares against a fixed-node AD baseline. Also records energy
consumption via Cray PM counters. Reproduces **Figure 6** in the paper.

## Structure

```
fig6_resource_efficiency/
├── ad_baseline/           # AD on fixed 4 nodes
│   ├── ap1/
│   │   ├── run.sbatch
│   │   ├── run_inner.sh
│   │   └── outputs/
│   ├── wa1/
│   ├── wa2/
│   └── sa1/
├── fd_scaling/            # FD at varying node counts
│   ├── ap1/
│   │   ├── run_4n.sbatch
│   │   ├── run_8n.sbatch
│   │   ├── run_16n.sbatch
│   │   ├── run_32n.sbatch
│   │   ├── run_64n.sbatch
│   │   ├── run_128n.sbatch
│   │   ├── run_inner.sh
│   │   └── outputs/
│   ├── wa1/
│   ├── wa2/
│   └── sa1/
└── results/
```

## What It Measures

- **AD baseline**: Per-gradient time on 4 nodes using JAX autodiff (GPU)
- **FD scaling**: Per-gradient time on 4, 8, 16, 32, 64, 128 nodes using
  finite differences (CPU-only)
- **Energy**: Node power (W) and energy per gradient (J) via Cray PM counters

## Resources

| Job type | Nodes | Runtime | Count |
|----------|-------|---------|-------|
| AD baseline | 4 | ~2 hours | 4 models |
| FD scaling | 4--128 | ~4 hours each | 4 models x 6 node counts |

Total: ~12 hours if all jobs run in parallel (requires up to 128 nodes for
the largest FD runs).

## How to Run

**AD baseline** (4 nodes each):
```bash
cd ad_baseline
sbatch ap1/run.sbatch
sbatch wa1/run.sbatch
sbatch wa2/run.sbatch
sbatch sa1/run.sbatch
```

**FD scaling** (4 to 128 nodes):
```bash
cd fd_scaling/ap1
sbatch run_4n.sbatch
sbatch run_8n.sbatch
sbatch run_16n.sbatch
sbatch run_32n.sbatch
sbatch run_64n.sbatch
sbatch run_128n.sbatch
# Repeat for wa1, wa2, sa1
```

All scripts call `fig2_wallclock/{model}/run.py` with `--benchmark_mode` and
either `--benchmark_method jax_autodiff` or `--benchmark_method finite_diff`.

## Output

All jobs append to `results/resource_efficiency_all.csv` with fields:
`model`, `method`, `n_nodes`, `n_tasks`, `n_hyperparams`,
`gradient_time_mean`, `gradient_time_std`, `energy_mean_j`, `energy_std_j`,
`power_mean_w`, `power_std_w`, `timestamp`.

A derived `minimum_resources_comparison.csv` is also produced (AD at 4 nodes
vs FD at the minimum node count needed). This file is used by Figure 2 and
Figure 4 as well.

## Dependencies

- Uses `fig2_wallclock/{model}/run.py` as the benchmark script
- Output CSVs are consumed by Figure 4 (`minimum_resources_comparison.csv`)
  and the plotting scripts for Figure 6
