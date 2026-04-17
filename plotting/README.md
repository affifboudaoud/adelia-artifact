# Plotting

Generates all paper figures from CSV data. Reference CSVs are provided in
`data/` so figures can be reproduced without re-running experiments.

## Quick Start

```bash
bash generate_all_figures.sh
```

Output PDFs are written to `figures/`.

## Individual Plot Scripts

Paper figure numbers reflect the current `writing/` source (Figures 1--2
are the overview teaser and the two-phase method diagram, so the
evaluation plots start at Figure 3).

| Script | Paper Element | Input CSVs |
|--------|--------------|------------|
| `scripts/plot_structure_comparison.py` | Table 2 | `structure_comparison.csv` |
| `plot_convergence_all.py` | Figure 3 | per-run L-BFGS traces |
| `scripts/plot_pipeline_wallclock_merged.py` | Figure 4 | `pipeline_wallclock_merged.csv` |
| `scripts/plot_scaling_study.py` | Figure 5 | `scaling_breakdown.csv` |
| `scripts/plot_framework_decomposition.py` | Figure 6 | `single_eval_comparison.csv`, `distributed_single_eval.csv`, `memory_breakdown.csv`, `minimum_resources_comparison.csv` |
| `scripts/plot_resource_energy_efficiency.py` | Figure 7 | `resource_efficiency_all.csv` |
| `scripts/plot_performance_analysis.py` | Figure 8 | `distributed_breakdown.csv` |

## Data Flow

Experiment directory names (e.g. `fig2_wallclock/`) are historical and
do **not** reflect current paper figure numbers; use the table above as
the source of truth.

| CSV | Produced by | Consumed by |
|-----|-------------|-------------|
| `structure_comparison.csv` | `tab2_structure_comparison/` | Table 2 |
| `pipeline_wallclock.csv` | `fig2_wallclock/` | Figure 4 |
| `pipeline_wallclock_merged.csv` | `fig2_wallclock/` | Figure 4 |
| `scaling_breakdown.csv` | `fig5_scaling_study/` | Figure 5 |
| `single_eval_comparison.csv` | `fig4_framework_decomposition/` | Figure 6 |
| `distributed_single_eval.csv` | `fig4_framework_decomposition/` | Figure 6 |
| `memory_breakdown.csv` | `fig4_framework_decomposition/` | Figure 6 |
| `minimum_resources_comparison.csv` | `fig6_resource_efficiency/` | Figures 4, 6, 7 |
| `resource_efficiency_all.csv` | `fig6_resource_efficiency/` | Figure 7 |
| `distributed_breakdown.csv` | `fig7_performance_analysis/` | Figure 8 |

## Regenerating from Experiment Outputs

If you re-run experiments, copy the resulting CSVs into `data/` (or point the
`--output_csv` flags at `plotting/data/` directly), then run
`generate_all_figures.sh`.
