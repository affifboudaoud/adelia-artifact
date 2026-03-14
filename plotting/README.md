# Plotting

Generates all paper figures from CSV data. Reference CSVs are provided in
`data/` so figures can be reproduced without re-running experiments.

## Quick Start

```bash
bash generate_all_figures.sh
```

Output PDFs are written to `figures/`.

## Individual Plot Scripts

| Script | Paper Element | Input CSVs |
|--------|--------------|------------|
| `plot_structure_comparison.py` | Table 2 | `structure_comparison.csv` |
| `plot_pipeline_wallclock.py` | Figure 2a (speedup) | `pipeline_wallclock.csv` |
| `plot_pipeline_wallclock_merged.py` | Figure 2b (convergence) | `pipeline_wallclock_merged.csv` |
| `plot_framework_decomposition.py` | Figure 4 | `single_eval_comparison.csv`, `distributed_single_eval.csv`, `memory_breakdown.csv`, `minimum_resources_comparison.csv` |
| `plot_scaling_study.py` | Figure 5a,b | `scaling_breakdown.csv` |
| `plot_resource_efficiency.py` | Figure 6a (FD scaling) | `resource_efficiency_all.csv` |
| `plot_resource_energy_efficiency.py` | Figure 6b (energy) | `resource_efficiency_all.csv` |
| `plot_performance_analysis.py` | Figure 7 | `distributed_breakdown.csv` |

## Data Flow

| CSV | Produced by | Consumed by |
|-----|-------------|-------------|
| `pipeline_wallclock.csv` | `fig2_wallclock/` | Figure 2a |
| `pipeline_wallclock_merged.csv` | `fig2_wallclock/` | Figure 2b |
| `structure_comparison.csv` | `tab2_structure_comparison/` | Table 2 |
| `single_eval_comparison.csv` | `fig4_framework_decomposition/` | Figure 4 |
| `distributed_single_eval.csv` | `fig4_framework_decomposition/` | Figure 4 |
| `memory_breakdown.csv` | `fig4_framework_decomposition/` | Figure 4 |
| `minimum_resources_comparison.csv` | `fig6_resource_efficiency/` | Figures 2, 4, 6 |
| `resource_efficiency_all.csv` | `fig6_resource_efficiency/` | Figure 6 |
| `scaling_breakdown.csv` | `fig5_scaling_study/` | Figure 5 |
| `distributed_breakdown.csv` | `fig7_performance_analysis/` | Figure 7 |

## Regenerating from Experiment Outputs

If you re-run experiments, copy the resulting CSVs into `data/` (or point the
`--output_csv` flags at `plotting/data/` directly), then run
`generate_all_figures.sh`.
