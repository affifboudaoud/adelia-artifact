# ADELIA: Automatic Differentiation for Efficient Latent Inference Approximation

## Artifact for SC26 Paper

This artifact accompanies the paper *"ADELIA: Automatic Differentiation for Efficient Latent Inference Approximation"* submitted to SC 2026.

### Contributions and Artifacts

| Contribution | Description | Artifacts |
|---|---|---|
| **C1** | Structure-preserving backward pass for BTA Cholesky | A1, A2 |
| **C2** | Extension to multivariate coregional models | A1 |
| **C3** | Multi-GPU distributed automatic differentiation | A1, A2 |

| Artifact | Description | Location |
|---|---|---|
| **A1** | ADELIA/DALIA framework with JAX AD | `DALIA/` |
| **A2** | Serinv structured sparse solver | `serinv/` |
| **A3** | Benchmark suite (this artifact) | `experiments/`, `data/`, `plotting/` |

### Hardware Requirements

All experiments were conducted on CSCS Alps (Switzerland) using NVIDIA GH200 Grace Hopper Superchip nodes:
- **GPU**: NVIDIA H100 96 GB HBM3
- **CPU**: NVIDIA Grace (ARM Neoverse V2), 72 cores, 128 GB LPDDR5X
- **Interconnect**: HPE Slingshot-11, 200 Gb/s per direction

### Data Download

Input datasets (~1.5 GB) are hosted externally. Download them before running experiments:

```bash
pip install gdown
./scripts/download_data.sh
```

### Quick Start

```bash
# 1. Install (see INSTALL.md for detailed instructions)
# Option A: via submodules (requires git-lfs)
git submodule update --init
# Option B: clone directly (if git-lfs is unavailable)
git clone -b cleaning https://github.com/affifboudaoud/DALIA.git
git clone -b serinv_jax https://github.com/affifboudaoud/serinv.git
# Then install both
pip install -e DALIA/ -e serinv/

# 2. Quick validate (1 GH200 node, ~5 min)
cd validation/gst_small
sbatch validate.sbatch
```

### Directory Layout

```
artifact/
├── INSTALL.md              # Full installation guide
├── validation/             # Gradient correctness checks
├── experiments/
│   ├── common/             # Shared utilities and environment setup
│   ├── tab2_structure_comparison/   # Table 2
│   ├── fig2_wallclock/              # Figure 2
│   ├── fig4_framework_decomposition/# Figure 4
│   ├── fig5_scaling_study/          # Figure 5
│   ├── fig6_resource_efficiency/    # Figure 6
│   └── fig7_performance_analysis/   # Figure 7
├── plotting/               # Figure generation from CSVs
├── data/                   # Input datasets (see below)
├── scripts/                # Utility scripts
├── DALIA/                  # DALIA framework (git submodule)
└── serinv/                 # Serinv structured sparse solver (git submodule)
```

Each experiment directory contains its own **README.md** with goals,
resource requirements, reproduction steps, and output descriptions.

### Experiments Overview

| Paper Element | Directory | Nodes | Est. Runtime |
|---|---|---|---|
| Gradient validation (Sec. V) | [`validation/`](validation/README.md) | 1 | ~1 h |
| Table 2: Strategy comparison | [`experiments/tab2_structure_comparison/`](experiments/tab2_structure_comparison/README.md) | 1 | ~2 h |
| Figure 2: Wallclock timing | [`experiments/fig2_wallclock/`](experiments/fig2_wallclock/README.md) | 1--4 | ~8 h total |
| Figure 4: Framework decomposition | [`experiments/fig4_framework_decomposition/`](experiments/fig4_framework_decomposition/README.md) | 1--4 | ~3 h |
| Figure 5: Scaling study | [`experiments/fig5_scaling_study/`](experiments/fig5_scaling_study/README.md) | 4 | ~6 h |
| Figure 6: Resource efficiency | [`experiments/fig6_resource_efficiency/`](experiments/fig6_resource_efficiency/README.md) | 4--128 | ~12 h |
| Figure 7: Performance analysis | [`experiments/fig7_performance_analysis/`](experiments/fig7_performance_analysis/README.md) | 4 | ~1 h |


### Plotting

Reference CSV data from the paper is provided in `plotting/data/` so figures
can be regenerated without re-running experiments:

```bash
cd plotting/scripts
python plot_pipeline_wallclock_merged.py   # Figure 2
python plot_framework_decomposition.py     # Figure 4
python plot_scaling_study.py               # Figure 5
python plot_resource_energy_efficiency.py   # Figure 6
python plot_performance_analysis.py        # Figure 7
```

Each plotting script reads reference CSVs from `plotting/data/` and writes
PDFs to `plotting/figures/`.
See [`plotting/README.md`](plotting/README.md) for the full data-flow map
from experiments to CSVs to figures.

### Data Provenance

| Dataset | Description | Source |
|---|---|---|
| gst_small | Synthetic Gaussian ST, ns=92, nt=5 | Generated via SPDE |
| gst_medium | Synthetic Gaussian ST, ns=716, nt=50 | Generated via SPDE |
| gst_large | Synthetic Gaussian ST, ns=4002, nt=250 | Generated via SPDE |
| gst_coreg2_small | Synthetic 2-variate, ns=354, nt=12 | Generated via SPDE |
| gst_coreg3_small | Synthetic 3-variate, ns=354, nt=8 | Generated via SPDE |
| gst_temperature | Temperature dataset, ns=945, nt=365 | Generated via SPDE |
| SA_1 | Benchmark 3-variate, ns=1673, nt=192 | DALIA Zenodo |
| AP1 | Air pollution (Italy), ns=4210, nt=48 | DALIA Zenodo |
| WA_1 | Benchmark 3-variate, ns=1247, nt=512 | DALIA Zenodo |
| WA_2 | Benchmark 3-variate, ns=4485, nt=48 | DALIA Zenodo |

