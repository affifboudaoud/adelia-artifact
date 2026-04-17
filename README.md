# ADELIA: Automatic Differentiation for Efficient Laplace Inference Approximations

Reproduction artifact for the ADELIA paper. It bundles the framework
(DALIA + Serinv as pinned submodules), the per-experiment SLURM
scripts, the input datasets, and the plotting scripts that regenerate
every figure and table in the paper.

## Reproducing the paper

- **Artifact description:** [`artifact_description.pdf`](artifact_description.pdf)
  (LaTeX source: [`artifact_description.tex`](artifact_description.tex);
  rebuild with `make` or `tectonic artifact_description.tex`).
- **Full reproduction** is ~28 GPU-hours on NVIDIA GH200. For a
  smaller end-to-end check that exercises every code path on a single
  GH200, the four steps below suffice; each is documented in more
  detail in the linked README:

  1. **Install the conda environment and build `mpi4py`/`mpi4jax`**
     via [`scripts/install_alps.sh`](scripts/install_alps.sh) on a GPU
     node; see [`INSTALL.md`](INSTALL.md) for the step-by-step
     walkthrough and troubleshooting.
  2. **Gradient correctness** on a small univariate and a small
     multivariate model (`gst_small`, `gst_coreg3_small`); see
     [`validation/README.md`](validation/README.md).
  3. **One end-to-end wall-clock run** on the same two small models
     (`experiments/fig4_wallclock/gst_small`,
     `experiments/fig4_wallclock/gst_coreg3_small`); see
     [`experiments/fig4_wallclock/README.md`](experiments/fig4_wallclock/README.md).
  4. **Regenerate all paper figures** from the reference CSVs shipped
     in [`plotting/data/`](plotting/data/) by running
     `cd plotting && bash generate_all_figures.sh`; see
     [`plotting/README.md`](plotting/README.md).

- **Configuration:** before submitting any job, set your CSCS
  allocation and conda env name in
  [`experiments/common/env_setup.sh`](experiments/common/env_setup.sh)
  (`SBATCH_ACCOUNT`, `CONDA_ENV`).
- **Submodule pins:** `DALIA/` at `9bdf6e6` (branch `cleaning`);
  `serinv/` at `5fa19bc` (branch `serinv_jax`).

## Hardware

All runs target CSCS Alps NVIDIA GH200 Grace Hopper nodes (four Hopper
GPUs, 96 GiB HBM3 each; 72-core ARM Neoverse V2 Grace CPU with 480 GiB
LPDDR5X; NVLink-C2C between CPU and GPUs; HPE Slingshot-11 between
nodes). Experiments use 1–128 GPUs depending on the figure.

## Data

Input datasets (~1.5 GB) are hosted externally. Download them before
running experiments:

```bash
pip install gdown
./scripts/download_data.sh
```

## Quick start

```bash
# 1. Install (see INSTALL.md for the full walkthrough)
#    Option A: via submodules (requires git-lfs)
git submodule update --init
#    Option B: clone directly (if git-lfs is unavailable)
git clone -b cleaning https://github.com/affifboudaoud/DALIA.git
git clone -b serinv_jax https://github.com/affifboudaoud/serinv.git
pip install -e DALIA/ -e serinv/

# 2. Quick validate (one GH200 node, ~5 min)
cd validation/gst_small && sbatch validate.sbatch
```

## Directory layout

```
artifact/
├── INSTALL.md              # Full installation guide
├── validation/             # Gradient correctness checks
├── experiments/
│   ├── common/             # Shared utilities and environment setup
│   ├── tab2_structure_comparison/
│   ├── fig4_wallclock/
│   ├── fig5_scaling_study/
│   ├── fig6_framework_decomposition/
│   ├── fig7_resource_efficiency/
│   └── fig8_performance_analysis/
├── plotting/               # Figure generation from CSVs
├── data/                   # Input datasets (see below)
├── scripts/                # Utility scripts
├── DALIA/                  # DALIA framework (git submodule)
└── serinv/                 # Serinv structured sparse solver (git submodule)
```

Each experiment directory is named after the paper element it
reproduces (`tab2_` → Table 2, `figN_` → Figure N) and contains its
own `README.md` with goals, resource requirements, reproduction
steps, and output descriptions.

## Experiments overview

| Paper element | Directory | Nodes | Est. runtime |
|---|---|---|---|
| Gradient validation | [`validation/`](validation/README.md) | 1 | ~1 h |
| Table 2 – Strategy comparison | [`experiments/tab2_structure_comparison/`](experiments/tab2_structure_comparison/README.md) | 1 | ~2 h |
| Figure 4 – End-to-end wall-clock | [`experiments/fig4_wallclock/`](experiments/fig4_wallclock/README.md) | 1–4 | ~8 h |
| Figure 5 – Problem-size scaling | [`experiments/fig5_scaling_study/`](experiments/fig5_scaling_study/README.md) | 1 | ~6 h |
| Figure 6 – Framework decomposition | [`experiments/fig6_framework_decomposition/`](experiments/fig6_framework_decomposition/README.md) | 1–4 | ~3 h |
| Figure 7 – Resource & energy efficiency | [`experiments/fig7_resource_efficiency/`](experiments/fig7_resource_efficiency/README.md) | 1–128 | ~12 h |
| Figure 8 – Per-stage performance analysis | [`experiments/fig8_performance_analysis/`](experiments/fig8_performance_analysis/README.md) | 4 | ~1 h |

## Plotting

Reference CSVs from the paper are shipped under
[`plotting/data/`](plotting/data/) so every figure can be regenerated
without re-running the experiments:

```bash
cd plotting && bash generate_all_figures.sh
```

Each plotting script reads reference CSVs from `plotting/data/` and
writes PDFs to `plotting/figures/`. See
[`plotting/README.md`](plotting/README.md) for the full script ↔ figure
map.

## Data provenance

| Dataset | Description | Source |
|---|---|---|
| gst_small | Synthetic Gaussian ST, ns=92, nt=5 | Generated via SPDE |
| gst_medium | Synthetic Gaussian ST, ns=716, nt=50 | Generated via SPDE |
| gst_large | Synthetic Gaussian ST, ns=4002, nt=250 | Generated via SPDE |
| gst_coreg2_small | Synthetic 2-variate, ns=354, nt=12 | Generated via SPDE |
| gst_coreg3_small | Synthetic 3-variate, ns=354, nt=8 | Generated via SPDE |
| gst_temperature | Temperature dataset, ns=945, nt=365 | INLADIST paper |
| SA_1 | Benchmark 3-variate, ns=1673, nt=192 | DALIA Zenodo |
| AP1 | Air pollution (Italy), ns=4210, nt=48 | DALIA Zenodo |
| WA_1 | Benchmark 3-variate, ns=1247, nt=512 | DALIA Zenodo |
| WA_2 | Benchmark 3-variate, ns=4485, nt=48 | DALIA Zenodo |
