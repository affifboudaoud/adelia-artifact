# Table 2: Structure-Preserving Differentiation Comparison

Compares five gradient computation strategies on models of increasing size,
measuring per-gradient wall time and peak GPU memory. Reproduces **Table 2**
in the paper.

## Strategies

| Strategy | Description |
|----------|-------------|
| FD | Central finite differences (2d+1 evaluations) |
| AD-Dense | JAX AD through dense Cholesky on full N x N matrix |
| AD-Loop | JAX AD through `lax.scan` BTA Cholesky (all carries stored) |
| AD-Loop-Ckpt | Same with `jax.checkpoint` (recomputes during backward) |
| AD-BTA | Custom backward pass exploiting BTA structure |

## Models

| Model | Type | Dimensions (nt x ns) |
|-------|------|---------------------|
| GST-S | Univariate | 5 x 92 |
| GST-C2 | 2-variate coregional | 12 x 708 |
| GST-C3 | 3-variate coregional | 8 x 1062 |
| GST-M | Univariate | 100 x 812 |
| GST-L | Univariate | 250 x 4002 |

AD-Dense and AD-Loop run out of memory on larger models; those cells are
marked "OOM" in the output.

## Resources

- **Nodes**: 1
- **Runtime**: ~2 hours
- **Benchmark runs**: 20 (+ 2 warmup)

## How to Run

```bash
sbatch run.sbatch
```

Edit `--account` in `run.sbatch` to match your allocation.

## Output

Results are written to `tab2_results.csv` with fields:
`model`, `dims`, `strategy`, `time_mean`, `time_std`, `mem_gib`, `timestamp`.

This CSV is also copied to `../../plotting/data/structure_comparison.csv` for
figure generation.

## Dependencies

None. Single-node, standalone experiment.
