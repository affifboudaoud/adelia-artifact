"""Generate pipeline_wallclock.csv for Figure 2 from benchmark results.

Reads per-gradient times from benchmark_results.csv and pipeline_results.csv,
then estimates total wallclock (optimization + Hessian + JIT) using the same
number of iterations for both AD and FD (conservative estimate).

Formulas:
    AD: wallclock = JIT + n_iters * t_grad_AD + 2 * n_hp * t_grad_AD
    FD: wallclock = n_iters * t_grad_FD + ceil(no_eval_hess / F) * t_eval_f
    where:
        no_eval_hess = 1 + 2*n_hp + 2*n_hp*(n_hp-1)  (2nd-order FD of objective)
        t_eval_f = t_grad_FD / ceil((2*n_hp+1) / F)   (single objective eval)
        F = number of F()-level parallel ranks (= nodes for distributed)

Usage:
    python generate_csv.py [--output_dir DIR]
"""

import argparse
import csv
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTTING_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "plotting", "data")


def load_benchmark_entries():
    entries = []
    path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            entries.append(row)
    return entries


def get_latest_entry(entries, model, method):
    """Get the latest entry for a model+method."""
    candidates = [
        e for e in entries
        if e["model"] == model and e["method"] == method
    ]
    if not candidates:
        return None
    # If timestamps available, use latest; otherwise return last entry
    if "timestamp" in candidates[0] and candidates[0]["timestamp"]:
        return max(candidates, key=lambda x: x.get("timestamp", ""))
    return candidates[-1]


# Model definitions: (csv_name, display_name, n_hp, nodes, jit_type)
MODELS = [
    ("gst_small",        "GST-S",  4, 1, "monolithic"),
    ("gst_medium",       "GST-M",  4, 1, "monolithic"),
    ("gst_large",        "GST-L",  4, 1, "monolithic"),
    ("gst_coreg2_small", "GST-C2", 9, 1, "monolithic"),
    ("gst_coreg3_small", "GST-C3", 15, 1, "monolithic"),
    ("sa1",              "SA1",    15, 1, "monolithic"),
    ("wa1",              "WA1",    15, 4, "two-phase"),
    ("ap1",              "AP1",    15, 4, "two-phase"),
    ("wa2",              "WA2",    15, 4, "two-phase"),
]


def compute_wallclock(ad_entry, fd_entry, n_hp, nodes):
    """Compute estimated wallclock for AD and FD at equal iteration count."""
    t_grad_ad = float(ad_entry["per_gradient_mean"])
    t_grad_fd = float(fd_entry["per_gradient_mean"])
    n_iters_ad = int(ad_entry["n_iterations"])
    n_iters_fd = int(fd_entry["n_iterations"])
    jit_time = float(ad_entry.get("jit_time", 0) or 0)

    # Use AD iteration count for both (conservative for FD)
    n_iters = n_iters_ad

    # AD wallclock
    ad_opt = n_iters * t_grad_ad
    ad_hess = 2 * n_hp * t_grad_ad
    ad_total = jit_time + ad_opt + ad_hess

    # FD wallclock
    # F()-level parallelism: for distributed models with FD, F() = nodes
    # For single-node, F() = 1
    n_feval = nodes  # F()-level parallel ranks for FD

    # Single objective eval time
    n_perturbations = 2 * n_hp + 1
    evals_per_grad = math.ceil(n_perturbations / n_feval)
    t_eval_f = t_grad_fd / evals_per_grad

    fd_opt = n_iters * t_grad_fd

    # FD Hessian: 2nd-order finite differences of objective
    no_eval_hess = 1 + 2 * n_hp + 4 * n_hp * (n_hp - 1) // 2
    fd_hess = math.ceil(no_eval_hess / n_feval) * t_eval_f

    fd_total = fd_opt + fd_hess

    return {
        "n_iters_fd": n_iters_fd,
        "n_iters_ad": n_iters_ad,
        "t_per_grad_fd": t_grad_fd,
        "t_per_grad_ad": t_grad_ad,
        "t_jit_ad": jit_time,
        "t_hessian_fd": fd_hess,
        "t_hessian_ad": ad_hess,
        "t_optim_fd": fd_opt,
        "t_optim_ad": ad_opt,
        "t_total_fd": fd_total,
        "t_total_ad": ad_total,
        "speedup_optim": fd_opt / ad_opt if ad_opt > 0 else 0,
        "speedup_hessian": fd_hess / ad_hess if ad_hess > 0 else 0,
        "speedup_total": fd_total / ad_total if ad_total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline_wallclock.csv")
    parser.add_argument("--output_dir", default=PLOTTING_DATA_DIR)
    args = parser.parse_args()

    entries = load_benchmark_entries()

    output_path = os.path.join(args.output_dir, "pipeline_wallclock.csv")

    fieldnames = [
        "model", "d", "nodes", "jit_type",
        "n_iters_fd", "n_iters_ad",
        "t_per_grad_fd", "t_per_grad_ad", "t_jit_ad",
        "t_hessian_fd", "t_hessian_ad",
        "t_optim_fd", "t_optim_ad",
        "t_total_fd", "t_total_ad",
        "speedup_optim", "speedup_hessian", "speedup_total",
    ]

    rows = []
    missing = []

    for csv_name, display_name, n_hp, nodes, jit_type in MODELS:
        ad = get_latest_entry(entries, csv_name, "jax_autodiff")
        fd = get_latest_entry(entries, csv_name, "finite_diff")

        if not ad or not fd:
            method = "AD" if not ad else "FD"
            missing.append(f"{display_name} ({method})")
            continue

        wc = compute_wallclock(ad, fd, n_hp, nodes)

        row = {"model": display_name, "d": n_hp, "nodes": nodes, "jit_type": jit_type}
        row.update(wc)
        rows.append(row)

    if missing:
        print(f"WARNING: Missing data for: {', '.join(missing)}")
        print("Run the missing benchmarks first.")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Format floats
            for k, v in row.items():
                if isinstance(v, float):
                    if k.startswith("speedup"):
                        row[k] = f"{v:.1f}"
                    elif k.startswith("t_jit") or k.startswith("t_per"):
                        row[k] = f"{v:.6f}"
                    else:
                        row[k] = f"{v:.4f}"
            writer.writerow(row)

    print(f"Wrote {len(rows)} models to {output_path}")

    # Print summary
    print(f"\n{'Model':8s} {'AD wall':>9s} {'FD wall':>9s} {'Speedup':>8s}")
    print("-" * 40)
    for row in rows:
        ad_t = float(row["t_total_ad"])
        fd_t = float(row["t_total_fd"])
        sp = float(row["speedup_total"])
        def fmt(s):
            if s < 60: return f"{s:.0f}s"
            if s < 3600: return f"{s/60:.1f}m"
            return f"{s/3600:.1f}h"
        print(f"{row['model']:8s} {fmt(ad_t):>9s} {fmt(fd_t):>9s} {sp:>7.1f}x")


if __name__ == "__main__":
    main()
