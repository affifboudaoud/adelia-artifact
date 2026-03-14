"""Utility functions for gradient computation benchmarking."""

import gc
import os
import time
from datetime import datetime

import numpy as np

from dalia.utils import print_msg


def cleanup_gpu_memory():
    """Free GPU memory pools between benchmark phases."""
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass


def time_gradient(
    dalia, n_runs=10, warmup_runs=2, measure_power=False, energy_monitor=None
):
    """Time gradient computation for a DALIA instance.

    Parameters
    ----------
    dalia : DALIA
        The DALIA instance to benchmark.
    n_runs : int
        Number of timing runs.
    warmup_runs : int
        Number of warmup runs before timing.
    measure_power : bool
        If True, sample GPU power draw during timed runs.
    energy_monitor : EnergyMonitor or None
        If provided, measure energy via Cray PM hardware counters over
        the entire batch of n_runs, then divide by n_runs for per-gradient values.

    Returns
    -------
    dict
        Dictionary with timing results (mean, std, all_times) and
        optionally power/energy and hardware energy fields.
    """
    theta = dalia.model.theta.copy()
    n_hyperparameters = dalia.model.n_hyperparameters

    if n_hyperparameters == 0:
        print_msg("No hyperparameters to compute gradient for.")
        return {"mean": 0.0, "std": 0.0, "all_times": []}

    is_jax = dalia.config.gradient_method == "jax_autodiff"

    if is_jax:
        return _time_jax_gradient(
            dalia, theta, n_runs, warmup_runs, measure_power, energy_monitor
        )
    else:
        return _time_fd_gradient(
            dalia, theta, n_runs, warmup_runs, measure_power, energy_monitor
        )


def _time_jax_gradient(
    dalia, theta, n_runs, warmup_runs, measure_power=False, energy_monitor=None
):
    """Time JAX autodiff gradient computation."""
    for _ in range(warmup_runs):
        _ = dalia.jax_grad_func(theta)

    sampler = _maybe_start_power_sampler(measure_power)
    if energy_monitor is not None:
        energy_monitor.mark_start()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = dalia.jax_grad_func(theta)
        times.append(time.perf_counter() - t0)

    energy_sample = None
    if energy_monitor is not None:
        energy_sample = energy_monitor.mark_end(label="batch")

    result = {
        "mean": np.mean(times),
        "std": np.std(times),
        "all_times": times,
    }
    if energy_sample is not None:
        result["node_joules_per_grad"] = energy_sample["node_joules"] / n_runs
        result["gpu_joules_per_grad"] = energy_sample["gpu_joules"] / n_runs
        result["other_joules_per_grad"] = energy_sample["other_joules"] / n_runs
    _maybe_stop_power_sampler(sampler, result, sum(times))
    return result


def _time_fd_gradient(
    dalia, theta, n_runs, warmup_runs, measure_power=False, energy_monitor=None
):
    """Time finite difference gradient computation using DALIA's internal
    MPI-parallel _objective_function."""
    dalia.iter = 0
    for _ in range(warmup_runs):
        _ = dalia._objective_function(theta)

    sampler = _maybe_start_power_sampler(measure_power)
    if energy_monitor is not None:
        energy_monitor.mark_start()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = dalia._objective_function(theta)
        times.append(time.perf_counter() - t0)

    energy_sample = None
    if energy_monitor is not None:
        energy_sample = energy_monitor.mark_end(label="batch")

    result = {
        "mean": np.mean(times),
        "std": np.std(times),
        "all_times": times,
    }
    if energy_sample is not None:
        result["node_joules_per_grad"] = energy_sample["node_joules"] / n_runs
        result["gpu_joules_per_grad"] = energy_sample["gpu_joules"] / n_runs
        result["other_joules_per_grad"] = energy_sample["other_joules"] / n_runs
    _maybe_stop_power_sampler(sampler, result, sum(times))
    return result


def _maybe_start_power_sampler(measure_power):
    """Start a PowerSampler if requested, returning it or None."""
    if not measure_power:
        return None
    from examples_utils.power_utils import PowerSampler

    sampler = PowerSampler(interval=0.1)
    sampler.start()
    return sampler


def _maybe_stop_power_sampler(sampler, result, total_time):
    """Stop the sampler and merge power/energy fields into result."""
    if sampler is None:
        return
    from examples_utils.power_utils import compute_energy

    sampler.stop()
    power = sampler.result()
    energy = compute_energy(power, total_time)
    result["power_mean_w"] = power["power_mean_w"]
    result["power_std_w"] = power["power_std_w"]
    result["energy_mean_j"] = energy["energy_mean_j"]
    result["energy_std_j"] = energy["energy_std_j"]


def write_benchmark_result(output_file, result_dict):
    """Write benchmark result to CSV file (MPI-aware).

    Only rank 0 writes to file. Creates file with header if it doesn't exist.

    Parameters
    ----------
    output_file : str
        Path to output CSV file.
    result_dict : dict
        Dictionary with keys: model, method, n_nodes, n_tasks, n_hyperparams,
        gradient_time_mean, gradient_time_std, timestamp.
        Optionally: power_mean_w, power_std_w, energy_mean_j, energy_std_j.
    """
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    if rank != 0:
        return

    header = (
        "model,method,n_nodes,n_tasks,n_hyperparams,"
        "gradient_time_mean,gradient_time_std,"
        "power_mean_w,power_std_w,energy_mean_j,energy_std_j,"
        "node_joules_per_grad,gpu_joules_per_grad,other_joules_per_grad,"
        "timestamp\n"
    )
    file_exists = os.path.exists(output_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a") as f:
        if not file_exists:
            f.write(header)

        line = (
            f"{result_dict['model']},"
            f"{result_dict['method']},"
            f"{result_dict['n_nodes']},"
            f"{result_dict['n_tasks']},"
            f"{result_dict['n_hyperparams']},"
            f"{result_dict['gradient_time_mean']:.6f},"
            f"{result_dict['gradient_time_std']:.6f},"
            f"{result_dict.get('power_mean_w', 0.0):.2f},"
            f"{result_dict.get('power_std_w', 0.0):.2f},"
            f"{result_dict.get('energy_mean_j', 0.0):.2f},"
            f"{result_dict.get('energy_std_j', 0.0):.2f},"
            f"{result_dict.get('node_joules_per_grad', 0.0):.2f},"
            f"{result_dict.get('gpu_joules_per_grad', 0.0):.2f},"
            f"{result_dict.get('other_joules_per_grad', 0.0):.2f},"
            f"{result_dict['timestamp']}\n"
        )
        f.write(line)


def get_mpi_info():
    """Get MPI rank, size, and node count information.

    Returns
    -------
    dict
        Dictionary with rank, size, n_nodes keys.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        rank = 0
        size = 1

    n_nodes = int(os.environ.get("SLURM_NNODES", 1))

    return {"rank": rank, "size": size, "n_nodes": n_nodes}


def run_benchmark(dalia_fd, dalia_jax, args, model_name):
    """Run gradient benchmark for FD and/or JAX methods.

    Parameters
    ----------
    dalia_fd : DALIA or None
        DALIA instance configured for finite differences. Can be None if not benchmarking FD.
    dalia_jax : DALIA or None
        DALIA instance configured for JAX autodiff. Can be None if not benchmarking JAX.
    args : argparse.Namespace
        Parsed arguments with benchmark_mode, n_benchmark_runs, output_csv.
    model_name : str
        Name of the model being benchmarked.

    Returns
    -------
    tuple
        (fd_results, jax_results) where each is a dict from time_gradient
        or None if not benchmarked.
    """
    fd_results = None
    jax_results = None
    mpi_info = get_mpi_info()
    dalia_ref = dalia_fd if dalia_fd is not None else dalia_jax
    n_hyperparams = dalia_ref.model.n_hyperparameters
    timestamp = datetime.now().isoformat()
    measure_power = getattr(args, "measure_power", False)
    use_energy_monitor = getattr(args, "energy_monitor", False)

    energy_monitor = None
    if use_energy_monitor:
        from energy_monitor import EnergyMonitor

        energy_monitor = EnergyMonitor(rank=mpi_info["rank"])
        if not energy_monitor.available:
            if mpi_info["rank"] == 0:
                print_msg("WARNING: Cray PM counters not available, disabling energy monitor")
            energy_monitor = None

    if mpi_info["rank"] == 0:
        print_msg(f"\n{'='*70}")
        print_msg(f"GRADIENT BENCHMARK: {model_name}")
        print_msg(f"{'='*70}")
        print_msg(f"Nodes: {mpi_info['n_nodes']}, Tasks: {mpi_info['size']}")
        print_msg(f"Hyperparameters: {n_hyperparams}")
        print_msg(f"Benchmark runs: {args.n_benchmark_runs}")
        if measure_power:
            print_msg("Power measurement: enabled")
        if energy_monitor is not None:
            print_msg("Energy monitor: enabled (Cray PM counters)")

    if args.benchmark_method in ["finite_diff", "both"] and dalia_fd is not None:
        if mpi_info["rank"] == 0:
            print_msg("\nTiming finite difference gradient...")

        try:
            fd_results = time_gradient(
                dalia_fd,
                n_runs=args.n_benchmark_runs,
                warmup_runs=2,
                measure_power=measure_power,
                energy_monitor=energy_monitor,
            )
        except Exception as e:
            if mpi_info["rank"] == 0:
                print_msg(f"  FD benchmark failed: {e}")
                print_msg("  Continuing without FD results.")
            fd_results = None
            try:
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()
            except ImportError:
                pass

        if fd_results is not None and mpi_info["rank"] == 0:
            print_msg(f"  Mean: {fd_results['mean']:.4f}s")
            print_msg(f"  Std:  {fd_results['std']:.4f}s")
            if measure_power and fd_results.get("power_mean_w", 0) > 0:
                print_msg(f"  Power: {fd_results['power_mean_w']:.1f} W")
                print_msg(f"  Energy: {fd_results['energy_mean_j']:.1f} J")
            if fd_results.get("node_joules_per_grad", 0) > 0:
                print_msg(f"  Node energy/grad: {fd_results['node_joules_per_grad']:.1f} J")
                print_msg(f"  GPU energy/grad:  {fd_results['gpu_joules_per_grad']:.1f} J")
                print_msg(f"  Other energy/grad: {fd_results['other_joules_per_grad']:.1f} J")

        if fd_results is not None and args.output_csv:
            result_row = {
                "model": model_name,
                "method": "finite_diff",
                "n_nodes": mpi_info["n_nodes"],
                "n_tasks": mpi_info["size"],
                "n_hyperparams": n_hyperparams,
                "gradient_time_mean": fd_results["mean"],
                "gradient_time_std": fd_results["std"],
                "timestamp": timestamp,
            }
            if measure_power:
                result_row["power_mean_w"] = fd_results.get("power_mean_w", 0.0)
                result_row["power_std_w"] = fd_results.get("power_std_w", 0.0)
                result_row["energy_mean_j"] = fd_results.get("energy_mean_j", 0.0)
                result_row["energy_std_j"] = fd_results.get("energy_std_j", 0.0)
            result_row["node_joules_per_grad"] = fd_results.get("node_joules_per_grad", 0.0)
            result_row["gpu_joules_per_grad"] = fd_results.get("gpu_joules_per_grad", 0.0)
            result_row["other_joules_per_grad"] = fd_results.get("other_joules_per_grad", 0.0)
            write_benchmark_result(args.output_csv, result_row)

    if args.benchmark_method in ["jax_autodiff", "both"] and dalia_jax is not None:
        if mpi_info["rank"] == 0:
            print_msg("\nTiming JAX autodiff gradient...")

        jax_results = time_gradient(
            dalia_jax,
            n_runs=args.n_benchmark_runs,
            warmup_runs=2,
            measure_power=measure_power,
            energy_monitor=energy_monitor,
        )

        if mpi_info["rank"] == 0:
            print_msg(f"  Mean: {jax_results['mean']:.4f}s")
            print_msg(f"  Std:  {jax_results['std']:.4f}s")
            if measure_power and jax_results.get("power_mean_w", 0) > 0:
                print_msg(f"  Power: {jax_results['power_mean_w']:.1f} W")
                print_msg(f"  Energy: {jax_results['energy_mean_j']:.1f} J")
            if jax_results.get("node_joules_per_grad", 0) > 0:
                print_msg(f"  Node energy/grad: {jax_results['node_joules_per_grad']:.1f} J")
                print_msg(f"  GPU energy/grad:  {jax_results['gpu_joules_per_grad']:.1f} J")
                print_msg(f"  Other energy/grad: {jax_results['other_joules_per_grad']:.1f} J")

        if args.output_csv:
            result_row = {
                "model": model_name,
                "method": "jax_autodiff",
                "n_nodes": mpi_info["n_nodes"],
                "n_tasks": mpi_info["size"],
                "n_hyperparams": n_hyperparams,
                "gradient_time_mean": jax_results["mean"],
                "gradient_time_std": jax_results["std"],
                "timestamp": timestamp,
            }
            if measure_power:
                result_row["power_mean_w"] = jax_results.get("power_mean_w", 0.0)
                result_row["power_std_w"] = jax_results.get("power_std_w", 0.0)
                result_row["energy_mean_j"] = jax_results.get("energy_mean_j", 0.0)
                result_row["energy_std_j"] = jax_results.get("energy_std_j", 0.0)
            result_row["node_joules_per_grad"] = jax_results.get("node_joules_per_grad", 0.0)
            result_row["gpu_joules_per_grad"] = jax_results.get("gpu_joules_per_grad", 0.0)
            result_row["other_joules_per_grad"] = jax_results.get("other_joules_per_grad", 0.0)
            write_benchmark_result(args.output_csv, result_row)

    if mpi_info["rank"] == 0:
        print_msg(f"\n{'='*70}")
        print_msg("Benchmark completed")
        if args.output_csv:
            print_msg(f"Results written to: {args.output_csv}")
        print_msg(f"{'='*70}")

    return fd_results, jax_results


def write_scaling_csv_row(output_file, model_name, model_info, ad_results,
                          fd_results, n_runs, n_nodes):
    """Write or merge one row in a scaling-study CSV.

    If a row for ``model_name`` already exists (e.g. from a separate FD or AD
    job), non-zero columns from this run are merged into the existing row.

    Parameters
    ----------
    output_file : str
        Path to the shared CSV file.
    model_name : str
        Model identifier matching the plotting convention (e.g. "WA1-nt32").
    model_info : dict
        Model metadata with keys: likelihood, nv, ns, nt, latent_dim,
        n_hyperparams, block_size, n_blocks, solver.
    ad_results : dict or None
        Return value of time_gradient for AD, or None.
    fd_results : dict or None
        Return value of time_gradient for FD, or None.
    n_runs : int
        Number of benchmark runs.
    n_nodes : int
        Number of SLURM nodes used.
    """
    import csv

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    if rank != 0:
        return

    fieldnames = [
        "model", "likelihood", "nv", "ns", "nt", "latent_dim", "n_hyperparams",
        "block_size", "n_blocks", "solver", "min_nodes_ad",
        "ad_gradient_time_mean", "ad_gradient_time_std", "ad_n_samples",
        "fd_gradient_time_mean", "fd_gradient_time_std", "fd_n_samples",
        "fd_n_nodes", "per_gradient_speedup", "node_seconds_ad",
        "node_seconds_fd", "single_eval_fits_1gpu", "notes",
    ]

    ad_mean = ad_results["mean"] if ad_results else 0.0
    ad_std = ad_results["std"] if ad_results else 0.0
    ad_n = n_runs if ad_results else 0
    fd_mean = fd_results["mean"] if fd_results else 0.0
    fd_std = fd_results["std"] if fd_results else 0.0
    fd_n = n_runs if fd_results else 0
    fits_1gpu = "yes" if n_nodes == 1 else "no"

    new_row = {
        "model": model_name,
        "likelihood": model_info["likelihood"],
        "nv": model_info["nv"],
        "ns": model_info["ns"],
        "nt": model_info["nt"],
        "latent_dim": model_info["latent_dim"],
        "n_hyperparams": model_info["n_hyperparams"],
        "block_size": model_info["block_size"],
        "n_blocks": model_info["n_blocks"],
        "solver": model_info["solver"],
        "min_nodes_ad": n_nodes,
        "ad_gradient_time_mean": ad_mean,
        "ad_gradient_time_std": ad_std,
        "ad_n_samples": ad_n,
        "fd_gradient_time_mean": fd_mean,
        "fd_gradient_time_std": fd_std,
        "fd_n_samples": fd_n,
        "fd_n_nodes": n_nodes,
        "per_gradient_speedup": 0.0,
        "node_seconds_ad": 0.0,
        "node_seconds_fd": 0.0,
        "single_eval_fits_1gpu": fits_1gpu,
        "notes": "",
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read existing rows, merge if model already present
    rows = []
    merged = False
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["model"] == model_name:
                    # Merge: fill in whichever side (AD or FD) is new
                    if ad_results and float(row.get("ad_gradient_time_mean", 0)) == 0:
                        row["ad_gradient_time_mean"] = ad_mean
                        row["ad_gradient_time_std"] = ad_std
                        row["ad_n_samples"] = ad_n
                    elif ad_results:
                        row["ad_gradient_time_mean"] = ad_mean
                        row["ad_gradient_time_std"] = ad_std
                        row["ad_n_samples"] = ad_n
                    if fd_results and float(row.get("fd_gradient_time_mean", 0)) == 0:
                        row["fd_gradient_time_mean"] = fd_mean
                        row["fd_gradient_time_std"] = fd_std
                        row["fd_n_samples"] = fd_n
                        row["fd_n_nodes"] = n_nodes
                    elif fd_results:
                        row["fd_gradient_time_mean"] = fd_mean
                        row["fd_gradient_time_std"] = fd_std
                        row["fd_n_samples"] = fd_n
                        row["fd_n_nodes"] = n_nodes
                    merged = True
                rows.append(row)

    if not merged:
        rows.append(new_row)

    # Recompute derived columns for all rows
    for row in rows:
        ad_t = float(row["ad_gradient_time_mean"])
        fd_t = float(row["fd_gradient_time_mean"])
        n = int(row.get("min_nodes_ad", 1) or 1)
        fd_n_nodes = int(row.get("fd_n_nodes", n) or n)
        row["per_gradient_speedup"] = f"{fd_t / ad_t:.1f}" if ad_t > 0 and fd_t > 0 else "0.0"
        row["node_seconds_ad"] = f"{ad_t * n:.6f}" if ad_t > 0 else "0.000000"
        row["node_seconds_fd"] = f"{fd_t * fd_n_nodes:.6f}" if fd_t > 0 else "0.000000"

    # Write back
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print_msg(f"Scaling result written to: {output_file}")


def run_framework_comparison(dalia_jax, dalia_fd, args, model_name):
    """Measure single-eval JAX vs CuPy times and forward/backward breakdown.

    Writes results to CSV files for use by the framework decomposition plot.
    Produces three measurements:
      1. JAX single forward eval time
      2. CuPy single forward eval time (via serinv)
      3. JAX forward vs backward time breakdown

    Parameters
    ----------
    dalia_jax : DALIA
        DALIA instance configured with jax_autodiff.
    dalia_fd : DALIA
        DALIA instance configured with finite_diff (uses CuPy/serinv).
    args : argparse.Namespace
        Must have n_benchmark_runs and output_dir attributes.
    model_name : str
        Model identifier for CSV output.
    """
    mpi_info = get_mpi_info()
    n_runs = args.n_benchmark_runs
    dalia_ref = dalia_jax if dalia_jax is not None else dalia_fd
    n_hyper = dalia_ref.model.n_hyperparameters

    # --- 1. JAX forward only ---
    jax_fwd_times = []
    if dalia_jax is not None:
        theta_jax = np.asarray(
            dalia_jax.model.theta.get()
            if hasattr(dalia_jax.model.theta, "get")
            else dalia_jax.model.theta
        )
        for _ in range(2):
            dalia_jax.jax_objective(theta_jax)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            dalia_jax.jax_objective(theta_jax)
            jax_fwd_times.append(time.perf_counter() - t0)

    # --- 2. CuPy/serinv forward only ---
    cupy_fwd_times = []
    if dalia_fd is not None:
        theta_fd = dalia_fd.model.theta.copy()
        dalia_fd.iter = 0
        for _ in range(2):
            dalia_fd._evaluate_f(theta_fd)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            dalia_fd._evaluate_f(theta_fd)
            cupy_fwd_times.append(time.perf_counter() - t0)

    # --- 3. JAX forward+backward (full gradient) ---
    jax_grad_times = []
    if dalia_jax is not None:
        for _ in range(2):
            dalia_jax.jax_grad_func(theta_jax)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            dalia_jax.jax_grad_func(theta_jax)
            jax_grad_times.append(time.perf_counter() - t0)

    jax_fwd_mean = np.mean(jax_fwd_times) if jax_fwd_times else 0.0
    jax_fwd_std = np.std(jax_fwd_times) if jax_fwd_times else 0.0
    cupy_fwd_mean = np.mean(cupy_fwd_times) if cupy_fwd_times else 0.0
    cupy_fwd_std = np.std(cupy_fwd_times) if cupy_fwd_times else 0.0
    jax_grad_mean = np.mean(jax_grad_times) if jax_grad_times else 0.0
    jax_grad_std = np.std(jax_grad_times) if jax_grad_times else 0.0
    jax_bwd_mean = jax_grad_mean - jax_fwd_mean if jax_fwd_times else 0.0
    ratio = cupy_fwd_mean / jax_fwd_mean if jax_fwd_mean > 0 and cupy_fwd_times else 0.0

    if mpi_info["rank"] == 0:
        print_msg(f"\nFramework comparison: {model_name}")
        if jax_fwd_times:
            print_msg(f"  JAX forward:   {jax_fwd_mean:.6f} +/- {jax_fwd_std:.6f}s")
        else:
            print_msg(f"  JAX forward:   skipped (no AD instance)")
        if cupy_fwd_times:
            print_msg(f"  CuPy forward:  {cupy_fwd_mean:.6f} +/- {cupy_fwd_std:.6f}s")
        else:
            print_msg(f"  CuPy forward:  skipped (no FD instance)")
        if jax_fwd_times and cupy_fwd_times:
            print_msg(f"  r (CuPy/JAX):  {ratio:.2f}")
        if jax_grad_times:
            print_msg(f"  JAX gradient:  {jax_grad_mean:.6f} +/- {jax_grad_std:.6f}s")
            print_msg(f"  JAX backward:  {jax_bwd_mean:.6f}s (total - forward)")

    if mpi_info["rank"] != 0:
        return

    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Write single_eval CSV
    is_distributed = mpi_info["n_nodes"] > 1
    eval_csv = os.path.join(
        output_dir,
        "distributed_single_eval.csv" if is_distributed else "single_eval_comparison.csv",
    )
    header = "model,n_hyperparams,jax_fwd_time_mean,jax_fwd_time_std,cupy_eval_time_mean,cupy_eval_time_std,ratio_cupy_over_jax,n_runs\n"
    row = (
        f"{model_name},{n_hyper},"
        f"{jax_fwd_mean:.6f},{jax_fwd_std:.6f},"
        f"{cupy_fwd_mean:.6f},{cupy_fwd_std:.6f},"
        f"{ratio:.2f},{n_runs}\n"
    )
    file_exists = os.path.exists(eval_csv)
    with open(eval_csv, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(row)
    print_msg(f"  Written to: {eval_csv}")

    # Write memory_breakdown CSV (forward/backward split)
    if not is_distributed:
        mem_csv = os.path.join(output_dir, "memory_breakdown.csv")
        mem_header = "model,method,forward_time,backward_time,total_time,total_time_std,peak_memory_bytes,n_runs\n"
        # AD row
        ad_row = None
        if dalia_jax is not None:
            ad_row = (
                f"{model_name},AD,"
                f"{jax_fwd_mean},{jax_bwd_mean},{jax_grad_mean},{jax_grad_std},0,{n_runs}\n"
            )
        # FD row (no backward pass — total time is the full FD gradient)
        fd_row = None
        if dalia_fd is not None:
            theta_fd = dalia_fd.model.theta.copy()
            fd_grad_times = []
            dalia_fd.iter = 0
            for _ in range(2):
                dalia_fd._objective_function(theta_fd)
            for _ in range(n_runs):
                t0 = time.perf_counter()
                dalia_fd._objective_function(theta_fd)
                fd_grad_times.append(time.perf_counter() - t0)
            fd_mean = np.mean(fd_grad_times)
            fd_std = np.std(fd_grad_times)
            fd_row = f"{model_name},FD,{fd_mean},0.0,{fd_mean},{fd_std},0,{n_runs}\n"

        file_exists = os.path.exists(mem_csv)
        with open(mem_csv, "a") as f:
            if not file_exists:
                f.write(mem_header)
            if ad_row is not None:
                f.write(ad_row)
            if fd_row is not None:
                f.write(fd_row)
        print_msg(f"  Written to: {mem_csv}")


def _flatten(d, prefix=""):
    """Flatten nested timing dicts, skipping parent totals that have detail dicts.

    E.g. {"chol_fwd_sub": 10.0, "chol_detail": {"local": 8.0, "rs": 2.0}}
    becomes {"chol_fwd_sub": 10.0, "chol_detail.local": 8.0, "chol_detail.rs": 2.0}
    but only non-detail keys count toward the total.
    """
    flat = {}
    detail_parents = {k.replace("_detail", "") for k in d if k.endswith("_detail")}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _leaf_total(flat):
    """Sum only top-level stage times, excluding detail sub-keys."""
    return sum(v for k, v in flat.items() if "_detail" not in k)


def run_breakdown_benchmark(dalia_jax, args, model_name):
    """Run per-stage timing breakdown for distributed AD gradient.

    Calls ``dalia.jax_grad_func.timed(theta)`` which returns per-stage
    wall-clock times without affecting the normal code path.

    Parameters
    ----------
    dalia_jax : DALIA
        DALIA instance configured with jax_autodiff (distributed).
    args : argparse.Namespace
        Must have n_benchmark_runs, breakdown_csv.
    model_name : str
        Model identifier for CSV output.
    """
    mpi_info = get_mpi_info()
    n_runs = args.n_benchmark_runs

    timed_func = getattr(dalia_jax.jax_grad_func, "timed", None)
    if timed_func is None:
        if mpi_info["rank"] == 0:
            print_msg("ERROR: .timed not available on jax_grad_func "
                       "(only distributed split_jit/two_phase support this)")
        return

    theta = dalia_jax.model.theta.copy()
    theta_np = np.asarray(theta.get() if hasattr(theta, "get") else theta)

    verbose = mpi_info["rank"] == 0

    # Dry runs (JIT already compiled; these warm GPU caches)
    for i in range(2):
        _, _, _, timing = timed_func(theta_np)
        if verbose:
            flat = _flatten(timing)
            total = _leaf_total(flat)
            parts = "  ".join(f"{k}={v:.2f}s" for k, v in flat.items())
            print_msg(f"  dry run {i+1}: {parts}  (total={total:.2f}s)", flush=True)

    # Collect per-stage timings
    all_timings = []
    for i in range(n_runs):
        _, _, _, timing = timed_func(theta_np)
        all_timings.append(timing)
        if verbose:
            flat = _flatten(timing)
            total = _leaf_total(flat)
            parts = "  ".join(f"{k}={v:.2f}s" for k, v in flat.items())
            print_msg(f"  run {i+1}/{n_runs}: {parts}  (total={total:.2f}s)", flush=True)

    flat_timings = [_flatten(t) for t in all_timings]

    stages = list(flat_timings[0].keys())
    stage_stats = {}
    for stage in stages:
        vals = [t[stage] for t in flat_timings]
        stage_stats[stage] = {"mean": np.mean(vals), "std": np.std(vals)}

    if mpi_info["rank"] == 0:
        print_msg(f"\nPer-stage breakdown: {model_name} ({n_runs} runs)")
        total = sum(s["mean"] for k, s in stage_stats.items() if "_detail" not in k)
        for stage in stages:
            s = stage_stats[stage]
            is_detail = "_detail" in stage
            pct = s["mean"] / total * 100 if total > 0 and not is_detail else 0
            print_msg(f"  {stage:20s}: {s['mean']:.4f}s +/- {s['std']:.4f}s ({pct:.1f}%)")
        print_msg(f"  {'total':20s}: {total:.4f}s")

    if mpi_info["rank"] != 0:
        return

    output_csv = getattr(args, "breakdown_csv", None)
    if output_csv is None:
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    header = "model,n_nodes,stage,time_mean,time_std,n_runs\n"
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a") as f:
        if not file_exists:
            f.write(header)
        for stage in stages:
            s = stage_stats[stage]
            f.write(f"{model_name},{mpi_info['n_nodes']},{stage},"
                    f"{s['mean']:.4f},{s['std']:.4f},{n_runs}\n")

    print_msg(f"  Written to: {output_csv}")


def write_optimization_result(output_file, model_name, method, results, n_nodes,
                              n_hyperparams, jit_time=None):
    """Append a row to the shared benchmark results CSV.

    Only rank 0 writes. Safe for concurrent appends from parallel SLURM jobs.

    Parameters
    ----------
    output_file : str
        Path to the shared CSV file.
    model_name : str
        Model identifier (e.g. "gst_small", "sa1").
    method : str
        "jax_autodiff" or "finite_diff".
    results : dict
        Return value of ``dalia.run()``.
    n_nodes : int
        Number of SLURM nodes.
    n_hyperparams : int
        Number of hyperparameters (d).
    jit_time : float or None
        Time for first forward+gradient (JIT compilation), AD only.
    """
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    if rank != 0:
        return

    obj_times = results.get("objective_function_time", [])
    n_iter = len(obj_times)
    grad_mean = float(np.mean(obj_times)) if n_iter > 0 else 0.0
    grad_std = float(np.std(obj_times)) if n_iter > 0 else 0.0

    header = (
        "model,method,n_nodes,n_hyperparams,n_iterations,"
        "per_gradient_mean,per_gradient_std,"
        "optimization_time,hessian_time,marginals_time,wallclock_time,"
        "jit_time,final_f,timestamp\n"
    )

    jit_str = f"{jit_time:.4f}" if jit_time is not None else ""
    row = (
        f"{model_name},{method},{n_nodes},{n_hyperparams},{n_iter},"
        f"{grad_mean:.6f},{grad_std:.6f},"
        f"{results.get('t_optimization', 0.0):.4f},"
        f"{results.get('t_hessian', 0.0):.4f},"
        f"{results.get('t_marginals', 0.0):.4f},"
        f"{results.get('t_wallclock', 0.0):.4f},"
        f"{jit_str},{results.get('f', 0.0):.6f},"
        f"{datetime.now().isoformat()}\n"
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    file_exists = os.path.exists(output_file)
    with open(output_file, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(row)
