"""Structured result collection and output for paper data generation."""

import json
import os
import socket
import subprocess
import time
from datetime import datetime

import numpy as np

from dalia.utils import print_msg


def _to_serializable(obj):
    """Recursively convert arrays and scalars to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "get"):
        return obj.get().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return str(obj)


def get_gpu_memory_info():
    """Query GPU memory usage from CuPy memory pool.

    Returns
    -------
    dict
        peak_bytes, used_bytes, total_bytes. All -1 if unavailable.
    """
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        device = cp.cuda.Device()
        free, total = device.mem_info
        return {
            "peak_bytes": pool.total_bytes(),
            "used_bytes": pool.used_bytes(),
            "total_bytes": total,
        }
    except Exception:
        return {"peak_bytes": -1, "used_bytes": -1, "total_bytes": -1}


def get_gpu_name():
    """Get GPU model name from nvidia-smi.

    Returns
    -------
    str
        GPU name string, or "unknown" if unavailable.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip().split("\n")[0].strip()
    except Exception:
        return "unknown"


def _get_mpi_rank():
    """Return MPI rank, or 0 if MPI is unavailable."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def collect_optimization_results(
    dalia,
    model_name,
    method_name,
    total_time,
    jit_time=-1.0,
    power_result=None,
):
    """Collect all optimization results from a completed DALIA run.

    Parameters
    ----------
    dalia : DALIA
        A DALIA instance after .run() or .minimize() has been called.
    model_name : str
        Model identifier (e.g., "gst_small").
    method_name : str
        Gradient method ("jax_autodiff" or "finite_diff").
    total_time : float
        Wall-clock optimization time in seconds (excluding JIT).
    jit_time : float
        Time for first forward+gradient call (JIT compilation).
    power_result : dict or None
        Power measurement dict from PowerSampler.result().

    Returns
    -------
    dict
        Complete results dictionary suitable for JSON serialization.
    """
    model = dalia.model
    result = dalia.minimization_result

    n_hyper = model.n_hyperparameters
    n_latent = model.n_latent_parameters

    n_iterations = len(result.get("f_values", []))

    gpu_mem = get_gpu_memory_info()

    data = {
        "metadata": {
            "model_name": model_name,
            "gradient_method": method_name,
            "solver_type": dalia.config.solver.type,
            "precision": os.environ.get("JAX_DEFAULT_DTYPE_FLOAT", "float64"),
            "n_hyperparameters": n_hyper,
            "n_latent_parameters": n_latent,
            "n_nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "n_tasks": int(os.environ.get("SLURM_NTASKS", 1)),
            "hostname": socket.gethostname(),
            "gpu_type": get_gpu_name(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "timestamp": datetime.now().isoformat(),
        },
        "jit": {
            "time_s": jit_time,
        },
        "optimization": {
            "total_time_s": total_time,
            "n_iterations": n_iterations,
            "final_f": _to_serializable(result.get("f")),
            "final_theta": _to_serializable(result.get("theta")),
            "converged": n_iterations > 0,
        },
        "per_iteration": {
            "f_values": _to_serializable(result.get("f_values", [])),
            "theta_values": _to_serializable(result.get("theta_values", [])),
            "objective_function_time": _to_serializable(
                getattr(dalia, "objective_function_time", [])
            ),
            "solver_time": _to_serializable(
                getattr(dalia, "solver_time", [])
            ),
            "construction_time": _to_serializable(
                getattr(dalia, "construction_time", [])
            ),
        },
        "gpu_memory": _to_serializable(gpu_mem),
    }

    if power_result is not None and power_result.get("power_mean_w", 0) > 0:
        data["power"] = {
            "power_mean_w": power_result["power_mean_w"],
            "power_std_w": power_result["power_std_w"],
            "energy_j": power_result.get("energy_mean_j", 0.0),
        }

    return data


def write_results_json(results, output_dir, model_name, method_name):
    """Write optimization results to a JSON file (rank 0 only).

    Parameters
    ----------
    results : dict
        Results from collect_optimization_results().
    output_dir : str
        Directory for output files.
    model_name : str
        Model identifier.
    method_name : str
        Method identifier.

    Returns
    -------
    str or None
        Path to the written file, or None if not rank 0.
    """
    if _get_mpi_rank() != 0:
        return None

    os.makedirs(output_dir, exist_ok=True)

    method_short = "ad" if "jax" in method_name else "fd"
    precision = results.get("metadata", {}).get("precision", "float64")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{method_short}_{precision}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(_to_serializable(results), f, indent=2)

    print_msg(f"Results written to: {filepath}")
    return filepath


def append_speedup_csv(
    model_name,
    n_hyper,
    fd_time,
    jax_time,
    gpu_type,
    csv_path,
):
    """Append a row to the speedup optimization time CSV.

    Parameters
    ----------
    model_name : str
        Model identifier.
    n_hyper : int
        Number of hyperparameters.
    fd_time : float
        FD optimization time in seconds.
    jax_time : float
        JAX optimization time in seconds.
    gpu_type : str
        GPU identifier ("A100" or "GH200").
    csv_path : str
        Path to output CSV.
    """
    if _get_mpi_rank() != 0:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    speedup = fd_time / jax_time if jax_time > 0 else float("inf")

    header = "Example,N_hyper,GPU,FD (s),JAX (s),Speedup\n"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(
            f"{model_name},{n_hyper},{gpu_type},"
            f"{fd_time:.2f},{jax_time:.2f},{speedup:.2f}x\n"
        )
