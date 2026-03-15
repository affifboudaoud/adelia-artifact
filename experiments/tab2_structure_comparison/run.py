"""Table 2: Structure-preserving differentiation comparison.

Compares five gradient strategies across five models of increasing size.
Measures per-gradient time (s) and peak GPU memory (GiB).

Strategies:
  FD          - Central finite differences (2d+1 evaluations)
  AD-Dense    - JAX AD through dense Cholesky on full N x N matrix
  AD-Loop     - JAX AD through lax.scan BTA Cholesky (all carries stored)
  AD-Loop-Ckpt - Same with jax.checkpoint (recomputes during backward)
  AD-BTA      - Custom backward pass exploiting BTA structure

Models:
  GST-S   - Univariate, 5 x 92
  GST-C2  - Coregional 2-variate, 12 x 708
  GST-C3  - Coregional 3-variate, 8 x 1062
  GST-M   - Univariate, 100 x 812
  GST-L   - Univariate, 250 x 4002
"""

import sys
import os
import time
import gc
import subprocess
import threading
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from parser_utils import parse_args

args = parse_args()

import jax
import jax.numpy as jnp

if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()

from dalia.core.jax_autodiff import configure_jax_precision

configure_jax_precision(args.precision)

import numpy as np

from dalia import xp, backend_flags
from dalia.configs import likelihood_config, models_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.core.jax_autodiff import (
    create_pure_jax_objective,
    create_pure_jax_objective_coregional,
)
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.models import CoregionalModel
from dalia.utils import print_msg
from benchmark_utils import time_gradient

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

PERTURBATION_C2 = [
    0.18197867, -0.12551227, 0.19998896,
    0.17226796, 0.14656176, -0.11864931,
    0.17817371, -0.13006157, 0.19308036,
]

PERTURBATION_C3 = [
    0.18197867, -0.12551227, 0.19998896,
    0.17226796, 0.14656176, -0.11864931,
    0.17817371, -0.13006157, 0.19308036,
    0.12317955, -0.14182536, 0.15686513,
    0.17168868, -0.0365025, 0.13315897,
]

STRATEGIES = ["FD", "AD-Dense", "AD-Loop", "AD-Loop-Ckpt", "AD-BTA"]


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_gst_small():
    data_dir = os.path.join(ARTIFACT_DIR, "data", "gst_small")
    st = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{data_dir}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0, "r_t": 0, "sigma_st": 0,
        "manifold": "sphere",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
    }))
    reg = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{data_dir}/inputs_regression",
        "n_fixed_effects": 6, "fixed_effects_prior_precision": 0.001,
    }))
    return Model(
        submodels=[reg, st],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": 4,
            "prior_hyperparameters": {"type": "penalized_complexity", "alpha": 0.01, "u": 4},
        }),
    )


def create_gst_coreg2_small():
    data_dir = os.path.join(ARTIFACT_DIR, "data", "gst_coreg2_small")
    input_dir = f"{data_dir}/inputs_nv2_ns354_nt12_nb2"
    theta_ref = np.load(f"{input_dir}/reference_outputs/theta_ref.npy")
    theta_initial = theta_ref + np.array(PERTURBATION_C2)

    st1 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_1/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0], "r_t": theta_initial[1], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[0], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[1], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    reg1 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_1/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model1 = Model(
        submodels=[reg1, st1],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[2],
            "prior_hyperparameters": {"type": "gaussian", "mean": theta_initial[2], "precision": 0.5},
        }),
    )

    st2 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_2/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[3], "r_t": theta_initial[4], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[3], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[4], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    reg2 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_2/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model2 = Model(
        submodels=[st2, reg2],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[5],
            "prior_hyperparameters": {"type": "gaussian", "mean": theta_ref[5], "precision": 0.5},
        }),
    )

    return CoregionalModel(
        models=[model1, model2],
        coregional_model_config=models_config.parse_config({
            "type": "coregional", "n_models": 2,
            "sigmas": [theta_initial[6], theta_initial[7]],
            "lambdas": [theta_initial[8]],
            "ph_sigmas": [
                {"type": "gaussian", "mean": theta_ref[6], "precision": 0.5},
                {"type": "gaussian", "mean": theta_ref[7], "precision": 0.5},
            ],
            "ph_lambdas": [
                {"type": "gaussian", "mean": 0.0, "precision": 0.5},
            ],
        }),
    )


def create_gst_coreg3_small():
    data_dir = os.path.join(ARTIFACT_DIR, "data", "gst_coreg3_small")
    input_dir = f"{data_dir}/inputs_nv3_ns354_nt8_nb3"
    theta_ref = np.load(f"{input_dir}/reference_outputs/theta_ref.npy")
    theta_initial = theta_ref + np.array(PERTURBATION_C3)

    st1 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_1/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0], "r_t": theta_initial[1], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[0], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[1], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    reg1 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_1/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model1 = Model(
        submodels=[reg1, st1],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[2],
            "prior_hyperparameters": {"type": "gaussian", "mean": theta_initial[2], "precision": 0.5},
        }),
    )

    st2 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_2/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[3], "r_t": theta_initial[4], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[3], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[4], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    reg2 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_2/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model2 = Model(
        submodels=[st2, reg2],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[5],
            "prior_hyperparameters": {"type": "gaussian", "mean": theta_ref[5], "precision": 0.5},
        }),
    )

    st3 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_3/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[6], "r_t": theta_initial[7], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[6], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[7], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    reg3 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_3/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model3 = Model(
        submodels=[st3, reg3],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[8],
            "prior_hyperparameters": {"type": "gaussian", "mean": theta_ref[8], "precision": 0.5},
        }),
    )

    return CoregionalModel(
        models=[model1, model2, model3],
        coregional_model_config=models_config.parse_config({
            "type": "coregional", "n_models": 3,
            "sigmas": [theta_initial[9], theta_initial[10], theta_initial[11]],
            "lambdas": [theta_initial[12], theta_initial[13], theta_initial[14]],
            "ph_sigmas": [
                {"type": "gaussian", "mean": theta_ref[6], "precision": 0.5},
                {"type": "gaussian", "mean": theta_ref[7], "precision": 0.5},
                {"type": "gaussian", "mean": theta_ref[8], "precision": 0.5},
            ],
            "ph_lambdas": [
                {"type": "gaussian", "mean": 0.0, "precision": 0.5},
                {"type": "gaussian", "mean": 0.0, "precision": 0.5},
                {"type": "gaussian", "mean": 0.0, "precision": 0.5},
            ],
        }),
    )


def create_gst_medium():
    data_dir = os.path.join(ARTIFACT_DIR, "data", "gst_medium")
    st = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{data_dir}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0, "r_t": 0, "sigma_st": 0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": 0.03972077083991806, "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": 2.3931471805599456, "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 1.4379142862353824, "precision": 0.5},
    }))
    reg = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{data_dir}/inputs_regression",
        "n_fixed_effects": 6, "fixed_effects_prior_precision": 0.001,
    }))
    return Model(
        submodels=[reg, st],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": 4,
            "prior_hyperparameters": {"type": "gaussian", "mean": 1.4, "precision": 0.5},
        }),
    )


def create_gst_large():
    data_dir = os.path.join(ARTIFACT_DIR, "data", "gst_large")
    st = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{data_dir}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": -0.960279229160082,
        "r_t": -0.3068528194400548,
        "sigma_st": -2.112085713764618,
        "manifold": "sphere",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
    }))
    reg = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{data_dir}/inputs_regression",
        "n_fixed_effects": 6, "fixed_effects_prior_precision": 0.001,
    }))
    return Model(
        submodels=[reg, st],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": 4,
            "prior_hyperparameters": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        }),
    )


MODELS = OrderedDict([
    ("GST-S",  {"create": create_gst_small,        "dims": r"5 \times 92",    "coregional": False}),
    ("GST-C2", {"create": create_gst_coreg2_small,  "dims": r"12 \times 708",  "coregional": True}),
    ("GST-C3", {"create": create_gst_coreg3_small,  "dims": r"8 \times 1062",  "coregional": True}),
    ("GST-M",  {"create": create_gst_medium,        "dims": r"100 \times 812", "coregional": False}),
    ("GST-L",  {"create": create_gst_large,         "dims": r"250 \times 4002","coregional": False}),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cleanup_gpu():
    """Release GPU memory between runs."""
    gc.collect()
    jax.clear_caches()
    gc.collect()
    try:
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def get_gpu_memory_used_gib():
    """Query current GPU memory usage in GiB via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=5,
        )
        return float(result.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        return 0.0


def _cuda_used_bytes():
    """Current GPU memory usage in bytes via CUDA runtime (device-global)."""
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


def _track_jax_peak_memory(fn, *args, **kwargs):
    """Run *fn* while sampling GPU memory in a background thread.

    Uses ``cudaMemGetInfo`` (device-global, independent of allocator type) so
    the measurement is correct with both BFC and platform allocators.

    Returns ``(result, peak_gib)`` where *peak_gib* is the peak memory above
    the pre-call baseline in GiB.
    """
    baseline = _cuda_used_bytes()
    peak_bytes = [baseline]
    stop_event = threading.Event()

    def _sampler():
        while not stop_event.is_set():
            cur = _cuda_used_bytes()
            if cur > peak_bytes[0]:
                peak_bytes[0] = cur
            stop_event.wait(0.001)

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    result = fn(*args, **kwargs)
    cur = _cuda_used_bytes()
    if cur > peak_bytes[0]:
        peak_bytes[0] = cur
    stop_event.set()
    t.join(timeout=2.0)
    return result, max(0.0, peak_bytes[0] - baseline) / (1024**3)


def make_dalia_config(solver_type, gradient_method, n_hyperparameters):
    return dalia_config.parse_config({
        "solver": {"type": solver_type},
        "gradient_method": gradient_method,
        "minimize": {
            "max_iter": 1, "gtol": 1e-3,
            "disp": False, "maxcor": max(n_hyperparameters, 1),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    })


def is_oom_error(exc):
    """Check if an exception is an out-of-memory error."""
    msg = str(exc).lower()
    oom_markers = ["out of memory", "resource_exhausted", "oom", "alloc",
                   "outofmemoryerror", "failed to allocate"]
    return any(m in msg for m in oom_markers)


def run_strategy(model_name, model_info, strategy, n_runs):
    """Run one (model, strategy) combination.

    Returns
    -------
    dict or None
        ``{"mean": float, "std": float, "mem_gib": float}`` on success,
        ``None`` on OOM.
    """
    create_fn = model_info["create"]
    is_coreg = model_info["coregional"]
    dalia = None

    cleanup_gpu()
    mem_before = get_gpu_memory_used_gib()

    try:
        model = create_fn()
        n_hp = model.n_hyperparameters

        if strategy == "FD":
            config = make_dalia_config("serinv", "finite_diff", n_hp)
            dalia = DALIA(model=model, config=config)

        elif strategy == "AD-Dense":
            config = make_dalia_config("dense", "jax_autodiff", n_hp)
            dalia = DALIA(model=model, config=config)

        else:
            ad_mode_map = {
                "AD-Loop": "scan",
                "AD-Loop-Ckpt": "scan_ckpt",
                "AD-BTA": "default",
            }
            ad_mode = ad_mode_map[strategy]
            config = make_dalia_config("serinv", "jax_autodiff", n_hp)
            dalia = DALIA(model=model, config=config)

            if ad_mode != "default":
                factory = (create_pure_jax_objective_coregional
                           if is_coreg else create_pure_jax_objective)
                dalia.jax_objective, dalia.jax_grad_func = factory(
                    dalia, ad_mode=ad_mode)

        if strategy == "FD":
            result = time_gradient(dalia, n_runs=n_runs, warmup_runs=2)
            mem_after = get_gpu_memory_used_gib()
            peak_mem = mem_after - mem_before
            if peak_mem < 0:
                peak_mem = mem_after
        else:
            # Measure peak memory during a dedicated warmup call so the
            # sampler thread does not interfere with the timed runs.
            theta = dalia.model.theta.copy()
            if hasattr(theta, "get"):
                theta = theta.get()
            _, peak_mem = _track_jax_peak_memory(
                dalia.jax_grad_func, theta
            )
            result = time_gradient(dalia, n_runs=n_runs, warmup_runs=2)

        return {
            "mean": result["mean"],
            "std": result["std"],
            "mem_gib": peak_mem,
        }

    except Exception as exc:
        if is_oom_error(exc):
            return None
        raise

    finally:
        if dalia is not None:
            del dalia
        cleanup_gpu()


def format_cell(result, show_std=False):
    """Format a result cell for the table."""
    if result is None:
        return "OOM", "OOM"
    t = result["mean"]
    mem = result["mem_gib"]
    if show_std:
        t_str = f"{t:.3f} +/- {result['std']:.3f}"
    else:
        t_str = f"{t:.3f}" if t >= 0.01 else f"{t:.4f}"
    mem_str = f"{mem:.1f}" if mem >= 0.1 else "<0.1"
    return t_str, mem_str


def write_csv(results, output_csv):
    """Write results to CSV."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    header = "model,dims,strategy,time_mean,time_std,mem_gib,timestamp\n"
    timestamp = datetime.now().isoformat()
    with open(output_csv, "w") as f:
        f.write(header)
        for model_name, strategies in results.items():
            dims = MODELS[model_name]["dims"]
            for strategy_name, res in strategies.items():
                if res is None:
                    f.write(f"{model_name},{dims},{strategy_name},OOM,OOM,OOM,{timestamp}\n")
                else:
                    f.write(f"{model_name},{dims},{strategy_name},"
                            f"{res['mean']:.6f},{res['std']:.6f},{res['mem_gib']:.2f},"
                            f"{timestamp}\n")
    print_msg(f"Results written to: {output_csv}")


def print_summary_table(results):
    """Print formatted summary matching Table 2 layout."""
    col_w = 14
    print_msg(f"\n{'=' * 100}")
    print_msg("Table 2: Per-gradient time (s) and peak GPU memory (GiB)")
    print_msg(f"{'=' * 100}")

    header = f"{'Model':<8} {'n x b':<14}"
    for s in STRATEGIES:
        header += f" {s + ' t':>{col_w}} {s + ' mem':>{col_w}}"
    print_msg(header)
    print_msg("-" * len(header))

    for model_name, strategies in results.items():
        dims = MODELS[model_name]["dims"].replace(r"\times", "x").replace(" ", "")
        row = f"{model_name:<8} {dims:<14}"
        for strategy_name in STRATEGIES:
            res = strategies.get(strategy_name)
            t_str, mem_str = format_cell(res)
            row += f" {t_str:>{col_w}} {mem_str:>{col_w}}"
        print_msg(row)

    print_msg(f"{'=' * 100}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_msg("--- Table 2: Structure-Preserving vs Generic Differentiation ---")
    n_runs = args.n_benchmark_runs

    results = OrderedDict()

    for model_name, model_info in MODELS.items():
        results[model_name] = OrderedDict()
        print_msg(f"\n{'=' * 60}")
        print_msg(f"Model: {model_name} ({model_info['dims']})")
        print_msg(f"{'=' * 60}")

        for strategy in STRATEGIES:
            print(f"  {strategy}... ", end="", flush=True)
            res = run_strategy(model_name, model_info, strategy, n_runs)
            results[model_name][strategy] = res

            if res is None:
                print("OOM", flush=True)
            else:
                print(f"{res['mean']:.4f}s (+/- {res['std']:.4f}), "
                      f"{res['mem_gib']:.1f} GiB", flush=True)

    print_summary_table(results)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(__file__), "tab2_results.csv")
    write_csv(results, output_csv)

    print_msg("--- Finished ---")
