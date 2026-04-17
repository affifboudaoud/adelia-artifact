"""Framework decomposition for gst_temperature.

Runs JAX and CuPy evaluations in separate phases to avoid OOM
from having both JAX buffers and serinv solver on GPU simultaneously.
"""
import sys
import os
import time
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))

from parser_utils import parse_args
args = parse_args()

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()

from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision(args.precision)

import numpy as np
from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import print_msg

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
DATA_DIR = os.path.join(ARTIFACT_DIR, "data", "gst_temperature")

n_runs = args.n_benchmark_runs
output_dir = args.output_dir


def create_model():
    st = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{DATA_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": -0.960279229160082, "r_t": -0.3068528194400548, "sigma_st": -2.112085713764618,
        "manifold": "plane",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
    }))
    reg = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{DATA_DIR}/inputs_regression",
        "n_fixed_effects": 4, "fixed_effects_prior_precision": 0.001,
    }))
    return Model(
        submodels=[reg, st],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": 4,
            "prior_hyperparameters": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        }),
    )


if __name__ == "__main__":
    print_msg("=== Framework decomposition: gst_temperature (split phases) ===")

    # --- Phase 1: JAX forward and forward+backward ---
    print_msg("\n--- Phase 1: JAX measurements ---")
    model_jax = create_model()
    dalia_jax = DALIA(
        model=model_jax,
        config=dalia_config.parse_config({
            "solver": {"type": "serinv"},
            "gradient_method": "jax_autodiff",
            "minimize": {"max_iter": 1, "gtol": 1e-3, "disp": False, "maxcor": 4},
            "f_reduction_tol": 1e-3, "theta_reduction_tol": 1e-4,
            "inner_iteration_max_iter": 50, "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3, "simulation_dir": ".",
        }),
    )

    theta_jax = np.asarray(
        dalia_jax.model.theta.get() if hasattr(dalia_jax.model.theta, "get")
        else dalia_jax.model.theta
    )

    # Warmup
    for _ in range(2):
        dalia_jax.jax_objective(theta_jax)
    for _ in range(2):
        dalia_jax.jax_grad_func(theta_jax)

    # JAX forward only
    jax_fwd_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        dalia_jax.jax_objective(theta_jax)
        jax_fwd_times.append(time.perf_counter() - t0)

    # JAX forward + backward
    jax_grad_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        dalia_jax.jax_grad_func(theta_jax)
        jax_grad_times.append(time.perf_counter() - t0)

    jax_fwd_mean = np.mean(jax_fwd_times)
    jax_fwd_std = np.std(jax_fwd_times)
    jax_grad_mean = np.mean(jax_grad_times)
    jax_grad_std = np.std(jax_grad_times)
    jax_bwd_mean = jax_grad_mean - jax_fwd_mean

    print_msg(f"  JAX forward:     {jax_fwd_mean:.6f} +/- {jax_fwd_std:.6f}s")
    print_msg(f"  JAX fwd+bwd:     {jax_grad_mean:.6f} +/- {jax_grad_std:.6f}s")
    print_msg(f"  JAX backward:    {jax_bwd_mean:.6f}s")

    # Clean up JAX
    del dalia_jax, model_jax
    gc.collect()
    jax.clear_caches()
    gc.collect()

    # --- Phase 2: CuPy/Serinv forward ---
    print_msg("\n--- Phase 2: CuPy/Serinv measurements ---")
    model_fd = create_model()
    dalia_fd = DALIA(
        model=model_fd,
        config=dalia_config.parse_config({
            "solver": {"type": "serinv"},
            "gradient_method": "finite_diff",
            "minimize": {"max_iter": 1, "gtol": 1e-3, "disp": False, "maxcor": 4},
            "f_reduction_tol": 1e-3, "theta_reduction_tol": 1e-4,
            "inner_iteration_max_iter": 50, "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3, "simulation_dir": ".",
        }),
    )

    theta_fd = dalia_fd.model.theta.copy()
    dalia_fd.iter = 0

    # Warmup
    for _ in range(2):
        dalia_fd._evaluate_f(theta_fd)

    cupy_fwd_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        dalia_fd._evaluate_f(theta_fd)
        cupy_fwd_times.append(time.perf_counter() - t0)

    cupy_fwd_mean = np.mean(cupy_fwd_times)
    cupy_fwd_std = np.std(cupy_fwd_times)
    ratio = cupy_fwd_mean / jax_fwd_mean if jax_fwd_mean > 0 else 0.0

    print_msg(f"  CuPy forward:    {cupy_fwd_mean:.6f} +/- {cupy_fwd_std:.6f}s")
    print_msg(f"  Ratio (CuPy/JAX): {ratio:.2f}")

    # --- Write results ---
    n_hp = 4
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "single_eval_comparison.csv")
        needs_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if needs_header:
                f.write("model,n_hyperparams,jax_fwd_time_mean,jax_fwd_time_std,"
                        "cupy_eval_time_mean,cupy_eval_time_std,ratio_cupy_over_jax,n_runs\n")
            f.write(f"gst_temperature,{n_hp},{jax_fwd_mean:.6f},{jax_fwd_std:.6f},"
                    f"{cupy_fwd_mean:.6f},{cupy_fwd_std:.6f},{ratio:.2f},{n_runs}\n")
        print_msg(f"Results appended to {csv_path}")

    print_msg("\n--- Summary ---")
    print_msg(f"  JAX fwd: {jax_fwd_mean:.4f}s, CuPy fwd: {cupy_fwd_mean:.4f}s, "
              f"r={ratio:.2f}, beta={jax_bwd_mean/jax_fwd_mean:.2f}")
    print_msg("--- Finished ---")
