import sys
import os
import time

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

from dalia import xp, backend_flags
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import print_msg
from jax_utils import get_first_forward_and_gradient
from benchmark_utils import run_benchmark, write_optimization_result

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
DATA_DIR = os.path.join(ARTIFACT_DIR, "data", "gst_temperature")


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
        "n_fixed_effects": 4,
        "fixed_effects_prior_precision": 0.001,
    }))
    model = Model(
        submodels=[reg, st],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": 4,
            "prior_hyperparameters": {
                "type": "penalized_complexity", "alpha": 0.01, "u": 5,
            },
        }),
    )
    return model


def run_with_method(model, gradient_method, max_iter, verbose=True):
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
            "solver": {"type": "serinv"},
            "gradient_method": gradient_method,
            "minimize": {
                "max_iter": max_iter, "gtol": 1e-3,
                "disp": verbose, "maxcor": len(model.theta),
            },
            "f_reduction_tol": 1e-3,
            "theta_reduction_tol": 1e-4,
            "inner_iteration_max_iter": 50,
            "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3,
            "simulation_dir": ".",
        }),
    )
    return dalia


if __name__ == "__main__":
    print_msg("--- gst_temperature: Gaussian Spatio-Temporal Model ---")

    model = create_model()

    if args.benchmark_mode:
        dalia_fd = dalia_jax = None
        if args.benchmark_method in ["finite_diff", "both"]:
            dalia_fd = run_with_method(create_model(), "finite_diff", args.max_iter, verbose=False)
        if args.benchmark_method in ["jax_autodiff", "both"]:
            dalia_jax = run_with_method(create_model(), "jax_autodiff", args.max_iter, verbose=False)
        run_benchmark(dalia_fd, dalia_jax, args, "gst_temperature")
        exit(0)

    if args.framework_comparison:
        from benchmark_utils import run_framework_comparison
        dalia_jax = run_with_method(create_model(), "jax_autodiff", 1, verbose=False)
        dalia_fd = run_with_method(create_model(), "finite_diff", 1, verbose=False)
        run_framework_comparison(dalia_jax, dalia_fd, args, "gst_temperature")
        exit(0)

    model_jax = create_model()
    dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)

    t0 = time.perf_counter()
    f_jax, grad_jax = get_first_forward_and_gradient(dalia_jax)
    t_first_jax = time.perf_counter() - t0

    print_msg(f"First forward+gradient: f={f_jax:.6f}, time={t_first_jax:.4f}s")

    t0 = time.perf_counter()
    results_jax = dalia_jax.run()
    t_total_jax = time.perf_counter() - t0

    print_msg(f"JAX optimization: {t_total_jax:.2f}s, f={results_jax['f']:.6f}")
    write_optimization_result(
        args.results_csv, "gst_temperature", "jax_autodiff", results_jax,
        n_nodes=1, n_hyperparams=model_jax.n_hyperparameters, jit_time=t_first_jax,
    )

    if not args.skip_fd:
        del dalia_jax, model_jax
        import gc; gc.collect()

        model_fd = create_model()
        dalia_fd = run_with_method(model_fd, "finite_diff", args.max_iter, verbose=False)

        t0 = time.perf_counter()
        f_fd, grad_fd = get_first_forward_and_gradient(dalia_fd)
        t_first_fd = time.perf_counter() - t0

        t0 = time.perf_counter()
        results_fd = dalia_fd.run()
        t_total_fd = time.perf_counter() - t0

        print_msg(f"FD optimization: {t_total_fd:.2f}s, f={results_fd['f']:.6f}")
        print_msg(f"Speedup: {t_total_fd / t_total_jax:.1f}x")
        write_optimization_result(
            args.results_csv, "gst_temperature", "finite_diff", results_fd,
            n_nodes=1, n_hyperparams=model_fd.n_hyperparameters,
        )

    if args.output_dir:
        from results_utils import collect_optimization_results, write_results_json, append_speedup_csv, get_gpu_name
        gpu_name = get_gpu_name()
        jax_data = collect_optimization_results(dalia_jax if 'dalia_jax' in dir() else dalia_fd, "gst_temperature", "jax_autodiff", t_total_jax, jit_time=t_first_jax)
        write_results_json(jax_data, args.output_dir, "gst_temperature", "jax_autodiff")

    print_msg("--- Finished ---")
