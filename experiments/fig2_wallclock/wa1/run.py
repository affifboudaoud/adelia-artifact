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
from dalia.configs import likelihood_config, models_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.models import CoregionalModel
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import print_msg
from jax_utils import get_first_forward_and_gradient
from benchmark_utils import run_benchmark, write_optimization_result
from energy_monitor import EnergyMonitor

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
DATA_DIR = os.path.join(ARTIFACT_DIR, "data", "wa1")

NV = 3
NS = 1247
NT = 512
NB = 3
DIM_THETA = 15


def _load_theta_ref():
    input_dir = f"{DATA_DIR}/inputs_nv{NV}_ns{NS}_nt{NT}_nb{NB}"
    theta_ref_file = (
        f"{input_dir}/reference_outputs"
        f"/theta_interpretS_original_DALIA_perm_{DIM_THETA}_1.dat"
    )
    if os.path.exists(theta_ref_file):
        return np.loadtxt(theta_ref_file)
    return np.array([
        1.0, 0.5, 8.0,
        1.0, 0.5, 8.0,
        1.0, 0.5, 8.0,
        0.0, 1.0, -0.5,
        -1.0, 1.0, 0.5,
    ])


def create_model():
    input_dir = f"{DATA_DIR}/inputs_nv{NV}_ns{NS}_nt{NT}_nb{NB}"
    theta_ref = _load_theta_ref()
    np.random.seed(6)
    theta_initial = theta_ref + 0.1 * np.random.randn(DIM_THETA)

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
            "prior_hyperparameters": {
                "type": "gaussian", "mean": theta_initial[2], "precision": 0.5,
            },
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
            "prior_hyperparameters": {
                "type": "gaussian", "mean": theta_ref[5], "precision": 0.5,
            },
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
            "prior_hyperparameters": {
                "type": "gaussian", "mean": theta_ref[8], "precision": 0.5,
            },
        }),
    )

    coreg_model = CoregionalModel(
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
    return coreg_model


def run_with_method(model, gradient_method, max_iter, verbose=True):
    solver_cfg = {"type": "serinv"}
    if args.solver_min_p > 1:
        solver_cfg["min_processes"] = args.solver_min_p
    dalia_cfg = {
        "solver": solver_cfg,
        "gradient_method": gradient_method,
        "minimize": {
            "max_iter": max_iter, "gtol": 1e-3,
            "disp": verbose, "maxcor": len(model.theta),
        },
        "f_reduction_tol": 1e-3,
        "f_reduction_lag": 10,
        "theta_reduction_lag": 10,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_cfg),
    )
    from dalia import comm_rank
    dalia.energy_monitor = EnergyMonitor(rank=comm_rank)
    return dalia


if __name__ == "__main__":
    print_msg("--- wa1: Coregional (3-variate) Spatio-Temporal Model (distributed, 4 nodes) ---")

    model = create_model()

    if args.benchmark_mode:
        dalia_fd = dalia_jax = None
        if args.benchmark_method in ["finite_diff", "both"]:
            dalia_fd = run_with_method(create_model(), "finite_diff", args.max_iter, verbose=False)
        if args.benchmark_method in ["jax_autodiff", "both"]:
            dalia_jax = run_with_method(create_model(), "jax_autodiff", args.max_iter, verbose=False)
        run_benchmark(dalia_fd, dalia_jax, args, "wa1")
        exit(0)

    if args.framework_comparison:
        from benchmark_utils import run_framework_comparison
        ad_only = getattr(args, "framework_ad_only", False)
        fd_only = getattr(args, "framework_fd_only", False)
        dalia_jax = None if fd_only else run_with_method(create_model(), "jax_autodiff", 1, verbose=False)
        dalia_fd = None if ad_only else run_with_method(create_model(), "finite_diff", 1, verbose=False)
        run_framework_comparison(dalia_jax, dalia_fd, args, "wa1")
        exit(0)

    if args.breakdown:
        from benchmark_utils import run_breakdown_benchmark
        dalia_jax = run_with_method(create_model(), "jax_autodiff", 1, verbose=False)
        run_breakdown_benchmark(dalia_jax, args, "WA1")
        exit(0)

    t_total_jax = t_total_fd = None
    n_nodes = int(os.environ.get("SLURM_NNODES", 1))

    if not args.skip_jax:
        model_jax = create_model()
        dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)
        n_hp = model_jax.n_hyperparameters

        t0 = time.perf_counter()
        f_jax, grad_jax = get_first_forward_and_gradient(dalia_jax)
        t_first_jax = time.perf_counter() - t0

        print_msg(f"First forward+gradient (AD): f={f_jax:.6f}, time={t_first_jax:.4f}s")

        t0 = time.perf_counter()
        results_jax = dalia_jax.run()
        t_total_jax = time.perf_counter() - t0

        print_msg(f"JAX optimization: {t_total_jax:.2f}s, f={results_jax['f']:.6f}")
        write_optimization_result(
            args.results_csv, "wa1", "jax_autodiff", results_jax,
            n_nodes=n_nodes, n_hyperparams=n_hp, jit_time=t_first_jax,
        )

        del dalia_jax, model_jax
        import gc; gc.collect()

    if not args.skip_fd:
        model_fd = create_model()
        dalia_fd = run_with_method(model_fd, "finite_diff", args.max_iter, verbose=False)

        t0 = time.perf_counter()
        f_fd, grad_fd = get_first_forward_and_gradient(dalia_fd)
        t_first_fd = time.perf_counter() - t0

        t0 = time.perf_counter()
        results_fd = dalia_fd.run()
        t_total_fd = time.perf_counter() - t0

        print_msg(f"FD optimization: {t_total_fd:.2f}s, f={results_fd['f']:.6f}")
        write_optimization_result(
            args.results_csv, "wa1", "finite_diff", results_fd,
            n_nodes=n_nodes, n_hyperparams=model_fd.n_hyperparameters,
        )

    if t_total_jax and t_total_fd:
        print_msg(f"Speedup: {t_total_fd / t_total_jax:.1f}x")

    print_msg("--- Finished ---")
