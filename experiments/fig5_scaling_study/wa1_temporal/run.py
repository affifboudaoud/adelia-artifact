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
from dalia.configs import (
    likelihood_config,
    models_config,
    dalia_config,
    submodels_config,
)

print(f"Array module: {xp.__name__}, Backend flags: {backend_flags}")
print(f"JAX devices: {jax.devices()}")

from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.models import CoregionalModel
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import print_msg
from jax_utils import get_first_forward_and_gradient
from benchmark_utils import run_benchmark, write_scaling_csv_row, cleanup_gpu_memory

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)
DATA_DIR = os.path.join(ARTIFACT_DIR, "data", "scaling_wa1_temporal")

NV = 3
NS = 1247
NT = args.nt if args.nt is not None else 2
NB = 3
DIM_THETA = 15
INPUT_DIR = f"{DATA_DIR}/inputs_nv{NV}_ns{NS}_nt{NT}_nb{NB}"


def create_model():
    """Create the trivariate coregional model for WA1 temporal scaling."""
    theta_ref_file = (
        f"{INPUT_DIR}/reference_outputs/"
        f"theta_interpretS_original_DALIA_perm_{DIM_THETA}_1.dat"
    )
    if os.path.exists(theta_ref_file):
        theta_ref = np.loadtxt(theta_ref_file)
    else:
        theta_ref = np.array([
            1.0, 0.5, 8.0,
            1.0, 0.5, 8.0,
            1.0, 0.5, 8.0,
            0.0, 1.0, -0.5,
            -1.0, 1.0, 0.5,
        ])
    np.random.seed(6)
    theta_initial = theta_ref + 0.1 * np.random.randn(DIM_THETA)

    st1 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{INPUT_DIR}/model_1/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0], "r_t": theta_initial[1], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[0], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[1], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    r1 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{INPUT_DIR}/model_1/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model1 = Model(
        submodels=[r1, st1],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[2],
            "prior_hyperparameters": {
                "type": "gaussian", "mean": theta_initial[2], "precision": 0.5,
            },
        }),
    )

    st2 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{INPUT_DIR}/model_2/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[3], "r_t": theta_initial[4], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[3], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[4], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    r2 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{INPUT_DIR}/model_2/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model2 = Model(
        submodels=[st2, r2],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[5],
            "prior_hyperparameters": {
                "type": "gaussian", "mean": theta_ref[5], "precision": 0.5,
            },
        }),
    )

    st3 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{INPUT_DIR}/model_3/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[6], "r_t": theta_initial[7], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": theta_ref[6], "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": theta_ref[7], "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }))
    r3 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{INPUT_DIR}/model_3/inputs_regression",
        "n_fixed_effects": 1, "fixed_effects_prior_precision": 0.001,
    }))
    model3 = Model(
        submodels=[st3, r3],
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
            "sigmas": list(theta_initial[9:12]),
            "lambdas": list(theta_initial[12:15]),
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
    return coreg_model, theta_ref


def run_with_method(model, gradient_method, max_iter, verbose=True,
                    solver_min_p=1):
    """Set up DALIA with the specified gradient method."""
    dalia_dict = {
        "solver": {"type": "serinv", "min_processes": solver_min_p},
        "gradient_method": gradient_method,
        "minimize": {
            "max_iter": max_iter,
            "gtol": 1e-3,
            "disp": verbose,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "eps_hessian_f": 5e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )
    return dalia


if __name__ == "__main__":
    N_latent = NV * NS * NT + NB
    print_msg("--- fig5: WA1 Temporal Scaling ---")
    print_msg(f"--- WA1: Trivariate Coregional (ns={NS}, nt={NT}, N={N_latent:,}) ---\n")

    model, theta_ref = create_model()
    print_msg(model)

    if args.benchmark_mode:
        model_label = "WA1-small" if NT == 2 else f"WA1-nt{NT}"
        fd_results = None
        jax_results = None

        if args.benchmark_method in ["finite_diff", "both"]:
            model_fd, _ = create_model()
            dalia_fd = run_with_method(model_fd, "finite_diff", args.max_iter, verbose=False,
                                       solver_min_p=args.solver_min_p)
            fd_results, _ = run_benchmark(dalia_fd, None, args, model_label)
            del dalia_fd, model_fd
            cleanup_gpu_memory()

        if args.benchmark_method in ["jax_autodiff", "both"]:
            model_jax, _ = create_model()
            dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)
            _, jax_results = run_benchmark(None, dalia_jax, args, model_label)

        results_csv = os.path.join(
            os.path.dirname(__file__), "..", "results", "scaling_results.csv"
        )
        model_info = {
            "likelihood": "gaussian", "nv": NV, "ns": NS, "nt": NT,
            "latent_dim": N_latent, "n_hyperparams": DIM_THETA,
            "block_size": NV * NS, "n_blocks": NT, "solver": "BTA",
        }
        n_nodes = int(os.environ.get("SLURM_NNODES", 1))
        write_scaling_csv_row(
            results_csv, model_label, model_info,
            jax_results, fd_results, args.n_benchmark_runs, n_nodes,
        )
        exit(0)

    model_jax, _ = create_model()
    dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)

    t0 = time.perf_counter()
    f_jax, grad_jax = get_first_forward_and_gradient(dalia_jax)
    t_first_jax = time.perf_counter() - t0

    print_msg(f"First forward+gradient: f={f_jax:.6f}, time={t_first_jax:.4f}s")

    t0 = time.perf_counter()
    results_jax = dalia_jax.run()
    t_total_jax = time.perf_counter() - t0

    print_msg(f"JAX optimization: {t_total_jax:.2f}s, f={results_jax['f']:.6f}")
    print_msg("--- Finished ---")
