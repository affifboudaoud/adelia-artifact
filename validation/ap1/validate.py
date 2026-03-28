import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--precision", default="float64", choices=["float32", "float64"])
parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument("--solver_min_p", type=int, default=1)
parser.add_argument("--jax_fd", action="store_true")
parser.add_argument("--fd_eps", type=float, default=1e-3)
args = parser.parse_args()

if args.phase != 1:
    import jax
    import jax.numpy as jnp
    if jax.default_backend() == "gpu":
        jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
    from dalia.core.jax_autodiff import configure_jax_precision
    configure_jax_precision(args.precision)

import numpy as np

from dalia.configs import likelihood_config, models_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.models import CoregionalModel
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(ARTIFACT_DIR, "data", "ap1")

NV = 3
NS = 4210
NT = 48
NB = 6
DIM_THETA = 15

THETA_INITIAL = [
    0.362219, 5.099211, 2.418104,
    0.142032, 3.762768, 2.953950,
    0.299849, 5.076667, 3.128995,
    2.536107, 1.119993, 2.445060,
    1.123385, -0.650153, 0.012496,
]


def create_model():
    input_dir = f"{DATA_DIR}/inputs_nv{NV}_ns{NS}_nt{NT}_nb{NB}"
    theta_initial = np.array(THETA_INITIAL)

    st1 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_1/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0], "r_t": theta_initial[1], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.99, "u": 3},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.99, "u": 6},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
    }))
    reg1 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_1/inputs_regression",
        "n_fixed_effects": 2, "fixed_effects_prior_precision": 0.001,
    }))
    model1 = Model(
        submodels=[reg1, st1],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[2],
            "prior_hyperparameters": {
                "type": "penalized_complexity", "alpha": 0.01, "u": 1,
            },
        }),
    )

    st2 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_2/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[3], "r_t": theta_initial[4], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.99, "u": 3},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.99, "u": 6},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
    }))
    reg2 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_2/inputs_regression",
        "n_fixed_effects": 2, "fixed_effects_prior_precision": 0.001,
    }))
    model2 = Model(
        submodels=[st2, reg2],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[5],
            "prior_hyperparameters": {
                "type": "penalized_complexity", "alpha": 0.01, "u": 1,
            },
        }),
    )

    st3 = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{input_dir}/model_3/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[6], "r_t": theta_initial[7], "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.99, "u": 3},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.99, "u": 6},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
    }))
    reg3 = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{input_dir}/model_3/inputs_regression",
        "n_fixed_effects": 2, "fixed_effects_prior_precision": 0.001,
    }))
    model3 = Model(
        submodels=[st3, reg3],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": theta_initial[8],
            "prior_hyperparameters": {
                "type": "penalized_complexity", "alpha": 0.01, "u": 1,
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
                {"type": "gaussian", "mean": 1, "precision": 0.5},
                {"type": "gaussian", "mean": 1, "precision": 0.5},
                {"type": "gaussian", "mean": 1, "precision": 0.5},
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
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
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
        }),
    )
    return dalia


if __name__ == "__main__":
    from validate_core import run_validation
    run_validation(
        model_name="ap1",
        create_model_fn=create_model,
        run_with_method_fn=run_with_method,
        include_ad_loop=False,
        precision=args.precision,
        phase=args.phase,
        jax_fd=args.jax_fd,
        fd_eps=args.fd_eps,
    )
