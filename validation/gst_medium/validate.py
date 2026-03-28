import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--precision", default="float64", choices=["float32", "float64"])
args = parser.parse_args()

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()

from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision(args.precision)

import numpy as np

from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(ARTIFACT_DIR, "data", "gst_medium")


def create_model():
    st = SpatioTemporalSubModel(config=submodels_config.parse_config({
        "type": "spatio_temporal",
        "input_dir": f"{DATA_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0, "r_t": 0, "sigma_st": 0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": 0.03972077083991806, "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": 2.3931471805599456, "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 1.4379142862353824, "precision": 0.5},
    }))
    reg = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": f"{DATA_DIR}/inputs_regression",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }))
    model = Model(
        submodels=[reg, st],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian", "prec_o": 4,
            "prior_hyperparameters": {
                "type": "gaussian", "mean": 1.4, "precision": 0.5,
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
    from validate_core import run_validation
    run_validation(
        model_name="gst_medium",
        create_model_fn=create_model,
        run_with_method_fn=run_with_method,
        include_ad_loop=False,
        precision=args.precision,
    )
