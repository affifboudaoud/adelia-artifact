"""Model setup for gst_coreg3_small_p2 (shared by validate.py and converge.py)."""

import os
import numpy as np

from dalia.configs import likelihood_config, models_config, submodels_config
from dalia.core.model import Model
from dalia.models import CoregionalModel
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(ARTIFACT_DIR, "data", "gst_coreg3_small")

NV = 3
NS = 354
NT = 8
NB = 3

PERTURBATION = [
    0.18197867, -0.12551227, 0.19998896,
    0.17226796, 0.14656176, -0.11864931,
    0.17817371, -0.13006157, 0.19308036,
    0.12317955, -0.14182536, 0.15686513,
    0.17168868, -0.0365025, 0.13315897,
]


def load_theta_ref():
    input_dir = f"{DATA_DIR}/inputs_nv{NV}_ns{NS}_nt{NT}_nb{NB}"
    return np.load(f"{input_dir}/reference_outputs/theta_ref.npy")


def create_model():
    input_dir = f"{DATA_DIR}/inputs_nv{NV}_ns{NS}_nt{NT}_nb{NB}"
    theta_ref = load_theta_ref()
    theta_initial = theta_ref + np.array(PERTURBATION)

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
