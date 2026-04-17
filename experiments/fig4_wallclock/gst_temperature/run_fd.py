import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))

from parser_utils import parse_args

args = parse_args()

os.environ["ARRAY_MODULE"] = "cupy"

import numpy as np
from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import print_msg
from benchmark_utils import write_optimization_result

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


if __name__ == "__main__":
    print_msg("--- gst_temperature FD: Gaussian Spatio-Temporal Model ---")

    model_fd = create_model()
    dalia_fd = DALIA(
        model=model_fd,
        config=dalia_config.parse_config({
            "solver": {"type": "serinv"},
            "gradient_method": "finite_diff",
            "minimize": {
                "max_iter": args.max_iter, "gtol": 1e-3,
                "disp": True, "maxcor": len(model_fd.theta),
            },
            "inner_iteration_max_iter": 50,
            "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3,
            "simulation_dir": ".",
        }),
    )

    t0 = time.perf_counter()
    results_fd = dalia_fd.run()
    t_total_fd = time.perf_counter() - t0

    print_msg(f"FD optimization: {t_total_fd:.2f}s, f={results_fd['f']:.6f}")
    print_msg(f"  theta={results_fd['theta']}")
    write_optimization_result(
        args.results_csv, "gst_temperature", "finite_diff", results_fd,
        n_nodes=1, n_hyperparams=model_fd.n_hyperparameters,
    )

    print_msg("--- Finished ---")
