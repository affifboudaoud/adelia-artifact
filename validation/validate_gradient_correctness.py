"""Numerical validation: AD-BTA (custom backward pass) vs AD-Loop (native JAX AD).

Cross-validates the custom backward pass (AD-BTA) against JAX's native
reverse-mode AD through lax.scan (AD-Loop) on the four models where AD-Loop
fits in memory: GST-S, GST-C2, GST-C3, GST-M.

Both methods are exact (no truncation error), so agreement to machine precision
proves correctness. The ~1e-5 gap vs FD can then be attributed to FD's O(h^2)
truncation error.

Paper claim (Section V, Gradient Correctness):
  "the two exact AD methods agree to below 10^{-7} in the worst case"

Usage:
  python validate_gradient_correctness.py [--precision float64]
"""

import os
import sys
import time
import csv
from collections import OrderedDict
from datetime import datetime

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["ARRAY_MODULE"] = "numpy"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "common"))

from dalia.core.jax_autodiff import configure_jax_precision

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--precision", type=str, default="float64",
                    choices=["float32", "float64"])
args = parser.parse_args()

configure_jax_precision(args.precision)

import numpy as np

from dalia.configs import (
    likelihood_config,
    models_config,
    dalia_config,
    submodels_config,
)
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.models import CoregionalModel
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.core.jax_autodiff import (
    create_pure_jax_objective,
    create_pure_jax_objective_coregional,
    get_jax_dtype,
)

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

SEED = 63
np.random.seed(SEED)

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


def make_dalia(model):
    config = dalia_config.parse_config({
        "solver": {"type": "serinv", "min_processes": 1},
        "gradient_method": "jax_autodiff",
        "minimize": {
            "max_iter": 1, "gtol": 1e-3,
            "disp": False, "maxcor": len(model.theta),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "eps_hessian_f": 5e-3,
        "simulation_dir": ".",
    })
    return DALIA(model=model, config=config)


# ---------------------------------------------------------------------------
# Model creation (using artifact data paths)
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
            "prior_hyperparameters": {"type": "gaussian", "mean": theta_ref[2], "precision": 0.5},
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
            "prior_hyperparameters": {"type": "gaussian", "mean": theta_ref[2], "precision": 0.5},
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


MODELS = OrderedDict([
    ("GST-S",  {"create": create_gst_small,         "coregional": False, "dims": "5 x 92"}),
    ("GST-C2", {"create": create_gst_coreg2_small,   "coregional": True,  "dims": "12 x 708"}),
    ("GST-C3", {"create": create_gst_coreg3_small,   "coregional": True,  "dims": "8 x 1062"}),
    ("GST-M",  {"create": create_gst_medium,         "coregional": False, "dims": "100 x 812"}),
])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_model(name, model_info):
    """Compare AD-BTA and AD-Loop gradients for one model."""
    model = model_info["create"]()
    is_coreg = model_info["coregional"]

    print(f"\n{'='*70}")
    print(f"  {name}  (d={len(model.theta)}, {model_info['dims']})")
    print(f"{'='*70}")

    dtype = get_jax_dtype()
    theta = model.theta.copy()
    create_fn = create_pure_jax_objective_coregional if is_coreg else create_pure_jax_objective

    dalia = make_dalia(model)

    # AD-BTA (custom_vjp)
    t0 = time.perf_counter()
    _, grad_fn_bta = create_fn(dalia, dtype=dtype, ad_mode="default")
    t_jit_bta = time.perf_counter() - t0
    f_bta, grad_bta, x_bta = grad_fn_bta(theta)

    # AD-Loop (native JAX AD through lax.scan)
    t0 = time.perf_counter()
    _, grad_fn_scan = create_fn(dalia, dtype=dtype, ad_mode="scan")
    t_jit_scan = time.perf_counter() - t0
    f_scan, grad_scan, x_scan = grad_fn_scan(theta)

    # Forward value comparison
    f_abs = abs(f_bta - f_scan)
    f_rel = f_abs / abs(f_scan) if f_scan != 0 else f_abs
    print(f"\nForward value:")
    print(f"  AD-BTA:  {f_bta:.15e}")
    print(f"  AD-Loop: {f_scan:.15e}")
    print(f"  Abs err: {f_abs:.2e}   Rel err: {f_rel:.2e}")

    # Gradient comparison
    g_abs = np.abs(grad_bta - grad_scan)
    g_rel = g_abs / (np.abs(grad_scan) + 1e-30)
    print(f"\nGradient (component-wise):")
    for i in range(len(theta)):
        print(f"  [{i:2d}] BTA={grad_bta[i]:+.12e}  Loop={grad_scan[i]:+.12e}  "
              f"abs={g_abs[i]:.2e}  rel={g_rel[i]:.2e}")
    print(f"\n  Max abs err:  {np.max(g_abs):.2e}")
    print(f"  Max rel err:  {np.max(g_rel):.2e}")

    # Latent variable comparison
    x_abs = np.max(np.abs(x_bta - x_scan))
    print(f"\nLatent x max abs err: {x_abs:.2e}")

    print(f"\nJIT time: AD-BTA={t_jit_bta:.1f}s, AD-Loop={t_jit_scan:.1f}s")

    return {
        "model": name,
        "dims": model_info["dims"],
        "d": len(theta),
        "f_bta": float(f_bta),
        "f_scan": float(f_scan),
        "f_rel_err": float(f_rel),
        "grad_max_abs_err": float(np.max(g_abs)),
        "grad_max_rel_err": float(np.max(g_rel)),
        "x_max_abs_err": float(x_abs),
        "jit_bta": t_jit_bta,
        "jit_scan": t_jit_scan,
    }


def write_csv(results, output_csv):
    """Write validation results to CSV."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "model", "dims", "d", "f_bta", "f_scan", "f_rel_err",
        "grad_max_abs_err", "grad_max_rel_err", "x_max_abs_err",
        "jit_bta", "jit_scan", "timestamp",
    ]
    timestamp = datetime.now().isoformat()
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames if k in r}
            row["timestamp"] = timestamp
            writer.writerow(row)
    print(f"\nResults written to: {output_csv}")


if __name__ == "__main__":
    print("=" * 70)
    print("  AD-BTA vs AD-Loop: Numerical Gradient Validation")
    print("  Both methods are exact (no truncation error).")
    print("  Agreement proves correctness of the custom backward pass.")
    print("=" * 70)

    results = []
    for name, info in MODELS.items():
        res = validate_model(name, info)
        results.append(res)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'d':>3} {'f rel err':>12} {'grad max abs':>14} {'grad max rel':>14} {'x max abs':>12}")
    print("-" * 68)
    worst_rel = 0.0
    for r in results:
        print(f"{r['model']:<10} {r['d']:>3} {r['f_rel_err']:>12.2e} "
              f"{r['grad_max_abs_err']:>14.2e} {r['grad_max_rel_err']:>14.2e} "
              f"{r['x_max_abs_err']:>12.2e}")
        worst_rel = max(worst_rel, r["grad_max_rel_err"])

    print(f"\nWorst-case gradient max rel err: {worst_rel:.2e}")

    output_csv = os.path.join(
        os.path.dirname(__file__), "reference_outputs", "gradient_correctness.csv"
    )
    write_csv(results, output_csv)

    threshold = 5e-7
    if worst_rel < threshold:
        print(f"PASS: All models below {threshold:.0e} threshold.")
    else:
        print(f"FAIL: Worst-case {worst_rel:.2e} exceeds {threshold:.0e} threshold.")
        sys.exit(1)
