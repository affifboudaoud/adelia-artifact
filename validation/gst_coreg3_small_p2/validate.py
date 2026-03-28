import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--precision", default="float64", choices=["float32", "float64"])
parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                    help="0=both, 1=FD reference, 2=JAX AD, 3=JAX FD+AD")
parser.add_argument("--solver_min_p", type=int, default=1)
parser.add_argument("--jax_fd", action="store_true",
                    help="Compute FD through jax_objective instead of CuPy")
parser.add_argument("--fd_eps", type=float, default=1e-3,
                    help="Finite difference step size")
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


def create_model():
    input_dir = f"{DATA_DIR}/inputs_nv{NV}_ns{NS}_nt{NT}_nb{NB}"
    theta_ref = np.load(f"{input_dir}/reference_outputs/theta_ref.npy")
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
            "theta_reduction_tol": 1e-4,
            "inner_iteration_max_iter": 50,
            "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3,
            "simulation_dir": ".",
        }),
    )
    return dalia


def compare_with_native(results):
    """Compare P=2 AD results against verified single-node AD reference."""
    ref_path = os.path.join(
        os.path.dirname(__file__), "..", "gst_coreg3_small", "outputs",
        "validation_gst_coreg3_small.npz",
    )
    if not os.path.exists(ref_path):
        from dalia.utils import print_msg
        print_msg(f"\nSkipping native comparison: {ref_path} not found")
        return

    from dalia.utils import print_msg
    ref = np.load(ref_path)
    f_ref = float(ref["f_ad"])
    g_ref = ref["grad_ad"]
    f_p2 = results["f_ad"]
    g_p2 = results["grad_ad"]
    n_hp = len(g_ref)

    print_msg("\n" + "-" * 70)
    print_msg("COMPARISON: P=2 distributed AD vs P=1 native AD (verified reference)")
    print_msg("-" * 70)
    f_rtol = abs(f_p2 - f_ref) / max(abs(f_ref), 1e-30)
    print_msg(f"  f (P=1 native): {f_ref:.15e}")
    print_msg(f"  f (P=2 dist):   {f_p2:.15e}")
    print_msg(f"  f relative err: {f_rtol:.6e}")

    print_msg(f"\n  {'i':<6} {'P=1 native':>15} {'P=2 dist':>15} {'abs err':>12} {'rel err':>12}")
    max_rtol = 0.0
    for i in range(n_hp):
        ae = abs(g_ref[i] - g_p2[i])
        denom = max(abs(g_ref[i]), abs(g_p2[i]), 1e-30)
        re = ae / denom
        max_rtol = max(max_rtol, re)
        print_msg(f"  {i:<6} {g_ref[i]:>15.6e} {g_p2[i]:>15.6e} {ae:>12.3e} {re:>12.3e}")

    cos = float(np.dot(g_ref, g_p2) / (np.linalg.norm(g_ref) * np.linalg.norm(g_p2)))
    nr = float(np.linalg.norm(g_p2) / np.linalg.norm(g_ref))
    print_msg(f"\n  Cosine:     {cos:.12f}")
    print_msg(f"  Norm ratio: {nr:.6f}")
    print_msg(f"  Max rtol:   {max_rtol:.6e}")


if __name__ == "__main__":
    from validate_core import run_validation
    results = run_validation(
        model_name="gst_coreg3_small_p2",
        create_model_fn=create_model,
        run_with_method_fn=run_with_method,
        include_ad_loop=False,
        precision=args.precision,
        phase=args.phase,
        jax_fd=args.jax_fd,
        fd_eps=args.fd_eps,
    )
    if results is not None and "f_ad" in results:
        compare_with_native(results)
