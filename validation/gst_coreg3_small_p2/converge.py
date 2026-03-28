import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--precision", default="float64", choices=["float32", "float64"])
parser.add_argument("--max_iter", type=int, default=200)
args = parser.parse_args()

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision(args.precision)

from model_setup import create_model, load_theta_ref

theta_ref = load_theta_ref()

if __name__ == "__main__":
    from converge_core import run_convergence
    run_convergence(
        model_name="gst_coreg3_small_p2",
        create_model_fn=create_model,
        solver_cfg={"type": "serinv", "min_processes": 2},
        max_iter=args.max_iter,
        theta_ref=theta_ref,
        precision=args.precision,
    )
