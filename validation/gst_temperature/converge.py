import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--precision", default="float64", choices=["float32", "float64"])
parser.add_argument("--max_iter", type=int, default=200)
args = parser.parse_args()

# Overwrite sys.argv before importing validate.py (which calls parse_args at module level)
sys.argv = [sys.argv[0], "--precision", args.precision]

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision(args.precision)

from validate import create_model

theta_ref = None
try:
    from validate import _load_theta_ref
    theta_ref = _load_theta_ref()
except (ImportError, Exception):
    pass
if theta_ref is None:
    try:
        from validate import load_theta_ref
        theta_ref = load_theta_ref()
    except (ImportError, Exception):
        pass

if __name__ == "__main__":
    from converge_core import run_convergence
    run_convergence(
        model_name=os.path.basename(os.path.dirname(os.path.abspath(__file__))),
        create_model_fn=create_model,
        solver_cfg={"type": "serinv"},
        max_iter=args.max_iter,
        theta_ref=theta_ref,
        precision=args.precision,
    )
