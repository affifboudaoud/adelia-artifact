import sys
import os
import time

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

import numpy as np

from dalia.configs import dalia_config
from dalia.core.dalia import DALIA
from dalia.utils import print_msg
from jax_utils import get_first_forward_and_gradient
from model_setup import create_model, load_theta_ref


if __name__ == "__main__":
    theta_ref = load_theta_ref()
    model = create_model()
    n_hp = model.n_hyperparameters

    theta_init = model.theta.copy()
    if hasattr(theta_init, "get"):
        theta_init = theta_init.get()
    theta_init = np.asarray(theta_init, dtype=np.float64)

    print_msg(f"SA1 convergence: n_hp={n_hp}, max_iter={args.max_iter}", flush=True)
    print_msg(f"  theta_init: {theta_init}", flush=True)
    print_msg(f"  theta_ref:  {theta_ref}", flush=True)
    print_msg(f"  ||theta_init - theta_ref||: {np.linalg.norm(theta_init - theta_ref):.6e}", flush=True)

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
            "solver": {"type": "serinv"},
            "gradient_method": "jax_autodiff",
            "minimize": {
                "max_iter": args.max_iter,
                "gtol": 1e-3,
                "disp": False,
                "maxcor": n_hp,
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

    t0 = time.perf_counter()
    f_jax, grad_jax = get_first_forward_and_gradient(dalia)
    t_first = time.perf_counter() - t0
    print_msg(f"First forward+gradient: f={f_jax:.6f}, time={t_first:.4f}s")

    t0 = time.perf_counter()
    results = dalia.run()
    t_total = time.perf_counter() - t0

    print_msg(f"Optimization: {t_total:.2f}s, f={results['f']:.6f}")

    # Save convergence data
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    theta_final = np.asarray(results["theta"], dtype=np.float64)
    grad_final = np.asarray(results.get("grad_f", np.zeros(n_hp)), dtype=np.float64)
    f_values = np.array(results.get("f_values", []))
    theta_values = np.array(results.get("theta_values", []))
    iter_times = np.array(results.get("objective_function_time", []))

    print_msg(f"\n  Final f:        {results['f']:.10f}")
    print_msg(f"  ||grad_final||: {np.linalg.norm(grad_final):.6e}")
    print_msg(f"  Iterations:     {len(f_values)}")

    if theta_ref is not None:
        print_msg(f"  ||theta_final - theta_ref||: {np.linalg.norm(theta_final - theta_ref):.6e}")
        for i in range(n_hp):
            print_msg(
                f"  {i:<4} theta={theta_final[i]:>12.6f}  ref={theta_ref[i]:>12.6f}  "
                f"diff={theta_final[i]-theta_ref[i]:>12.6e}  grad={grad_final[i]:>12.6e}"
            )

    npz_path = os.path.join(output_dir, "convergence_sa1.npz")
    np.savez(
        npz_path,
        f_values=f_values,
        theta_values=theta_values,
        theta_final=theta_final,
        grad_final=grad_final,
        theta_init=theta_init,
        theta_ref=theta_ref,
        iter_times=iter_times,
        n_iterations=len(f_values),
        wall_time=t_total,
    )
    print_msg(f"Results saved to {npz_path}")
