"""SA1 convergence: follows validate_jaxfd pattern exactly."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
from dalia.core.jax_autodiff import configure_jax_precision, get_jax_dtype
configure_jax_precision("float64")

import numpy as np
from model_setup import create_model, load_theta_ref
from dalia.utils import print_msg

theta_ref = load_theta_ref()

# Same DALIA creation as validate.py
from dalia.configs import dalia_config
from dalia.core.dalia import DALIA

model = create_model()
n_hp = model.n_hyperparameters
theta_init = model.theta.copy()
if hasattr(theta_init, "get"):
    theta_init = theta_init.get()
theta_init = np.asarray(theta_init, dtype=np.float64)

dalia = DALIA(
    model=model,
    config=dalia_config.parse_config({
        "solver": {"type": "serinv"},
        "gradient_method": "jax_autodiff",
        "minimize": {
            "max_iter": 1, "gtol": 1e-3,
            "disp": False, "maxcor": n_hp,
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }),
)

dtype = get_jax_dtype()
theta_jax = jnp.asarray(theta_init, dtype=dtype)

# Same pattern as validate_core: forward first, then grad
t0 = time.perf_counter()
f_fwd, _ = dalia.jax_objective(theta_jax)
print_msg(f"Forward warmup: f={float(f_fwd):.10f} ({time.perf_counter()-t0:.2f}s)")

t0 = time.perf_counter()
f_check, g_check, _ = dalia.jax_grad_func(theta_jax)
f_check = float(f_check)
print_msg(f"Sanity check: f={f_check:.10f}, ||grad||={np.linalg.norm(np.asarray(g_check)):.4e} ({time.perf_counter()-t0:.2f}s)")

if np.isnan(f_check):
    print_msg("ERROR: f is NaN. Aborting.")
    sys.exit(1)

# L-BFGS-B convergence
from scipy import optimize
grad_func = dalia.jax_grad_func
f_values = []
iter_times = []

def objective_and_grad(theta_np):
    t0 = time.perf_counter()
    f_val, grad_val, _ = grad_func(jnp.asarray(theta_np, dtype=dtype))
    iter_times.append(time.perf_counter() - t0)
    return float(f_val), np.asarray(grad_val, dtype=np.float64)

last_grad = [None]

def wrapper(theta_np):
    f_out, g_out = objective_and_grad(theta_np)
    last_grad[0] = g_out
    return f_out, g_out

def callback(intermediate_result):
    f_i = intermediate_result.fun
    f_values.append(f_i)
    g_norm = np.linalg.norm(last_grad[0]) if last_grad[0] is not None else float('nan')
    dt = iter_times[-1] if iter_times else 0
    print_msg(f"  iter {len(f_values):3d} | f={f_i:>16.6f} | ||grad||={g_norm:>12.4e} | t={dt:.2f}s")

print_msg(f"\nRunning L-BFGS-B (max_iter=200) ...")
t0_total = time.perf_counter()

result = optimize.minimize(
    fun=wrapper,
    x0=theta_init,
    method="L-BFGS-B",
    jac=True,
    options={"maxiter": 200, "maxcor": n_hp, "gtol": 1e-4, "disp": False},
    callback=callback,
)

wall_time = time.perf_counter() - t0_total
print_msg(f"\nStatus: {result.message}")
print_msg(f"Iterations: {len(f_values)}, Wall time: {wall_time:.1f}s")
if f_values:
    print_msg(f"Final f: {f_values[-1]:.10f}")
print_msg(f"theta_final: {result.x}")
if theta_ref is not None:
    print_msg(f"||theta_final - theta_ref||: {np.linalg.norm(result.x - theta_ref):.6e}")
