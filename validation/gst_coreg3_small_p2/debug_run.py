"""Debug: isolate why _objective_function_jax returns NaN but jax_grad_func works."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision("float64")

import numpy as np
from dalia.core.dalia import DALIA
from dalia.configs import dalia_config
from dalia.utils import print_msg
from dalia import xp
from model_setup import create_model, load_theta_ref

theta_ref = load_theta_ref()
model = create_model()
n_hp = model.n_hyperparameters

theta_init = model.theta.copy()
if hasattr(theta_init, "get"):
    theta_init = theta_init.get()
theta_init = np.asarray(theta_init, dtype=np.float64)

dalia = DALIA(
    model=model,
    config=dalia_config.parse_config({
        "solver": {"type": "serinv", "min_processes": 2},
        "gradient_method": "jax_autodiff",
        "minimize": {"max_iter": 5, "gtol": 1e-4, "disp": True, "maxcor": n_hp},
        "f_reduction_tol": 1e-4,
        "f_reduction_lag": 5,
        "theta_reduction_tol": 1e-5,
        "theta_reduction_lag": 5,
        "simulation_dir": ".",
    }),
)

dtype = jnp.float64
theta_jax = jnp.asarray(theta_init, dtype=dtype)

# Test 1: Direct jax_grad_func call
print_msg("\n--- Test 1: Direct jax_grad_func(theta_jax) ---")
f1, g1, x1 = dalia.jax_grad_func(theta_jax)
print_msg(f"  f={float(f1):.10f}, ||g||={np.linalg.norm(np.asarray(g1)):.4e}")

# Test 2: Direct jax_grad_func with numpy input (like scipy passes)
print_msg("\n--- Test 2: Direct jax_grad_func(theta_np) ---")
f2, g2, x2 = dalia.jax_grad_func(theta_init)
print_msg(f"  f={float(f2):.10f}, ||g||={np.linalg.norm(np.asarray(g2)):.4e}")

# Test 3: Simulate what _objective_function_jax does step by step
print_msg("\n--- Test 3: Simulating _objective_function_jax ---")
dalia.solver.t_cholesky = 0.0
dalia.solver.t_solve = 0.0
print_msg("  solver timers reset")

dalia.model.theta[:] = xp.asarray(theta_init)
print_msg(f"  model.theta updated, theta[:3]={dalia.model.theta[:3]}")

from dalia.utils.multiprocessing import synchronize
synchronize(comm=dalia.comm_world)
print_msg("  synchronized comm_world")

f3, g3, x3 = dalia.jax_grad_func(theta_init)
print_msg(f"  f={float(f3):.10f}, ||g||={np.linalg.norm(np.asarray(g3)):.4e}")

dalia.model.x[:] = xp.asarray(x3)
print_msg("  model.x updated")

synchronize(comm=dalia.comm_world)
print_msg("  synchronized again")

# Test 4: Call _objective_function_jax directly
print_msg("\n--- Test 4: dalia._objective_function_jax(theta_init) ---")
dalia.iter = 0
dalia.objective_function_time = []
dalia.solver_time = []
dalia.construction_time = []
dalia.objective_function_energy = []
dalia.energy_monitor = None
result4 = dalia._objective_function_jax(theta_init)
print_msg(f"  f={result4[0]}, ||g||={np.linalg.norm(np.asarray(result4[1])):.4e}")

# Test 5: dalia.run() with max_iter=3
print_msg("\n--- Test 5: dalia.run() with max_iter=3 ---")
dalia2 = DALIA(
    model=create_model(),
    config=dalia_config.parse_config({
        "solver": {"type": "serinv", "min_processes": 2},
        "gradient_method": "jax_autodiff",
        "minimize": {"max_iter": 3, "gtol": 1e-4, "disp": True, "maxcor": n_hp},
        "f_reduction_tol": 1e-4,
        "f_reduction_lag": 5,
        "theta_reduction_tol": 1e-5,
        "theta_reduction_lag": 5,
        "simulation_dir": ".",
    }),
)
results5 = dalia2.run()
print_msg(f"  f={results5['f']}")
