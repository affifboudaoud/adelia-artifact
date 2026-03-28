"""Shared convergence harness for running DALIA models to convergence."""

import os
import time

import numpy as np
from scipy import optimize

from dalia.utils import print_msg
from dalia.core.dalia import DALIA
from dalia.configs import dalia_config


def run_convergence(
    model_name,
    create_model_fn,
    solver_cfg,
    max_iter=200,
    theta_ref=None,
    precision="float64",
):
    """Run a model to convergence with JAX AD.

    Calls jax_grad_func directly in scipy.optimize.minimize, bypassing
    _objective_function_jax which has NaN issues in distributed mode.

    Parameters
    ----------
    model_name : str
    create_model_fn : callable
    solver_cfg : dict
    max_iter : int
    theta_ref : ndarray or None
    precision : str
    """
    import jax.numpy as jnp
    from dalia.core.jax_autodiff import get_jax_dtype

    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print_msg("=" * 70)
    print_msg(f"CONVERGENCE RUN: {model_name}")
    print_msg("=" * 70)

    model = create_model_fn()
    n_hp = model.n_hyperparameters

    theta_init_host = model.theta.copy()
    if hasattr(theta_init_host, "get"):
        theta_init_host = theta_init_host.get()
    theta_init_host = np.asarray(theta_init_host, dtype=np.float64)

    print_msg(f"  n_hyperparameters: {n_hp}")
    print_msg(f"  theta_initial: {theta_init_host}")
    if theta_ref is not None:
        print_msg(f"  theta_ref:     {theta_ref}")
        print_msg(f"  ||theta_init - theta_ref||: {np.linalg.norm(theta_init_host - theta_ref):.6e}")

    # Initialize DALIA to get jax_grad_func
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
            "solver": solver_cfg,
            "gradient_method": "jax_autodiff",
            "minimize": {"max_iter": 1, "disp": False, "maxcor": n_hp},
            "f_reduction_tol": 1e-4,
            "f_reduction_lag": 5,
            "theta_reduction_tol": 1e-5,
            "theta_reduction_lag": 5,
            "simulation_dir": ".",
        }),
    )

    dtype = get_jax_dtype()
    grad_func = dalia.jax_grad_func

    # Sanity check — pass theta same way as get_first_forward_and_gradient
    f_check, g_check, _ = grad_func(model.theta.copy())
    f_check = float(f_check)
    print_msg(f"\n  Sanity check: f(theta_init) = {f_check:.10f}, ||grad|| = {np.linalg.norm(np.asarray(g_check)):.4e}")
    if np.isnan(f_check):
        print_msg("  ERROR: f is NaN at initial theta. Aborting.")
        return None

    # L-BFGS-B with jax_grad_func called directly
    f_values = []
    theta_values = []
    iter_times = []
    theta_init = theta_init_host

    def objective_and_grad(theta_np):
        t0 = time.perf_counter()
        f_val, grad_val, _ = grad_func(theta_np)
        dt = time.perf_counter() - t0
        iter_times.append(dt)
        return float(f_val), np.asarray(grad_val, dtype=np.float64)

    last_grad = [None]

    def objective_and_grad_wrapper(theta_np):
        f_out, g_out = objective_and_grad(theta_np)
        last_grad[0] = g_out
        return f_out, g_out

    def callback(intermediate_result):
        theta_i = intermediate_result.x.copy()
        f_i = intermediate_result.fun
        f_values.append(f_i)
        theta_values.append(theta_i)
        n = len(f_values)
        g_norm = np.linalg.norm(last_grad[0]) if last_grad[0] is not None else float('nan')
        dt = iter_times[-1] if iter_times else 0
        print_msg(
            f"  iter {n:3d} | f={f_i:>16.6f} | ||grad||={g_norm:>12.4e} | t={dt:.2f}s"
        )

    print_msg(f"\nRunning L-BFGS-B (max_iter={max_iter}) ...")
    t0_total = time.perf_counter()

    result = optimize.minimize(
        fun=objective_and_grad_wrapper,
        x0=theta_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "maxcor": n_hp, "gtol": 1e-4, "disp": False},
        callback=callback,
    )

    wall_time = time.perf_counter() - t0_total
    n_iter = len(f_values)
    theta_final = result.x
    grad_final = result.jac if result.jac is not None else np.zeros(n_hp)

    # --- Summary ---
    print_msg("\n" + "=" * 70)
    print_msg("CONVERGENCE SUMMARY")
    print_msg("=" * 70)
    print_msg(f"  Status:        {result.message}")
    print_msg(f"  Iterations:    {n_iter}")
    print_msg(f"  Wall time:     {wall_time:.1f}s ({wall_time/max(n_iter,1):.2f}s/iter)")
    if f_values:
        print_msg(f"  Final f:       {f_values[-1]:.10f}")
    print_msg(f"  ||grad_final||: {np.linalg.norm(grad_final):.6e}")

    if n_iter >= 5:
        print_msg(f"  f decrease (last 5): {f_values[max(0,n_iter-5)] - f_values[-1]:.6e}")

    print_msg(f"\n  {'i':<6} {'theta_final':>14} {'grad_final':>14}", end="")
    if theta_ref is not None:
        print_msg(f" {'theta_ref':>14} {'diff':>14}")
    else:
        print_msg("")
    for i in range(n_hp):
        line = f"  {i:<6} {theta_final[i]:>14.6f} {grad_final[i]:>14.6e}"
        if theta_ref is not None:
            line += f" {theta_ref[i]:>14.6f} {theta_final[i]-theta_ref[i]:>14.6e}"
        print_msg(line)

    if theta_ref is not None:
        print_msg(f"\n  ||theta_final - theta_ref||: {np.linalg.norm(theta_final - theta_ref):.6e}")

    # --- Save ---
    npz_path = os.path.join(output_dir, f"convergence_{model_name}.npz")
    save_dict = {
        "f_values": np.array(f_values),
        "theta_values": np.array(theta_values) if theta_values else np.array([]),
        "theta_final": theta_final,
        "grad_final": grad_final,
        "theta_init": theta_init,
        "n_iterations": n_iter,
        "wall_time": wall_time,
        "iter_times": np.array(iter_times),
    }
    if theta_ref is not None:
        save_dict["theta_ref"] = theta_ref
    np.savez(npz_path, **save_dict)
    print_msg(f"\nResults saved to {npz_path}")

    return result
