"""Shared validation harness for numerical correctness checks."""

import gc
import os
import sys
import time

import numpy as np

from dalia.utils import print_msg
from jax_utils import compute_finite_diff_gradient


FWD_RTOL = 1e-6
GRAD_MAX_RTOL = 1e-2
GRAD_COSINE_MIN = 0.999
AD_LOOP_RTOL = 1e-10


def _rel_err(a, b):
    denom = max(abs(float(a)), abs(float(b)), 1e-30)
    return abs(float(a) - float(b)) / denom


def _component_rel_err(a, b):
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-30, 1e-30, denom)
    return np.abs(a - b) / denom


def _cosine_similarity(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def _print_comparisons(results, include_ad_loop):
    """Print comparison results and return (checks_passed, checks_total)."""
    f_ad = results["f_ad"]
    f_ad_fwd = results["f_ad_fwd"]
    f_cupy = results["f_cupy"]
    grad_ad = results["grad_ad"]
    grad_fd = results["grad_fd"]
    n_hp = len(grad_ad)

    checks_passed = 0
    checks_total = 0

    # Check 1: Forward pass (AD vs reference)
    checks_total += 1
    fwd_abs = abs(f_ad - f_cupy)
    fwd_rtol = _rel_err(f_ad, f_cupy)
    fwd_label = "JAX fwd" if abs(f_ad_fwd - f_cupy) < 1e-6 else "CuPy/serinv"
    print_msg("\n" + "-" * 70)
    print_msg(f"1. FORWARD PASS COMPARISON (JAX AD vs {fwd_label})")
    print_msg("-" * 70)
    print_msg(f"  f (JAX AD):      {f_ad:.15e}")
    print_msg(f"  f ({fwd_label:>12s}): {f_cupy:.15e}")
    print_msg(f"  Absolute error:  {fwd_abs:.6e}")
    print_msg(f"  Relative error:  {fwd_rtol:.6e}")
    if fwd_rtol < FWD_RTOL:
        print_msg(f"  [PASS] Forward values match (rtol {fwd_rtol:.2e} < {FWD_RTOL:.0e})")
        checks_passed += 1
    else:
        print_msg(f"  [FAIL] Forward values differ (rtol {fwd_rtol:.2e} >= {FWD_RTOL:.0e})")

    fwd_int_abs = abs(f_ad - f_ad_fwd)
    print_msg(f"\n  Internal check: |f_grad - f_fwd| = {fwd_int_abs:.6e}")

    # Check 2: Gradient comparison (AD vs FD)
    checks_total += 1
    comp_rtol = _component_rel_err(grad_ad, grad_fd)
    max_rtol = float(np.max(comp_rtol))
    cosine = _cosine_similarity(grad_ad, grad_fd)
    norm_ratio = np.linalg.norm(grad_ad) / max(np.linalg.norm(grad_fd), 1e-30)

    print_msg("\n" + "-" * 70)
    print_msg("2. GRADIENT COMPARISON (JAX AD vs Finite Differences)")
    print_msg("-" * 70)
    print_msg(f"  {'Component':<12} {'AD gradient':>15} {'FD gradient':>15} {'Abs err':>12} {'Rel err':>12}")
    for i in range(n_hp):
        print_msg(
            f"  theta[{i}]    {grad_ad[i]:>15.6e} {grad_fd[i]:>15.6e}"
            f" {abs(grad_ad[i] - grad_fd[i]):>12.3e} {comp_rtol[i]:>12.3e}"
        )
    print_msg(f"\n  Max component-wise relative error: {max_rtol:.6e}")
    print_msg(f"  Cosine similarity:  {cosine:.10f}")
    print_msg(f"  Norm ratio ||AD||/||FD||: {norm_ratio:.6f}")

    grad_pass = max_rtol < GRAD_MAX_RTOL and cosine > GRAD_COSINE_MIN
    if grad_pass:
        print_msg(f"  [PASS] Gradients match (max rtol {max_rtol:.2e} < {GRAD_MAX_RTOL:.0e}, cosine {cosine:.6f} > {GRAD_COSINE_MIN})")
        checks_passed += 1
    else:
        print_msg(f"  [FAIL] Gradients differ (max rtol {max_rtol:.2e}, cosine {cosine:.6f})")

    # Check 3: AD-loop (optional)
    if include_ad_loop and "grad_loop" in results:
        checks_total += 1
        f_loop = results["f_loop"]
        grad_loop = results["grad_loop"]
        loop_f_rtol = _rel_err(f_ad, f_loop)
        loop_grad_diff = np.linalg.norm(grad_ad - grad_loop)
        loop_grad_rtol = loop_grad_diff / max(np.linalg.norm(grad_ad), 1e-30)

        print_msg("\n" + "-" * 70)
        print_msg("3. AD-LOOP COMPARISON (optimized JAX vs jax.value_and_grad)")
        print_msg("-" * 70)
        print_msg(f"  f (optimized):   {f_ad:.15e}")
        print_msg(f"  f (AD loop):     {f_loop:.15e}")
        print_msg(f"  f relative error: {loop_f_rtol:.6e}")
        print_msg(f"  Gradient L2 relative error: {loop_grad_rtol:.6e}")

        if loop_grad_rtol < AD_LOOP_RTOL:
            print_msg(f"  [PASS] AD loop matches (grad rtol {loop_grad_rtol:.2e} < {AD_LOOP_RTOL:.0e})")
            checks_passed += 1
        else:
            print_msg(f"  [FAIL] AD loop differs (grad rtol {loop_grad_rtol:.2e} >= {AD_LOOP_RTOL:.0e})")

    return checks_passed, checks_total


def _compute_jax_fd_gradient(jax_objective, theta_jax, n_hp, fd_eps, dtype):
    """Compute FD gradient through the JAX objective (works distributed)."""
    import jax.numpy as jnp
    import time as _time

    grad = np.zeros(n_hp)
    t0 = _time.perf_counter()
    for i in range(n_hp):
        theta_plus = theta_jax.at[i].add(fd_eps)
        theta_minus = theta_jax.at[i].add(-fd_eps)
        f_plus, _ = jax_objective(theta_plus)
        f_minus, _ = jax_objective(theta_minus)
        grad[i] = (float(f_plus) - float(f_minus)) / (2 * fd_eps)
        elapsed = _time.perf_counter() - t0
        print_msg(f"    FD [{i+1}/{n_hp}] grad={grad[i]:.6e} ({elapsed:.1f}s)", flush=True)
    return grad


def run_validation(
    model_name,
    create_model_fn,
    run_with_method_fn,
    include_ad_loop=False,
    precision="float64",
    fd_eps=1e-3,
    output_dir=None,
    phase=0,
    jax_fd=False,
):
    """Run numerical validation.

    Parameters
    ----------
    phase : int
        0 = run both phases in same process (default, works for single-node).
        1 = CuPy/serinv FD only (reference), save results to npz.
        2 = JAX AD only, load phase-1 npz and compare.
        3 = JAX FD + JAX AD in one job (for distributed models).
    jax_fd : bool
        If True with phase=2, compute FD through jax_objective instead of
        loading CuPy FD reference. Useful for distributed models where the
        CuPy distributed solver has higher numerical noise.
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"validation_{model_name}.npz")

    print_msg("=" * 70)
    print_msg(f"NUMERICAL VALIDATION: {model_name}" + (f" (phase {phase})" if phase else ""))
    print_msg("=" * 70)

    # phase 0: FD then AD in same process
    # phase 1: FD only (reference), save npz
    # phase 2: AD only, load npz, compare (or jax_fd=True to compute FD via JAX)
    # phase 3: JAX FD + JAX AD in one job
    run_fd = phase in (0, 1)
    run_ad = phase in (0, 2, 3)

    results = {}

    # --- FD phase (reference) ---
    if run_fd:
        print_msg("\nPhase 1: CuPy/serinv forward + FD gradient (reference)")
        model_fd = create_model_fn()
        dalia_fd = run_with_method_fn(model_fd, "finite_diff", 1, verbose=False)

        theta_np = dalia_fd.model.theta.copy()
        if hasattr(theta_np, "get"):
            theta_np = theta_np.get()
        theta_np = np.asarray(theta_np, dtype=np.float64)
        n_hp = len(theta_np)

        print_msg(f"  n_hyperparameters: {n_hp}")
        print_msg(f"  theta: {theta_np}")

        t0 = time.perf_counter()
        f_cupy = float(dalia_fd._evaluate_f(theta_np))
        t_cupy = time.perf_counter() - t0
        print_msg(f"  CuPy forward: f={f_cupy:.10f} ({t_cupy:.2f}s)")

        print_msg(f"  Computing FD gradient ({2 * n_hp + 1} evaluations, eps={fd_eps}) ...")
        t0 = time.perf_counter()
        grad_fd = compute_finite_diff_gradient(dalia_fd, theta_np, eps=fd_eps)
        grad_fd = np.asarray(grad_fd, dtype=np.float64)
        t_fd = time.perf_counter() - t0
        print_msg(f"  FD gradient done ({t_fd:.2f}s)")

        results["f_cupy"] = f_cupy
        results["grad_fd"] = grad_fd
        results["theta"] = theta_np

        del dalia_fd, model_fd
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except ImportError:
            pass

        if phase == 1:
            save_dict = {k: v for k, v in results.items() if v is not None}
            np.savez(npz_path, **save_dict)
            print_msg(f"\nPhase 1 (FD reference) saved to {npz_path}")
            return results

    # --- AD phase ---
    if run_ad:
        import jax
        import jax.numpy as jnp
        from dalia.core.jax_autodiff import get_jax_dtype
        from jax_utils import _get_jax_objective_fn

        if phase == 2 and not jax_fd:
            print_msg(f"\nLoading FD reference from {npz_path}")
            loaded = np.load(npz_path)
            results = {k: loaded[k] for k in loaded.files}
            if "f_cupy" in results:
                results["f_cupy"] = float(results["f_cupy"])
            theta_np = results["theta"]
        elif phase == 3 or (phase == 2 and jax_fd):
            # Extract theta from a fresh model
            model_tmp = create_model_fn()
            theta_np = model_tmp.theta.copy()
            if hasattr(theta_np, "get"):
                theta_np = theta_np.get()
            theta_np = np.asarray(theta_np, dtype=np.float64)
            del model_tmp

        print_msg("\nPhase 2: JAX AD forward + gradient")
        model_jax = create_model_fn()
        dalia_jax = run_with_method_fn(model_jax, "jax_autodiff", 1, verbose=False)

        print_msg(f"  precision: {precision}")

        dtype = get_jax_dtype()
        theta_jax = jnp.asarray(theta_np, dtype=dtype)

        t0 = time.perf_counter()
        f_ad_fwd, _ = dalia_jax.jax_objective(theta_jax)
        f_ad_fwd = float(f_ad_fwd)
        t_fwd = time.perf_counter() - t0
        print_msg(f"  JAX forward: f={f_ad_fwd:.10f} ({t_fwd:.2f}s)")

        t0 = time.perf_counter()
        f_ad_raw, grad_ad_raw, x_ad_raw = dalia_jax.jax_grad_func(theta_jax)
        f_ad = float(f_ad_raw)
        grad_ad = np.array(grad_ad_raw, dtype=np.float64)
        t_grad = time.perf_counter() - t0
        print_msg(f"  JAX forward+grad: f={f_ad:.10f} ({t_grad:.2f}s)")

        results["f_ad"] = f_ad
        results["f_ad_fwd"] = f_ad_fwd
        results["grad_ad"] = grad_ad
        results["theta"] = theta_np

        # JAX-based FD (for distributed models)
        if jax_fd or phase == 3:
            n_hp = len(theta_np)
            results["f_cupy"] = f_ad_fwd  # use JAX forward as reference
            print_msg(f"\n  Computing JAX-based FD gradient ({2 * n_hp} evaluations, eps={fd_eps}) ...")
            t0 = time.perf_counter()
            grad_fd = _compute_jax_fd_gradient(
                dalia_jax.jax_objective, theta_jax, n_hp, fd_eps, dtype)
            grad_fd = np.asarray(grad_fd, dtype=np.float64)
            t_fd = time.perf_counter() - t0
            print_msg(f"  JAX FD gradient done ({t_fd:.2f}s)")
            results["grad_fd"] = grad_fd

        if include_ad_loop:
            print_msg("\n  AD-loop: computing pure jax.value_and_grad ...")
            objective_fn, _ = _get_jax_objective_fn(dalia_jax)

            def forward_only(th):
                obj, _ = objective_fn(th)
                return obj

            vag_fn = jax.jit(jax.value_and_grad(forward_only))
            t0 = time.perf_counter()
            f_loop_raw, grad_loop_raw = vag_fn(theta_jax)
            jax.block_until_ready((f_loop_raw, grad_loop_raw))
            f_loop = float(f_loop_raw)
            grad_loop = np.array(grad_loop_raw, dtype=np.float64)
            t_loop = time.perf_counter() - t0
            print_msg(f"  AD-loop: f={f_loop:.10f} ({t_loop:.2f}s)")
            results["f_loop"] = f_loop
            results["grad_loop"] = grad_loop

        del dalia_jax, model_jax
        gc.collect()

    # --- Comparisons ---
    checks_passed, checks_total = _print_comparisons(results, include_ad_loop)

    # Summary
    print_msg("\n" + "=" * 70)
    status = "PASS" if checks_passed == checks_total else "FAIL"
    print_msg(f"OVERALL RESULT: {status} ({checks_passed}/{checks_total} checks passed)")
    print_msg("=" * 70)

    save_dict = {k: v for k, v in results.items() if v is not None}
    np.savez(npz_path, **save_dict)
    print_msg(f"\nResults saved to {npz_path}")

    return results
