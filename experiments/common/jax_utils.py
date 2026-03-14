"""Shared JAX utilities for DALIA examples."""

import os
import time
import numpy as np

import jax
import jax.numpy as jnp

from dalia.utils import print_msg
from dalia.core.jax_autodiff import get_jax_dtype


def compute_finite_diff_gradient(dalia, theta, eps=1e-3):
    """Compute gradient using central finite differences."""
    n = len(theta)
    grad = np.zeros(n)
    for i in range(n):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        f_plus = dalia._evaluate_f(theta_plus)

        theta_minus = theta.copy()
        theta_minus[i] -= eps
        f_minus = dalia._evaluate_f(theta_minus)

        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def get_first_forward_and_gradient(dalia):
    """Get the first forward pass value and gradient."""
    theta = dalia.model.theta.copy()

    if dalia.config.gradient_method == "jax_autodiff":
        f_val, grad, x = dalia.jax_grad_func(theta)
        return float(f_val), np.array(grad)
    else:
        f_val = dalia._evaluate_f(theta)
        grad = compute_finite_diff_gradient(dalia, theta, eps=dalia.eps_gradient_f)
        return float(f_val), np.array(grad)


def get_first_forward_value(dalia):
    """Get the first forward pass value (for zero-hyperparameter models)."""
    theta = dalia.model.theta.copy()
    f_val = dalia._evaluate_f(theta)
    return float(f_val)


def _get_jax_objective_fn(dalia):
    """Get the appropriate JAX objective function based on model type."""
    is_coregional = hasattr(dalia.model, 'models')

    if is_coregional:
        from dalia.core.jax_autodiff import create_pure_jax_objective_coregional
        return create_pure_jax_objective_coregional(dalia)
    else:
        from dalia.core.jax_autodiff import create_pure_jax_objective
        return create_pure_jax_objective(dalia)


def profile_jax_execution(dalia, output_dir=None, num_runs=5):
    """Profile JAX forward and backward pass execution times.

    Handles regular models, CoregionalModel, and zero-hyperparameter models.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    n_hyperparameters = dalia.model.n_hyperparameters
    theta = jnp.asarray(dalia.model.theta, dtype=get_jax_dtype())

    objective_fn, _ = _get_jax_objective_fn(dalia)

    def forward_only(theta):
        obj, x = objective_fn(theta)
        return obj

    forward_jit = jax.jit(forward_only)

    print_msg("Warming up JIT compilation...")
    _ = forward_jit(theta).block_until_ready()

    # For zero-hyperparameter models, only profile forward pass
    if n_hyperparameters == 0:
        print_msg(f"Profiling with {num_runs} runs each...")

        forward_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            result = forward_jit(theta)
            result.block_until_ready()
            forward_times.append(time.perf_counter() - t0)

        print_msg("\n" + "-" * 70)
        print_msg("JAX EXECUTION PROFILING RESULTS")
        print_msg("-" * 70)

        print_msg(f"\nForward pass only:")
        print_msg(f"  Mean: {np.mean(forward_times)*1000:.2f} ms")
        print_msg(f"  Std:  {np.std(forward_times)*1000:.2f} ms")

        print_msg("\nNote: No gradient computation for zero-hyperparameter case")

        return {
            'forward_mean_ms': np.mean(forward_times) * 1000,
        }

    # For models with hyperparameters, profile forward, backward, and combined
    grad_fn_pure = jax.grad(forward_only)
    value_and_grad_fn = jax.value_and_grad(forward_only)

    grad_jit = jax.jit(grad_fn_pure)
    value_and_grad_jit = jax.jit(value_and_grad_fn)

    _ = grad_jit(theta).block_until_ready()
    val, grad = value_and_grad_jit(theta)
    val.block_until_ready()
    grad.block_until_ready()

    print_msg(f"Profiling with {num_runs} runs each...")

    forward_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = forward_jit(theta)
        result.block_until_ready()
        forward_times.append(time.perf_counter() - t0)

    grad_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = grad_jit(theta)
        result.block_until_ready()
        grad_times.append(time.perf_counter() - t0)

    value_and_grad_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        val, grad = value_and_grad_jit(theta)
        val.block_until_ready()
        grad.block_until_ready()
        value_and_grad_times.append(time.perf_counter() - t0)

    print_msg("\n" + "-" * 70)
    print_msg("JAX EXECUTION PROFILING RESULTS")
    print_msg("-" * 70)

    print_msg(f"\nForward pass only:")
    print_msg(f"  Mean: {np.mean(forward_times)*1000:.2f} ms")
    print_msg(f"  Std:  {np.std(forward_times)*1000:.2f} ms")

    print_msg(f"\nBackward pass only (grad):")
    print_msg(f"  Mean: {np.mean(grad_times)*1000:.2f} ms")
    print_msg(f"  Std:  {np.std(grad_times)*1000:.2f} ms")

    print_msg(f"\nValue and grad combined:")
    print_msg(f"  Mean: {np.mean(value_and_grad_times)*1000:.2f} ms")
    print_msg(f"  Std:  {np.std(value_and_grad_times)*1000:.2f} ms")

    backward_only = np.mean(value_and_grad_times) - np.mean(forward_times)
    print_msg(f"\nDerived metrics:")
    print_msg(f"  Backward pass time (derived): {backward_only*1000:.2f} ms")
    if np.mean(forward_times) > 0:
        print_msg(f"  Backward/Forward ratio: {backward_only/np.mean(forward_times):.2f}x")
        print_msg(f"  Grad-only/Forward ratio: {np.mean(grad_times)/np.mean(forward_times):.2f}x")

    trace_dir = os.path.join(output_dir, "jax_profile_trace")
    print_msg(f"\nGenerating Chrome trace in: {trace_dir}")

    with jax.profiler.trace(trace_dir):
        for _ in range(3):
            val, grad = value_and_grad_jit(theta)
            val.block_until_ready()
            grad.block_until_ready()

    print_msg("View trace with: chrome://tracing (load the .json file)")

    return {
        'forward_mean_ms': np.mean(forward_times) * 1000,
        'grad_mean_ms': np.mean(grad_times) * 1000,
        'value_and_grad_mean_ms': np.mean(value_and_grad_times) * 1000,
        'backward_forward_ratio': backward_only / np.mean(forward_times) if np.mean(forward_times) > 0 else 0,
    }


def print_jax_ir(dalia, output_file=None):
    """Print the JAX IR (jaxpr) for the forward and backward pass.

    Handles regular models, CoregionalModel, and zero-hyperparameter models.
    """
    n_hyperparameters = dalia.model.n_hyperparameters
    theta = jnp.asarray(dalia.model.theta, dtype=get_jax_dtype())

    objective_fn, _ = _get_jax_objective_fn(dalia)

    def forward_only(theta):
        obj, x = objective_fn(theta)
        return obj

    forward_jaxpr = jax.make_jaxpr(forward_only)(theta)

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("JAX INTERMEDIATE REPRESENTATION (JAXPR)")
    output_lines.append("=" * 70)

    output_lines.append("\n" + "-" * 70)
    output_lines.append("FORWARD PASS JAXPR")
    output_lines.append("-" * 70)
    output_lines.append(f"Number of equations: {len(forward_jaxpr.eqns)}")
    output_lines.append(f"Input variables: {forward_jaxpr.in_avals}")
    output_lines.append(f"Output variables: {forward_jaxpr.out_avals}")
    output_lines.append("\nFull JAXPR:")
    output_lines.append(str(forward_jaxpr))

    def count_primitives(jaxpr):
        counts = {}
        for eqn in jaxpr.eqns:
            prim_name = eqn.primitive.name
            counts[prim_name] = counts.get(prim_name, 0) + 1
        return counts

    # For models with hyperparameters, also show gradient jaxpr
    if n_hyperparameters > 0:
        grad_jaxpr = jax.make_jaxpr(jax.grad(forward_only))(theta)
        value_and_grad_fn = jax.value_and_grad(forward_only)
        value_and_grad_jaxpr = jax.make_jaxpr(value_and_grad_fn)(theta)

        output_lines.append("\n" + "-" * 70)
        output_lines.append("BACKWARD PASS (GRAD) JAXPR")
        output_lines.append("-" * 70)
        output_lines.append(f"Number of equations: {len(grad_jaxpr.eqns)}")
        output_lines.append(f"Input variables: {grad_jaxpr.in_avals}")
        output_lines.append(f"Output variables: {grad_jaxpr.out_avals}")
        output_lines.append("\nFull JAXPR:")
        output_lines.append(str(grad_jaxpr))

        output_lines.append("\n" + "-" * 70)
        output_lines.append("VALUE_AND_GRAD JAXPR")
        output_lines.append("-" * 70)
        output_lines.append(f"Number of equations: {len(value_and_grad_jaxpr.eqns)}")
        output_lines.append("\nFull JAXPR:")
        output_lines.append(str(value_and_grad_jaxpr))

    output_lines.append("\n" + "-" * 70)
    output_lines.append("PRIMITIVE OPERATION COUNTS")
    output_lines.append("-" * 70)

    forward_counts = count_primitives(forward_jaxpr)

    output_lines.append("\nForward pass primitives:")
    for prim, count in sorted(forward_counts.items(), key=lambda x: -x[1]):
        output_lines.append(f"  {prim}: {count}")

    if n_hyperparameters > 0:
        grad_counts = count_primitives(grad_jaxpr)
        output_lines.append("\nBackward pass primitives:")
        for prim, count in sorted(grad_counts.items(), key=lambda x: -x[1]):
            output_lines.append(f"  {prim}: {count}")

    output_text = "\n".join(output_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
        print_msg(f"JAX IR written to: {output_file}")
    else:
        print(output_text)

    if n_hyperparameters > 0:
        return forward_jaxpr, grad_jaxpr
    return forward_jaxpr
