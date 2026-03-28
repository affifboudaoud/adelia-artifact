"""Table 2 structure comparison for GST-T (gst_temperature) only."""

import sys
import os
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from parser_utils import parse_args
args = parse_args()

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()

from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision(args.precision)

from run import (
    create_gst_temperature, STRATEGIES, run_strategy,
    print_summary_table, write_csv, MODELS,
)
from dalia.utils import print_msg

if __name__ == "__main__":
    print_msg("--- Table 2: GST-T (gst_temperature) only ---")
    n_runs = args.n_benchmark_runs

    results = OrderedDict()
    model_name = "GST-T"
    model_info = MODELS[model_name]
    results[model_name] = OrderedDict()

    print_msg(f"Model: {model_name} ({model_info['dims']})")
    for strategy in STRATEGIES:
        print(f"  {strategy}... ", end="", flush=True)
        res = run_strategy(model_name, model_info, strategy, n_runs)
        results[model_name][strategy] = res
        if res is None:
            print("OOM", flush=True)
        else:
            print(f"{res['mean']:.4f}s (+/- {res['std']:.4f}), "
                  f"{res['mem_gib']:.1f} GiB", flush=True)

    print_summary_table(results)
    write_csv(results, os.path.join(os.path.dirname(__file__), "tab2_results_gst_t.csv"))
    print_msg("--- Finished ---")
