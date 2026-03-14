import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="DALIA example parameters")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=250,
        help="Maximum number of iterations in the optimization process.",
    )
    parser.add_argument(
        "--solver_min_p",
        type=int,
        default=1,
        help="Minimum number of processes for the solver. If greater than 1 a distributed solver is used.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable JAX execution profiling.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="JAX computation precision (float32 or float64).",
    )
    parser.add_argument(
        "--benchmark_mode",
        action="store_true",
        help="Run gradient benchmark instead of full optimization.",
    )
    parser.add_argument(
        "--n_benchmark_runs",
        type=int,
        default=10,
        help="Number of runs for timing in benchmark mode.",
    )
    parser.add_argument(
        "--benchmark_method",
        type=str,
        choices=["finite_diff", "jax_autodiff", "both"],
        default="both",
        help="Which gradient method to benchmark.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file for benchmark results.",
    )
    parser.add_argument(
        "--measure_power",
        action="store_true",
        help="Measure GPU power draw during benchmark using nvidia-smi.",
    )
    parser.add_argument(
        "--energy_monitor",
        action="store_true",
        help="Measure energy via Cray PM hardware counters.",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=None,
        help="Number of time steps (overrides script default).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for structured JSON/CSV result output.",
    )
    parser.add_argument(
        "--skip_fd",
        action="store_true",
        help="Skip finite differences run (JAX AD only).",
    )
    parser.add_argument(
        "--skip_jax",
        action="store_true",
        help="Skip JAX autodiff run (FD only).",
    )
    parser.add_argument(
        "--distributed_method",
        type=str,
        choices=["split_jit", "two_phase"],
        default="two_phase",
        help="Distributed AD method for coregional models.",
    )
    parser.add_argument(
        "--theta_initial",
        type=float,
        nargs="+",
        default=None,
        help="Initial theta values for warm-starting optimization.",
    )
    parser.add_argument(
        "--framework_comparison",
        action="store_true",
        help="Run framework comparison (JAX vs CuPy forward eval) for Figure 4.",
    )
    parser.add_argument(
        "--framework_ad_only",
        action="store_true",
        help="Framework comparison: only create JAX AD instance (skip FD to save memory).",
    )
    parser.add_argument(
        "--framework_fd_only",
        action="store_true",
        help="Framework comparison: only create FD instance (skip JAX AD).",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Run per-stage timing breakdown for distributed AD (Figure 7).",
    )
    parser.add_argument(
        "--breakdown_csv",
        type=str,
        default=None,
        help="Path to output CSV file for per-stage breakdown results.",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "fig2_wallclock", "results", "benchmark_results.csv",
        ),
        help="Path to shared CSV for benchmark results (all models append here).",
    )
    args = parser.parse_args()
    print("Parsed parameters:")
    print(f"  max_iter: {args.max_iter}")
    print(f"  solver_min_p: {args.solver_min_p}")
    print(f"  profile: {args.profile}")
    print(f"  precision: {args.precision}")
    if args.benchmark_mode:
        print(f"  benchmark_mode: {args.benchmark_mode}")
        print(f"  n_benchmark_runs: {args.n_benchmark_runs}")
        print(f"  benchmark_method: {args.benchmark_method}")
        print(f"  output_csv: {args.output_csv}")
    if args.output_dir:
        print(f"  output_dir: {args.output_dir}")
    if args.skip_fd:
        print(f"  skip_fd: {args.skip_fd}")
    if args.skip_jax:
        print(f"  skip_jax: {args.skip_jax}")
    return args