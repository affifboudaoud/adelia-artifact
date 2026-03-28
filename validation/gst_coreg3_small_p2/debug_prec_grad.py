"""Diagnose precision gradient errors in distributed coregional AD.

Decomposes the P=2 gradient into grad_cond_lik, grad_quad_lik, grad_scalar
and compares each against a dense reference computed from Q_cond^{-1}.

Usage (P=2 GPU):
    srun -n 2 python debug_prec_grad.py

Usage (P=1 reference on single rank):
    python debug_prec_grad.py --p1
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--p1", action="store_true", help="Run P=1 reference (single rank)")
args = parser.parse_args()

import jax
import jax.numpy as jnp
if jax.default_backend() == "gpu":
    jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision("float64")

import numpy as np
import scipy.sparse as sp

from model_setup import create_model, load_theta_ref, PERTURBATION, NV, NS, NT, NB
from dalia.configs import dalia_config
from dalia.core.dalia import DALIA
from dalia.utils import print_msg


def dense_grad_cond_lik(dalia_inst, theta, x):
    """Compute exact tr(Sigma * AtA_m) * prec_m from dense Q_cond^{-1}."""
    model = dalia_inst.model
    n_models = model.n_models

    a_scipy = sp.csr_matrix(model.a.get() if hasattr(model.a, 'get') else model.a)
    n_observations_idx = [int(v) for v in model.n_observations_idx]
    hyperparameters_idx = [int(v) for v in model.hyperparameters_idx]

    likelihood_precs = np.array([
        np.exp(theta[hyperparameters_idx[m + 1] - 1]) for m in range(n_models)
    ])

    y = np.asarray(model.y.get() if hasattr(model.y, 'get') else model.y).ravel()

    a_dense = a_scipy.toarray()
    N = a_dense.shape[1]

    ata_full = a_dense.T @ a_dense
    per_model_ata = []
    for m in range(n_models):
        obs_s = n_observations_idx[m]
        obs_e = n_observations_idx[m + 1]
        a_m = a_dense[obs_s:obs_e, :]
        per_model_ata.append(a_m.T @ a_m)

    from dalia.core.autodiff.data_extraction import _extract_static_data_coregional
    sd = _extract_static_data_coregional(dalia_inst)

    from dalia.core.autodiff.spatial_precompute import (
        precompute_spatial_components_coregional,
        _reconstruct_coregional_diag_block,
        _reconstruct_coregional_lower_block,
    )

    ns = sd['ns']
    nt = sd['nt']
    block_size = sd['block_size']
    n_fe = sd['n_fixed_effects_total']
    models_data = sd['models_data']
    manifolds = [md.get('manifold', 'plane') for md in models_data]
    per_model_offsets = sd['per_model_offsets']
    theta_keys = sd['theta_keys']

    theta_jax = jnp.asarray(theta)
    sc_list_raw, coreg_w_jax = precompute_spatial_components_coregional(
        theta_jax, n_models, ns, models_data, hyperparameters_idx,
        theta_keys, manifolds)
    sc_list = []
    for m in range(n_models):
        sc_m = sc_list_raw[m]
        sc_padded = {**sc_m}
        for key in ['m0_subdiag', 'm1_subdiag', 'm2_subdiag']:
            sc_padded[key] = jnp.concatenate([sc_m[key], jnp.zeros(1)])
        sc_list.append(sc_padded)

    Q_cond = np.zeros((N, N))
    for t in range(nt):
        diag = np.array(_reconstruct_coregional_diag_block(
            sc_list, coreg_w_jax, n_models, ns, t))
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            rows = np.asarray(sd['per_model_ata_diag_rows'][m][t])
            cols = np.asarray(sd['per_model_ata_diag_cols'][m][t])
            vals = np.asarray(sd['per_model_ata_diag_vals'][m][t])
            for r, c, v in zip(rows, cols, vals):
                diag[m_off + r, m_off + c] += likelihood_precs[m] * v
        r0 = t * block_size
        Q_cond[r0:r0 + block_size, r0:r0 + block_size] = diag

        if t < nt - 1:
            lower = np.array(_reconstruct_coregional_lower_block(
                sc_list, coreg_w_jax, n_models, ns, t))
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                rows = np.asarray(sd['per_model_ata_lower_rows'][m][t])
                cols = np.asarray(sd['per_model_ata_lower_cols'][m][t])
                vals = np.asarray(sd['per_model_ata_lower_vals'][m][t])
                for r, c, v in zip(rows, cols, vals):
                    lower[m_off + r, m_off + c] += likelihood_precs[m] * v
            c0 = (t + 1) * block_size
            Q_cond[c0:c0 + block_size, r0:r0 + block_size] = lower
            Q_cond[r0:r0 + block_size, c0:c0 + block_size] = lower.T

        arrow = np.zeros((n_fe, block_size))
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            rows = np.asarray(sd['per_model_ata_arrow_rows'][m][t])
            cols = np.asarray(sd['per_model_ata_arrow_cols'][m][t])
            vals = np.asarray(sd['per_model_ata_arrow_vals'][m][t])
            for r, c, v in zip(rows, cols, vals):
                arrow[r, m_off + c] += likelihood_precs[m] * v
        a0 = nt * block_size
        Q_cond[a0:a0 + n_fe, r0:r0 + block_size] = arrow
        Q_cond[r0:r0 + block_size, a0:a0 + n_fe] = arrow.T

    fe_prec = sd['fixed_effects_precision']
    for i in range(n_fe):
        Q_cond[a0 + i, a0 + i] += fe_prec
    for m in range(n_models):
        tip = np.asarray(sd['per_model_ata_tip'][m])
        Q_cond[a0:a0 + n_fe, a0:a0 + n_fe] += likelihood_precs[m] * tip

    L = np.linalg.cholesky(Q_cond)
    Sigma = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N)))

    grad_cond_lik_dense = np.zeros(n_models)
    grad_quad_lik_dense = np.zeros(n_models)
    for m in range(n_models):
        prec_m = likelihood_precs[m]
        ata_m = per_model_ata[m]
        grad_cond_lik_dense[m] = prec_m * np.sum(Sigma * ata_m)

        obs_s = n_observations_idx[m]
        obs_e = n_observations_idx[m + 1]
        y_m_w = np.zeros(a_dense.shape[0])
        y_m_w[obs_s:obs_e] = y[obs_s:obs_e]
        rhs_m = a_dense.T @ y_m_w
        xTAmy = np.dot(x, rhs_m)
        xAtAx = x @ ata_m @ x
        grad_quad_lik_dense[m] = prec_m * (2.0 * xTAmy - xAtAx)

    return grad_cond_lik_dense, grad_quad_lik_dense


def run_p2():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    model = create_model()
    theta_ref = load_theta_ref()
    theta = theta_ref + np.array(PERTURBATION)

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
            "solver": {"type": "serinv"},
            "gradient_method": "jax_autodiff",
            "minimize": {"max_iter": 1, "gtol": 1e-3, "disp": False, "maxcor": 15},
            "f_reduction_tol": 1e-3, "theta_reduction_tol": 1e-4,
            "inner_iteration_max_iter": 50, "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3, "simulation_dir": ".",
        }),
    )

    f_val, grad_val, x_val = dalia.jax_grad_func(theta)
    dc = dalia.jax_grad_func._debug_components
    grad_cond_lik_p2 = dc['grad_cond_lik']
    grad_quad_lik_p2 = dc['grad_quad_lik']
    grad_scalar_p2 = dc['grad_scalar']

    n_models = model.n_models
    hp_idx = [int(v) for v in model.hyperparameters_idx]

    if rank == 0:
        print_msg(f"\n{'='*70}")
        print_msg(f"P=2 DISTRIBUTED GRADIENT DECOMPOSITION")
        print_msg(f"{'='*70}")
        print_msg(f"f = {f_val:.15e}")
        for m in range(n_models):
            prec_idx = hp_idx[m + 1] - 1
            print_msg(f"\nModel {m}: prec_idx={prec_idx}")
            print_msg(f"  grad_cond_lik[{m}] = {grad_cond_lik_p2[m]:>20.12e}")
            print_msg(f"  grad_quad_lik[{m}] = {grad_quad_lik_p2[m]:>20.12e}")
            print_msg(f"  grad_scalar[{prec_idx}]  = {grad_scalar_p2[prec_idx]:>20.12e}")
            total = 0.5 * grad_cond_lik_p2[m] - 0.5 * grad_quad_lik_p2[m] + grad_scalar_p2[prec_idx]
            print_msg(f"  combined           = {total:>20.12e}")
            print_msg(f"  grad_val[{prec_idx}]       = {grad_val[prec_idx]:>20.12e}")

        print_msg(f"\nComputing dense reference on rank 0...")
        gcl_ref, gql_ref = dense_grad_cond_lik(dalia, theta, x_val)

        print_msg(f"\n{'='*70}")
        print_msg(f"COMPARISON: P=2 distributed vs Dense reference")
        print_msg(f"{'='*70}")
        print_msg(f"{'Component':<18} {'m':>2} {'P=2':>20} {'Dense':>20} {'abs err':>12} {'rel err':>12}")
        print_msg("-" * 88)
        for m in range(n_models):
            ae = abs(grad_cond_lik_p2[m] - gcl_ref[m])
            re = ae / max(abs(gcl_ref[m]), 1e-30)
            print_msg(f"{'grad_cond_lik':<18} {m:>2} {grad_cond_lik_p2[m]:>20.12e} "
                      f"{gcl_ref[m]:>20.12e} {ae:>12.3e} {re:>12.3e}")
        for m in range(n_models):
            ae = abs(grad_quad_lik_p2[m] - gql_ref[m])
            re = ae / max(abs(gql_ref[m]), 1e-30)
            print_msg(f"{'grad_quad_lik':<18} {m:>2} {grad_quad_lik_p2[m]:>20.12e} "
                      f"{gql_ref[m]:>20.12e} {ae:>12.3e} {re:>12.3e}")

        print_msg(f"\nPer-model prec gradient error breakdown:")
        print_msg(f"{'m':>2} {'cond_lik err':>14} {'quad_lik err':>14} {'total err':>14} {'total rel':>12}")
        for m in range(n_models):
            prec_idx = hp_idx[m + 1] - 1
            cl_err = 0.5 * (grad_cond_lik_p2[m] - gcl_ref[m])
            ql_err = -0.5 * (grad_quad_lik_p2[m] - gql_ref[m])
            total_err = cl_err + ql_err
            ref_total = 0.5 * gcl_ref[m] - 0.5 * gql_ref[m] + grad_scalar_p2[prec_idx]
            total_rel = abs(total_err) / max(abs(ref_total), 1e-30)
            print_msg(f"{m:>2} {cl_err:>14.6e} {ql_err:>14.6e} {total_err:>14.6e} {total_rel:>12.3e}")

        np.savez(os.path.join(os.path.dirname(__file__), "debug_p2_components.npz"),
                 f=f_val, grad=grad_val, x=x_val,
                 grad_cond_lik=grad_cond_lik_p2,
                 grad_quad_lik=grad_quad_lik_p2,
                 grad_scalar=grad_scalar_p2,
                 gcl_dense_ref=gcl_ref, gql_dense_ref=gql_ref)

    comm.Barrier()


def run_p1():
    model = create_model()
    theta_ref = load_theta_ref()
    theta = theta_ref + np.array(PERTURBATION)

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
            "solver": {"type": "serinv"},
            "gradient_method": "jax_autodiff",
            "minimize": {"max_iter": 1, "gtol": 1e-3, "disp": False, "maxcor": 15},
            "f_reduction_tol": 1e-3, "theta_reduction_tol": 1e-4,
            "inner_iteration_max_iter": 50, "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3, "simulation_dir": ".",
        }),
    )

    f_val, grad_val, x_val = dalia.jax_grad_func(theta)
    n_models = model.n_models
    hp_idx = [int(v) for v in model.hyperparameters_idx]

    print_msg(f"\n{'='*70}")
    print_msg(f"P=1 AD-LOOP REFERENCE")
    print_msg(f"{'='*70}")
    print_msg(f"f = {f_val:.15e}")

    gcl_ref, gql_ref = dense_grad_cond_lik(dalia, theta, x_val)

    for m in range(n_models):
        prec_idx = hp_idx[m + 1] - 1
        print_msg(f"\nModel {m}: prec_idx={prec_idx}")
        print_msg(f"  grad_val[{prec_idx}]        = {grad_val[prec_idx]:>20.12e}")
        print_msg(f"  dense grad_cond_lik[{m}] = {gcl_ref[m]:>20.12e}")
        print_msg(f"  dense grad_quad_lik[{m}] = {gql_ref[m]:>20.12e}")

    np.savez(os.path.join(os.path.dirname(__file__), "debug_p1_ref.npz"),
             f=f_val, grad=grad_val, x=x_val,
             gcl_dense_ref=gcl_ref, gql_dense_ref=gql_ref)
    print_msg(f"\nSaved to debug_p1_ref.npz")


if __name__ == "__main__":
    if args.p1:
        run_p1()
    else:
        run_p2()
