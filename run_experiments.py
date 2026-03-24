"""
Main experiment script — RTX 4500 Ada / Windows 11 edition.
Full replication of Table 3 from Yuan et al. (2018).

Usage (PowerShell / CMD):
    python run_experiments.py --dataset movielens
    python run_experiments.py --dataset filmtrust
    python run_experiments.py --dataset ciao
    python run_experiments.py --all
"""

import argparse
import os
import json
import time

# ── GPU setup MUST come before any other TF import ───────────────────────────
from device_config import configure, get_device, optimal_batch, print_device_info
gpu_found = configure(memory_growth=True, enable_tf32=True,
                      enable_xla=True, verbose=True)

import numpy as np
import tensorflow as tf

from data_utils  import (load_movielens, load_filmtrust, load_ciao,
                          leave_one_out_split, get_train_matrix_dense)
from model       import CAE, ACAE
from baselines   import ItemPop, MFBPR, CDAE, NeuMF, AMF
from trainer     import (train_itempop, train_bpr_model,
                          train_autoencoder, train_neumf, train_acae,
                          evaluate_model_gpu, _pin_rating_matrix)
from metrics     import evaluate_model, print_results


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (match paper Table 3 settings)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "movielens": dict(
        latent_dim=50,   gamma=0.01,  lr=0.001,
        pretrain_epochs=500, adv_epochs=1000,
        epsilon=1.0,  lambda1=1.0, lambda2=10.0,
        cdae_corruption=0.5,
        mf_dim=16,    mlp_layers=(64, 32, 16, 8),
        bpr_latent=64, amf_epsilon=0.5, amf_lambda=1.0,
    ),
    "filmtrust": dict(
        latent_dim=50,   gamma=0.001, lr=0.001,
        pretrain_epochs=500, adv_epochs=1000,
        epsilon=2.0,  lambda1=1.0, lambda2=10.0,
        cdae_corruption=0.5,
        mf_dim=16,    mlp_layers=(64, 32, 16, 8),
        bpr_latent=64, amf_epsilon=0.5, amf_lambda=1.0,
    ),
    "ciao": dict(
        latent_dim=50,   gamma=0.001, lr=0.001,
        pretrain_epochs=500, adv_epochs=1000,
        epsilon=0.5,  lambda1=1.0, lambda2=1.0,
        cdae_corruption=0.5,
        mf_dim=16,    mlp_layers=(64, 32, 16, 8),
        bpr_latent=64, amf_epsilon=0.3, amf_lambda=1.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(name: str) -> dict:
    print(f"\n{'='*66}")
    print(f"  Dataset : {name.upper()}")
    print(f"  Device  : {get_device()}")
    print(f"{'='*66}")

    cfg    = CONFIGS[name]
    device = get_device()
    t0     = time.time()

    # ── Load & split ──────────────────────────────────────────────────────────
    if name == "movielens":
        df, n_users, n_items = load_movielens()
    elif name == "filmtrust":
        df, n_users, n_items = load_filmtrust()
    else:
        df, n_users, n_items = load_ciao()

    print(f"  Users {n_users:,}  Items {n_items:,}  Ratings {len(df):,}")

    train_data, test_data = leave_one_out_split(
        df, n_users, n_items, num_negatives=200, seed=42)
    rating_matrix = get_train_matrix_dense(train_data, n_users, n_items)
    print(f"  Train interactions : {len(train_data):,}")
    print(f"  Test users         : {len(test_data):,}")

    # Pin rating matrix to GPU once — reused by every model
    rm_gpu = _pin_rating_matrix(rating_matrix, device)

    all_results = {}

    # ── Batch sizes from device_config ──────────────────────────────────────────
    bs_ae  = optimal_batch(name, "acae",  fallback=512)
    bs_bpr = optimal_batch(name, "bpr",   fallback=1024)
    bs_nmf = optimal_batch(name, "neumf", fallback=1024)
    print(f"  Batch sizes → AE:{bs_ae}  BPR:{bs_bpr}  NeuMF:{bs_nmf}\n")

    # ── 1. ItemPop ────────────────────────────────────────────────────────────
    print("--- ItemPop ---")
    itempop = ItemPop()
    train_itempop(itempop, train_data, n_items)
    ip_res = evaluate_model(
        lambda u, c: itempop.predict(u, c), test_data, n_items)
    print_results(ip_res, "ItemPop")
    all_results["ItemPop"] = ip_res

    # ── 2. MF-BPR ─────────────────────────────────────────────────────────────
    print("\n--- MF-BPR ---")
    mfbpr   = MFBPR(n_users, n_items,
                    latent_dim=cfg["bpr_latent"], reg=cfg["gamma"])
    bpr_res = train_bpr_model(
        mfbpr, train_data, n_users, n_items, test_data, rating_matrix,
        n_epochs=1000, batch_size=bs_bpr, lr=cfg["lr"],
        verbose_every=200, device=device, seed=42)
    all_results["MF-BPR"] = bpr_res

    # ── 3. CDAE ───────────────────────────────────────────────────────────────
    print("\n--- CDAE ---")
    cdae     = CDAE(n_users, n_items,
                    latent_dim=cfg["latent_dim"], gamma=cfg["gamma"],
                    corruption=cfg["cdae_corruption"])
    cdae_res = train_autoencoder(
        cdae, rating_matrix, n_users, n_items, test_data,
        n_epochs=1000, batch_size=bs_ae, lr=cfg["lr"],
        verbose_every=200, device=device, seed=42)
    all_results["CDAE"] = cdae_res

    # ── 4. NeuMF ──────────────────────────────────────────────────────────────
    print("\n--- NeuMF ---")
    neumf     = NeuMF(n_users, n_items,
                      mf_dim=cfg["mf_dim"], layers=cfg["mlp_layers"])
    neumf_res = train_neumf(
        neumf, train_data, n_users, n_items, test_data, rating_matrix,
        n_epochs=1000, batch_size=bs_nmf, lr=cfg["lr"],
        verbose_every=200, num_neg=4, device=device, seed=42)
    all_results["NeuMF"] = neumf_res

    # ── 5. AMF ────────────────────────────────────────────────────────────────
    print("\n--- AMF ---")
    amf     = AMF(n_users, n_items,
                  latent_dim=cfg["bpr_latent"], reg=cfg["gamma"],
                  epsilon=cfg["amf_epsilon"], lam=cfg["amf_lambda"])
    amf_res = train_bpr_model(
        amf, train_data, n_users, n_items, test_data, rating_matrix,
        n_epochs=1000, batch_size=bs_bpr, lr=cfg["lr"],
        verbose_every=200, device=device, seed=42)
    all_results["AMF"] = amf_res

    # ── 6. CAE  (pre-train only — used for Figure 3 comparison) ──────────────
    print("\n--- CAE ---")
    cae      = CAE(n_users, n_items,
                   latent_dim=cfg["latent_dim"], gamma=cfg["gamma"])
    cae_res, cae_trace = train_acae(
        cae, rating_matrix, n_users, n_items, test_data,
        pretrain_epochs=cfg["pretrain_epochs"], pretrain_lr=cfg["lr"],
        adv_epochs=0,
        batch_size=bs_ae, device=device, seed=42)
    all_results["CAE"] = cae_res

    # ── 7. ACAE ───────────────────────────────────────────────────────────────
    print("\n--- ACAE ---")
    acae     = ACAE(n_users, n_items,
                    latent_dim=cfg["latent_dim"], gamma=cfg["gamma"],
                    lambda1=cfg["lambda1"], lambda2=cfg["lambda2"],
                    epsilon=cfg["epsilon"])
    acae_res, trace = train_acae(
        acae, rating_matrix, n_users, n_items, test_data,
        pretrain_epochs=cfg["pretrain_epochs"], pretrain_lr=cfg["lr"],
        adv_epochs=cfg["adv_epochs"],           adv_verbose_every=50,
        batch_size=bs_ae, device=device, seed=42)
    all_results["ACAE"] = acae_res

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed/60:.1f} min")

    # ── Print Table 3 for this dataset ───────────────────────────────────────
    _print_table(name, all_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = f"results/{name}_results.json"
    with open(out, "w") as f:
        json.dump({"dataset": name, "n_users": n_users, "n_items": n_items,
                   "elapsed_min": round(elapsed / 60, 1),
                   "results": all_results,
                   "cae_trace": cae_trace, "acae_trace": trace}, f, indent=2)
    print(f"  Results saved → {out}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print Table 3
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(dataset_name, results):
    metrics = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]
    methods = ["ItemPop", "MF-BPR", "CDAE", "NeuMF", "AMF", "ACAE"]
    print(f"\n{'─'*64}")
    print(f"  TABLE 3  –  {dataset_name.upper()}")
    print(f"{'─'*64}")
    header = f"{'Method':<12}" + "".join(f"{m:>13}" for m in metrics)
    print(header)
    print("─" * len(header))
    for method in methods:
        r   = results.get(method, {})
        row = f"{method:<12}" + "".join(f"{r.get(m,0):>13.6f}" for m in metrics)
        print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACAE Replication – RTX 4500 Ada")
    parser.add_argument("--dataset", choices=["movielens", "filmtrust", "ciao"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print_device_info()

    if not gpu_found:
        print("WARNING: No CUDA GPU detected. See setup_cuda_windows.md")

    datasets = (["movielens", "filmtrust", "ciao"] if args.all
                else ([args.dataset] if args.dataset else []))
    if not datasets:
        parser.print_help(); exit(1)

    combined = {}
    for ds in datasets:
        combined[ds] = run_dataset(ds)

    if len(datasets) > 1:
        print(f"\n{'='*66}\n  COMBINED TABLE 3\n{'='*66}")
        for ds in datasets:
            _print_table(ds, combined[ds])
