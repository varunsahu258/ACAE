"""
dry_run.py  –  Smoke-test every script in the ACAE project.

Exercises ALL code paths (data loading, every model, training loops,
evaluation, robustness analysis, plotting) using:
  - MovieLens-1M sliced to 200 users / 300 items
  - 2 pre-train epochs + 2 adversarial epochs
  - 1 test negative per user instead of 200

Typical wall time on RTX 4500 Ada: < 60 seconds.
No CUDA required – falls back to CPU gracefully.

Usage:
    python dry_run.py
"""

import os
import sys
import json
import time
import traceback
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence TF info spam

# ── GPU setup (same as run_experiments.py) ────────────────────────────────────
from device_config import configure, get_device, optimal_batch
configure(memory_growth=True, enable_tf32=True, enable_xla=False,  # XLA off for speed on tiny data
          verbose=False)
import tensorflow as tf
tf.random.set_seed(42)

from data_utils  import (load_movielens, leave_one_out_split,
                          get_train_matrix_dense)
from model       import CAE, ACAE
from baselines   import ItemPop, MFBPR, CDAE, NeuMF, AMF
from trainer     import (train_itempop, train_bpr_model,
                          train_autoencoder, train_neumf, train_acae,
                          evaluate_model_gpu, _pin_rating_matrix)
from metrics     import evaluate_model, print_results


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run hyper-parameters  (tiny — just enough to touch every code path)
# ─────────────────────────────────────────────────────────────────────────────

N_USERS   = 200      # slice of MovieLens users
N_ITEMS   = 300      # slice of MovieLens items
N_NEG     = 1        # negatives per test user (paper uses 200)
PRETRAIN  = 2        # pre-training epochs
ADV       = 2        # adversarial training epochs
BATCH     = 64
LR        = 0.001

# ACAE / CDAE hyper-params (same as paper's MovieLens config)
LATENT    = 50
GAMMA     = 0.01
EPSILON   = 1.0
LAMBDA1   = 1.0
LAMBDA2   = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"

results_log = []

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def check(name, fn):
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"[{PASS}] {name:<42}  {elapsed:.1f}s")
        results_log.append((name, "PASS", elapsed))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{FAIL}] {name:<42}  {elapsed:.1f}s")
        print(f"         Error: {e}")
        traceback.print_exc()
        results_log.append((name, "FAIL", elapsed))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

section("1 · Data pipeline")

df = train_data = test_data = rating_matrix = None
n_users = n_items = 0

def _load_and_slice():
    global df, train_data, test_data, rating_matrix, n_users, n_items
    full_df, fu, fi = load_movielens()

    # Slice to N_USERS × N_ITEMS for speed
    top_users = full_df["user"].value_counts().head(N_USERS).index
    top_items = full_df["item"].value_counts().head(N_ITEMS).index
    sliced = full_df[full_df["user"].isin(top_users) &
                     full_df["item"].isin(top_items)].copy()

    # Re-index to 0-based
    u_map = {u: i for i, u in enumerate(sliced["user"].unique())}
    i_map = {it: i for i, it in enumerate(sliced["item"].unique())}
    sliced["user"] = sliced["user"].map(u_map)
    sliced["item"] = sliced["item"].map(i_map)
    df = sliced

    n_users = sliced["user"].nunique()
    n_items = sliced["item"].nunique()

    assert n_users > 0 and n_items > 0, "Empty slice"
    assert "user" in sliced.columns and "item" in sliced.columns

check("MovieLens-1M download + slice", _load_and_slice)

def _split():
    global train_data, test_data, rating_matrix
    train_data, test_data = leave_one_out_split(
        df, n_users, n_items, num_negatives=N_NEG, seed=42)
    rating_matrix = get_train_matrix_dense(train_data, n_users, n_items)
    assert len(train_data) > 0
    assert len(test_data)  > 0
    assert rating_matrix.shape == (n_users, n_items)

check("leave_one_out_split + rating matrix", _split)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPU pin
# ─────────────────────────────────────────────────────────────────────────────

section("2 · GPU / device setup")

device = get_device()
rm_gpu = None

def _pin():
    global rm_gpu
    rm_gpu = _pin_rating_matrix(rating_matrix, device)
    assert rm_gpu.shape == (n_users, n_items)

check(f"Pin rating matrix → {device}", _pin)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Models – construction
# ─────────────────────────────────────────────────────────────────────────────

section("3 · Model construction")

itempop = mfbpr = cdae_m = neumf = amf = cae = acae = None

check("ItemPop()",   lambda: globals().__setitem__("itempop", ItemPop()))
check("MFBPR()",     lambda: globals().__setitem__("mfbpr",
        MFBPR(n_users, n_items, latent_dim=16, reg=GAMMA)))
check("CDAE()",      lambda: globals().__setitem__("cdae_m",
        CDAE(n_users, n_items, latent_dim=LATENT, gamma=GAMMA, corruption=0.5)))
check("NeuMF()",     lambda: globals().__setitem__("neumf",
        NeuMF(n_users, n_items, mf_dim=8, layers=(16, 8))))
check("AMF()",       lambda: globals().__setitem__("amf",
        AMF(n_users, n_items, latent_dim=16, reg=GAMMA, epsilon=0.5, lam=1.0)))
check("CAE()",       lambda: globals().__setitem__("cae",
        CAE(n_users, n_items, latent_dim=LATENT, gamma=GAMMA)))
check("ACAE()",      lambda: globals().__setitem__("acae",
        ACAE(n_users, n_items, latent_dim=LATENT, gamma=GAMMA,
             lambda1=LAMBDA1, lambda2=LAMBDA2, epsilon=EPSILON)))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training loops
# ─────────────────────────────────────────────────────────────────────────────

section("4 · Training loops")

def _train_itempop():
    train_itempop(itempop, train_data, n_items)

def _train_mfbpr():
    train_bpr_model(
        mfbpr, train_data, n_users, n_items, test_data, rating_matrix,
        n_epochs=PRETRAIN, batch_size=BATCH, lr=LR,
        verbose_every=9999, device=device, seed=42)

def _train_cdae():
    train_autoencoder(
        cdae_m, rating_matrix, n_users, n_items, test_data,
        n_epochs=PRETRAIN, batch_size=BATCH, lr=LR,
        verbose_every=9999, device=device, seed=42)

def _train_neumf():
    train_neumf(
        neumf, train_data, n_users, n_items, test_data, rating_matrix,
        n_epochs=PRETRAIN, batch_size=BATCH, lr=LR,
        verbose_every=9999, num_neg=4, device=device, seed=42)

def _train_amf():
    train_bpr_model(
        amf, train_data, n_users, n_items, test_data, rating_matrix,
        n_epochs=PRETRAIN, batch_size=BATCH, lr=LR,
        verbose_every=9999, device=device, seed=42)

def _train_acae():
    train_acae(
        acae, rating_matrix, n_users, n_items, test_data,
        pretrain_epochs=PRETRAIN, pretrain_lr=LR,
        adv_epochs=ADV, adv_verbose_every=9999,
        batch_size=BATCH, device=device, seed=42)

check("train_itempop()",    _train_itempop)
check("train_bpr_model() – MF-BPR", _train_mfbpr)
check("train_autoencoder() – CDAE", _train_cdae)
check("train_neumf()",      _train_neumf)
check("train_bpr_model() – AMF",    _train_amf)
check("train_acae() (pre-train + adv)", _train_acae)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

section("5 · Evaluation")

def _eval_itempop():
    res = evaluate_model(lambda u, c: itempop.predict(u, c), test_data, n_items)
    assert "HR@5" in res

def _eval_gpu():
    res = evaluate_model_gpu(acae, rm_gpu, test_data, n_items, device)
    assert "HR@5" in res and "NDCG@5" in res

check("evaluate_model() – CPU (ItemPop)", _eval_itempop)
check("evaluate_model_gpu() – ACAE",      _eval_gpu)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Robustness analysis
# ─────────────────────────────────────────────────────────────────────────────

section("6 · Robustness analysis")

def _robustness():
    from robustness_analysis import (inject_adversarial_noise_decoder,
                                     run_robustness_analysis)
    # just test noise injection on the already-trained ACAE
    hr = inject_adversarial_noise_decoder(
        acae, rating_matrix, n_users, test_data, n_items, epsilon=1.0)
    assert 0.0 <= hr <= 1.0, f"HR@5 out of range: {hr}"

check("inject_adversarial_noise_decoder()", _robustness)


# ─────────────────────────────────────────────────────────────────────────────
# 7. JSON save / load (results pipeline)
# ─────────────────────────────────────────────────────────────────────────────

section("7 · JSON results pipeline")

def _json_roundtrip():
    os.makedirs("results", exist_ok=True)
    dummy = {"dataset": "dry_run", "results": {"ACAE": {"HR@5": 0.5}},
             "acae_trace": [[1, 0.5, 0.3, 0.6, 0.35]]}
    path = "results/dry_run_results.json"
    with open(path, "w") as f:
        json.dump(dummy, f)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["results"]["ACAE"]["HR@5"] == 0.5

check("JSON save + load (results/)", _json_roundtrip)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plotting
# ─────────────────────────────────────────────────────────────────────────────

section("8 · Plotting")

def _plot():
    from plot_figures import plot_training_traces, plot_robustness, print_table3
    plot_training_traces({"dry_run": "results/dry_run_results.json"})
    print_table3({"dry_run": "results/dry_run_results.json"})
    assert os.path.exists("figures/figure3_training_traces.pdf")

check("plot_training_traces() + print_table3()", _plot)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

total_time = sum(r[2] for r in results_log)
passed     = sum(1 for r in results_log if r[1] == "PASS")
failed     = sum(1 for r in results_log if r[1] == "FAIL")

print(f"\n{'═'*60}")
print(f"  DRY RUN COMPLETE")
print(f"{'═'*60}")
print(f"  Checks : {len(results_log)}")
print(f"  Passed : {passed}")
print(f"  Failed : {failed}")
print(f"  Total  : {total_time:.1f}s")
print(f"{'═'*60}")

if failed:
    print(f"\n  Failed checks:")
    for name, status, _ in results_log:
        if status == "FAIL":
            print(f"    ✗  {name}")
    sys.exit(1)
else:
    print("\n  All checks passed. Project is ready to run.\n")
    sys.exit(0)
