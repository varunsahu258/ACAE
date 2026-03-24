"""
Training routines — RTX 4500 Ada optimised.

Key CUDA speedups vs the original:
  1. GPU-pinned rating matrix  — tf.constant on GPU at startup, never
     re-uploaded.  Batch slicing done via tf.gather on the GPU.
  2. tf.data pipeline          — prefetch(AUTOTUNE) overlaps CPU data prep
     with GPU compute.  Eliminates the Python loop overhead that was the
     main bottleneck in the original batch iterator.
  3. Compiled train_step       — each model exposes a @tf.function-decorated
     train_step / adv_train_step that is called directly here; no tape
     overhead inside the Python training loop.
  4. Large batches             — device_config.BATCH[] provides dataset/model-
     specific sizes tuned for 24 GB VRAM.
  5. tqdm progress bars        — show per-epoch GPU throughput (samples/sec).
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from metrics import evaluate_model, print_results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pin_rating_matrix(rating_matrix, device):
    """Upload the dense rating matrix to GPU once; reuse for every batch."""
    with tf.device(device):
        return tf.constant(rating_matrix, dtype=tf.float32)


def _ae_dataset(n_users, batch_size, seed=42):
    """
    tf.data pipeline for auto-encoder models.
    Yields shuffled user-id batches; the actual rating rows are gathered
    on-GPU from the pinned matrix.
    """
    ds = tf.data.Dataset.range(n_users)
    ds = ds.shuffle(n_users, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _bpr_dataset(user_pos_arrays, n_items, batch_size, seed=42):
    """
    tf.data pipeline for BPR-style models.
    Pre-samples one positive + one negative per user per epoch;
    returns (users, pos_items, neg_items) tensor batches.
    """
    rng = np.random.RandomState(seed)

    def generator():
        users_with_pos = [u for u, pos in enumerate(user_pos_arrays)
                          if len(pos) > 0]
        rng.shuffle(users_with_pos)
        for u in users_with_pos:
            pos  = user_pos_arrays[u]
            pi   = pos[rng.randint(len(pos))]
            ni   = rng.randint(n_items)
            while ni in set(pos):       # rejection sample
                ni = rng.randint(n_items)
            yield np.int32(u), np.int32(pi), np.int32(ni)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# GPU-accelerated evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_gpu(model, rating_matrix_gpu, test_data, n_items,
                       device, batch_size=512):
    """
    Batch evaluation on GPU: scores all test users in chunks instead of
    one-by-one, keeping tensors on device throughout.
    """
    import math
    from metrics import hit_ratio, ndcg

    results = {"HR@5": [], "HR@10": [], "NDCG@5": [], "NDCG@10": []}

    # Process test users in GPU batches
    n_test  = len(test_data)
    for start in range(0, n_test, batch_size):
        chunk = test_data[start:start + batch_size]
        uids  = [td[0] for td in chunk]

        with tf.device(device):
            y_batch  = tf.gather(rating_matrix_gpu, uids)        # (B, I)
            uid_t    = tf.constant(uids, dtype=tf.int32)
            scores_b = model.predict_batch(y_batch, uid_t)       # (B, I)

        scores_np = scores_b.numpy()

        for i, (user, pos_item, neg_items) in enumerate(chunk):
            candidates = [pos_item] + neg_items                  # 201 items
            sc         = scores_np[i][candidates]
            order      = np.argsort(-sc)
            ranked     = [candidates[j] for j in order]

            for k in (5, 10):
                top_k = ranked[:k]
                results[f"HR@{k}"].append(1.0 if pos_item in top_k else 0.0)
                results[f"NDCG@{k}"].append(
                    (math.log(2) / math.log(top_k.index(pos_item) + 2))
                    if pos_item in top_k else 0.0)

    return {k: float(np.mean(v)) for k, v in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. ItemPop
# ─────────────────────────────────────────────────────────────────────────────

def train_itempop(model, train_data, n_items):
    model.fit(train_data, n_items)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MF-BPR / AMF
# ─────────────────────────────────────────────────────────────────────────────

def train_bpr_model(model, train_data, n_users, n_items,
                    test_data, rating_matrix,
                    n_epochs=1000, batch_size=4096,
                    lr=0.001, verbose_every=100,
                    device="/GPU:0", seed=42):

    optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
    rating_matrix_gpu = _pin_rating_matrix(rating_matrix, device)

    # Build per-user positive lists (numpy arrays for fast sampling)
    user_pos = [[] for _ in range(n_users)]
    for u, i, r in train_data:
        if r > 0:
            user_pos[u].append(i)
    user_pos = [np.array(p, dtype=np.int32) for p in user_pos]

    best_hr5, best_results = 0.0, {}

    for epoch in tqdm(range(1, n_epochs + 1),
                      desc=f"{type(model).__name__}", unit="epoch"):
        ds = _bpr_dataset(user_pos, n_items, batch_size, seed=seed + epoch)

        with tf.device(device):
            for u_t, pi_t, ni_t in ds:
                model.train_step(u_t, pi_t, ni_t, optimiser)

        if epoch % verbose_every == 0:
            results = evaluate_model_gpu(
                model, rating_matrix_gpu, test_data, n_items, device)
            print_results(results, method_name=type(model).__name__, epoch=epoch)
            if results.get("HR@5", 0) > best_hr5:
                best_hr5, best_results = results["HR@5"], results

    return best_results


# ─────────────────────────────────────────────────────────────────────────────
# 3. CDAE / CAE
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(model, rating_matrix, n_users, n_items,
                      test_data,
                      n_epochs=1000, batch_size=2048,
                      lr=0.001, verbose_every=100,
                      device="/GPU:0", seed=42):

    optimiser         = tf.keras.optimizers.Adam(learning_rate=lr)
    rating_matrix_gpu = _pin_rating_matrix(rating_matrix, device)

    best_hr5, best_results = 0.0, {}

    for epoch in tqdm(range(1, n_epochs + 1),
                      desc=type(model).__name__, unit="epoch"):
        ds = _ae_dataset(n_users, batch_size, seed=seed + epoch)

        with tf.device(device):
            for uid_batch in ds:
                y_batch = tf.gather(rating_matrix_gpu, uid_batch)
                model.train_step(y_batch, uid_batch, optimiser)

        if epoch % verbose_every == 0:
            results = evaluate_model_gpu(
                model, rating_matrix_gpu, test_data, n_items, device)
            print_results(results, method_name=type(model).__name__, epoch=epoch)
            if results.get("HR@5", 0) > best_hr5:
                best_hr5, best_results = results["HR@5"], results

    return best_results


# ─────────────────────────────────────────────────────────────────────────────
# 4. NeuMF
# ─────────────────────────────────────────────────────────────────────────────

def train_neumf(model, train_data, n_users, n_items,
                test_data, rating_matrix,
                n_epochs=1000, batch_size=4096,
                lr=0.001, verbose_every=100,
                num_neg=4, device="/GPU:0", seed=42):

    optimiser         = tf.keras.optimizers.Adam(learning_rate=lr)
    rating_matrix_gpu = _pin_rating_matrix(rating_matrix, device)

    user_pos = [[] for _ in range(n_users)]
    for u, i, r in train_data:
        if r > 0:
            user_pos[u].append(i)
    user_pos = [np.array(p, dtype=np.int32) for p in user_pos]

    rng = np.random.RandomState(seed)

    def _build_epoch_samples():
        us, its, ls = [], [], []
        for u in range(n_users):
            if len(user_pos[u]) == 0:
                continue
            pos_set = set(user_pos[u].tolist())
            for pi in user_pos[u]:
                us.append(u);  its.append(pi);  ls.append(1.0)
                for _ in range(num_neg):
                    ni = rng.randint(n_items)
                    while ni in pos_set:
                        ni = rng.randint(n_items)
                    us.append(u);  its.append(ni);  ls.append(0.0)
        idx = np.arange(len(us))
        rng.shuffle(idx)
        return (np.array(us,  dtype=np.int32)[idx],
                np.array(its, dtype=np.int32)[idx],
                np.array(ls,  dtype=np.float32)[idx])

    best_hr5, best_results = 0.0, {}

    for epoch in tqdm(range(1, n_epochs + 1), desc="NeuMF", unit="epoch"):
        us, its, ls = _build_epoch_samples()

        ds = (tf.data.Dataset
              .from_tensor_slices((us, its, ls))
              .batch(batch_size, drop_remainder=False)
              .prefetch(tf.data.AUTOTUNE))

        with tf.device(device):
            for u_t, i_t, l_t in ds:
                model.train_step(u_t, i_t, l_t, optimiser)

        if epoch % verbose_every == 0:
            results = evaluate_model_gpu(
                model, rating_matrix_gpu, test_data, n_items, device)
            print_results(results, method_name="NeuMF", epoch=epoch)
            if results.get("HR@5", 0) > best_hr5:
                best_hr5, best_results = results["HR@5"], results

    return best_results


# ─────────────────────────────────────────────────────────────────────────────
# 5. ACAE  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def train_acae(model, rating_matrix, n_users, n_items,
               test_data,
               pretrain_epochs=500, pretrain_lr=0.001,
               adv_epochs=500,      adv_verbose_every=50,
               batch_size=2048,     device="/GPU:0", seed=42):
    """
    Algorithm 1 — two-stage ACAE training, fully GPU-resident.

    Stage 1: Pre-train CAE  (Adam, fixed LR)
    Stage 2: Adversarial fine-tune  (Adagrad)

    Returns (best_results, trace) where trace entries are
    (epoch, HR@5, NDCG@5, HR@10, NDCG@10).  Pretrain epochs are stored
    with negative indices so both stages can be plotted together (Figure 3).
    Pass adv_epochs=0 to train CAE only (no adversarial stage).
    """
    rating_matrix_gpu = _pin_rating_matrix(rating_matrix, device)

    # ── Stage 1: Pre-training ─────────────────────────────────────────────────
    print("\n[ACAE] Stage 1: Pre-training ...")
    opt_pre = tf.keras.optimizers.Adam(learning_rate=pretrain_lr)

    best_hr5, best_results = 0.0, {}
    trace = []

    for epoch in tqdm(range(1, pretrain_epochs + 1),
                      desc="ACAE-pretrain", unit="epoch"):
        ds = _ae_dataset(n_users, batch_size, seed=seed + epoch)
        with tf.device(device):
            for uid_batch in ds:
                y_batch = tf.gather(rating_matrix_gpu, uid_batch)
                model.train_step(y_batch, uid_batch, opt_pre)

        if epoch % 100 == 0:
            res = evaluate_model_gpu(
                model, rating_matrix_gpu, test_data, n_items, device)
            print_results(res, method_name="ACAE-pretrain", epoch=epoch)
            # Negative epoch index distinguishes pretrain from adv in the trace
            trace.append((-pretrain_epochs + epoch,
                          res.get("HR@5",    0),
                          res.get("NDCG@5",  0),
                          res.get("HR@10",   0),
                          res.get("NDCG@10", 0)))
            if res.get("HR@5", 0) > best_hr5:
                best_hr5, best_results = res["HR@5"], res

    if adv_epochs == 0:
        return best_results, trace

    # ── Stage 2: Adversarial training ─────────────────────────────────────────
    print("\n[ACAE] Stage 2: Adversarial training ...")
    opt_adv = tf.keras.optimizers.Adagrad(learning_rate=pretrain_lr)

    for epoch in tqdm(range(1, adv_epochs + 1),
                      desc="ACAE-adv", unit="epoch"):
        ds = _ae_dataset(n_users, batch_size, seed=seed + epoch + pretrain_epochs)
        with tf.device(device):
            for uid_batch in ds:
                y_batch = tf.gather(rating_matrix_gpu, uid_batch)
                model.adv_train_step(y_batch, uid_batch, opt_adv)

        if epoch % adv_verbose_every == 0:
            results = evaluate_model_gpu(
                model, rating_matrix_gpu, test_data, n_items, device)
            print_results(results, method_name="ACAE", epoch=epoch)
            trace.append((epoch,
                          results.get("HR@5",    0),
                          results.get("NDCG@5",  0),
                          results.get("HR@10",   0),
                          results.get("NDCG@10", 0)))
            if results.get("HR@5", 0) > best_hr5:
                best_hr5, best_results = results["HR@5"], results

    return best_results, trace
