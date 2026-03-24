"""
Robustness analysis – replicates Figure 4 and Table 2 from the paper.

Tests HR@5 degradation when adversarial noise is injected on decoder
weights (W2) at different epsilon levels, for models trained with
various adversarial noise strengths.
"""

import numpy as np
import tensorflow as tf
import json
import os
import argparse


def inject_adversarial_noise_decoder(model, rating_matrix, n_users,
                                      test_data, n_items, epsilon):
    """
    Evaluate HR@5 when adversarial noise (ε) is added to W2 at test time.
    Uses fast-gradient method direction from the clean loss.
    """
    from metrics import evaluate_model

    # Compute gradient of loss w.r.t. W2 over all users (in batches)
    batch_size = 256
    user_ids   = list(range(n_users))
    grad_accum = tf.zeros_like(model.W2)
    n_batches  = 0

    for start in range(0, n_users, batch_size):
        batch_u = user_ids[start:start + batch_size]
        y   = tf.constant(rating_matrix[batch_u], dtype=tf.float32)
        uid = tf.constant(batch_u, dtype=tf.int32)
        with tf.GradientTape() as tape:
            logits = model.forward(y, uid)
            loss   = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
        grad = tape.gradient(loss, model.W2)
        grad_accum = grad_accum + grad
        n_batches += 1

    grad_accum = grad_accum / n_batches
    N2 = epsilon * grad_accum / (tf.norm(grad_accum) + 1e-8)

    # Evaluate with perturbed W2
    W2_orig = model.W2
    model.W2 = tf.Variable(W2_orig + N2)

    def score_fn(u, cands):
        return model.predict(u, cands, rating_matrix)

    from metrics import evaluate_model
    results = evaluate_model(score_fn, test_data, n_items)
    model.W2 = W2_orig   # restore
    return results.get("HR@5", 0.0)


def run_robustness_analysis(dataset_name, rating_matrix, test_data,
                             n_users, n_items, cfg):
    """
    Replicates Table 2 and Figure 4:
    - Train ACAE at epsilon = 1, 7, 15  (low / medium / high)
    - Test each with noise ε ∈ {0,1,...,15} on W2
    - Also test the no-adversarial-training baseline (ε_train = 0)
    """
    from model   import ACAE, CAE
    from trainer import train_autoencoder, train_acae

    epsilons_train = [0, 1, 7, 15]   # 0 = no adversarial training (CAE)
    epsilons_test  = list(range(0, 16))

    results_table = {}   # {eps_train: {eps_test: hr5}}

    for eps_train in epsilons_train:
        print(f"\n  Training with epsilon_train = {eps_train} ...")

        if eps_train == 0:
            # Plain CAE (no adversarial training)
            model = CAE(n_users, n_items,
                        latent_dim=cfg["latent_dim"], gamma=cfg["gamma"])
            train_autoencoder(
                model, rating_matrix, n_users, n_items, test_data,
                n_epochs=cfg["pretrain_epochs"],
                batch_size=cfg["batch_size"], lr=cfg["lr"],
                verbose_every=9999, seed=42)
        else:
            model = ACAE(n_users, n_items,
                         latent_dim=cfg["latent_dim"], gamma=cfg["gamma"],
                         lambda1=cfg["lambda1"], lambda2=cfg["lambda2"],
                         epsilon=float(eps_train))
            train_acae(
                model, rating_matrix, n_users, n_items, test_data,
                pretrain_epochs=cfg["pretrain_epochs"],
                pretrain_lr=cfg["lr"],
                adv_epochs=cfg["adv_epochs"],
                adv_verbose_every=9999,
                batch_size=cfg["batch_size"], seed=42)

        row = {}
        for eps_test in epsilons_test:
            if eps_test == 0:
                from metrics import evaluate_model
                def score_fn(u, c):
                    return model.predict(u, c, rating_matrix)
                res = evaluate_model(score_fn, test_data, n_items)
                row[eps_test] = res.get("HR@5", 0.0)
            else:
                row[eps_test] = inject_adversarial_noise_decoder(
                    model, rating_matrix, n_users, test_data, n_items,
                    epsilon=float(eps_test))
            print(f"    ε_test={eps_test:2d}  HR@5={row[eps_test]:.4f}")

        results_table[eps_train] = row

    # Print Table 2 equivalent
    print(f"\n\n--- Table 2 Equivalent ({dataset_name}) ---")
    print(f"{'ε_train':<12} {'HR@5 (clean)':>14} {'HR@5 (ε=8)':>12} {'% drop':>8}")
    print("─" * 50)
    for eps_train in epsilons_train:
        hr_clean = results_table[eps_train][0]
        hr_noisy = results_table[eps_train][8]
        pct_drop = (hr_clean - hr_noisy) / hr_clean * 100
        label = "W/O" if eps_train == 0 else f"ε={eps_train}"
        print(f"{label:<12} {hr_clean:>14.4f} {hr_noisy:>12.4f} {pct_drop:>7.2f}%")

    # Save
    os.makedirs("results", exist_ok=True)
    out = f"results/{dataset_name}_robustness.json"
    with open(out, "w") as f:
        # Convert int keys to strings for JSON
        json.dump({str(k): {str(k2): v2 for k2, v2 in v.items()}
                   for k, v in results_table.items()}, f, indent=2)
    print(f"\nRobustness results saved → {out}")
    return results_table


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_utils import (load_movielens, load_filmtrust, load_ciao,
                             leave_one_out_split, get_train_matrix_dense)
    from run_experiments import CONFIGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movielens",
                        choices=["movielens", "filmtrust", "ciao"])
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(42)

    name = args.dataset
    if name == "movielens":
        df, n_users, n_items = load_movielens()
    elif name == "filmtrust":
        df, n_users, n_items = load_filmtrust()
    else:
        df, n_users, n_items = load_ciao()

    train_data, test_data = leave_one_out_split(df, n_users, n_items, seed=42)
    rating_matrix = get_train_matrix_dense(train_data, n_users, n_items)

    run_robustness_analysis(name, rating_matrix, test_data,
                             n_users, n_items, CONFIGS[name])
