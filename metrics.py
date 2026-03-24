"""
Evaluation metrics: HR@N and NDCG@N
Following the leave-one-out protocol from NeuMF (He et al., 2017).
"""

import numpy as np
import math


def hit_ratio(ranked_list, test_item):
    """1 if test_item appears in ranked_list, else 0."""
    return 1.0 if test_item in ranked_list else 0.0


def ndcg(ranked_list, test_item):
    """Normalised Discounted Cumulative Gain for a single test item."""
    if test_item in ranked_list:
        idx = ranked_list.index(test_item)
        return math.log(2) / math.log(idx + 2)   # position is 0-indexed
    return 0.0


def evaluate_model(model_scores_fn, test_data, n_items, top_k_list=(5, 10)):
    """
    Evaluate a model on the test set.

    Parameters
    ----------
    model_scores_fn : callable
        Takes (user_id, candidate_items) → numpy array of scores
        where candidate_items is a list [pos_item] + neg_items (201 items).
    test_data : list of (user, pos_item, neg_items)
    top_k_list : tuple of ints, e.g. (5, 10)

    Returns
    -------
    dict  {HR@k: float, NDCG@k: float, ...}
    """
    results = {f"HR@{k}":   [] for k in top_k_list}
    results.update({f"NDCG@{k}": [] for k in top_k_list})

    for user, pos_item, neg_items in test_data:
        candidates = [pos_item] + neg_items          # 201 items
        scores = model_scores_fn(user, candidates)   # shape (201,)

        # rank descending
        sorted_idx = np.argsort(-scores)
        ranked_items = [candidates[i] for i in sorted_idx]

        for k in top_k_list:
            top_k = ranked_items[:k]
            results[f"HR@{k}"].append(hit_ratio(top_k, pos_item))
            results[f"NDCG@{k}"].append(ndcg(top_k, pos_item))

    return {key: float(np.mean(vals)) for key, vals in results.items()}


def print_results(results, method_name="", epoch=None):
    prefix = f"[{method_name}]" if method_name else ""
    ep_str = f" Epoch {epoch}" if epoch is not None else ""
    print(f"{prefix}{ep_str}  " +
          "  ".join(f"{k}={v:.6f}" for k, v in sorted(results.items())))
