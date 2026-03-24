"""
Plot figures from the paper using saved result JSONs.

Reproduces:
  - Figure 3: Training traces (HR@5 and NDCG@5 for CAE vs ACAE)
  - Figure 4: Robustness against adversarial noise
  - Table 2:  Printed to console
"""

import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "CAE":    "#1f77b4",
    "ACAE":   "#ff7f0e",
    "W/O":    "#1f77b4",
    "eps1":   "#ff7f0e",
    "eps7":   "#2ca02c",
    "eps15":  "#d62728",
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Training traces
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_traces(results_files):
    """
    Reproduces Figure 3 from the paper: training traces for CAE and ACAE.

    results_files: dict {dataset_name: path_to_json}
    Each JSON must contain:
      - 'cae_trace':  [(epoch, HR@5, NDCG@5, HR@10, NDCG@10), ...]  (pretrain only)
      - 'acae_trace': [(epoch, HR@5, NDCG@5, HR@10, NDCG@10), ...]  (both stages)
    Epoch values in cae_trace/acae_trace use negative numbers for the pretrain
    stage and positive for the adversarial stage so both can share an x-axis.
    """
    datasets   = list(results_files.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 8))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(datasets):
        with open(results_files[ds]) as f:
            data = json.load(f)

        cae_trace  = data.get("cae_trace",  [])
        acae_trace = data.get("acae_trace", [])
        if not acae_trace:
            continue

        cae_epochs  = [t[0] for t in cae_trace]
        cae_hr5     = [t[1] for t in cae_trace]
        cae_ndcg5   = [t[2] for t in cae_trace]

        acae_epochs = [t[0] for t in acae_trace]
        acae_hr5    = [t[1] for t in acae_trace]
        acae_ndcg5  = [t[2] for t in acae_trace]

        # HR@5
        ax = axes[0][col]
        if cae_epochs:
            ax.plot(cae_epochs, cae_hr5,
                    color=COLORS["CAE"], label="CAE", linewidth=1.5)
        ax.plot(acae_epochs, acae_hr5,
                color=COLORS["ACAE"], label="ACAE", linewidth=1.5)
        ax.set_title(f"({chr(97+col)}) HR@5 for {ds.capitalize()}", fontsize=10)
        ax.set_xlabel("Epochs"); ax.set_ylabel("HR@5")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # NDCG@5
        ax = axes[1][col]
        if cae_epochs:
            ax.plot(cae_epochs, cae_ndcg5,
                    color=COLORS["CAE"], label="CAE", linewidth=1.5)
        ax.plot(acae_epochs, acae_ndcg5,
                color=COLORS["ACAE"], label="ACAE", linewidth=1.5)
        ax.set_title(f"({chr(97+n_datasets+col)}) NDCG@5 for {ds.capitalize()}",
                     fontsize=10)
        ax.set_xlabel("Epochs"); ax.set_ylabel("NDCG@5")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 3: Adversarial Training Traces", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = "figures/figure3_training_traces.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Robustness against adversarial noise
# ─────────────────────────────────────────────────────────────────────────────

def plot_robustness(robustness_files):
    """
    robustness_files: dict {dataset_name: path_to_robustness_json}
    """
    datasets   = [d for d in ["movielens", "filmtrust"] if d in robustness_files]
    n_datasets = len(datasets)
    if n_datasets == 0:
        print("No robustness data found.")
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 4))
    if n_datasets == 1:
        axes = [axes]

    eps_labels = {
        "0":  ("W/O Adv Training", COLORS["W/O"],   "o"),
        "1":  ("Epsilon = 1",       COLORS["eps1"],  "s"),
        "7":  ("Epsilon = 7",       COLORS["eps7"],  "^"),
        "15": ("Epsilon = 15",      COLORS["eps15"], "D"),
    }

    for ax, ds in zip(axes, datasets):
        with open(robustness_files[ds]) as f:
            data = json.load(f)

        for train_eps, (label, color, marker) in eps_labels.items():
            if train_eps not in data:
                continue
            row = data[train_eps]
            x = sorted(int(k) for k in row.keys())
            y = [row[str(k)] for k in x]
            ax.plot(x, y, label=label, color=color, marker=marker,
                    markersize=4, linewidth=1.5)

        ax.set_title(ds.capitalize(), fontsize=11)
        ax.set_xlabel("ε (test noise level)"); ax.set_ylabel("HR@5")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 4: Robustness Against Adversarial Noise",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = "figures/figure4_robustness.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 reprint from saved JSON
# ─────────────────────────────────────────────────────────────────────────────

def print_table3(results_files):
    metrics = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]
    methods = ["ItemPop", "MF-BPR", "CDAE", "NeuMF", "AMF", "ACAE"]
    datasets = list(results_files.keys())

    print(f"\n{'='*80}")
    print("  TABLE 3: Comparison with Baselines")
    print(f"{'='*80}")

    col_w = 11
    print(f"{'Method':<10}", end="")
    for ds in datasets:
        for m in metrics:
            print(f"{ds[:3].upper()+'-'+m:>{col_w}}", end="")
    print()
    print("─" * (10 + col_w * len(metrics) * len(datasets)))

    for method in methods:
        print(f"{method:<10}", end="")
        for ds in datasets:
            with open(results_files[ds]) as f:
                data = json.load(f)
            r = data.get("results", {}).get(method, {})
            for m in metrics:
                print(f"{r.get(m, 0.0):>{col_w}.6f}", end="")
        print()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    rdir = args.results_dir
    datasets = ["movielens", "ciao", "filmtrust"]

    results_files    = {}
    robustness_files = {}

    for ds in datasets:
        r_path = os.path.join(rdir, f"{ds}_results.json")
        rob_path = os.path.join(rdir, f"{ds}_robustness.json")
        if os.path.exists(r_path):
            results_files[ds] = r_path
        if os.path.exists(rob_path):
            robustness_files[ds] = rob_path

    if results_files:
        print_table3(results_files)
        plot_training_traces(results_files)
    else:
        print("No results JSONs found. Run run_experiments.py first.")

    if robustness_files:
        plot_robustness(robustness_files)
    else:
        print("No robustness JSONs found. Run robustness_analysis.py first.")
