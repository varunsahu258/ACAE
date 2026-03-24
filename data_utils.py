"""
Data utilities for ACAE paper replication.
Handles MovieLens-1M, CiaoDVD, and FilmTrust datasets.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
import os
import zipfile
import urllib.request


# ─────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────

def download_movielens(data_dir="data/movielens"):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(path):
        print("Downloading MovieLens-1M...")
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(data_dir, "ml-1m.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        import shutil
        for f in ["ratings.dat", "users.dat", "movies.dat"]:
            src = os.path.join(data_dir, "ml-1m", f)
            dst = os.path.join(data_dir, f)
            if os.path.exists(src):
                shutil.copy(src, dst)
        os.remove(zip_path)
    return path


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_movielens(data_dir="data/movielens"):
    """Load MovieLens-1M. Ratings >3 → 1, else 0."""
    path = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(path):
        download_movielens(data_dir)

    df = pd.read_csv(path, sep="::", header=None,
                     names=["user", "item", "rating", "timestamp"],
                     engine="python")
    df["rating"] = (df["rating"] > 3).astype(int)
    # remap ids to 0-based
    users = {u: i for i, u in enumerate(df["user"].unique())}
    items = {it: i for i, it in enumerate(df["item"].unique())}
    df["user"] = df["user"].map(users)
    df["item"] = df["item"].map(items)
    return df, len(users), len(items)


def load_filmtrust(data_dir="data/filmtrust"):
    """Load FilmTrust. Ratings >2 → 1, else 0. No timestamp → random split."""
    path = os.path.join(data_dir, "ratings.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FilmTrust ratings.txt not found at {path}.\n"
            "Download from https://www.librec.net/datasets.html and extract to data/filmtrust/"
        )
    df = pd.read_csv(path, sep=" ", header=None,
                     names=["user", "item", "rating"])
    df["rating"] = (df["rating"] > 2).astype(int)
    users = {u: i for i, u in enumerate(df["user"].unique())}
    items = {it: i for i, it in enumerate(df["item"].unique())}
    df["user"] = df["user"].map(users)
    df["item"] = df["item"].map(items)
    df["timestamp"] = None   # no timestamp
    return df, len(users), len(items)


def load_ciao(data_dir="data/ciao"):
    """Load CiaoDVDs. Merge repeated ratings to earliest. Ratings >3 → 1."""
    path = os.path.join(data_dir, "movie-ratings.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CiaoDVD ratings file not found at {path}.\n"
            "Download from https://www.librec.net/datasets.html and extract to data/ciao/"
        )
    # columns vary by file version – try a few separators
    try:
        df = pd.read_csv(path, sep="\t", header=None)
    except Exception:
        df = pd.read_csv(path, sep=",", header=None)

    # Expected columns: userID, itemID, categoryID, rating, helpfulness, date
    if df.shape[1] >= 4:
        df = df.iloc[:, [0, 1, 3]]
        df.columns = ["user", "item", "rating"]
    else:
        df.columns = ["user", "item", "rating"]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating"], inplace=True)
    # keep earliest rating per (user, item) pair
    df = df.sort_values("rating").drop_duplicates(subset=["user", "item"], keep="first")
    df["rating"] = (df["rating"] > 3).astype(int)

    users = {u: i for i, u in enumerate(df["user"].unique())}
    items = {it: i for i, it in enumerate(df["item"].unique())}
    df["user"] = df["user"].map(users)
    df["item"] = df["item"].map(items)
    df["timestamp"] = None
    return df, len(users), len(items)


# ─────────────────────────────────────────────
# Leave-one-out split
# ─────────────────────────────────────────────

def leave_one_out_split(df, n_users, n_items, num_negatives=200, seed=42):
    """
    Leave-one-out evaluation protocol (He et al. NeuMF, 2017).
    For each user: hold out latest (or random) positive interaction,
    sample 200 negative items as test negatives.
    """
    rng = np.random.RandomState(seed)
    has_ts = "timestamp" in df.columns and df["timestamp"].notna().any()

    train_data = []
    test_data  = []   # list of (user, pos_item, [200 neg_items])

    # keep only positive interactions
    pos_df = df[df["rating"] == 1].copy()

    for u in range(n_users):
        u_pos = pos_df[pos_df["user"] == u]["item"].tolist()
        if len(u_pos) == 0:
            continue

        if has_ts:
            # pick latest interaction as test item
            u_df = pos_df[pos_df["user"] == u].sort_values("timestamp")
            test_item = u_df.iloc[-1]["item"]
            train_items = u_df.iloc[:-1]["item"].tolist()
        else:
            # random pick
            idx = rng.randint(len(u_pos))
            test_item = u_pos[idx]
            train_items = [it for it in u_pos if it != test_item]

        for it in train_items:
            train_data.append((u, it, 1))

        # sample 200 negatives not in u_pos
        pos_set = set(u_pos)
        negs = []
        while len(negs) < num_negatives:
            cand = rng.randint(n_items)
            if cand not in pos_set and cand not in negs:
                negs.append(cand)

        test_data.append((u, test_item, negs))

    return train_data, test_data


# ─────────────────────────────────────────────
# Sparse rating matrix
# ─────────────────────────────────────────────

def build_rating_matrix(train_data, n_users, n_items):
    """Build user×item sparse binary rating matrix from train_data."""
    rows, cols, vals = [], [], []
    for u, i, r in train_data:
        rows.append(u)
        cols.append(i)
        vals.append(float(r))
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return mat


def get_train_matrix_dense(train_data, n_users, n_items):
    """Return dense numpy array (n_users × n_items)."""
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r in train_data:
        mat[u, i] = float(r)
    return mat
