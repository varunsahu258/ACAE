"""
Baseline recommendation models — RTX 4500 Ada optimised.

All hot-path methods decorated with @tf.function(jit_compile=True) so XLA
fuses them into single CUDA kernels.  Embedding lookups and BPR scoring
are fully vectorised; no Python loops inside any compiled function.

Models:
  1. ItemPop  – popularity ranking (CPU, trivial)
  2. MFBPR    – Matrix Factorisation + BPR loss
  3. CDAE     – Collaborative Denoising Auto-Encoder
  4. NeuMF    – Neural Collaborative Filtering (GMF + MLP)
  5. AMF      – Adversarial Matrix Factorisation
"""

import numpy as np
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# 1. ItemPop
# ─────────────────────────────────────────────────────────────────────────────

class ItemPop:
    def fit(self, train_data, n_items):
        self.popularity = np.zeros(n_items, dtype=np.float32)
        for _, item, rating in train_data:
            if rating > 0:
                self.popularity[item] += 1.0

    def predict(self, user_id, item_ids, rating_matrix=None):
        return self.popularity[item_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MF-BPR
# ─────────────────────────────────────────────────────────────────────────────

class MFBPR(tf.keras.Model):
    def __init__(self, n_users, n_items, latent_dim=64, reg=0.01):
        super().__init__()
        init = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        self.user_emb = tf.Variable(init([n_users, latent_dim]), name="U")
        self.item_emb = tf.Variable(init([n_items, latent_dim]), name="V")
        self.reg = tf.constant(reg, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def bpr_loss(self, users, pos_items, neg_items):
        u  = tf.nn.embedding_lookup(self.user_emb, users)
        pi = tf.nn.embedding_lookup(self.item_emb, pos_items)
        ni = tf.nn.embedding_lookup(self.item_emb, neg_items)
        diff = tf.reduce_sum(u * (pi - ni), axis=1)
        loss = -tf.reduce_mean(tf.math.log_sigmoid(diff))
        reg  = self.reg * (tf.reduce_sum(tf.square(u))  +
                           tf.reduce_sum(tf.square(pi)) +
                           tf.reduce_sum(tf.square(ni)))
        return loss + reg

    @tf.function(jit_compile=True)
    def train_step(self, users, pos_items, neg_items, optimiser):
        train_vars = [self.user_emb, self.item_emb]
        with tf.GradientTape() as tape:
            loss = self.bpr_loss(users, pos_items, neg_items)
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if pairs:
            optimiser.apply_gradients(pairs)
        return loss

    def predict(self, user_id, item_ids, rating_matrix=None):
        u = self.user_emb[user_id]
        v = tf.nn.embedding_lookup(self.item_emb, item_ids)
        return tf.linalg.matvec(v, u).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. CDAE
# ─────────────────────────────────────────────────────────────────────────────

class CDAE(tf.keras.Model):
    def __init__(self, n_users, n_items, latent_dim=50,
                 gamma=0.01, corruption=0.5):
        super().__init__()
        init = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        self.W1 = tf.Variable(init([n_items,    latent_dim]), name="W1")
        self.b1 = tf.Variable(tf.zeros([latent_dim]),         name="b1")
        self.W2 = tf.Variable(init([latent_dim, n_items]),    name="W2")
        self.b2 = tf.Variable(tf.zeros([n_items]),            name="b2")
        self.P  = tf.Variable(init([n_users,    latent_dim]), name="P")
        self.gamma      = tf.constant(gamma,      dtype=tf.float32)
        self.corruption = tf.constant(corruption, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def _forward(self, y, user_ids):
        pu = tf.nn.embedding_lookup(self.P, user_ids)
        h  = tf.sigmoid(tf.matmul(y, self.W1) + pu + self.b1)
        return tf.matmul(h, self.W2) + self.b2

    @tf.function(jit_compile=True)
    def loss_fn(self, y, user_ids):
        # Multiplicative mask-out noise on input (training only)
        mask    = tf.cast(
            tf.random.uniform(tf.shape(y)) > self.corruption, tf.float32)
        y_noisy = y * mask
        logits  = self._forward(y_noisy, user_ids)
        ce  = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
        reg = self.gamma * (
            tf.reduce_sum(tf.square(self.W1)) +
            tf.reduce_sum(tf.square(self.W2)) +
            tf.reduce_sum(tf.square(self.b1)) +
            tf.reduce_sum(tf.square(self.b2)) +
            tf.reduce_sum(tf.square(self.P)))
        return ce + reg

    @tf.function(jit_compile=True)
    def train_step(self, y, user_ids, optimiser):
        train_vars = [self.W1, self.b1, self.W2, self.b2, self.P]
        with tf.GradientTape() as tape:
            loss = self.loss_fn(y, user_ids)
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if pairs:
            optimiser.apply_gradients(pairs)
        return loss

    def predict(self, user_id, item_ids, rating_matrix):
        y   = tf.constant(rating_matrix[user_id:user_id+1], dtype=tf.float32)
        uid = tf.constant([user_id], dtype=tf.int32)
        scores = tf.sigmoid(self._forward(y, uid)[0])
        return tf.gather(scores, item_ids).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. NeuMF
# ─────────────────────────────────────────────────────────────────────────────

class NeuMF(tf.keras.Model):
    def __init__(self, n_users, n_items,
                 mf_dim=16, layers=(64, 32, 16, 8), reg=0.0):
        super().__init__()
        init    = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        mlp_dim = layers[0] // 2

        self.gmf_user = tf.Variable(init([n_users, mf_dim]),   name="gmf_u")
        self.gmf_item = tf.Variable(init([n_items, mf_dim]),   name="gmf_i")
        self.mlp_user = tf.Variable(init([n_users, mlp_dim]),  name="mlp_u")
        self.mlp_item = tf.Variable(init([n_items, mlp_dim]),  name="mlp_i")

        regulariser = tf.keras.regularizers.l2(reg) if reg > 0 else None
        self.mlp_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(d, activation="relu",
                                  kernel_regularizer=regulariser)
            for d in layers[1:]
        ])
        self.out_layer = tf.keras.layers.Dense(
            1, activation=None, kernel_regularizer=regulariser)

    def _forward(self, user_ids, item_ids):
        # GMF branch
        gmf = (tf.nn.embedding_lookup(self.gmf_user, user_ids) *
               tf.nn.embedding_lookup(self.gmf_item, item_ids))
        # MLP branch
        mlp = tf.concat([
            tf.nn.embedding_lookup(self.mlp_user, user_ids),
            tf.nn.embedding_lookup(self.mlp_item, item_ids)], axis=-1)
        mlp = self.mlp_dense(mlp)
        return tf.squeeze(self.out_layer(tf.concat([gmf, mlp], axis=-1)), -1)

    # NeuMF uses dynamic input shapes (variable batch), so jit_compile may
    # retrace; we use tf.function without XLA here for safety.
    @tf.function
    def train_step(self, users, items, labels, optimiser):
        with tf.GradientTape() as tape:
            logits = self._forward(users, items)
            loss   = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=logits))
            loss  += sum(self.losses)
        train_vars = ([self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]
                      + self.mlp_dense.trainable_variables
                      + self.out_layer.trainable_variables)
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if pairs:
            optimiser.apply_gradients(pairs)
        return loss

    def predict(self, user_id, item_ids, rating_matrix=None):
        uids = tf.constant([user_id] * len(item_ids), dtype=tf.int32)
        iids = tf.constant(item_ids,                  dtype=tf.int32)
        return tf.sigmoid(self._forward(uids, iids)).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 5. AMF  (Adversarial Matrix Factorisation)
# ─────────────────────────────────────────────────────────────────────────────

class AMF(tf.keras.Model):
    def __init__(self, n_users, n_items, latent_dim=64,
                 reg=0.01, epsilon=0.5, lam=1.0):
        super().__init__()
        init = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        self.user_emb = tf.Variable(init([n_users, latent_dim]), name="U")
        self.item_emb = tf.Variable(init([n_items, latent_dim]), name="V")
        self.reg     = tf.constant(reg,     dtype=tf.float32)
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.lam     = tf.constant(lam,     dtype=tf.float32)

    def _bpr_score(self, u, pi, ni):
        return tf.reduce_sum(u * (pi - ni), axis=1)

    def _adv_noise(self, grad):
        return self.epsilon * grad / (tf.norm(grad) + 1e-8)

    @tf.function(jit_compile=True)
    def amf_loss(self, users, pos_items, neg_items):
        u  = tf.nn.embedding_lookup(self.user_emb, users)
        pi = tf.nn.embedding_lookup(self.item_emb, pos_items)
        ni = tf.nn.embedding_lookup(self.item_emb, neg_items)

        # Standard BPR
        bpr_loss = -tf.reduce_mean(
            tf.math.log_sigmoid(self._bpr_score(u, pi, ni)))

        # Gradient for adversarial noise direction
        with tf.GradientTape() as noise_tape:
            noise_tape.watch([u, pi, ni])
            base = -tf.reduce_mean(
                tf.math.log_sigmoid(self._bpr_score(u, pi, ni)))
        du, dpi, dni = noise_tape.gradient(base, [u, pi, ni])

        # Perturbed BPR
        adv_loss = -tf.reduce_mean(
            tf.math.log_sigmoid(
                self._bpr_score(u  + self._adv_noise(du),
                                pi + self._adv_noise(dpi),
                                ni + self._adv_noise(dni))))

        reg = self.reg * (tf.reduce_sum(tf.square(u))  +
                          tf.reduce_sum(tf.square(pi)) +
                          tf.reduce_sum(tf.square(ni)))
        return bpr_loss + self.lam * adv_loss + reg

    @tf.function(jit_compile=True)
    def train_step(self, users, pos_items, neg_items, optimiser):
        train_vars = [self.user_emb, self.item_emb]
        with tf.GradientTape() as tape:
            loss = self.amf_loss(users, pos_items, neg_items)
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if pairs:
            optimiser.apply_gradients(pairs)
        return loss

    def predict(self, user_id, item_ids, rating_matrix=None):
        u = self.user_emb[user_id]
        v = tf.nn.embedding_lookup(self.item_emb, item_ids)
        return tf.linalg.matvec(v, u).numpy()
