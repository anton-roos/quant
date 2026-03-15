"""
Elastic Weight Consolidation (EWC) for continual learning.

Prevents catastrophic forgetting during walk-forward retraining by adding
a penalty that keeps weights close to their previous values, weighted by
their importance (Fisher Information diagonal).

Reference: Kirkpatrick et al. 2017, "Overcoming catastrophic forgetting
in neural networks." PNAS.

Usage in walk_forward_retrain.py:
    ewc = EWC(old_model, val_gen, n_batches=20)

    @tf.function
    def train_step(x, y, sample_weight=None):
        with tf.GradientTape() as tape:
            preds = new_model(x, training=True)
            task_loss = focal_loss_fn(y, preds)
            ewc_penalty = ewc.penalty(new_model)
            loss = task_loss + ewc_lambda * ewc_penalty
        grads = tape.gradient(loss, new_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, new_model.trainable_weights))
        return loss
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EWC:
    """Elastic Weight Consolidation regulariser.

    Computes the Fisher Information diagonal for a trained model on a
    reference dataset, then provides a penalty function for subsequent
    training that resists changes to important weights.
    """

    def __init__(self, model, val_gen, n_batches: int = 20):
        """
        Parameters
        ----------
        model : keras.Model
            The *trained* model whose weights we want to protect.
        val_gen : keras.utils.Sequence
            Validation generator to estimate the Fisher Information diagonal.
        n_batches : int
            Number of batches to use for Fisher estimation (default 20).
            More batches → better estimate but slower.
        """
        import tensorflow as tf

        self._tf = tf
        # Snapshot of old parameters
        self.params_old: Dict[str, np.ndarray] = {
            w.name: w.numpy().copy() for w in model.trainable_weights
        }
        # Fisher Information diagonal (same shape as params_old)
        self.fisher: Dict[str, np.ndarray] = self._compute_fisher(model, val_gen, n_batches)
        logger.info(
            f"EWC: Fisher computed over {min(n_batches, len(val_gen))} batches, "
            f"{len(self.fisher)} weight tensors"
        )

    def _compute_fisher(self, model, val_gen, n_batches: int) -> Dict[str, np.ndarray]:
        """Estimate the diagonal Fisher Information matrix.

        For each sample, compute the squared gradient of the log-likelihood
        w.r.t. each weight.  Average over all samples gives the Fisher diag.
        """
        tf = self._tf
        fisher: Dict[str, np.ndarray] = {
            w.name: np.zeros(w.shape, dtype=np.float32) for w in model.trainable_weights
        }
        n_total = 0

        for batch_idx in range(min(n_batches, len(val_gen))):
            batch = val_gen[batch_idx]
            x_batch = batch[0]
            # We use the model's own predictions as pseudo-labels
            with tf.GradientTape() as tape:
                log_probs = tf.math.log(
                    tf.clip_by_value(model(x_batch, training=False), 1e-7, 1.0 - 1e-7)
                )
                # Sum over outputs for each sample, mean over batch
                log_lik = tf.reduce_mean(tf.reduce_sum(log_probs, axis=-1))

            grads = tape.gradient(log_lik, model.trainable_weights)
            for w, g in zip(model.trainable_weights, grads):
                if g is not None:
                    fisher[w.name] += (g.numpy() ** 2)
            n_total += 1

        if n_total > 0:
            for name in fisher:
                fisher[name] /= n_total

        return fisher

    def penalty(self, model) -> "tf.Tensor":
        """Compute the EWC regularisation penalty.

        Returns a scalar tensor:
            Σ_i  F_i * (θ_i - θ*_i)^2

        Parameters
        ----------
        model : keras.Model
            The model currently being trained.
        """
        tf = self._tf
        penalty_terms = []
        for w in model.trainable_weights:
            if w.name in self.params_old and w.name in self.fisher:
                old = tf.constant(self.params_old[w.name], dtype=tf.float32)
                fish = tf.constant(self.fisher[w.name], dtype=tf.float32)
                penalty_terms.append(tf.reduce_sum(fish * tf.square(w - old)))
        if penalty_terms:
            return tf.add_n(penalty_terms)
        return tf.constant(0.0)
