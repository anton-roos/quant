"""
Custom Keras layers used by the LSTM forecasting model.

MCDropout and Attention are defined here *once* so that training,
inference, walk-forward retraining, and the live bot all share
the exact same implementation.  This avoids silent weight-loading
bugs caused by diverging class definitions.
"""

import tensorflow as tf
from keras.layers import Dropout, Layer


# ---------------------------------------------------------------------------
# Focal Loss — handles severe class imbalance (e.g. 2-7% positive rate)
# ---------------------------------------------------------------------------

def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    """Binary focal loss for multi-label sigmoid outputs.

    Focal loss down-weights easy (well-classified) examples so the model
    focuses on hard positives — critical when positive rate is ~5-7%.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  Higher values suppress easy negatives more.
        ``gamma=0`` recovers standard binary cross-entropy.
    alpha : float
        Balance factor for positives.  With ~5% positive rate set alpha
        high (0.75-0.85) to compensate for class imbalance — otherwise
        the model finds the degenerate "predict near-zero for everything"
        solution that minimises focal loss for the majority class.

    Returns
    -------
    loss_fn
        A Keras-compatible loss function ``f(y_true, y_pred) -> scalar``.
    """
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Standard BCE components
        bce_pos = -y_true * tf.math.log(y_pred)
        bce_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)

        # Focal modulating factor
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Alpha balance
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        loss = alpha_t * focal_weight * (bce_pos + bce_neg)
        return tf.reduce_mean(loss)

    focal_loss_fn.__name__ = "binary_focal_loss"
    return focal_loss_fn


def pnl_weighted_focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    """P&L-weighted focal loss for multi-label sigmoid outputs.

    Identical to binary_focal_loss but scales each sample's loss by an
    ATR-normalised reward weight appended as the 9th output channel:

        y_true shape: (batch, 9)  — first 8 = class labels, last 1 = ATR weight
        y_pred shape: (batch, 8)  — sigmoid outputs (unchanged)

    The ATR weight  w = clip(ATR_14 / close, 0.5, 2.0)  gives larger
    instruments (higher relative ATR) proportionally higher gradients,
    so the model focuses on high-magnitude-move opportunities.

    Enable via config:  PNL_LOSS_ENABLED: true
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Split off the weight channel (last column)
        atr_weight = y_true[:, -1:]          # (batch, 1)
        y_labels = y_true[:, :-1]            # (batch, 8)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce_pos = -y_labels * tf.math.log(y_pred)
        bce_neg = -(1.0 - y_labels) * tf.math.log(1.0 - y_pred)

        p_t = y_labels * y_pred + (1.0 - y_labels) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        alpha_t = y_labels * alpha + (1.0 - y_labels) * (1.0 - alpha)

        # Base focal loss per sample (mean over output heads)
        per_sample_loss = tf.reduce_mean(
            alpha_t * focal_weight * (bce_pos + bce_neg), axis=-1, keepdims=True
        )  # (batch, 1)

        # Scale by ATR weight (clipped to [0.5, 2.0] during label creation)
        weighted_loss = per_sample_loss * (1.0 + atr_weight)
        return tf.reduce_mean(weighted_loss)

    loss_fn.__name__ = "pnl_weighted_focal_loss"
    return loss_fn


class MCDropout(Dropout):
    """Monte-Carlo Dropout — keeps dropout active during inference.

    Used to produce uncertainty estimates via repeated stochastic
    forward passes (MC Dropout sampling).
    """

    def call(self, inputs, training=None):
        # Always apply dropout, regardless of the ``training`` flag.
        return super().call(inputs, training=True)


class Attention(Layer):
    """Bahdanau-style additive attention for sequence summarisation.

    Computes a scalar attention weight per timestep, then returns
    the weighted sum across the time axis.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)  # attention weights
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
