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

def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Binary focal loss for multi-label sigmoid outputs.

    Focal loss down-weights easy (well-classified) examples so the model
    focuses on hard positives — critical when positive rate is ~5-7%.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  Higher values suppress easy negatives more.
        ``gamma=0`` recovers standard binary cross-entropy.
    alpha : float
        Balance factor for positives.  ``alpha=0.25`` means positives get
        weight 0.25 and negatives 0.75 (counterintuitively, lower alpha
        for the minority class works well with focal loss because gamma
        already up-weights hard examples).

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
