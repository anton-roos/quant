"""
Custom Keras layers used by the LSTM forecasting model.

MCDropout and Attention are defined here *once* so that training,
inference, walk-forward retraining, and the live bot all share
the exact same implementation.  This avoids silent weight-loading
bugs caused by diverging class definitions.
"""

import tensorflow as tf
from keras.layers import Dropout, Layer


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
