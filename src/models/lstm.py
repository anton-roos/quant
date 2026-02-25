"""
Factory function for the Conv1D + LSTM forecasting model.

Defined in one place so that initial training, walk-forward retraining,
and any future experiment scripts all produce architecturally identical
models.

Uses the Keras Functional API to support a secondary symbol-ID input
that feeds a learned embedding.  This lets the model produce
symbol-specific predictions instead of near-constant outputs.
"""

from keras.models import Model
from keras.layers import (
    Concatenate, Conv1D, Dense, Embedding, Flatten, GaussianNoise,
    Input, LSTM, BatchNormalization, Reshape,
)
from keras.regularizers import l2

from src.models.layers import MCDropout


def build_model(
    n_features: int,
    window_size: int,
    n_outputs: int = 8,
    n_symbols: int = 0,
    symbol_embed_dim: int = 8,
) -> Model:
    """Build and return the Conv1D + LSTM classification model.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    window_size : int
        Length of the look-back window (the ``+ 1`` is handled by the
        caller when creating the input data).
    n_outputs : int
        Number of sigmoid outputs (default 8 = 4 upside + 4 downside).
    n_symbols : int
        Total number of distinct instruments.  When > 0 the model
        accepts a second input ``symbol_id`` (shape ``(batch, 1)``)
        and learns a per-symbol embedding that is concatenated with
        the LSTM output before the Dense head.
    symbol_embed_dim : int
        Dimensionality of the learned symbol embedding (default 8).

    Returns
    -------
    Model
        An **uncompiled** Keras Functional-API model.
    """
    _l2 = l2(1e-4)

    # --- Primary input: time-series window ---
    ts_input = Input(shape=(window_size + 1, n_features), name="ts_input")

    x = GaussianNoise(0.01)(ts_input)

    # Temporal convolution to capture local patterns
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=_l2)(x)
    x = BatchNormalization()(x)
    x = MCDropout(0.4)(x)

    # Second conv layer for higher-level patterns
    x = Conv1D(32, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=_l2)(x)
    x = BatchNormalization()(x)
    x = MCDropout(0.3)(x)

    # Stacked LSTMs for long-range dependencies
    x = LSTM(128, return_sequences=True, kernel_regularizer=_l2)(x)
    x = MCDropout(0.4)(x)
    x = LSTM(64, return_sequences=False, kernel_regularizer=_l2)(x)
    x = MCDropout(0.4)(x)

    # --- Optional symbol embedding ---
    if n_symbols > 0:
        sym_input = Input(shape=(1,), dtype="int32", name="symbol_id")
        sym_emb = Embedding(n_symbols, symbol_embed_dim, name="symbol_embedding")(sym_input)
        sym_emb = Flatten()(sym_emb)
        x = Concatenate()([x, sym_emb])
        inputs = [ts_input, sym_input]
    else:
        inputs = ts_input

    # Dense head
    x = Dense(64, activation="relu", kernel_regularizer=_l2)(x)
    x = BatchNormalization()(x)
    x = MCDropout(0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=_l2)(x)
    outputs = Dense(n_outputs, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
