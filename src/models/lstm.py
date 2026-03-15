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
    Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, GaussianNoise,
    Input, LSTM, BatchNormalization, Reshape,
)
from keras.regularizers import l2

from src.models.layers import MCDropout


def build_encoder(
    n_features: int,
    window_size: int,
) -> Model:
    """Build just the Conv1D+LSTM encoder trunk (without Dense head).

    Used by self-supervised pre-training and the GNN overlay.
    The output is the LSTM's final hidden state, shape (batch, 64).
    """
    _l2 = l2(1e-5)
    ts_input = Input(shape=(window_size + 1, n_features), name="ts_input")
    x = GaussianNoise(0.01)(ts_input)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=_l2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=_l2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True, kernel_regularizer=_l2)(x)
    x = Dropout(0.35)(x)
    x = LSTM(64, return_sequences=False, kernel_regularizer=_l2)(x)
    return Model(inputs=ts_input, outputs=x, name="encoder")


def build_encoder_decoder(
    n_features: int,
    window_size: int,
    n_future: int = 5,
) -> Model:
    """Build encoder + decoder for self-supervised next-candle pre-training.

    The decoder predicts the log-returns for the next ``n_future`` candles
    across all ``n_features`` dimensions, giving the model a rich signal
    to learn price dynamics before fine-tuning on the classification task.

    Parameters
    ----------
    n_features : int
    window_size : int
    n_future : int
        Number of future timesteps to predict (default 5 = one week).

    Returns
    -------
    Model
        Uncompiled Keras model with MSE-friendly Dense outputs.
    """
    encoder = build_encoder(n_features, window_size)
    ts_input = encoder.input
    z = encoder.output                                      # (batch, 64)
    out = Dense(128, activation="relu")(z)
    out = Dense(n_future * n_features)(out)                 # flat prediction
    out = Reshape((n_future, n_features), name="future_pred")(out)
    return Model(inputs=ts_input, outputs=out, name="encoder_decoder")


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
    # Lighter regularisation — l2(1e-4) dominated the tiny focal-loss values
    # and pushed weights to zero, causing the "predict everything near-zero" collapse.
    _l2 = l2(1e-5)

    # --- Primary input: time-series window ---
    ts_input = Input(shape=(window_size + 1, n_features), name="ts_input")

    x = GaussianNoise(0.01)(ts_input)

    # Temporal convolution to capture local patterns.
    # Use standard Dropout (not MCDropout) so validation metrics are stable —
    # MC-Dropout inference is handled by calling model(x, training=True) externally.
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=_l2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Second conv layer for higher-level patterns
    x = Conv1D(32, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=_l2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Stacked LSTMs for long-range dependencies
    x = LSTM(128, return_sequences=True, kernel_regularizer=_l2)(x)
    x = Dropout(0.35)(x)
    x = LSTM(64, return_sequences=False, kernel_regularizer=_l2)(x)
    x = Dropout(0.35)(x)

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
    x = Dropout(0.25)(x)
    x = Dense(32, activation="relu", kernel_regularizer=_l2)(x)
    outputs = Dense(n_outputs, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
