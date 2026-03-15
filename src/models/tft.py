"""
Temporal Fusion Transformer (TFT) for multi-horizon classification.

A simplified TFT implementation that can serve as a drop-in replacement for
the Conv1D+LSTM model (same signature as ``build_model`` in lstm.py).

Architecture components:
  - Variable Selection Network (VSN): soft-gating per feature at each timestep
  - Gated Residual Network (GRN): core building block with gate-add-norm
  - LSTM encoder-decoder: captures sequential dependencies
  - Interpretable Multi-Head Self-Attention: long-range temporal modelling
  - Point-wise output layer: 8 sigmoid outputs (4 up + 4 down horizons)

Enable via config:  MODEL_TYPE: "tft"

Reference:
  Lim et al. 2021 "Temporal Fusion Transformers for interpretable
  multi-horizon time series forecasting." IJF.
"""

from __future__ import annotations

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Add, Concatenate, Dense, Dropout, Embedding, Flatten,
    Input, LSTM, LayerNormalization, MultiHeadAttention, Reshape,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _grn(x, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1,
         context=None, name: str = "grn"):
    """Gated Residual Network.

    GRN(x, c) = LayerNorm(x + GLU(ELU(W1·[x, c]) + W2·x))

    Parameters
    ----------
    x : tensor, shape (..., d)
    hidden_dim : int
    output_dim : int
    dropout_rate : float
    context : tensor or None  (optional context vector, same leading dims as x)
    name : str
    """
    residual = x
    if x.shape[-1] != output_dim:
        residual = Dense(output_dim, use_bias=False, name=f"{name}_proj")(x)

    if context is not None:
        x_ctx = Concatenate(name=f"{name}_cat")([x, context])
    else:
        x_ctx = x

    h = Dense(hidden_dim, activation="elu", name=f"{name}_h1")(x_ctx)
    h = Dense(output_dim * 2, name=f"{name}_h2")(h)  # gate + value in one shot

    # GLU: split into value + gate
    v, gate = h[..., :output_dim], h[..., output_dim:]
    gate = tf.keras.activations.sigmoid(gate)
    h = v * gate

    h = Dropout(dropout_rate, name=f"{name}_drop")(h)
    return LayerNormalization(name=f"{name}_norm")(residual + h)


def _vsn(inputs, n_inputs: int, hidden_dim: int, dropout_rate: float = 0.1,
         name: str = "vsn"):
    """Variable Selection Network.

    Learns a soft importance weight per input feature at each timestep.

    Parameters
    ----------
    inputs : tensor, shape (batch, time, n_inputs * hidden_dim)
        Processed features (each input already embedded/dense'd to hidden_dim).
    n_inputs : int
    hidden_dim : int
    name : str

    Returns
    -------
    output : tensor (batch, time, hidden_dim)  — weighted combination
    weights : tensor (batch, time, n_inputs)   — selection weights
    """
    # Flat representation for weight computation
    flat = Reshape((-1, n_inputs * hidden_dim), name=f"{name}_flat")(inputs)
    weights = _grn(flat, hidden_dim, n_inputs, dropout_rate, name=f"{name}_grn")
    weights = Dense(n_inputs, activation="softmax", name=f"{name}_softmax")(weights)
    # weights: (batch, time, n_inputs)

    # Reshape inputs back to (batch, time, n_inputs, hidden_dim)
    x_split = Reshape((-1, n_inputs, hidden_dim), name=f"{name}_split")(inputs)
    # Weighted sum
    weights_exp = tf.expand_dims(weights, axis=-1)  # (batch, time, n_inputs, 1)
    output = tf.reduce_sum(x_split * weights_exp, axis=2)  # (batch, time, hidden_dim)
    return output, weights


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_tft_model(
    n_features: int,
    window_size: int,
    n_outputs: int = 8,
    n_symbols: int = 0,
    symbol_embed_dim: int = 8,
    hidden_dim: int = 64,
    n_heads: int = 4,
    dropout_rate: float = 0.1,
) -> Model:
    """Build and return a simplified TFT classification model.

    Signature matches ``build_model`` in lstm.py so it can be used as a
    drop-in replacement by setting ``MODEL_TYPE: "tft"`` in the config.

    Parameters
    ----------
    n_features : int
    window_size : int
    n_outputs : int  (default 8)
    n_symbols : int  (>0 enables symbol embedding, same as LSTM model)
    symbol_embed_dim : int
    hidden_dim : int  TFT hidden dimension (default 64)
    n_heads : int  Attention heads (default 4)
    dropout_rate : float

    Returns
    -------
    Model  Uncompiled Keras Functional model.
    """
    # ---- Inputs ----
    ts_input = Input(shape=(window_size + 1, n_features), name="ts_input")

    if n_symbols > 0:
        sym_input = Input(shape=(1,), dtype="int32", name="symbol_id")
        sym_emb = Embedding(n_symbols, symbol_embed_dim, name="symbol_embedding")(sym_input)
        sym_emb = Flatten()(sym_emb)  # (batch, symbol_embed_dim)
        inputs = [ts_input, sym_input]
    else:
        sym_emb = None
        inputs = ts_input

    # ---- Feature embedding: project each feature to hidden_dim ----
    # (batch, time, n_features) → (batch, time, n_features, hidden_dim)
    # Implemented as a Dense applied to the last axis
    feat_emb = Dense(hidden_dim * n_features, name="feat_embed")(ts_input)
    # Note: Dense applies to last dim only, so shape is (batch, time, hidden_dim*n_features)

    # ---- Variable Selection Network ----
    selected, _ = _vsn(feat_emb, n_features, hidden_dim, dropout_rate, name="vsn")
    # selected: (batch, time, hidden_dim)

    # ---- LSTM encoder ----
    lstm_out = LSTM(hidden_dim, return_sequences=True, name="lstm_enc")(selected)
    lstm_out = Dropout(dropout_rate, name="lstm_drop")(lstm_out)

    # ---- GRN on LSTM output ----
    enriched = _grn(lstm_out, hidden_dim, hidden_dim, dropout_rate, name="enrich")

    # ---- Interpretable Multi-Head Self-Attention ----
    attn_out = MultiHeadAttention(
        num_heads=n_heads, key_dim=hidden_dim // n_heads, dropout=dropout_rate,
        name="mhsa"
    )(enriched, enriched)
    attn_out = LayerNormalization(name="attn_norm")(enriched + attn_out)

    # ---- Temporal GRN ----
    temporal = _grn(attn_out, hidden_dim, hidden_dim, dropout_rate, name="temporal")

    # ---- Aggregate sequence → single vector (last timestep) ----
    z = temporal[:, -1, :]  # (batch, hidden_dim)

    # ---- Inject symbol context if available ----
    if sym_emb is not None:
        z = Concatenate(name="sym_concat")([z, sym_emb])

    # ---- Output head ----
    z = _grn(z, hidden_dim, hidden_dim, dropout_rate, name="output_grn")
    outputs = Dense(n_outputs, activation="sigmoid", name="predictions")(z)

    return Model(inputs=inputs, outputs=outputs, name="tft_model")
