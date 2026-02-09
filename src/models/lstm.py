"""
Factory function for the Conv1D + LSTM forecasting model.

Defined in one place so that initial training, walk-forward retraining,
and any future experiment scripts all produce architecturally identical
models.
"""

from keras.models import Sequential
from keras.layers import Conv1D, Input, LSTM, Dense, BatchNormalization, GaussianNoise
from keras.regularizers import l2

from src.models.layers import MCDropout


def build_model(n_features: int, window_size: int, n_outputs: int = 8) -> Sequential:
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

    Returns
    -------
    Sequential
        An **uncompiled** Keras model.
    """
    _l2 = l2(1e-4)

    model = Sequential([
        # Explicit input shape
        Input(shape=(window_size + 1, n_features)),
        # Input noise for regularisation (active during training only)
        GaussianNoise(0.01),

        # Temporal convolution to capture local patterns
        Conv1D(
            64, kernel_size=3, activation="relu", padding="same",
            kernel_regularizer=_l2,
        ),
        BatchNormalization(),
        MCDropout(0.4),

        # Second conv layer for higher-level patterns
        Conv1D(32, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=_l2),
        BatchNormalization(),
        MCDropout(0.3),

        # Stacked LSTMs for long-range dependencies
        LSTM(128, return_sequences=True, kernel_regularizer=_l2),
        MCDropout(0.4),
        LSTM(64, return_sequences=False, kernel_regularizer=_l2),
        MCDropout(0.4),

        # Dense head
        Dense(64, activation="relu", kernel_regularizer=_l2),
        BatchNormalization(),
        MCDropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=_l2),
        Dense(n_outputs, activation="sigmoid"),
    ])
    return model
