"""
Graph Attention Network (GAT) layer for cross-asset contagion modelling.

Each instrument is a node.  Edges are defined by rolling pairwise Pearson
correlations (computed daily and cached).  The GAT layer aggregates
information from correlated instruments before the final Dense prediction
head, allowing the model to detect contagion signals.

Architecture:
  1. All instruments are processed through the LSTM encoder simultaneously.
  2. A rolling correlation matrix defines the adjacency (edges pruned by threshold).
  3. CrossAssetGNN (one GAT layer) updates each node's embedding using
     attention-weighted neighbour aggregation.
  4. Updated embeddings feed the existing Dense classification head.

Enable via config:  GNN_ENABLED: true

Usage:
    from src.models.gnn import CrossAssetGNN, build_correlation_adj
    adj = build_correlation_adj(price_dict, threshold=0.6)
    gnn = CrossAssetGNN(hidden_dim=64, dropout=0.1)
    updated_nodes = gnn(node_features, adj)  # (n_instruments, hidden_dim)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Adjacency matrix computation
# ---------------------------------------------------------------------------

def build_correlation_adj(
    price_dict: Dict[str, np.ndarray],
    threshold: float = 0.6,
    lookback: int = 60,
) -> Tuple[np.ndarray, List[str]]:
    """Compute a binary adjacency matrix from rolling return correlations.

    Parameters
    ----------
    price_dict : dict {symbol_name: close_price_array}
    threshold : float
        Minimum absolute correlation to include an edge.
    lookback : int
        Number of trailing candles for correlation computation.

    Returns
    -------
    adj : ndarray, shape (n, n)
        Symmetric binary adjacency matrix (self-loops excluded).
    names : list of str
        Instrument names in order (index matches adj matrix rows/cols).
    """
    names = sorted(price_dict.keys())
    n = len(names)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    # Build log-return matrix
    log_rets = {}
    for name in names:
        prices = np.asarray(price_dict[name], dtype=np.float64)
        if len(prices) >= lookback + 1:
            prices = prices[-lookback - 1:]
            log_rets[name] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-12))
        else:
            log_rets[name] = np.zeros(max(1, lookback), dtype=np.float64)

    # Correlation matrix
    matrix = np.column_stack([log_rets[n] for n in names])  # (T, n)
    corr = np.corrcoef(matrix.T)  # (n, n)
    np.fill_diagonal(corr, 0.0)   # remove self-loops

    # Threshold
    adj = (np.abs(corr) >= threshold).astype(np.float32)
    return adj, names


def save_adj_matrix(adj: np.ndarray, names: List[str], path: Path) -> None:
    np.savez(path, adj=adj, names=np.array(names))


def load_adj_matrix(path: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    return data["adj"], list(data["names"])


# ---------------------------------------------------------------------------
# Graph Attention Layer (single-head, simplified)
# ---------------------------------------------------------------------------

class CrossAssetGNN:
    """Single-layer Graph Attention Network over instruments.

    Implemented as a pure NumPy/TF operation (not a Keras layer) so it can
    be inserted into the inference path without retraining.  The learnable
    parameters (W_q, W_k, W_v) ARE trainable Keras variables so they can
    be saved/loaded with the model.

    Note: This is an *inference-time overlay*, not a full end-to-end trained
    module.  It enriches LSTM embeddings using today's correlation structure.
    """

    def __init__(self, hidden_dim: int = 64, dropout_rate: float = 0.1):
        """
        Parameters
        ----------
        hidden_dim : int
            Input (and output) dimension per node.
        dropout_rate : float
        """
        import tensorflow as tf
        self._tf = tf
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Trainable projection matrices
        self.W_q = tf.Variable(
            tf.random.normal([hidden_dim, hidden_dim], stddev=0.02), name="gnn_Wq"
        )
        self.W_k = tf.Variable(
            tf.random.normal([hidden_dim, hidden_dim], stddev=0.02), name="gnn_Wk"
        )
        self.W_v = tf.Variable(
            tf.random.normal([hidden_dim, hidden_dim], stddev=0.02), name="gnn_Wv"
        )
        self.bias = tf.Variable(tf.zeros([hidden_dim]), name="gnn_bias")

    @property
    def trainable_variables(self):
        return [self.W_q, self.W_k, self.W_v, self.bias]

    def __call__(
        self,
        node_features: "np.ndarray",
        adj: "np.ndarray",
        training: bool = False,
    ) -> "np.ndarray":
        """Apply one GAT message-passing step.

        Parameters
        ----------
        node_features : ndarray or tensor, shape (n_instruments, hidden_dim)
        adj : ndarray, shape (n_instruments, n_instruments)
            Symmetric binary adjacency matrix.
        training : bool

        Returns
        -------
        updated : ndarray, shape (n_instruments, hidden_dim)
        """
        tf = self._tf
        H = tf.cast(node_features, tf.float32)          # (n, d)
        A = tf.cast(adj, tf.float32)                    # (n, n)

        # Queries, keys, values
        Q = tf.matmul(H, self.W_q)                     # (n, d)
        K = tf.matmul(H, self.W_k)                     # (n, d)
        V = tf.matmul(H, self.W_v)                     # (n, d)

        # Attention scores: (n, n) = Q · K^T / sqrt(d), masked by adj
        scores = tf.matmul(Q, K, transpose_b=True) / (self.hidden_dim ** 0.5)
        # Mask non-edges with large negative value before softmax
        mask = tf.cast(A == 0, tf.float32) * -1e9
        scores = scores + mask
        attn = tf.nn.softmax(scores, axis=-1)           # (n, n)

        if training and self.dropout_rate > 0:
            attn = tf.nn.dropout(attn, rate=self.dropout_rate)

        # Aggregate
        out = tf.matmul(attn, V) + self.bias            # (n, d)
        # Residual connection + layer norm
        out = tf.nn.l2_normalize(H + out, axis=-1) * (self.hidden_dim ** 0.5)
        return out.numpy()

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_q=self.W_q.numpy(), W_k=self.W_k.numpy(),
            W_v=self.W_v.numpy(), bias=self.bias.numpy(),
        )

    def load_weights(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path)
        self.W_q.assign(data["W_q"])
        self.W_k.assign(data["W_k"])
        self.W_v.assign(data["W_v"])
        self.bias.assign(data["bias"])
        logger.info(f"GNN weights loaded from {path}")
        return True
