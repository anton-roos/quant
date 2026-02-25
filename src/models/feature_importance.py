"""
Feature importance analysis for the LSTM trading model.

Provides:
  - **Permutation importance**: shuffle one feature at a time and measure
    prediction degradation (model-agnostic, works with any Keras model).
  - **Gradient-based saliency**: compute input gradients to see which
    features the model is paying attention to (fast, differentiable).

Results are saved to ``outputs/models/feature_importance.csv`` and can
be used to:
  1. Prune noisy features that hurt confidence
  2. Verify that sentiment features are being used
  3. Identify which technical indicators matter most per horizon

Usage:
    from src.models.feature_importance import (
        permutation_importance, gradient_saliency, save_importance_report
    )
    imp = permutation_importance(model, X_val, y_val, feature_cols)
    save_importance_report(imp, feature_cols)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_PATH = PROJECT_ROOT / "outputs" / "models" / "feature_importance.csv"


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    n_repeats: int = 5,
    metric: str = "auc",
) -> pd.DataFrame:
    """Compute permutation importance for each feature.

    Parameters
    ----------
    model : Keras model
    X : ndarray, shape (n_samples, window+1, n_features)
    y : ndarray, shape (n_samples, n_outputs)
    feature_cols : list of feature names
    n_repeats : int
        Number of shuffles per feature.
    metric : str
        Either "auc" (sklearn) or "loss" (model.evaluate).

    Returns
    -------
    DataFrame with columns: feature, importance_mean, importance_std
    """
    from sklearn.metrics import roc_auc_score

    # Baseline score
    y_pred_base = model.predict(X, verbose=0)
    if metric == "auc":
        base_score = _safe_auc(y, y_pred_base)
    else:
        base_score = -model.evaluate(X, y, verbose=0)  # negate loss so higher=better

    n_features = X.shape[2]
    results = []

    for i in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # Shuffle feature i across all samples (keep temporal structure)
            perm_idx = np.random.permutation(X_perm.shape[0])
            X_perm[:, :, i] = X_perm[perm_idx, :, i]

            y_pred_perm = model.predict(X_perm, verbose=0)
            if metric == "auc":
                perm_score = _safe_auc(y, y_pred_perm)
            else:
                perm_score = -model.evaluate(X_perm, y, verbose=0)

            scores.append(base_score - perm_score)

        feat_name = feature_cols[i] if i < len(feature_cols) else f"feature_{i}"
        results.append({
            "feature": feat_name,
            "importance_mean": float(np.mean(scores)),
            "importance_std": float(np.std(scores)),
        })
        if (i + 1) % 20 == 0:
            logger.info(f"  Permutation importance: {i+1}/{n_features} features done")

    df = pd.DataFrame(results).sort_values("importance_mean", ascending=False)
    return df


def gradient_saliency(
    model,
    X: np.ndarray,
    output_index: int = 0,
) -> np.ndarray:
    """Compute gradient-based saliency map for a single output.

    Parameters
    ----------
    model : Keras model
    X : ndarray, shape (n_samples, window+1, n_features)
    output_index : int
        Which output head to compute gradients for.

    Returns
    -------
    ndarray, shape (n_features,) — mean absolute gradient per feature.
    """
    import tensorflow as tf

    X_tensor = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor, training=False)
        target_output = preds[:, output_index]
        loss = tf.reduce_mean(target_output)

    grads = tape.gradient(loss, X_tensor)
    # Mean absolute gradient across samples and timesteps
    saliency = np.abs(grads.numpy()).mean(axis=(0, 1))
    return saliency


def full_saliency_report(
    model,
    X: np.ndarray,
    feature_cols: List[str],
    n_outputs: int = 8,
) -> pd.DataFrame:
    """Run gradient saliency for all outputs and return a DataFrame.

    Returns DataFrame: feature, saliency_mean, saliency_buy, saliency_sell
    """
    all_saliency = []
    for i in range(n_outputs):
        s = gradient_saliency(model, X, output_index=i)
        all_saliency.append(s)

    stacked = np.stack(all_saliency, axis=0)  # (n_outputs, n_features)
    buy_mean = stacked[:4].mean(axis=0)   # outputs 0-3 = buy horizons
    sell_mean = stacked[4:].mean(axis=0)  # outputs 4-7 = sell horizons

    records = []
    for j, feat in enumerate(feature_cols):
        records.append({
            "feature": feat,
            "saliency_mean": float(stacked[:, j].mean()),
            "saliency_buy": float(buy_mean[j]),
            "saliency_sell": float(sell_mean[j]),
        })

    return pd.DataFrame(records).sort_values("saliency_mean", ascending=False)


def save_importance_report(
    df: pd.DataFrame,
    path: Optional[Path] = None,
):
    """Save feature importance report to CSV."""
    p = path or REPORT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Saved feature importance report: {p}")

    # Log top 15
    top = df.head(15)
    logger.info("Top 15 features by importance:")
    for _, row in top.iterrows():
        logger.info(f"  {row['feature']:35s}  {row.get('importance_mean', row.get('saliency_mean', 0)):.6f}")


def _safe_auc(y_true, y_pred):
    """Compute mean AUC across outputs, handling edge cases."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            if len(np.unique(y_true[:, i])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.5
