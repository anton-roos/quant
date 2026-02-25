"""
Uncertainty estimation beyond MC-Dropout.

Adds:
  - **Prediction intervals** via quantile analysis of MC samples
  - **Entropy-based uncertainty** (information-theoretic confidence)
  - **Composite confidence score** combining MC std, calibration,
    ensemble disagreement, and entropy into a single 0-1 score
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def prediction_intervals(
    mc_preds: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute prediction intervals from MC-Dropout samples.

    Parameters
    ----------
    mc_preds : ndarray, shape (n_mc_samples, batch, n_outputs)
        Raw MC-Dropout predictions.
    alpha : float
        Significance level. ``alpha=0.1`` gives 90% intervals.

    Returns
    -------
    lower, upper : ndarrays, shape (batch, n_outputs)
    """
    lower = np.percentile(mc_preds, 100 * (alpha / 2), axis=0)
    upper = np.percentile(mc_preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of predicted probabilities.

    Higher entropy = more uncertain.  For sigmoid outputs, max
    entropy is at p=0.5 (=0.693 nats).

    Parameters
    ----------
    mean_probs : ndarray, shape (batch, n_outputs)

    Returns
    -------
    ndarray, shape (batch, n_outputs) — entropy in [0, ~0.693]
    """
    p = np.clip(mean_probs, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def mutual_information(mc_preds: np.ndarray) -> np.ndarray:
    """Mutual information between model parameters and predictions.

    Captures *epistemic* uncertainty (what the model doesn't know),
    as opposed to *aleatoric* uncertainty (inherent randomness).

    MI = H[y|x] - E_theta[H[y|x,theta]]

    Parameters
    ----------
    mc_preds : ndarray, shape (n_mc_samples, batch, n_outputs)

    Returns
    -------
    ndarray, shape (batch, n_outputs)
    """
    mean_probs = mc_preds.mean(axis=0)
    total_entropy = predictive_entropy(mean_probs)

    # Expected entropy of individual samples
    p = np.clip(mc_preds, 1e-7, 1 - 1e-7)
    per_sample_entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    expected_entropy = per_sample_entropy.mean(axis=0)

    return total_entropy - expected_entropy


def composite_confidence(
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
    mc_preds: Optional[np.ndarray] = None,
    ensemble_disagreement: float = 0.0,
    calibrated: bool = False,
) -> np.ndarray:
    """Compute a composite confidence score in [0, 1].

    Combines multiple uncertainty signals into a single tradeable
    confidence metric.  Higher = more confident.

    Parameters
    ----------
    mean_probs : ndarray, shape (batch, n_outputs)
    std_probs : ndarray, shape (batch, n_outputs)
    mc_preds : optional ndarray, shape (n_mc_samples, batch, n_outputs)
    ensemble_disagreement : float
        From EnsemblePredictor.predict().
    calibrated : bool
        Whether the probs have been calibrated (gets a small bonus).

    Returns
    -------
    ndarray, shape (batch, n_outputs) — confidence in [0, 1]
    """
    # 1. Directional strength: how far from 0.5 (decision boundary)
    #    Raw: 0 at p=0.5, 0.5 at p=0 or p=1
    directional = np.abs(mean_probs - 0.5) * 2.0  # [0, 1]

    # 2. MC consistency: low std = high confidence
    #    Normalise std (max possible ~0.5 for Bernoulli)
    mc_consistency = 1.0 - np.clip(std_probs / 0.25, 0, 1)

    # 3. Entropy-based certainty
    entropy = predictive_entropy(mean_probs)
    max_entropy = np.log(2)  # ~0.693
    certainty = 1.0 - np.clip(entropy / max_entropy, 0, 1)

    # 4. Epistemic confidence (if MC samples available)
    if mc_preds is not None:
        mi = mutual_information(mc_preds)
        epistemic_conf = 1.0 - np.clip(mi / max_entropy, 0, 1)
    else:
        epistemic_conf = mc_consistency

    # 5. Ensemble agreement
    agreement = 1.0 - np.clip(ensemble_disagreement * 4, 0, 1)  # scale

    # 6. Calibration bonus
    cal_bonus = 1.05 if calibrated else 1.0

    # Weighted combination
    confidence = (
        0.30 * directional
        + 0.25 * mc_consistency
        + 0.20 * certainty
        + 0.15 * epistemic_conf
        + 0.10 * agreement
    ) * cal_bonus

    return np.clip(confidence, 0, 1)
