"""
Ensemble prediction — combine multiple models for robust inference.

Supports:
  - **Multi-model ensemble**: load several .keras models trained with
    different seeds / architectures and average their MC-Dropout outputs.
  - **Snapshot ensemble**: use checkpoints saved during training.

The ensemble reduces variance, improves calibration, and gives a natural
measure of model disagreement as an extra uncertainty signal.

Usage:
    from src.models.ensemble import EnsemblePredictor
    ensemble = EnsemblePredictor(model_dir="outputs/models")
    mean_probs, std_probs, disagreement = ensemble.predict(X, mc_samples=50)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class EnsemblePredictor:
    """Load and run inference across multiple LSTM models."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_paths: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        model_dir : str
            Directory containing *.keras model files.
            All files matching ``*model*.keras`` are loaded.
        model_paths : list of str
            Explicit list of model file paths (overrides model_dir).
        """
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from keras.models import load_model
        from src.models.layers import MCDropout, binary_focal_loss

        self.models = []
        self._mc_fns = []

        focal_loss = binary_focal_loss(gamma=2.0, alpha=0.75)
        custom_objects = {"MCDropout": MCDropout, "binary_focal_loss": focal_loss}

        if model_paths:
            paths = [Path(p) for p in model_paths]
        else:
            d = Path(model_dir) if model_dir else PROJECT_ROOT / "outputs" / "models"
            # Load numbered ensemble models: lstm_model_0.keras, lstm_model_1.keras, …
            paths = sorted(d.glob("lstm_model_[0-9]*.keras"))
            if not paths:
                # Fall back to any *model*.keras
                paths = sorted(d.glob("*model*.keras"))

        if not paths:
            logger.warning("No ensemble model files found — single-model mode")
            return

        for p in paths:
            try:
                m = load_model(str(p), custom_objects=custom_objects)
                self.models.append(m)
                has_sym = len(m.inputs) > 1
                if has_sym:
                    self._mc_fns.append(tf.function(lambda x, s, _m=m: _m([x, s], training=True)))
                else:
                    self._mc_fns.append(tf.function(lambda x, s, _m=m: _m(x, training=True)))
                logger.info(f"Ensemble: loaded {p.name} (multi-input={has_sym})")
            except Exception as e:
                logger.warning(f"Ensemble: failed to load {p.name}: {e}")

        logger.info(f"Ensemble: {len(self.models)} models loaded")

    @property
    def n_models(self) -> int:
        return len(self.models)

    def predict(
        self,
        X: np.ndarray,
        mc_samples: int = 50,
        symbol_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run MC-Dropout inference across all ensemble members.

        Parameters
        ----------
        X : ndarray, shape (batch, window+1, n_features)
        mc_samples : int
            Number of MC-Dropout forward passes per model.
        symbol_ids : ndarray, shape (batch, 1), optional
            Integer symbol IDs for models with embedding inputs.

        Returns
        -------
        mean_probs : ndarray, shape (batch, n_outputs)
        std_probs : ndarray, shape (batch, n_outputs)
        disagreement : float
            Std of per-model means (0 = full consensus, higher = disagreement).
        """
        if not self.models:
            raise RuntimeError("No models loaded for ensemble prediction")

        import tensorflow as tf
        sym = symbol_ids if symbol_ids is not None else np.zeros((X.shape[0], 1), dtype=np.int32)

        all_preds = []
        per_model_means = []

        for fn in self._mc_fns:
            model_preds = np.array([fn(X, sym).numpy() for _ in range(mc_samples)])
            all_preds.append(model_preds)
            per_model_means.append(model_preds.mean(axis=0))

        stacked = np.concatenate(all_preds, axis=0)
        mean_probs = stacked.mean(axis=0)
        std_probs = stacked.std(axis=0)

        if len(per_model_means) > 1:
            model_means = np.stack(per_model_means, axis=0)
            disagreement = float(model_means.std(axis=0).mean())
        else:
            disagreement = 0.0

        return mean_probs, std_probs, disagreement
