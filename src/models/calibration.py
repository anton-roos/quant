"""
Model calibration using Platt scaling (sigmoid) or isotonic regression.

After training, the raw sigmoid outputs from the LSTM may not be
well-calibrated probabilities.  This module fits a calibrator on the
validation set and applies it at inference time to produce better-
calibrated confidence scores.

Usage (training time):
    from src.models.calibration import fit_calibrator, save_calibrator
    calibrator = fit_calibrator(y_val, raw_probs_val, method="isotonic")
    save_calibrator(calibrator)

Usage (inference time):
    from src.models.calibration import load_calibrator, calibrate
    calibrator = load_calibrator()
    calibrated_probs = calibrate(calibrator, raw_probs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATOR_PATH = PROJECT_ROOT / "outputs" / "models" / "calibrator.joblib"


class ProbabilityCalibrator:
    """Wraps per-output isotonic or Platt calibrators."""

    def __init__(self, method: Literal["isotonic", "platt"] = "isotonic"):
        self.method = method
        self.calibrators = []  # one per output column
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit calibrators on validation data.

        Parameters
        ----------
        y_true : ndarray, shape (n_samples, n_outputs)
            Binary ground-truth labels for each output.
        y_prob : ndarray, shape (n_samples, n_outputs)
            Raw model probabilities for each output.
        """
        n_outputs = y_prob.shape[1]
        self.calibrators = []

        for i in range(n_outputs):
            if self.method == "isotonic":
                cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                cal.fit(y_prob[:, i], y_true[:, i])
            else:  # platt
                cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                cal.fit(y_prob[:, i].reshape(-1, 1), y_true[:, i])
            self.calibrators.append(cal)

        self.fitted = True
        logger.info(f"Fitted {self.method} calibrators for {n_outputs} outputs")

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities.

        Parameters
        ----------
        y_prob : ndarray, shape (n_samples, n_outputs)

        Returns
        -------
        ndarray, same shape — calibrated probabilities.
        """
        if not self.fitted:
            logger.warning("Calibrator not fitted — returning raw probabilities")
            return y_prob

        result = np.zeros_like(y_prob)
        for i, cal in enumerate(self.calibrators):
            if self.method == "isotonic":
                result[:, i] = cal.predict(y_prob[:, i])
            else:
                result[:, i] = cal.predict_proba(y_prob[:, i].reshape(-1, 1))[:, 1]
        return result


def fit_calibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: Literal["isotonic", "platt"] = "isotonic",
) -> ProbabilityCalibrator:
    """Convenience: fit and return a calibrator."""
    cal = ProbabilityCalibrator(method=method)
    cal.fit(y_true, y_prob)
    return cal


def save_calibrator(calibrator: ProbabilityCalibrator, path: Optional[Path] = None):
    """Persist fitted calibrator to disk."""
    p = path or CALIBRATOR_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, p)
    logger.info(f"Saved calibrator to {p}")


def load_calibrator(path: Optional[Path] = None) -> Optional[ProbabilityCalibrator]:
    """Load calibrator from disk. Returns None if not found."""
    p = path or CALIBRATOR_PATH
    if not p.exists():
        logger.info("No calibrator found — raw probabilities will be used")
        return None
    cal = joblib.load(p)
    logger.info(f"Loaded calibrator from {p}")
    return cal


def calibrate(
    calibrator: Optional[ProbabilityCalibrator],
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply calibration if a calibrator is available, else pass through."""
    if calibrator is None or not calibrator.fitted:
        return y_prob
    return calibrator.transform(y_prob)
