"""
Confidence metrics for backtesting — measures how well the model's
confidence tracks actual trade outcomes.

Adds to the existing backtester:
  - Reliability diagram (calibration curve)
  - Confidence-weighted Sharpe ratio
  - Confidence bucketing analysis
  - Expected Calibration Error (ECE)
  - Brier score decomposition

Usage:
    from src.inference.confidence_metrics import ConfidenceAnalyser
    analyser = ConfidenceAnalyser()
    analyser.record(adj_prob=0.72, pred_std=0.08, actual_return=0.015, side="BUY")
    ...
    analyser.generate_report()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfidenceAnalyser:
    """Collect prediction confidence vs realised outcomes during backtests."""

    def __init__(self):
        self.records: List[Dict] = []

    def record(
        self,
        symbol: str = "",
        side: str = "BUY",
        horizon: str = "1w",
        adj_prob: float = 0.0,
        pred_prob: float = 0.0,
        pred_std: float = 0.0,
        confidence: float = 0.0,
        actual_return: float = 0.0,
        was_correct: Optional[bool] = None,
    ):
        """Record a single prediction-vs-outcome pair."""
        if was_correct is None:
            # Determine correctness: BUY correct if return > 0, SELL if return < 0
            was_correct = (actual_return > 0) if side == "BUY" else (actual_return < 0)

        self.records.append({
            "symbol": symbol,
            "side": side,
            "horizon": horizon,
            "adj_prob": adj_prob,
            "pred_prob": pred_prob,
            "pred_std": pred_std,
            "confidence": confidence,
            "actual_return": actual_return,
            "was_correct": was_correct,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE).

        Lower is better.  ECE = 0 means perfectly calibrated.
        """
        if not self.records:
            return 0.0

        df = self.to_dataframe()
        probs = df["adj_prob"].values
        correct = df["was_correct"].astype(float).values

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = probs[mask].mean()
            bin_acc = correct[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)

        return ece / len(probs) if len(probs) > 0 else 0.0

    def brier_score(self) -> float:
        """Compute Brier score (mean squared error of probabilities)."""
        if not self.records:
            return 0.0
        df = self.to_dataframe()
        return float(((df["adj_prob"] - df["was_correct"].astype(float)) ** 2).mean())

    def confidence_buckets(self, n_buckets: int = 5) -> pd.DataFrame:
        """Bucket trades by confidence and compute hit rate per bucket.

        Returns a DataFrame with columns:
            bucket, n_trades, hit_rate, avg_return, avg_confidence
        """
        if not self.records:
            return pd.DataFrame()

        df = self.to_dataframe()
        df["bucket"] = pd.qcut(df["adj_prob"], n_buckets, labels=False, duplicates="drop")

        result = df.groupby("bucket").agg(
            n_trades=("was_correct", "count"),
            hit_rate=("was_correct", "mean"),
            avg_return=("actual_return", "mean"),
            avg_confidence=("adj_prob", "mean"),
            avg_std=("pred_std", "mean"),
        ).reset_index()

        return result

    def confidence_weighted_sharpe(self, risk_free: float = 0.0) -> float:
        """Sharpe ratio weighted by prediction confidence.

        Higher-confidence trades get more weight, so this measures
        whether confidence is a useful signal for trade quality.
        """
        if not self.records:
            return 0.0

        df = self.to_dataframe()
        weights = df["adj_prob"].values
        returns = df["actual_return"].values

        weighted_returns = weights * returns
        if weighted_returns.std() == 0:
            return 0.0

        return float(
            (weighted_returns.mean() - risk_free)
            / weighted_returns.std()
            * np.sqrt(252)
        )

    def reliability_data(self, n_bins: int = 10) -> pd.DataFrame:
        """Data for plotting a reliability (calibration) diagram.

        Returns DataFrame with: bin_center, predicted_prob, actual_freq, count
        """
        if not self.records:
            return pd.DataFrame()

        df = self.to_dataframe()
        probs = df["adj_prob"].values
        correct = df["was_correct"].astype(float).values

        bin_edges = np.linspace(0, 1, n_bins + 1)
        rows = []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            rows.append({
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                "predicted_prob": probs[mask].mean(),
                "actual_freq": correct[mask].mean(),
                "count": int(mask.sum()),
            })
        return pd.DataFrame(rows)

    def generate_report(self, save_dir: Optional[Path] = None) -> str:
        """Generate a text report + save CSV data."""
        if not self.records:
            return "No confidence data recorded."

        df = self.to_dataframe()
        save_dir = save_dir or PROJECT_ROOT / "outputs"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save raw data
        df.to_csv(save_dir / "confidence_analysis.csv", index=False)

        ece = self.expected_calibration_error()
        brier = self.brier_score()
        cw_sharpe = self.confidence_weighted_sharpe()
        buckets = self.confidence_buckets()

        lines = [
            "=" * 60,
            "CONFIDENCE ANALYSIS REPORT",
            "=" * 60,
            f"Total predictions: {len(self.records)}",
            f"Overall accuracy:  {df['was_correct'].mean():.3f}",
            f"Mean confidence:   {df['adj_prob'].mean():.3f}",
            f"",
            f"Expected Calibration Error (ECE): {ece:.4f}",
            f"Brier Score:                      {brier:.4f}",
            f"Confidence-Weighted Sharpe:        {cw_sharpe:.3f}",
            f"",
            "Confidence Buckets:",
        ]

        if len(buckets) > 0:
            buckets.to_csv(save_dir / "confidence_buckets.csv", index=False)
            for _, row in buckets.iterrows():
                lines.append(
                    f"  Bucket {int(row['bucket'])}: "
                    f"n={int(row['n_trades']):4d}  hit_rate={row['hit_rate']:.3f}  "
                    f"avg_ret={row['avg_return']:+.4f}  avg_conf={row['avg_confidence']:.3f}"
                )

        # Save reliability diagram data
        rel = self.reliability_data()
        if len(rel) > 0:
            rel.to_csv(save_dir / "reliability_diagram.csv", index=False)
            lines.append("")
            lines.append("Reliability Diagram (for plotting):")
            for _, row in rel.iterrows():
                lines.append(
                    f"  predicted={row['predicted_prob']:.3f} "
                    f"actual={row['actual_freq']:.3f} "
                    f"(n={int(row['count'])})"
                )

        report = "\n".join(lines)
        logger.info(report)

        # Save text report
        (save_dir / "confidence_report.txt").write_text(report)
        return report
