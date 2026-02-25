"""
Data quality and quantity checks for the trading pipeline.

Run before training or inference to catch problems early:
  - Missing / stale data files
  - NaN / inf contamination
  - Feature count mismatches
  - Insufficient history length
  - Sentiment data freshness
  - Distribution drift detection

Usage:
    from src.data.quality_checks import run_all_checks
    issues = run_all_checks()  # returns list of (severity, message)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
Severity = str  # "ERROR" | "WARNING" | "INFO"
Issue = Tuple[Severity, str]


def check_processed_data(
    processed_dir: Optional[Path] = None,
    min_rows: int = 200,
    max_nan_pct: float = 5.0,
) -> List[Issue]:
    """Validate all processed CSV files."""
    issues: List[Issue] = []
    d = processed_dir or PROJECT_ROOT / "src" / "data" / "indicators_data" / "processed"

    if not d.exists():
        issues.append(("ERROR", f"Processed data directory not found: {d}"))
        return issues

    csv_files = list(d.rglob("*_processed.csv"))
    if not csv_files:
        issues.append(("ERROR", f"No processed CSV files found in {d}"))
        return issues

    issues.append(("INFO", f"Found {len(csv_files)} processed data files"))

    for f in csv_files:
        try:
            df = pd.read_csv(f, parse_dates=["date"])
        except Exception as e:
            issues.append(("ERROR", f"{f.name}: failed to read — {e}"))
            continue

        # Row count
        if len(df) < min_rows:
            issues.append(("WARNING", f"{f.name}: only {len(df)} rows (need {min_rows}+)"))

        # NaN check
        nan_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        if nan_pct > max_nan_pct:
            issues.append(("WARNING", f"{f.name}: {nan_pct:.1f}% NaN values"))

        # Inf check
        numeric = df.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric.values).sum()
        if inf_count > 0:
            issues.append(("WARNING", f"{f.name}: {inf_count} inf values"))

        # Staleness (last date)
        if "date" in df.columns and len(df) > 0:
            last_date = pd.Timestamp(df["date"].max())
            days_stale = (pd.Timestamp.now(tz="UTC").tz_localize(None) - last_date).days
            if days_stale > 5:
                issues.append(("WARNING", f"{f.name}: data {days_stale} days stale (last: {last_date.date()})"))

    return issues


def check_sentiment_data(max_age_hours: int = 48) -> List[Issue]:
    """Check sentiment data freshness and quality."""
    issues: List[Issue] = []
    hist_path = PROJECT_ROOT / "outputs" / "sentiment" / "sentiment_history.csv"

    if not hist_path.exists():
        issues.append(("WARNING", "No sentiment_history.csv found — sentiment features will be zeros"))
        return issues

    try:
        df = pd.read_csv(hist_path, parse_dates=["date"])
    except Exception as e:
        issues.append(("ERROR", f"Failed to read sentiment history: {e}"))
        return issues

    issues.append(("INFO", f"Sentiment history: {len(df)} rows, {df['symbol'].nunique()} symbols"))

    # Freshness
    if len(df) > 0:
        latest = pd.Timestamp(df["date"].max())
        hours_old = (pd.Timestamp.now(tz="UTC").tz_localize(None) - latest).total_seconds() / 3600
        if hours_old > max_age_hours:
            issues.append(("WARNING", f"Sentiment data is {hours_old:.0f}h old (max: {max_age_hours}h)"))

    # Check for symbols with very few data points
    counts = df.groupby("symbol").size()
    sparse = counts[counts < 5]
    if len(sparse) > 0:
        issues.append(("WARNING", f"{len(sparse)} symbols have <5 sentiment data points"))

    return issues


def check_model_artifacts() -> List[Issue]:
    """Verify model artifacts exist and are consistent."""
    issues: List[Issue] = []
    models_dir = PROJECT_ROOT / "outputs" / "models"

    required_files = {
        "lstm_model.keras": "Trained model",
        "scaler.joblib": "Feature scaler",
        "feature_cols.json": "Feature column list",
    }

    for fname, desc in required_files.items():
        p = models_dir / fname
        if not p.exists():
            issues.append(("ERROR", f"Missing {desc}: {p}"))
        else:
            size_mb = p.stat().st_size / (1024 * 1024)
            issues.append(("INFO", f"{desc}: {fname} ({size_mb:.1f} MB)"))

    # Feature hash check
    hash_path = models_dir / "feature_hash.txt"
    cols_path = models_dir / "feature_cols.json"
    if hash_path.exists() and cols_path.exists():
        import hashlib
        with open(cols_path) as f:
            cols = json.load(f)
        actual = hashlib.sha256(json.dumps(sorted(cols)).encode()).hexdigest()[:16]
        expected = hash_path.read_text().strip()
        if actual != expected:
            issues.append(("ERROR", f"Feature hash mismatch: saved={expected}, current={actual}"))
        else:
            issues.append(("INFO", f"Feature hash OK: {actual}"))

    # Calibrator
    cal_path = models_dir / "calibrator.joblib"
    if cal_path.exists():
        issues.append(("INFO", "Calibrator found — predictions will be calibrated"))
    else:
        issues.append(("INFO", "No calibrator — raw model probabilities will be used"))

    return issues


def check_feature_distribution(
    processed_dir: Optional[Path] = None,
    reference_stats_path: Optional[Path] = None,
    drift_threshold: float = 3.0,
) -> List[Issue]:
    """Detect feature distribution drift by comparing current stats to a reference.

    If no reference exists, saves the current stats as the new reference.
    """
    issues: List[Issue] = []
    d = processed_dir or PROJECT_ROOT / "src" / "data" / "indicators_data" / "processed"
    ref_path = reference_stats_path or PROJECT_ROOT / "outputs" / "models" / "feature_stats_reference.csv"

    csv_files = list(d.rglob("*_processed.csv"))
    if not csv_files:
        return issues

    # Compute current aggregate stats
    all_dfs = []
    for f in csv_files[:10]:  # sample up to 10 files for speed
        try:
            df = pd.read_csv(f).select_dtypes(include=[np.number])
            all_dfs.append(df.tail(100))  # latest 100 rows each
        except Exception:
            continue

    if not all_dfs:
        return issues

    combined = pd.concat(all_dfs, ignore_index=True)
    current_stats = pd.DataFrame({
        "mean": combined.mean(),
        "std": combined.std(),
    })

    if ref_path.exists():
        ref = pd.read_csv(ref_path, index_col=0)
        # Check drift for common columns
        common = current_stats.index.intersection(ref.index)
        for col in common:
            ref_mean = ref.loc[col, "mean"]
            ref_std = ref.loc[col, "std"]
            cur_mean = current_stats.loc[col, "mean"]
            if ref_std > 0:
                z = abs(cur_mean - ref_mean) / ref_std
                if z > drift_threshold:
                    issues.append(("WARNING", f"Feature drift: {col} shifted {z:.1f} std from reference"))
    else:
        # Save as reference
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        current_stats.to_csv(ref_path)
        issues.append(("INFO", f"Saved feature distribution reference: {ref_path}"))

    return issues


def run_all_checks(**kwargs) -> List[Issue]:
    """Run all data quality checks and return a list of issues."""
    logger.info("Running data quality checks...")
    all_issues: List[Issue] = []

    all_issues.extend(check_processed_data())
    all_issues.extend(check_sentiment_data())
    all_issues.extend(check_model_artifacts())
    all_issues.extend(check_feature_distribution())

    # Summary
    errors = sum(1 for sev, _ in all_issues if sev == "ERROR")
    warnings = sum(1 for sev, _ in all_issues if sev == "WARNING")
    logger.info(f"Quality check complete: {errors} errors, {warnings} warnings")

    for sev, msg in all_issues:
        if sev == "ERROR":
            logger.error(f"  [ERROR] {msg}")
        elif sev == "WARNING":
            logger.warning(f"  [WARN]  {msg}")
        else:
            logger.info(f"  [INFO]  {msg}")

    return all_issues


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_all_checks()
