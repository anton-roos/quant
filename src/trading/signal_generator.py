"""
Signal generation — model inference and candidate ranking.

Extracted from TradingBot.generate_signals() so the logic is
independently testable and reusable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.constants import (
    CATEGORY_TYPE_TO_FOLDER,
    HORIZONS,
    HORIZON_DAYS,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    csv_path: Path,
    feature_cols: List[str],
    scaler,
    window_size: int,
    cycle_cache: Dict[str, Optional[pd.DataFrame]],
) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    """Read a processed CSV, apply the saved scaler, return the latest window.

    Returns ``(X_window, df)`` or ``None`` if data is insufficient.
    """
    key = str(csv_path)
    if key not in cycle_cache:
        if not csv_path.exists():
            cycle_cache[key] = None
            return None
        cycle_cache[key] = (
            pd.read_csv(csv_path, parse_dates=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
    df = cycle_cache.get(key)
    if df is None:
        return None

    df = df.copy()

    # Add target columns (pipeline expects them even during inference)
    df["Target_1d"] = np.log(df["close"].shift(-1) / df["close"])
    df["Target_1w"] = np.log(df["close"].shift(-5) / df["close"])
    df["Target_1m"] = np.log(df["close"].shift(-21) / df["close"])
    df["Target_6m"] = np.log(df["close"].shift(-126) / df["close"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    if len(df) < window_size + 1:
        return None

    features = df[feature_cols].values
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features_scaled = scaler.transform(features)

    X = features_scaled[-(window_size + 1):]
    X = X.reshape(1, window_size + 1, len(feature_cols))
    return X, df


def mc_predict(model_fn, X: np.ndarray, n_samples: int, symbol_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run MC-Dropout inference and return ``(mean_probs, std_probs)``.

    When ``symbol_id`` is not None the model is called with multi-input
    ``(X, sym_ids)`` to leverage the learned symbol embedding.
    """
    if symbol_id is not None:
        sym_ids = np.array([[symbol_id]], dtype=np.int32)
        preds = np.array([model_fn(X, sym_ids).numpy() for _ in range(n_samples)])
    else:
        preds = np.array([model_fn(X).numpy() for _ in range(n_samples)])
    return preds.mean(axis=0), preds.std(axis=0)


# ---------------------------------------------------------------------------
# Candidate building
# ---------------------------------------------------------------------------

def generate_signals(
    *,
    config: dict,
    symbols: List[Dict],
    model_fn,
    scaler,
    feature_cols: List[str],
    cycle_cache: Dict[str, Optional[pd.DataFrame]],
    processed_base: Path,
    symbol_ids: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Scan all instruments and return ranked trade candidates.

    Each candidate dict contains: symbol, mt5_name, side, horizon,
    horizon_days, pred_prob, pred_std, adj_prob, close, atr, type,
    weighted_score.

    Parameters
    ----------
    symbol_ids : dict, optional
        Mapping from symbol stem (e.g. ``EURUSD_daily_processed``) to
        integer ID for the model's embedding input.
    """
    candidates: List[Dict] = []
    n_horizons = len(HORIZONS)
    window_size = config["WINDOW_SIZE"]
    mc_samples = config["MC_DROPOUT_SAMPLES"]
    _symbol_ids = symbol_ids or {}

    for sym in symbols:
        mt5_name = sym["name"]
        safe_name = sanitize_filename(mt5_name)
        cat_folder = CATEGORY_TYPE_TO_FOLDER.get(sym.get("type", ""), "other")
        csv_path = processed_base / cat_folder / f"{safe_name}_daily_processed.csv"

        result = prepare_features(csv_path, feature_cols, scaler, window_size, cycle_cache)
        if result is None:
            continue

        X, df = result
        sym_stem = f"{safe_name}_daily_processed"
        sym_id = _symbol_ids.get(sym_stem, None)
        mean_probs, std_probs = mc_predict(model_fn, X, mc_samples, symbol_id=sym_id)

        latest_close = float(df["close"].iloc[-1])
        latest_atr = (
            float(df["ATR_14"].iloc[-1])
            if "ATR_14" in df.columns and len(df) > 0
            else 0.0
        )

        # Volatility / regime filter
        if config.get("VOLATILITY_FILTER_ENABLED", False) and "ATR_14" in df.columns:
            lookback = config.get("ATR_PERCENTILE_LOOKBACK", 252)
            atr_series = df["ATR_14"].dropna().iloc[-lookback:]
            if len(atr_series) > 20:
                pctile = float((atr_series < latest_atr).sum() / len(atr_series) * 100)
                lo = config.get("ATR_PERCENTILE_LOW", 10)
                hi = config.get("ATR_PERCENTILE_HIGH", 95)
                if pctile < lo or pctile >= hi:
                    logger.debug(
                        f"  Skipping {mt5_name}: ATR percentile {pctile:.0f}% "
                        f"outside [{lo}, {hi}]"
                    )
                    continue

        for i, h in enumerate(HORIZONS):
            # BUY signal (upside probability, outputs 0..3)
            pred_prob = float(mean_probs[0, i])
            pred_std = float(std_probs[0, i])
            adj_prob = pred_prob - config["STD_FACTOR"] * pred_std

            min_buy = config.get("MIN_ACCEPTED_BUY", config["MIN_ACCEPTED"])
            if adj_prob > min_buy:
                candidates.append(_candidate(
                    safe_name, mt5_name, "BUY", h, HORIZON_DAYS[h],
                    pred_prob, pred_std, adj_prob, latest_close, latest_atr, sym,
                ))

            # SELL signal (downside probability, outputs 4..7)
            pred_prob_down = float(mean_probs[0, n_horizons + i])
            pred_std_down = float(std_probs[0, n_horizons + i])
            adj_prob_down = pred_prob_down - config["STD_FACTOR"] * pred_std_down

            min_sell = config.get("MIN_ACCEPTED_SELL", config["MIN_ACCEPTED"])
            if adj_prob_down > min_sell:
                candidates.append(_candidate(
                    safe_name, mt5_name, "SELL", h, HORIZON_DAYS[h],
                    pred_prob_down, pred_std_down, adj_prob_down,
                    latest_close, latest_atr, sym,
                ))

    # Apply horizon weights to ranking score
    hw = config.get("HORIZON_WEIGHTS", {"1d": 1.5, "1w": 1.2, "1m": 1.0, "6m": 0.6})
    for c in candidates:
        c["weighted_score"] = c["adj_prob"] * hw.get(c["horizon"], 1.0)

    candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
    return candidates


def _candidate(
    safe_name, mt5_name, side, h, days, pred_prob, pred_std, adj_prob,
    close, atr, sym,
) -> Dict:
    return {
        "symbol": safe_name,
        "mt5_name": mt5_name,
        "side": side,
        "horizon": h,
        "horizon_days": days,
        "pred_prob": pred_prob,
        "pred_std": pred_std,
        "adj_prob": adj_prob,
        "close": close,
        "atr": atr,
        "type": sym.get("type", "Unknown"),
    }
