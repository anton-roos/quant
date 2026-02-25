"""
Sentiment feature helpers for the data processor.

Loads sentiment history from ``outputs/sentiment/sentiment_history.csv``
and merges it into a per-instrument DataFrame by (date, symbol).

Features produced per row:
  - ``sentiment_combined``       — blended heuristic+LLM score [-1, 1]
  - ``sentiment_heuristic``     — keyword-only score [-1, 1]
  - ``sentiment_llm``            — LLM score [-1, 1]
  - ``sentiment_llm_conf``       — LLM confidence [0, 1]
  - ``sentiment_headline_count`` — number of headlines found
  - ``sentiment_ma3``            — 3-day rolling mean of combined sentiment
  - ``sentiment_ma7``            — 7-day rolling mean of combined sentiment
  - ``sentiment_momentum``       — change in combined sentiment vs 3 days ago
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
HISTORY_PATH = PROJECT_ROOT / "outputs" / "sentiment" / "sentiment_history.csv"

# Module-level cache so we only read the CSV once per process
_sentiment_cache: Optional[pd.DataFrame] = None


def invalidate_sentiment_cache():
    """Clear the in-memory sentiment cache (call after new ingestion)."""
    global _sentiment_cache
    _sentiment_cache = None


def _load_history() -> pd.DataFrame:
    """Load and cache the full sentiment history."""
    global _sentiment_cache
    if _sentiment_cache is not None:
        return _sentiment_cache

    if not HISTORY_PATH.exists():
        logger.debug("No sentiment_history.csv found — sentiment features will be zeros")
        _sentiment_cache = pd.DataFrame()
        return _sentiment_cache

    try:
        _sentiment_cache = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
        logger.info(f"Loaded sentiment history: {len(_sentiment_cache)} rows")
    except Exception as e:
        logger.warning(f"Failed to load sentiment history: {e}")
        _sentiment_cache = pd.DataFrame()

    return _sentiment_cache


def merge_sentiment_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Merge sentiment features into an instrument DataFrame.

    Parameters
    ----------
    df : DataFrame
        The processed instrument DataFrame (must have a ``date`` column).
    symbol : str
        The instrument name as it appears in sentiment_history.csv.

    Returns
    -------
    DataFrame
        *df* with sentiment columns added (zero-filled where missing).
    """
    hist = _load_history()

    sent_cols = [
        "sentiment_combined", "sentiment_heuristic", "sentiment_llm",
        "sentiment_llm_conf", "sentiment_headline_count",
        "sentiment_ma3", "sentiment_ma7", "sentiment_momentum",
    ]

    if hist.empty or symbol not in hist["symbol"].values:
        for col in sent_cols:
            df[col] = 0.0
        return df

    sym_sent = hist[hist["symbol"] == symbol].copy()
    sym_sent = sym_sent.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Rename columns to match our feature naming
    rename_map = {
        "combined_sentiment": "sentiment_combined",
        "heuristic_sentiment": "sentiment_heuristic",
        "llm_sentiment": "sentiment_llm",
        "llm_confidence": "sentiment_llm_conf",
        "headline_count": "sentiment_headline_count",
    }
    sym_sent = sym_sent.rename(columns=rename_map)

    # Compute rolling features
    sym_sent["sentiment_ma3"] = sym_sent["sentiment_combined"].rolling(3, min_periods=1).mean()
    sym_sent["sentiment_ma7"] = sym_sent["sentiment_combined"].rolling(7, min_periods=1).mean()
    sym_sent["sentiment_momentum"] = sym_sent["sentiment_combined"] - sym_sent["sentiment_combined"].shift(3)

    # Select only the columns we need for merge
    merge_cols = ["date"] + [c for c in sent_cols if c in sym_sent.columns]
    sym_sent = sym_sent[merge_cols]

    # Merge by date
    df = pd.merge(df, sym_sent, on="date", how="left")

    # Forward-fill then zero-fill (sentiment from previous day is still informative)
    for col in sent_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    return df
