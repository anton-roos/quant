"""
Risk-management filters applied before opening new positions.

Extracted from TradingBot so each filter is independently testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.constants import CATEGORY_TYPE_TO_FOLDER, sanitize_filename

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------

def check_drawdown(
    config: dict,
    mt5_client,
    start_equity: Optional[float],
    peak_equity: Optional[float],
) -> Tuple[bool, float, float]:
    """Return ``(halt, start_equity, peak_equity)``.

    ``halt`` is True when the drawdown from peak exceeds the configured max.
    """
    try:
        acct = mt5_client.account_info()
        equity = acct["equity"]

        if start_equity is None:
            start_equity = equity
        if peak_equity is None:
            peak_equity = equity

        peak_equity = max(peak_equity, equity)

        if peak_equity > 0:
            drawdown_pct = ((peak_equity - equity) / peak_equity) * 100
            if drawdown_pct > config["MAX_DRAWDOWN_PCT"]:
                logger.critical(
                    f"DRAWDOWN LIMIT HIT: {drawdown_pct:.1f}% "
                    f"(max {config['MAX_DRAWDOWN_PCT']}%). HALTING."
                )
                return True, start_equity, peak_equity
        return False, start_equity, peak_equity
    except Exception as e:
        logger.error(f"Drawdown check failed: {e}")
        return False, start_equity or 0.0, peak_equity or 0.0


# ---------------------------------------------------------------------------
# Direction-aware correlation filter
# ---------------------------------------------------------------------------

def load_close_series(
    mt5_name: str,
    symbols: List[Dict],
    processed_base: Path,
    cycle_cache: Dict[str, Optional[pd.DataFrame]],
) -> Optional[pd.Series]:
    """Load the daily close series for *mt5_name* from processed data."""
    safe_name = sanitize_filename(mt5_name)
    sym_entry = next((s for s in symbols if s["name"] == mt5_name), None)
    if sym_entry is None:
        return None

    cat_folder = CATEGORY_TYPE_TO_FOLDER.get(sym_entry.get("type", ""), "other")
    csv_path = processed_base / cat_folder / f"{safe_name}_daily_processed.csv"

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
    try:
        return df.set_index("date")["close"]
    except Exception:
        return None


def is_too_correlated(
    config: dict,
    candidate_name: str,
    candidate_side: str,
    open_positions_info: List[Dict],
    symbols: List[Dict],
    processed_base: Path,
    cycle_cache: Dict[str, Optional[pd.DataFrame]],
) -> bool:
    """Direction-aware correlation filter.

    Rejects when the candidate would add concentrated directional exposure.
    """
    threshold = config.get("CORRELATION_THRESHOLD", 0.7)
    lookback = config.get("CORRELATION_LOOKBACK", 60)

    if not open_positions_info:
        return False

    cand_close = load_close_series(candidate_name, symbols, processed_base, cycle_cache)
    if cand_close is None or len(cand_close) < lookback:
        return False

    cand_ret = cand_close.pct_change().iloc[-lookback:]

    for open_pos in open_positions_info:
        open_name = open_pos["symbol"]
        open_side = open_pos.get("type", "BUY")

        open_close = load_close_series(open_name, symbols, processed_base, cycle_cache)
        if open_close is None or len(open_close) < lookback:
            continue

        open_ret = open_close.pct_change().iloc[-lookback:]
        combined = pd.concat(
            [cand_ret.rename("cand"), open_ret.rename("open")], axis=1
        ).dropna()
        if len(combined) < 20:
            continue

        corr = combined["cand"].corr(combined["open"])
        same_direction = candidate_side == open_side
        is_risky = (corr > threshold and same_direction) or (
            corr < -threshold and not same_direction
        )

        if is_risky:
            logger.info(
                f"  Skipping {candidate_name} {candidate_side} — corr={corr:.2f} with "
                f"open {open_name} {open_side} (threshold={threshold})"
            )
            return True
    return False


# ---------------------------------------------------------------------------
# Spread / liquidity gate
# ---------------------------------------------------------------------------

def check_spread(
    config: dict,
    mt5_client,
    mt5_name: str,
    atr: float,
    side: str,
    digits: int,
) -> Tuple[bool, float]:
    """Return ``(ok, live_price)``.  ``ok`` is False if spread is too wide."""
    try:
        live_quote = mt5_client.quote(mt5_name)
        bid = live_quote.get("bid", 0)
        ask = live_quote.get("ask", 0)
        spread = ask - bid
        max_spread = atr * config.get("MAX_SPREAD_ATR_RATIO", 0.15)

        if atr > 0 and spread > max_spread:
            logger.info(
                f"  Skipping {mt5_name} — spread {spread:.{digits}f} > "
                f"max {max_spread:.{digits}f} (ATR×{config.get('MAX_SPREAD_ATR_RATIO', 0.15)})"
            )
            return False, 0.0

        live_price = bid if side == "BUY" else ask
        if not live_price or live_price <= 0:
            live_price = 0.0
        return True, live_price
    except Exception as e:
        logger.debug(f"  Quote failed for {mt5_name}: {e}")
        return True, 0.0  # Allow trade with fallback price


# ---------------------------------------------------------------------------
# Current ATR helper
# ---------------------------------------------------------------------------

def get_current_atr(
    mt5_name: str,
    symbols: List[Dict],
    processed_base: Path,
    cycle_cache: Dict[str, Optional[pd.DataFrame]],
) -> float:
    """Return the latest ATR_14 value for *mt5_name* from processed data."""
    safe_name = sanitize_filename(mt5_name)
    sym_entry = next((s for s in symbols if s["name"] == mt5_name), None)
    if sym_entry is None:
        return 0.0

    cat_folder = CATEGORY_TYPE_TO_FOLDER.get(sym_entry.get("type", ""), "other")
    csv_path = processed_base / cat_folder / f"{safe_name}_daily_processed.csv"

    key = str(csv_path)
    if key not in cycle_cache:
        if not csv_path.exists():
            cycle_cache[key] = None
            return 0.0
        cycle_cache[key] = (
            pd.read_csv(csv_path, parse_dates=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
    df = cycle_cache.get(key)
    if df is None:
        return 0.0

    try:
        if "ATR_14" in df.columns and len(df) > 0:
            return float(df["ATR_14"].iloc[-1])
    except Exception:
        pass
    return 0.0
