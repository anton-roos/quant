"""
Position sizing, stop-loss / take-profit calculation, and
profit-currency conversion.

Extracted from TradingBot so the logic is independently testable.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SL / TP
# ---------------------------------------------------------------------------

def compute_sl_tp(
    config: dict,
    side: str,
    close: float,
    atr: float,
    digits: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute stop-loss and take-profit based on ATR multiples."""
    if atr <= 0:
        return None, None

    sl_dist = atr * config["ATR_SL_MULTIPLIER"]
    tp_dist = atr * config["ATR_TP_MULTIPLIER"]

    if side == "BUY":
        sl = round(close - sl_dist, digits)
        tp = round(close + tp_dist, digits)
    else:
        sl = round(close + sl_dist, digits)
        tp = round(close - tp_dist, digits)
    return sl, tp


# ---------------------------------------------------------------------------
# Profit-currency exchange rate
# ---------------------------------------------------------------------------

def get_profit_currency_rate(
    mt5_client,
    symbol_info: Dict,
    candidate_type: str,
) -> float:
    """Return multiplier to convert 1 unit of profit currency to account currency.

    Returns 1.0 as safe fallback.
    """
    try:
        acct = mt5_client.account_info()
        account_ccy = acct.get("currency", "USD").upper()
        profit_ccy = symbol_info.get("currency_profit", "").upper()

        if not profit_ccy or profit_ccy == "INT":
            name = symbol_info.get("name", "")
            if len(name) == 6 and name.isalpha():
                profit_ccy = name[3:6].upper()
            elif candidate_type == "Forex" and len(name) >= 6:
                profit_ccy = name[3:6].upper()
            else:
                profit_ccy = "USD"

        if profit_ccy == account_ccy:
            return 1.0

        for attempt_symbol, invert in [
            (f"{profit_ccy}{account_ccy}", False),
            (f"{account_ccy}{profit_ccy}", True),
        ]:
            try:
                q = mt5_client.quote(attempt_symbol)
                bid = q.get("bid", 0)
                if bid and bid > 0:
                    return (1.0 / bid) if invert else bid
            except Exception:
                continue

        logger.debug(
            f"Could not find conversion rate for {profit_ccy}→{account_ccy}, assuming 1.0"
        )
        return 1.0
    except Exception as e:
        logger.debug(f"Profit currency rate lookup failed: {e}, assuming 1.0")
        return 1.0


# ---------------------------------------------------------------------------
# Lot sizing
# ---------------------------------------------------------------------------

def compute_lot_size(
    config: dict,
    mt5_client,
    symbol_info: Dict,
    atr: float,
    close: float,
    candidate_type: str = "Forex",
) -> float:
    """Compute position size based on risk-% of equity and ATR-based stop.

    Converts the SL distance into account-currency risk using the profit
    currency exchange rate.
    """
    try:
        acct = mt5_client.account_info()
        equity = acct["equity"]

        # Use per-instrument-type risk % if configured, else fall back to global
        risk_per_type = config.get("RISK_PER_TYPE", {})
        risk_pct = risk_per_type.get(candidate_type, config["RISK_PER_TRADE_PCT"])
        risk_amount = equity * (risk_pct / 100.0)

        sl_distance = atr * config["ATR_SL_MULTIPLIER"]
        if sl_distance <= 0:
            return config["DEFAULT_LOT_SIZE"]

        contract_size = symbol_info.get("trade_contract_size", 100000)
        loss_per_lot_profit_ccy = sl_distance * contract_size
        profit_rate = get_profit_currency_rate(mt5_client, symbol_info, candidate_type)
        loss_per_lot_account = loss_per_lot_profit_ccy * profit_rate

        if loss_per_lot_account <= 0:
            return config["DEFAULT_LOT_SIZE"]

        lot_size = risk_amount / loss_per_lot_account

        vol_min = symbol_info.get("volume_min", 0.01)
        vol_max = symbol_info.get("volume_max", 100.0)
        vol_step = symbol_info.get("volume_step", 0.01)

        lot_size = max(vol_min, min(lot_size, vol_max, config["MAX_LOT_SIZE"]))
        if vol_step > 0:
            lot_size = round(lot_size / vol_step) * vol_step
        lot_size = round(lot_size, 2)
        return lot_size

    except Exception as e:
        logger.warning(f"Lot sizing failed: {e}, using default")
        return config["DEFAULT_LOT_SIZE"]
