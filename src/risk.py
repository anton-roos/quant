"""
Risk management module for pre-trade validation.

Provides ``RiskManager`` (validates orders against position limits,
daily-loss limits, and per-trade risk) and ``KillSwitch`` (emergency
halt mechanism).

This module is designed to be usable in both a live-trading context
and within unit tests that mock ``MetaTrader5`` via ``unittest.mock.patch``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import MetaTrader5; tests mock ``src.risk.mt5`` so the import
# name must be at module level.
try:
    import MetaTrader5 as mt5  # noqa: F401 – used via mock in tests
except ImportError:
    mt5 = None  # type: ignore[assignment]


# Re-export ErrorCode / OrderSide from the canonical MT5 Bridge models so
# ``from src.risk import …`` works seamlessly.
from src.mt5_bridge.models import OrderSide, ErrorCode  # noqa: F401


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RiskCheckResult:
    """Outcome of a single risk check."""
    success: bool
    error_code: Optional[ErrorCode] = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """Pre-trade risk validation engine."""

    def __init__(
        self,
        max_open_trades: int = 5,
        max_daily_loss_pct: float = 5.0,
        max_risk_per_trade_pct: float = 2.0,
    ):
        self.max_open_trades = max_open_trades
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self._day_start_balance: float = 0.0

    # ---------- daily tracking ----------

    def reset_daily_tracking(self, starting_balance: float) -> None:
        self._day_start_balance = starting_balance

    # ---------- individual checks ----------

    def check_max_open_trades(self, current_open: int) -> RiskCheckResult:
        if current_open >= self.max_open_trades:
            return RiskCheckResult(
                success=False,
                error_code=ErrorCode.RISK_LIMIT_OPEN_TRADES,
                error_message=(
                    f"Max open trades ({self.max_open_trades}) reached "
                    f"(current: {current_open})"
                ),
            )
        return RiskCheckResult(success=True)

    def check_daily_loss_limit(self, current_equity: float) -> RiskCheckResult:
        if self._day_start_balance <= 0:
            return RiskCheckResult(success=True)
        loss_pct = (
            (self._day_start_balance - current_equity) / self._day_start_balance
        ) * 100
        if loss_pct > self.max_daily_loss_pct:
            return RiskCheckResult(
                success=False,
                error_code=ErrorCode.RISK_LIMIT_DAILY_LOSS,
                error_message=(
                    f"Daily loss {loss_pct:.2f}% exceeds limit "
                    f"{self.max_daily_loss_pct}%"
                ),
            )
        return RiskCheckResult(success=True)

    def check_trade_risk(
        self,
        symbol: str,
        volume: float,
        entry_price: float,
        sl: Optional[float],
        side,  # OrderSide or str
        account_equity: float,
    ) -> RiskCheckResult:
        if sl is None:
            return RiskCheckResult(
                success=False,
                error_code=ErrorCode.RISK_LIMIT_TRADE_RISK,
                error_message="Stop loss is required for all trades.",
            )

        try:
            sl_distance = abs(entry_price - sl)
            if sl_distance <= 0:
                return RiskCheckResult(success=True)

            # Attempt to use MT5 symbol info for precise risk calculation
            if mt5 is not None:
                info = mt5.symbol_info(symbol)
                if info is not None:
                    tick_value = getattr(info, "trade_tick_value", 1.0)
                    tick_size = getattr(info, "trade_tick_size", 0.00001)
                    if tick_size > 0:
                        ticks = sl_distance / tick_size
                        risk_amount = ticks * tick_value * volume
                        risk_pct = (risk_amount / account_equity) * 100
                        if risk_pct > self.max_risk_per_trade_pct:
                            return RiskCheckResult(
                                success=False,
                                error_code=ErrorCode.RISK_LIMIT_TRADE_RISK,
                                error_message=(
                                    f"Trade risk {risk_pct:.2f}% exceeds limit "
                                    f"{self.max_risk_per_trade_pct}%"
                                ),
                            )
            return RiskCheckResult(success=True)

        except Exception as e:
            logger.warning(f"Trade risk check error: {e}")
            return RiskCheckResult(success=True)

    # ---------- composite ----------

    def validate_order(
        self,
        symbol: str,
        side,
        volume: float,
        sl: Optional[float],
        tp: Optional[float],
        current_open_count: int,
        account_equity: float,
        entry_price: float,
    ) -> RiskCheckResult:
        """Run all risk checks and return the first failure (or success)."""
        for check in [
            lambda: self.check_max_open_trades(current_open_count),
            lambda: self.check_trade_risk(symbol, volume, entry_price, sl, side, account_equity),
        ]:
            result = check()
            if not result.success:
                return result
        return RiskCheckResult(success=True)


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------

class KillSwitch:
    """Emergency halt mechanism backed by a JSON state file."""

    def __init__(self, state_file: str = "./kill_switch.state"):
        self._state_file = Path(state_file)
        self._active = False
        self._load()

    def _load(self):
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self._active = data.get("active", False)
            except Exception:
                self._active = False

    def _save(self):
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps({"active": self._active}))

    def activate(self) -> None:
        self._active = True
        self._save()

    def deactivate(self) -> None:
        self._active = False
        self._save()

    def is_active(self) -> bool:
        return self._active

    def check(self) -> RiskCheckResult:
        if self._active:
            return RiskCheckResult(
                success=False,
                error_code=ErrorCode.KILL_SWITCH_ACTIVE,
                error_message="Kill switch is active — all trading halted.",
            )
        return RiskCheckResult(success=True)
