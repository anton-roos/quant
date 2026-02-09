"""
Per-instrument-type trade slot allocation with a global cap.

Enforces both per-type limits (e.g., max 3 forex, max 1 crypto) and a
global maximum to prevent over-exposure when multiple categories are
active simultaneously.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default slot limits per instrument type
DEFAULT_SLOTS_PER_TYPE: Dict[str, int] = {
    "Index": 2,
    "Forex": 3,
    "Crypto": 1,
    "Commodity": 2,
}

# Default risk-per-trade by instrument type (fraction of equity)
DEFAULT_RISK_PER_TYPE: Dict[str, float] = {
    "Index": 1.0,       # 1.0% risk per trade
    "Forex": 0.75,      # 0.75% risk per trade
    "Crypto": 0.5,      # 0.5% risk per trade (higher vol = less risk)
    "Commodity": 1.0,    # 1.0% risk per trade
}

DEFAULT_GLOBAL_MAX_SLOTS = 5
DEFAULT_GLOBAL_MAX_RISK_PCT = 5.0  # 5% total portfolio risk across all open trades


class SlotManager:
    """Manages per-instrument-type trade slot allocation with a global cap."""

    def __init__(
        self,
        slots_per_type: Optional[Dict[str, int]] = None,
        risk_per_type: Optional[Dict[str, float]] = None,
        global_max_slots: int = DEFAULT_GLOBAL_MAX_SLOTS,
        global_max_risk_pct: float = DEFAULT_GLOBAL_MAX_RISK_PCT,
    ):
        self.slots_per_type = slots_per_type or DEFAULT_SLOTS_PER_TYPE.copy()
        self.risk_per_type = risk_per_type or DEFAULT_RISK_PER_TYPE.copy()
        self.global_max_slots = global_max_slots
        self.global_max_risk_pct = global_max_risk_pct

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def count_open_by_type(
        open_positions: List[Dict],
        symbols: List[Dict],
    ) -> Dict[str, int]:
        """Count open positions grouped by instrument type.

        Args:
            open_positions: List of open position dicts (must have 'symbol' key).
            symbols: List of symbol definitions from symbols.json.

        Returns:
            Dict mapping instrument type (e.g. "Forex") to count.
        """
        name_to_type: Dict[str, str] = {
            s["name"]: s.get("type", "Unknown") for s in symbols
        }
        counts: Dict[str, int] = {}
        for pos in open_positions:
            sym = pos.get("symbol", "")
            itype = name_to_type.get(sym, "Unknown")
            counts[itype] = counts.get(itype, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Slot queries
    # ------------------------------------------------------------------

    def type_slots_remaining(
        self,
        instrument_type: str,
        open_by_type: Dict[str, int],
    ) -> int:
        """Return how many more slots are available for *instrument_type*."""
        limit = self.slots_per_type.get(instrument_type, 0)
        used = open_by_type.get(instrument_type, 0)
        return max(0, limit - used)

    def global_slots_remaining(self, open_by_type: Dict[str, int]) -> int:
        """Return how many global slots are still available."""
        total_open = sum(open_by_type.values())
        return max(0, self.global_max_slots - total_open)

    def available_slots(
        self,
        instrument_type: str,
        open_by_type: Dict[str, int],
    ) -> int:
        """Available slots for *instrument_type*, respecting both limits."""
        return min(
            self.type_slots_remaining(instrument_type, open_by_type),
            self.global_slots_remaining(open_by_type),
        )

    def can_open_trade(
        self,
        instrument_type: str,
        open_by_type: Dict[str, int],
        current_total_risk_pct: float = 0.0,
    ) -> bool:
        """Check if a new trade is allowed for the given instrument type.

        Validates both slot availability and global risk budget.
        """
        if self.available_slots(instrument_type, open_by_type) <= 0:
            return False
        # Check if adding this trade would exceed global risk budget
        new_risk = current_total_risk_pct + self.get_risk_per_trade(instrument_type)
        return new_risk <= self.global_max_risk_pct

    def get_risk_per_trade(self, instrument_type: str) -> float:
        """Return the risk-per-trade (%) for the given instrument type."""
        return self.risk_per_type.get(instrument_type, 1.0)

    # ------------------------------------------------------------------
    # Logging / diagnostics
    # ------------------------------------------------------------------

    def log_slot_status(self, open_by_type: Dict[str, int]) -> None:
        """Log current slot usage across all instrument types."""
        total_open = sum(open_by_type.values())
        logger.info(
            f"Slot status: {total_open}/{self.global_max_slots} global | "
            + " | ".join(
                f"{itype}: {open_by_type.get(itype, 0)}/{limit}"
                for itype, limit in sorted(self.slots_per_type.items())
            )
        )

    @classmethod
    def from_config(cls, config: dict) -> "SlotManager":
        """Construct a SlotManager from a bot config dict."""
        return cls(
            slots_per_type=config.get("SLOTS_PER_TYPE", DEFAULT_SLOTS_PER_TYPE),
            risk_per_type=config.get("RISK_PER_TYPE", DEFAULT_RISK_PER_TYPE),
            global_max_slots=config.get("GLOBAL_MAX_SLOTS", DEFAULT_GLOBAL_MAX_SLOTS),
            global_max_risk_pct=config.get("GLOBAL_MAX_RISK_PCT", DEFAULT_GLOBAL_MAX_RISK_PCT),
        )
