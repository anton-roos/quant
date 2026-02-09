"""
Tests for per-instrument-type trade slot allocation (SlotManager).
"""
import pytest

from src.trading.slot_manager import SlotManager


@pytest.fixture
def slot_manager():
    """Moderate profile: Index=2, Forex=3, Crypto=1, Commodity=2, global=5."""
    return SlotManager(
        slots_per_type={"Index": 2, "Forex": 3, "Crypto": 1, "Commodity": 2},
        risk_per_type={"Index": 1.0, "Forex": 0.75, "Crypto": 0.5, "Commodity": 1.0},
        global_max_slots=5,
        global_max_risk_pct=5.0,
    )


@pytest.fixture
def symbols():
    return [
        {"name": "EURUSD", "type": "Forex"},
        {"name": "GBPUSD", "type": "Forex"},
        {"name": "USDJPY", "type": "Forex"},
        {"name": "US 500", "type": "Index"},
        {"name": "US Tech 100", "type": "Index"},
        {"name": "Bitcoin (BTCUSD)", "type": "Crypto"},
        {"name": "Brent Crude Oil", "type": "Commodity"},
        {"name": "Gold", "type": "Commodity"},
    ]


# ---------- count_open_by_type ----------

class TestCountOpenByType:
    def test_empty_positions(self, symbols):
        result = SlotManager.count_open_by_type([], symbols)
        assert result == {}

    def test_mixed_positions(self, symbols):
        open_positions = [
            {"symbol": "EURUSD"},
            {"symbol": "US 500"},
            {"symbol": "Bitcoin (BTCUSD)"},
            {"symbol": "GBPUSD"},
        ]
        result = SlotManager.count_open_by_type(open_positions, symbols)
        assert result == {"Forex": 2, "Index": 1, "Crypto": 1}

    def test_unknown_symbol(self, symbols):
        open_positions = [{"symbol": "UNKNOWN_SYMBOL"}]
        result = SlotManager.count_open_by_type(open_positions, symbols)
        assert result == {"Unknown": 1}


# ---------- type_slots_remaining ----------

class TestTypeSlotsRemaining:
    def test_no_open_positions(self, slot_manager):
        assert slot_manager.type_slots_remaining("Forex", {}) == 3
        assert slot_manager.type_slots_remaining("Crypto", {}) == 1
        assert slot_manager.type_slots_remaining("Index", {}) == 2

    def test_some_used(self, slot_manager):
        open_by_type = {"Forex": 2, "Index": 1}
        assert slot_manager.type_slots_remaining("Forex", open_by_type) == 1
        assert slot_manager.type_slots_remaining("Index", open_by_type) == 1

    def test_fully_used(self, slot_manager):
        open_by_type = {"Crypto": 1}
        assert slot_manager.type_slots_remaining("Crypto", open_by_type) == 0

    def test_unknown_type_returns_zero(self, slot_manager):
        assert slot_manager.type_slots_remaining("Unknown", {}) == 0


# ---------- global_slots_remaining ----------

class TestGlobalSlotsRemaining:
    def test_none_open(self, slot_manager):
        assert slot_manager.global_slots_remaining({}) == 5

    def test_some_open(self, slot_manager):
        assert slot_manager.global_slots_remaining({"Forex": 2, "Index": 1}) == 2

    def test_at_cap(self, slot_manager):
        assert slot_manager.global_slots_remaining({"Forex": 3, "Index": 2}) == 0

    def test_over_cap_returns_zero(self, slot_manager):
        # Shouldn't happen, but defensive
        assert slot_manager.global_slots_remaining({"Forex": 3, "Index": 3}) == 0


# ---------- available_slots ----------

class TestAvailableSlots:
    def test_both_limits_open(self, slot_manager):
        """No open positions => min(type_limit, global_remaining)."""
        # Forex: min(3, 5) = 3
        assert slot_manager.available_slots("Forex", {}) == 3
        # Crypto: min(1, 5) = 1
        assert slot_manager.available_slots("Crypto", {}) == 1

    def test_global_cap_is_binding(self, slot_manager):
        """Global cap reached even though type has room."""
        open_by_type = {"Forex": 2, "Index": 2, "Crypto": 1}  # total = 5
        assert slot_manager.available_slots("Commodity", open_by_type) == 0

    def test_type_cap_is_binding(self, slot_manager):
        """Type cap reached even though global has room."""
        open_by_type = {"Crypto": 1}  # total = 1, global has 4 left
        assert slot_manager.available_slots("Crypto", open_by_type) == 0

    def test_mixed_scenario(self, slot_manager):
        """4 open total, Forex at 2/3 => Forex=1, global=1 => min=1."""
        open_by_type = {"Forex": 2, "Index": 1, "Commodity": 1}
        assert slot_manager.available_slots("Forex", open_by_type) == 1


# ---------- can_open_trade ----------

class TestCanOpenTrade:
    def test_basic_allow(self, slot_manager):
        assert slot_manager.can_open_trade("Forex", {}) is True

    def test_blocked_by_type_slots(self, slot_manager):
        open_by_type = {"Crypto": 1}
        assert slot_manager.can_open_trade("Crypto", open_by_type) is False

    def test_blocked_by_global_slots(self, slot_manager):
        open_by_type = {"Forex": 3, "Index": 2}  # total = 5
        assert slot_manager.can_open_trade("Commodity", open_by_type) is False

    def test_blocked_by_risk_budget(self, slot_manager):
        """Even with slot room, exceeding risk budget blocks the trade."""
        open_by_type = {"Forex": 2}
        # Current risk already at 4.6%, adding Forex (0.75%) = 5.35% > 5.0%
        assert slot_manager.can_open_trade("Forex", open_by_type, current_total_risk_pct=4.6) is False

    def test_allowed_within_risk_budget(self, slot_manager):
        open_by_type = {"Forex": 1}
        # Current risk at 0.75%, adding Forex (0.75%) = 1.5% < 5.0%
        assert slot_manager.can_open_trade("Forex", open_by_type, current_total_risk_pct=0.75) is True


# ---------- get_risk_per_trade ----------

class TestGetRiskPerTrade:
    def test_known_types(self, slot_manager):
        assert slot_manager.get_risk_per_trade("Index") == 1.0
        assert slot_manager.get_risk_per_trade("Forex") == 0.75
        assert slot_manager.get_risk_per_trade("Crypto") == 0.5
        assert slot_manager.get_risk_per_trade("Commodity") == 1.0

    def test_unknown_type_returns_default(self, slot_manager):
        assert slot_manager.get_risk_per_trade("Unknown") == 1.0


# ---------- from_config ----------

class TestFromConfig:
    def test_from_default_config(self):
        config = {}
        sm = SlotManager.from_config(config)
        assert sm.global_max_slots == 5
        assert sm.slots_per_type["Forex"] == 3

    def test_from_custom_config(self):
        config = {
            "SLOTS_PER_TYPE": {"Forex": 4, "Index": 1, "Crypto": 0, "Commodity": 1},
            "RISK_PER_TYPE": {"Forex": 0.5, "Index": 0.5, "Crypto": 0.25, "Commodity": 0.5},
            "GLOBAL_MAX_SLOTS": 4,
            "GLOBAL_MAX_RISK_PCT": 3.0,
        }
        sm = SlotManager.from_config(config)
        assert sm.global_max_slots == 4
        assert sm.slots_per_type["Forex"] == 4
        assert sm.slots_per_type["Crypto"] == 0
        assert sm.global_max_risk_pct == 3.0
        # Crypto slots = 0 => can never open
        assert sm.can_open_trade("Crypto", {}) is False


# ---------- log_slot_status ----------

class TestLogSlotStatus:
    def test_logging(self, slot_manager, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            slot_manager.log_slot_status({"Forex": 2, "Index": 1})
        assert "Slot status:" in caplog.text
        assert "Forex: 2/3" in caplog.text
        assert "Index: 1/2" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
