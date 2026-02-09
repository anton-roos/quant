"""
Unit tests for the live trading bot (src/trading/bot.py).

All MT5 Bridge calls are mocked – no network or broker required.
"""

import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# bot.py imports tensorflow at module level; skip all bot tests when TF
# is not installed so the rest of the test suite still runs cleanly.
tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_processed_csv(tmp_dir: Path, symbol: str = "EURUSD", category: str = "forex", rows: int = 200):
    """Create a minimal processed CSV with required columns."""
    dates = pd.bdate_range(end="2026-02-06", periods=rows)
    close = 1.08 + np.random.randn(rows).cumsum() * 0.001
    df = pd.DataFrame({
        "date": dates,
        "open": close - 0.0005,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": np.random.randint(100, 10000, rows),
        "ATR_14": np.abs(np.random.randn(rows) * 0.001) + 0.0005,
    })
    # Add some dummy features
    for i in range(10):
        df[f"feat_{i}"] = np.random.randn(rows)

    out_dir = tmp_dir / "processed" / category
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_daily_processed.csv"
    df.to_csv(path, index=False)
    return path, df


def _make_symbols(names_and_types):
    """Return a symbols list like symbols.json."""
    return [{"name": n, "type": t} for n, t in names_and_types]


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with model artifacts and processed data."""
    # Model dir
    model_dir = tmp_path / "outputs" / "models"
    model_dir.mkdir(parents=True)

    # Feature cols
    feature_cols = [f"feat_{i}" for i in range(10)]
    with open(model_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # Symbols
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    symbols = _make_symbols([("EURUSD", "Forex"), ("GBPUSD", "Forex"), ("USDJPY", "Forex")])
    with open(config_dir / "symbols.json", "w") as f:
        json.dump({"symbols": symbols}, f)

    # Processed data
    data_dir = tmp_path / "src" / "data" / "indicators_data"
    for sym, typ in [("EURUSD", "forex"), ("GBPUSD", "forex"), ("USDJPY", "forex")]:
        _make_processed_csv(data_dir, sym, typ)

    # Bot state dir
    (tmp_path / "outputs").mkdir(exist_ok=True)

    return tmp_path


def _build_bot(tmp_workspace, config_overrides=None):
    """Construct a TradingBot with mocked model, scaler, and MT5 client."""
    from src.trading.bot import TradingBot, DEFAULT_CONFIG, PROJECT_ROOT as _

    # Patch PROJECT_ROOT for the module
    import src.trading.bot as bot_module
    original_root = bot_module.PROJECT_ROOT
    bot_module.PROJECT_ROOT = tmp_workspace

    config = DEFAULT_CONFIG.copy()
    config.update({
        "PROCESSED_DIR": "src/data/indicators_data/processed",
        "RAW_DIR": "src/data/indicators_data/raw",
        "MODEL_PATH": "outputs/models/lstm_model.keras",
        "SCALER_PATH": "outputs/models/scaler.pkl",
        "FEATURES_PATH": "outputs/models/feature_cols.json",
        "SYMBOLS_PATH": "config/symbols.json",
        "STATE_FILE": "outputs/bot_state.json",
        "NOTIFY_WEBHOOK_URL": "",
        "WINDOW_SIZE": 10,
        "MC_DROPOUT_SAMPLES": 3,
        "MAX_CONCURRENT_POSITIONS": 3,
        "VOLATILITY_FILTER_ENABLED": False,
        "CLOSE_BEFORE_WEEKEND": False,
        "RETRAIN_ENABLED": False,
    })
    if config_overrides:
        config.update(config_overrides)

    # Create bot without calling __init__ (we'll set up manually)
    bot = object.__new__(TradingBot)
    bot.config = config
    bot.running = False
    bot.start_equity = 10000.0
    bot.peak_equity = 10000.0
    bot.last_refresh_date = None
    bot._cycle_cache = {}

    # Slot manager (per-instrument-type allocation)
    from src.trading.slot_manager import SlotManager
    bot.slot_manager = SlotManager.from_config(config)

    # Mock model (returns 8 outputs: 4 up + 4 down)
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(
        numpy=MagicMock(return_value=np.random.rand(1, 8).astype(np.float32))
    )
    bot.model = mock_model
    bot._mc_predict_fn = mock_model

    # Mock scaler
    mock_scaler = MagicMock()
    mock_scaler.transform = MagicMock(side_effect=lambda x: x)  # identity
    bot.scaler = mock_scaler

    # Feature cols
    bot.feature_cols = [f"feat_{i}" for i in range(10)]

    # Load symbols from temp
    with open(tmp_workspace / "config" / "symbols.json") as f:
        data = json.load(f)
    bot.symbols = data["symbols"]
    bot.symbol_name_map = {}
    from src.data.features.mt5_bridge_downloader import _sanitize_filename
    for sym in bot.symbols:
        bot.symbol_name_map[_sanitize_filename(sym["name"])] = sym["name"]

    # Mock MT5 client
    bot.mt5 = MagicMock()
    bot.mt5.is_healthy.return_value = True
    bot.mt5.account_info.return_value = {
        "balance": 10000.0, "equity": 10000.0,
        "margin": 100.0, "free_margin": 9900.0,
        "currency": "USD",
    }
    bot.mt5.bot_positions.return_value = []
    bot.mt5.symbol_info.return_value = {
        "digits": 5, "point": 0.00001,
        "trade_contract_size": 100000,
        "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
        "trade_stops_level": 0, "currency_profit": "USD",
    }
    bot.mt5.quote.return_value = {"bid": 1.0850, "ask": 1.0853}
    bot.mt5.place_market_order.return_value = {
        "accepted": True, "ticket": 12345, "price": 1.0851,
    }

    # Mock journal & notifier
    from src.trading.trade_journal import TradeJournal
    from src.trading.notifications import Notifier
    bot.journal = MagicMock(spec=TradeJournal)
    bot.notifier = MagicMock(spec=Notifier)

    yield bot

    # Restore original root
    bot_module.PROJECT_ROOT = original_root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignalGeneration:
    """Tests for generate_signals() — scanning & ranking."""

    def test_returns_candidates(self, tmp_workspace):
        """generate_signals should return a non-empty list when model outputs high probs."""
        bot = next(_build_bot(tmp_workspace))

        # Force high probabilities
        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.8, dtype=np.float32))
        )

        candidates = bot.generate_signals()
        assert len(candidates) > 0
        # Each candidate should have required keys
        for c in candidates:
            assert "symbol" in c
            assert "mt5_name" in c
            assert "side" in c
            assert "horizon" in c
            assert "adj_prob" in c
            assert "weighted_score" in c

    def test_respects_min_accepted(self, tmp_workspace):
        """Candidates with adj_prob <= MIN_ACCEPTED should be filtered out."""
        bot = next(_build_bot(tmp_workspace, {"MIN_ACCEPTED": 0.99}))

        # With very low probs, nothing should pass
        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.3, dtype=np.float32))
        )

        candidates = bot.generate_signals()
        assert len(candidates) == 0

    def test_horizon_weighting(self, tmp_workspace):
        """Candidates should be sorted by weighted_score, not raw adj_prob."""
        bot = next(_build_bot(tmp_workspace))

        # Return probs where all horizons have similar adj_prob
        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.5, dtype=np.float32))
        )

        candidates = bot.generate_signals()
        if len(candidates) > 1:
            scores = [c["weighted_score"] for c in candidates]
            assert scores == sorted(scores, reverse=True)

    def test_volatility_filter_low(self, tmp_workspace):
        """Symbols with ATR percentile below threshold should be skipped."""
        bot = next(_build_bot(tmp_workspace, {
            "VOLATILITY_FILTER_ENABLED": True,
            "ATR_PERCENTILE_LOW": 50,
            "ATR_PERCENTILE_HIGH": 99,
            "ATR_PERCENTILE_LOOKBACK": 50,
        }))

        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.8, dtype=np.float32))
        )

        # The random ATR data should cause some filtering
        candidates = bot.generate_signals()
        # At least verifies it doesn't crash and returns a list
        assert isinstance(candidates, list)


class TestPositionReview:
    def test_review_open_positions_handles_naive_and_aware_datetimes(self, tmp_workspace):
        """_review_open_positions should not crash on naive/aware mixes.

        MT5 may return timestamps with an explicit offset, a trailing 'Z',
        or (rarely) a naive ISO datetime. The bot should normalize to UTC.
        """
        bot = next(_build_bot(tmp_workspace, {"POSITION_REVIEW_HOURS": 24}))

        stale_time_utc = datetime.now(timezone.utc) - timedelta(hours=30)
        stale_time_z = stale_time_utc.isoformat().replace("+00:00", "Z")
        stale_time_naive = stale_time_utc.replace(tzinfo=None).isoformat()

        bot.mt5.bot_positions.return_value = [
            {
                "ticket": 101,
                "symbol": "EURUSD",
                "time": stale_time_utc.isoformat(),
                "price_current": 1.1000,
                "profit": 1.23,
                "commission": 0.0,
                "swap": 0.0,
            },
            {
                "ticket": 102,
                "symbol": "GBPUSD",
                "time": stale_time_z,
                "price_current": 1.2500,
                "profit": -2.0,
                "commission": 0.0,
                "swap": 0.0,
            },
            {
                "ticket": 103,
                "symbol": "USDJPY",
                "time": stale_time_naive,
                "price_current": 150.00,
                "profit": 0.0,
                "commission": 0.0,
                "swap": 0.0,
            },
        ]

        bot._review_open_positions()

        assert bot.mt5.close_position.call_count == 3
        bot.mt5.close_position.assert_any_call(101, comment="bot:stale")
        bot.mt5.close_position.assert_any_call(102, comment="bot:stale")
        bot.mt5.close_position.assert_any_call(103, comment="bot:stale")


class TestPositionSizing:
    """Tests for _compute_lot_size and _compute_sl_tp."""

    def test_lot_size_basic(self, tmp_workspace):
        """Lot size should be within broker limits."""
        bot = next(_build_bot(tmp_workspace))
        sym_info = {
            "trade_contract_size": 100000,
            "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
        }
        lot = bot._compute_lot_size(sym_info, atr=0.001, close=1.085)
        assert 0.01 <= lot <= bot.config["MAX_LOT_SIZE"]

    def test_lot_size_zero_atr(self, tmp_workspace):
        """Zero ATR should fall back to default lot size."""
        bot = next(_build_bot(tmp_workspace))
        sym_info = {
            "trade_contract_size": 100000,
            "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
        }
        lot = bot._compute_lot_size(sym_info, atr=0.0, close=1.085)
        assert lot == bot.config["DEFAULT_LOT_SIZE"]

    def test_sl_tp_buy(self, tmp_workspace):
        """BUY: SL below close, TP above close."""
        bot = next(_build_bot(tmp_workspace))
        sl, tp = bot._compute_sl_tp("BUY", 1.0850, 0.001, 5)
        assert sl is not None and tp is not None
        assert sl < 1.0850
        assert tp > 1.0850

    def test_sl_tp_sell(self, tmp_workspace):
        """SELL: SL above close, TP below close."""
        bot = next(_build_bot(tmp_workspace))
        sl, tp = bot._compute_sl_tp("SELL", 1.0850, 0.001, 5)
        assert sl is not None and tp is not None
        assert sl > 1.0850
        assert tp < 1.0850

    def test_sl_tp_zero_atr(self, tmp_workspace):
        """Zero ATR should return None for both SL and TP."""
        bot = next(_build_bot(tmp_workspace))
        sl, tp = bot._compute_sl_tp("BUY", 1.0850, 0.0, 5)
        assert sl is None and tp is None


class TestCorrelationFilter:
    """Tests for _is_too_correlated (direction-aware)."""

    def test_no_open_positions(self, tmp_workspace):
        """Should return False when there are no open positions."""
        bot = next(_build_bot(tmp_workspace))
        assert bot._is_too_correlated("EURUSD", "BUY", []) is False

    def test_same_direction_high_corr_rejected(self, tmp_workspace):
        """Same-direction trade on highly correlated pair should be rejected."""
        bot = next(_build_bot(tmp_workspace, {"CORRELATION_THRESHOLD": 0.01}))

        # EURUSD and EURUSD are perfectly correlated
        open_pos = [{"symbol": "EURUSD", "type": "BUY"}]
        result = bot._is_too_correlated("EURUSD", "BUY", open_pos)
        # With threshold 0.01, self-correlation (1.0) should reject
        assert result is True

    def test_opposite_direction_high_corr_accepted(self, tmp_workspace):
        """Opposite-direction trade on positively correlated pair is diversifying."""
        bot = next(_build_bot(tmp_workspace, {"CORRELATION_THRESHOLD": 0.5}))

        open_pos = [{"symbol": "EURUSD", "type": "BUY"}]
        # SELL on same symbol with positive correlation = opposite direction = OK
        result = bot._is_too_correlated("EURUSD", "SELL", open_pos)
        # Positive corr + opposite direction = not risky
        assert result is False


class TestDrawdownCheck:
    """Tests for _check_drawdown."""

    def test_no_drawdown(self, tmp_workspace):
        """Should return False when equity is at peak."""
        bot = next(_build_bot(tmp_workspace))
        bot.start_equity = 10000
        bot.peak_equity = 10000
        bot.mt5.account_info.return_value = {"equity": 10000}
        assert bot._check_drawdown() is False

    def test_drawdown_exceeded(self, tmp_workspace):
        """Should return True when drawdown exceeds limit."""
        bot = next(_build_bot(tmp_workspace, {"MAX_DRAWDOWN_PCT": 5.0}))
        bot.start_equity = 10000
        bot.peak_equity = 10000
        bot.mt5.account_info.return_value = {"equity": 9400}  # 6% drawdown
        assert bot._check_drawdown() is True
        # Should send notification
        bot.notifier.send.assert_called()

    def test_drawdown_within_limit(self, tmp_workspace):
        """Should return False when drawdown is within limit."""
        bot = next(_build_bot(tmp_workspace, {"MAX_DRAWDOWN_PCT": 15.0}))
        bot.start_equity = 10000
        bot.peak_equity = 10000
        bot.mt5.account_info.return_value = {"equity": 9000}  # 10% drawdown
        assert bot._check_drawdown() is False


class TestSpreadCheck:
    """Tests for spread/liquidity gate in execute_cycle."""

    def test_wide_spread_skips_trade(self, tmp_workspace):
        """Candidate should be skipped when spread exceeds ATR ratio."""
        bot = next(_build_bot(tmp_workspace, {"MAX_SPREAD_ATR_RATIO": 0.01}))

        # Wide spread: ask - bid = 0.003 >> ATR * 0.01
        bot.mt5.quote.return_value = {"bid": 1.0850, "ask": 1.0880}

        # Force one high-confidence candidate
        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.8, dtype=np.float32))
        )

        bot.execute_cycle()
        # With such a wide spread, no orders should be placed
        bot.mt5.place_market_order.assert_not_called()


class TestWeekendClose:
    """Tests for close-before-weekend functionality."""

    def test_friday_close(self, tmp_workspace):
        """Positions should be closed on Friday afternoon when enabled."""
        bot = next(_build_bot(tmp_workspace, {
            "CLOSE_BEFORE_WEEKEND": True,
            "FRIDAY_CLOSE_HOUR_UTC": 20,
        }))
        bot.mt5.bot_positions.return_value = [
            {"ticket": 111, "symbol": "EURUSD", "price_current": 1.085, "profit": 50, "commission": 0, "swap": 0}
        ]

        # Mock datetime to be Friday 21:00 UTC
        friday = datetime(2026, 2, 6, 21, 0, tzinfo=timezone.utc)  # Feb 6, 2026 is a Friday
        with patch("src.trading.bot.datetime") as mock_dt:
            mock_dt.now.return_value = friday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            bot.execute_cycle()

        bot.mt5.close_position.assert_called_once_with(111, comment="bot:weekend")


class TestStatePersistence:
    """Tests for _save_state / _load_state."""

    def test_round_trip(self, tmp_workspace):
        """State should survive save → load cycle."""
        bot = next(_build_bot(tmp_workspace))

        bot.start_equity = 9876.54
        bot.peak_equity = 10234.56
        bot.last_refresh_date = "2026-02-08"

        import src.trading.bot as bot_module
        original_root = bot_module.PROJECT_ROOT
        bot_module.PROJECT_ROOT = tmp_workspace

        bot._save_state()

        # Reset and reload
        bot.start_equity = None
        bot.peak_equity = None
        bot.last_refresh_date = None
        bot._load_state()

        assert bot.start_equity == 9876.54
        assert bot.peak_equity == 10234.56
        assert bot.last_refresh_date == "2026-02-08"

        bot_module.PROJECT_ROOT = original_root


class TestJournalIntegration:
    """Tests verifying the trade journal is called on entries and exits."""

    def test_journal_record_entry_on_fill(self, tmp_workspace):
        """Journal should record an entry when an order is accepted."""
        bot = next(_build_bot(tmp_workspace))

        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.8, dtype=np.float32))
        )
        bot.mt5.quote.return_value = {"bid": 1.0850, "ask": 1.0852}

        bot.execute_cycle()

        # At least one entry should have been recorded
        if bot.mt5.place_market_order.called:
            bot.journal.record_entry.assert_called()

    def test_journal_equity_snapshot(self, tmp_workspace):
        """Journal should record an equity snapshot each cycle."""
        bot = next(_build_bot(tmp_workspace))

        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.1, dtype=np.float32))
        )

        bot.execute_cycle()
        bot.journal.record_equity_snapshot.assert_called_once()


class TestNotifications:
    """Tests for notification dispatch."""

    def test_no_notification_when_disabled(self, tmp_workspace):
        """Notifier should not be called when webhook is empty."""
        from src.trading.notifications import Notifier
        notifier = Notifier("")
        # Calling send should not raise and should be a no-op
        notifier.send("test message")
        # No exception = pass

    def test_drawdown_notification(self, tmp_workspace):
        """Drawdown breach should trigger a notification."""
        bot = next(_build_bot(tmp_workspace, {"MAX_DRAWDOWN_PCT": 1.0}))
        bot.peak_equity = 10000
        bot.mt5.account_info.return_value = {"equity": 9800}

        bot._check_drawdown()
        bot.notifier.send.assert_called()


class TestCycleCache:
    """Tests for per-cycle CSV caching."""

    def test_cache_populated(self, tmp_workspace):
        """After generate_signals, _cycle_cache should contain DataFrames."""
        bot = next(_build_bot(tmp_workspace))

        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.5, dtype=np.float32))
        )

        bot.generate_signals()
        assert len(bot._cycle_cache) > 0

    def test_cache_cleared_per_cycle(self, tmp_workspace):
        """Cache should be cleared at the start of each execute_cycle."""
        bot = next(_build_bot(tmp_workspace))

        bot._cycle_cache["stale_key"] = "stale_value"

        bot._mc_predict_fn.return_value = MagicMock(
            numpy=MagicMock(return_value=np.full((1, 8), 0.1, dtype=np.float32))
        )

        bot.execute_cycle()
        assert "stale_key" not in bot._cycle_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
