"""
ML pipeline tests — covers data processing, indicators, config validation,
and backtester determinism.
"""
import os
import sys
import tempfile
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# 1.  DATA-PROCESSING / INDICATOR TESTS
# =========================================================================

def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data that exercises every indicator."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    opn = close + rng.randn(n) * 0.3
    vol = rng.randint(100, 10000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "open": opn, "high": high,
        "low": low, "close": close, "volume": vol,
    })


class TestRSI:
    """Validate RSI calculation uses Wilder's exponential smoothing."""

    def test_rsi_bounded_0_100(self):
        df = _make_ohlcv(300)
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        valid = rsi.dropna()
        assert valid.min() >= 0, "RSI must be >= 0"
        assert valid.max() <= 100, "RSI must be <= 100"

    def test_rsi_monotone_up_gives_high_rsi(self):
        """A steadily rising series should yield RSI > 70."""
        n = 100
        df = pd.DataFrame({"close": np.linspace(100, 200, n)})
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        # After warm-up, RSI should be very high
        assert rsi.iloc[-1] > 70

    def test_rsi_monotone_down_gives_low_rsi(self):
        n = 100
        df = pd.DataFrame({"close": np.linspace(200, 100, n)})
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        assert rsi.iloc[-1] < 30


class TestMACD:
    def test_macd_crossover(self):
        """MACD line should cross signal line on a rising series."""
        n = 200
        close = pd.Series(np.linspace(100, 150, n))
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        # On a monotonically rising series, MACD should be positive
        assert macd.iloc[-1] > 0, "MACD should be positive for rising prices"
        # And histogram (MACD - signal) should be positive eventually
        hist = macd - signal
        assert hist.iloc[-1] > 0


class TestBollingerBands:
    def test_close_within_bands(self):
        """Close should generally fall within +/- 2 std Bollinger Bands."""
        df = _make_ohlcv(250)
        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        # Drop NaN warm-up
        valid = df["close"].iloc[20:]
        pct_within = ((valid <= upper.iloc[20:]) & (valid >= lower.iloc[20:])).mean()
        # ~95% of data should be within 2-sigma bands for normal data
        assert pct_within > 0.80, f"Only {pct_within:.0%} within bands"

    def test_bollinger_width_positive(self):
        df = _make_ohlcv(100)
        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        width = (2 * std20 * 2) / ma20
        assert (width.dropna() > 0).all()


class TestVolatility:
    def test_rolling_volatility_positive(self):
        df = _make_ohlcv(100)
        vol10 = df["close"].pct_change().rolling(10).std()
        assert (vol10.dropna() >= 0).all()

    def test_atr_positive(self):
        df = _make_ohlcv(100)
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift(1)).abs(),
            "lc": (df["low"] - df["close"].shift(1)).abs(),
        }).max(axis=1)
        atr = tr.rolling(14).mean()
        assert (atr.dropna() > 0).all()


class TestRegimeFeatures:
    def test_regime_features_added(self):
        """Verify add_regime_features adds expected columns."""
        from src.data.processor import add_regime_features
        df = _make_ohlcv(100)
        result = add_regime_features(df.copy())
        expected_cols = [
            "regime_high_vol", "regime_trend",
            "rolling_sharpe_20d", "rolling_sharpe_60d",
            "mean_reversion_zscore", "ret_autocorr_5d",
            "ATR_14", "ATR_norm",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestProcessFile:
    """Integration test: process a synthetic CSV through the full pipeline."""

    def test_process_file_smoke(self, tmp_path):
        from src.data.processor import process_file

        # Create synthetic raw CSV in a "forex" subfolder (so asset class is detected)
        raw_dir = tmp_path / "forex"
        raw_dir.mkdir()
        csv_path = raw_dir / "TEST_daily.csv"
        df = _make_ohlcv(200)
        df.to_csv(csv_path, index=False)

        out_path = tmp_path / "TEST_daily_processed.csv"

        # Patch cross-asset loader to avoid needing real reference files
        with patch("src.data.processor._load_cross_asset_series", return_value={}):
            process_file(str(csv_path), str(out_path))

        assert out_path.exists()
        result = pd.read_csv(out_path)
        assert len(result) > 50, "Should have many rows after dropping NaN warm-up"
        # Verify key processed columns exist
        assert "RSI" in result.columns
        assert "MACD" in result.columns
        assert "BollingerUpper" in result.columns
        assert "asset_forex" in result.columns
        # Raw OHLCV should be dropped
        for raw in ["open", "high", "low", "volume"]:
            assert raw not in result.columns

    def test_no_inf_values(self, tmp_path):
        from src.data.processor import process_file

        raw_dir = tmp_path / "indices"
        raw_dir.mkdir()
        csv_path = raw_dir / "IDX_daily.csv"
        df = _make_ohlcv(200)
        df["volume"] = 0  # zero volume → exercises the no-volume path
        df.to_csv(csv_path, index=False)

        out_path = tmp_path / "IDX_daily_processed.csv"
        with patch("src.data.processor._load_cross_asset_series", return_value={}):
            process_file(str(csv_path), str(out_path))

        result = pd.read_csv(out_path)
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any(), \
            "Processed data should contain no inf values"


# =========================================================================
# 2.  CONFIG VALIDATOR TESTS
# =========================================================================

class TestConfigValidator:
    """Test the runtime config validator catches bad configs."""

    def _base_config(self) -> dict:
        """Return a minimal valid config."""
        return {
            "MT5_HOST": "localhost",
            "MT5_PORT": 8787,
            "MAGIC": 24001,
            "MODEL_PATH": "outputs/models/model.keras",
            "SCALER_PATH": "outputs/models/scaler.pkl",
            "FEATURES_PATH": "outputs/models/feature_cols.json",
            "SYMBOLS_PATH": "config/symbols.json",
            "MIN_ACCEPTED": 0.50,
            "STD_FACTOR": 1.0,
            "MC_DROPOUT_SAMPLES": 30,
            "WINDOW_SIZE": 90,
            "MAX_CONCURRENT_POSITIONS": 3,
            "RISK_PER_TRADE_PCT": 1.0,
            "DEFAULT_LOT_SIZE": 0.01,
            "MAX_LOT_SIZE": 1.0,
            "ATR_SL_MULTIPLIER": 2.0,
            "ATR_TP_MULTIPLIER": 3.0,
            "MAX_DRAWDOWN_PCT": 15.0,
            "CORRELATION_THRESHOLD": 0.7,
            "CORRELATION_LOOKBACK": 50,
            "BREAKEVEN_AFTER_R": 1.0,
            "TRAILING_ATR_MULTIPLIER": 1.5,
            "MAX_SPREAD_ATR_RATIO": 0.3,
            "FRIDAY_CLOSE_HOUR_UTC": 20,
            "ATR_PERCENTILE_LOW": 10,
            "ATR_PERCENTILE_HIGH": 90,
            "ATR_PERCENTILE_LOOKBACK": 100,
            "RETRAIN_INTERVAL_DAYS": 30,
            "CHECK_INTERVAL_SECONDS": 300,
            "DAILY_REFRESH_HOUR": 0,
            "POSITION_REVIEW_HOURS": 24,
            "TRAILING_STOP_ENABLED": True,
            "CLOSE_BEFORE_WEEKEND": True,
            "VOLATILITY_FILTER_ENABLED": True,
            "RETRAIN_ENABLED": True,
            "NOTIFY_ON_TRADE": False,
            "NOTIFY_ON_DRAWDOWN": False,
            "HORIZON_WEIGHTS": {"1d": 0.15, "1w": 0.25, "1m": 0.35, "6m": 0.25},
        }

    def test_valid_config_passes(self):
        from src.trading.config_validator import validate_config
        errors = validate_config(self._base_config())
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_out_of_range_detected(self):
        from src.trading.config_validator import validate_config
        cfg = self._base_config()
        cfg["MIN_ACCEPTED"] = 2.0  # Out of [0, 1]
        errors = validate_config(cfg)
        assert any("MIN_ACCEPTED" in e for e in errors)

    def test_wrong_type_detected(self):
        from src.trading.config_validator import validate_config
        cfg = self._base_config()
        cfg["MAX_CONCURRENT_POSITIONS"] = "three"  # Should be int
        errors = validate_config(cfg)
        assert any("MAX_CONCURRENT_POSITIONS" in e for e in errors)

    def test_atr_sl_tp_cross_validation(self):
        from src.trading.config_validator import validate_config
        cfg = self._base_config()
        cfg["ATR_SL_MULTIPLIER"] = 5.0
        cfg["ATR_TP_MULTIPLIER"] = 2.0  # SL > TP → bad risk/reward
        errors = validate_config(cfg)
        assert any("ATR_SL" in e for e in errors)

    def test_validate_config_or_die_raises(self):
        from src.trading.config_validator import validate_config_or_die
        cfg = self._base_config()
        cfg["MT5_PORT"] = -1
        with pytest.raises(ValueError):
            validate_config_or_die(cfg)


# =========================================================================
# 3.  BACKTESTER TESTS
# =========================================================================

class TestBacktester:
    """Test the refactored Backtester class."""

    def _make_forecasts(self, tmp_path, n_symbols=3, n_days=60):
        """Create minimal synthetic forecast CSVs."""
        forecast_dir = tmp_path / "outputs" / "forecasts"
        forecast_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        rng = np.random.RandomState(123)

        for i in range(n_symbols):
            sym = f"SYM{i}"
            close = 100 + np.cumsum(rng.randn(n_days) * 0.5)
            df = pd.DataFrame({
                "Date": dates,
                "Close": close,
                "Pred_Prob_1d": rng.uniform(0.3, 0.8, n_days),
                "Pred_Prob_Std_1d": rng.uniform(0.01, 0.1, n_days),
                "Pred_Prob_1w": rng.uniform(0.3, 0.8, n_days),
                "Pred_Prob_Std_1w": rng.uniform(0.01, 0.1, n_days),
                "Pred_Prob_Down_1d": rng.uniform(0.3, 0.8, n_days),
                "Pred_Prob_Down_Std_1d": rng.uniform(0.01, 0.1, n_days),
            })
            df.to_csv(forecast_dir / f"{sym}_forecast.csv", index=False)
        return tmp_path

    def test_backtester_runs_end_to_end(self, tmp_path):
        from src.inference.run_backtest import Backtester, BacktestConfig
        root = self._make_forecasts(tmp_path)
        # Also need video_dir
        (root / "outputs" / "videos").mkdir(parents=True, exist_ok=True)

        cfg = BacktestConfig(
            project_root=root,
            random_runs=5,          # Fast
            min_accepted=0.40,      # Lower threshold to get some trades
            periods={"1d": 1, "1w": 5},
        )
        bt = Backtester(cfg)
        bt.run()

        assert len(bt.strategy_history) > 0
        assert bt.metrics.get("strat_final") is not None

    def test_backtester_deterministic(self, tmp_path):
        """Two runs with same seed should produce identical results."""
        from src.inference.run_backtest import Backtester, BacktestConfig
        root = self._make_forecasts(tmp_path)
        (root / "outputs" / "videos").mkdir(parents=True, exist_ok=True)

        cfg = BacktestConfig(
            project_root=root,
            random_runs=5,
            min_accepted=0.40,
            periods={"1d": 1},
        )

        bt1 = Backtester(cfg)
        bt1.load_forecasts()
        bt1.run_strategy()
        bt1.run_random_baseline(seed=99)

        bt2 = Backtester(cfg)
        bt2.load_forecasts()
        bt2.run_strategy()
        bt2.run_random_baseline(seed=99)

        np.testing.assert_array_equal(
            np.array(bt1.strategy_history),
            np.array(bt2.strategy_history),
        )
        np.testing.assert_array_equal(bt1.random_results, bt2.random_results)

    def test_no_trades_when_threshold_high(self, tmp_path):
        """With an impossibly high threshold, no trades should occur."""
        from src.inference.run_backtest import Backtester, BacktestConfig
        root = self._make_forecasts(tmp_path)
        (root / "outputs" / "videos").mkdir(parents=True, exist_ok=True)

        cfg = BacktestConfig(
            project_root=root,
            min_accepted=0.99,  # Impossible threshold
            periods={"1d": 1},
            random_runs=2,
        )
        bt = Backtester(cfg)
        bt.load_forecasts()
        bt.run_strategy()
        assert bt.total_trades == 0

    def test_metrics_computed(self, tmp_path):
        from src.inference.run_backtest import Backtester, BacktestConfig
        root = self._make_forecasts(tmp_path)
        (root / "outputs" / "videos").mkdir(parents=True, exist_ok=True)

        cfg = BacktestConfig(
            project_root=root, random_runs=3,
            min_accepted=0.40, periods={"1d": 1},
        )
        bt = Backtester(cfg)
        bt.load_forecasts()
        bt.run_strategy()
        bt.run_random_baseline(seed=42)
        m = bt.compute_metrics()

        for key in ["strat_final", "sharpe", "sortino", "calmar", "profit_factor", "max_dd"]:
            assert key in m, f"Missing metric: {key}"
            assert np.isfinite(m[key]), f"Metric {key} is not finite: {m[key]}"


# =========================================================================
# 4.  CROSS-ASSET CACHE INVALIDATION
# =========================================================================

class TestCrossAssetCache:
    def test_invalidate_clears_cache(self):
        from src.data.processor import _cross_asset_cache, invalidate_cross_asset_cache
        # Seed the cache with dummy data
        _cross_asset_cache["test"] = "dummy"
        invalidate_cross_asset_cache()
        from src.data.processor import _cross_asset_cache as refreshed
        assert len(refreshed) == 0
