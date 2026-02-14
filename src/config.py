"""
Unified configuration for the AI Market Predictor project.

Provides default values for both the *training pipeline* and the
*live trading bot*.  Values can be overridden from JSON config files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Training pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for run_forecast.py and walk_forward_retrain.py."""

    DATA_DIR: str = "src/data/indicators_data/processed"
    FORECAST_DIR: str = "outputs/forecasts"
    MODEL_DIR: str = "outputs/models"
    CACHE_DIR: str = "outputs/cache"

    WINDOW_SIZE: int = 90
    TRAIN_VAL_FRAC: float = 0.8
    VAL_FRAC_WITHIN_TRAIN: float = 0.2
    MC_DROPOUT_SAMPLES: int = 50
    PROB_THRESHOLD: float = 0.7

    MIN_VAL_AUC: float = 0.55
    BATCH_SIZE: int = 128
    MAX_EPOCHS: int = 500
    EARLY_STOP_PATIENCE: int = 15
    REDUCE_LR_PATIENCE: int = 5

    def resolve_paths(self, root: Optional[Path] = None) -> Dict[str, Path]:
        """Return resolved Path objects relative to *root* (default: PROJECT_ROOT)."""
        root = root or PROJECT_ROOT
        return {
            "DATA_DIR": root / self.DATA_DIR,
            "FORECAST_DIR": root / self.FORECAST_DIR,
            "MODEL_DIR": root / self.MODEL_DIR,
            "CACHE_DIR": root / self.CACHE_DIR,
        }


# ---------------------------------------------------------------------------
# Bot configuration
# ---------------------------------------------------------------------------

@dataclass
class BotConfig:
    """Configuration for the live trading bot (src/trading/bot.py)."""

    # MT5 Bridge connection
    MT5_HOST: str = "127.0.0.1"
    MT5_PORT: int = 8787
    MAGIC: int = 24001

    # Model artefacts (relative to project root)
    MODEL_PATH: str = "outputs/models/lstm_model.keras"
    SCALER_PATH: str = "outputs/models/scaler.pkl"
    FEATURES_PATH: str = "outputs/models/feature_cols.json"
    SYMBOLS_PATH: str = "config/symbols.json"

    # Data paths
    RAW_DIR: str = "src/data/indicators_data/raw"
    PROCESSED_DIR: str = "src/data/indicators_data/processed"

    # Strategy parameters
    MIN_ACCEPTED: float = 0.50
    MIN_ACCEPTED_BUY: float = 0.10
    MIN_ACCEPTED_SELL: float = 0.05
    STD_FACTOR: float = 1.0
    MC_DROPOUT_SAMPLES: int = 50
    WINDOW_SIZE: int = 90

    # Risk management
    RISK_PER_TRADE_PCT: float = 1.0
    DEFAULT_LOT_SIZE: float = 0.01
    MAX_LOT_SIZE: float = 1.0
    ATR_SL_MULTIPLIER: float = 2.0
    ATR_TP_MULTIPLIER: float = 3.0
    HORIZON_SL_SCALE: Dict[str, float] = field(
        default_factory=lambda: {"1d": 0.5, "1w": 1.0, "1m": 2.0, "6m": 4.0}
    )
    HORIZON_TP_SCALE: Dict[str, float] = field(
        default_factory=lambda: {"1d": 0.5, "1w": 1.0, "1m": 2.0, "6m": 4.0}
    )
    HORIZON_MAX_HOLD_HOURS: Dict[str, int] = field(
        default_factory=lambda: {"1d": 36, "1w": 168, "1m": 720, "6m": 4320}
    )
    MAX_DRAWDOWN_PCT: float = 15.0

    # Per-instrument-type slot allocation
    SLOTS_PER_TYPE: Dict[str, int] = field(
        default_factory=lambda: {"Index": 2, "Forex": 3, "Crypto": 1, "Commodity": 2}
    )
    RISK_PER_TYPE: Dict[str, float] = field(
        default_factory=lambda: {"Index": 1.0, "Forex": 0.75, "Crypto": 0.5, "Commodity": 1.0}
    )
    GLOBAL_MAX_SLOTS: int = 5
    GLOBAL_MAX_RISK_PCT: float = 5.0

    # Correlation filter
    CORRELATION_THRESHOLD: float = 0.7
    CORRELATION_LOOKBACK: int = 60

    # Trailing stop
    TRAILING_STOP_ENABLED: bool = True
    BREAKEVEN_AFTER_R: float = 1.0
    TRAILING_ATR_MULTIPLIER: float = 1.5

    # Signal reversal exit
    SIGNAL_REVERSAL_MARGIN: float = 0.04

    # Spread / liquidity gate
    MAX_SPREAD_ATR_RATIO: float = 0.15

    # Weekend gap-risk protection
    CLOSE_BEFORE_WEEKEND: bool = True
    FRIDAY_CLOSE_HOUR_UTC: int = 20

    # Regime / volatility filter
    VOLATILITY_FILTER_ENABLED: bool = True
    ATR_PERCENTILE_LOW: int = 10
    ATR_PERCENTILE_HIGH: int = 95
    ATR_PERCENTILE_LOOKBACK: int = 252

    # Horizon weighting
    HORIZON_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {"1d": 1.5, "1w": 1.2, "1m": 1.0, "6m": 0.6}
    )

    # Walk-forward retrain
    RETRAIN_ENABLED: bool = True
    RETRAIN_INTERVAL_DAYS: int = 30

    # Notifications
    NOTIFY_WEBHOOK_URL: str = ""
    NOTIFY_ON_TRADE: bool = True
    NOTIFY_ON_DRAWDOWN: bool = True
    NOTIFY_ON_ERROR: bool = True

    # Bridge logging
    BRIDGE_LOG_DIR: str = "outputs/bridge_logs"

    # State persistence
    STATE_FILE: str = "outputs/bot_state.json"

    # Scheduling
    CHECK_INTERVAL_SECONDS: int = 300
    DAILY_REFRESH_HOUR: int = 0
    POSITION_REVIEW_HOURS: int = 24


def load_training_config(path: Optional[Path] = None) -> TrainingConfig:
    """Load training configuration from JSON, falling back to defaults."""
    cfg = TrainingConfig()
    if path and path.exists():
        with open(path, "r") as f:
            overrides = json.load(f).get("training", {})
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        logger.info(f"Loaded training config overrides from {path}")
    return cfg


def load_bot_config(path: Optional[Path] = None) -> dict:
    """Load bot config from JSON into a plain dict (backward-compatible).

    Returns a dict matching the existing DEFAULT_CONFIG interface so
    callers don't need to change immediately.
    """
    cfg = BotConfig()
    config_dict = asdict(cfg)

    json_path = path or (PROJECT_ROOT / "config" / "bot_config.json")
    if json_path.exists():
        with open(json_path, "r") as f:
            user_cfg = json.load(f)
        config_dict.update(user_cfg)
        logger.info(f"Loaded bot config from {json_path}")
    else:
        logger.info("No bot_config.json found — using defaults")
    return config_dict
