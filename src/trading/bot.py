"""
Live autonomous trading bot using the trained LSTM model and MT5 Bridge.

Continuously monitors all instruments, generates predictions via MC-Dropout
inference, and opens/closes positions through the MT5 Bridge REST API.

Usage:
    python -m src.trading.bot            # From project root
    python src/trading/bot.py            # Direct execution
"""

import os
import sys
import json
import time
import logging
import signal
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Ensure project root is on the path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# TensorFlow (import after seeding)
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

import tensorflow as tf
from keras.models import load_model

from src.models.layers import MCDropout
from src.utils.constants import sanitize_filename, CATEGORY_TYPE_TO_FOLDER
from src.trading.mt5_client import MT5Client
from src.trading.trade_journal import TradeJournal
from src.trading.notifications import Notifier
from src.trading.config_validator import validate_config_or_die
from src.trading.slot_manager import SlotManager
from src.data.processor import process_file, invalidate_cross_asset_cache

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "outputs" / "bot_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"bot_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("trading_bot")

# MCDropout imported from src.models.layers (single definition)
# sanitize_filename imported from src.utils.constants

# Backward-compatible alias for code that still uses _sanitize_filename
_sanitize_filename = sanitize_filename


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # MT5 Bridge connection
    "MT5_HOST": "127.0.0.1",
    "MT5_PORT": 8787,
    "MAGIC": 24001,

    # Model artifacts (relative to project root)
    "MODEL_PATH": "outputs/models/lstm_model.keras",
    "SCALER_PATH": "outputs/models/scaler.joblib",
    "FEATURES_PATH": "outputs/models/feature_cols.json",
    "SYMBOLS_PATH": "config/symbols.json",

    # Data paths
    "RAW_DIR": "src/data/indicators_data/raw",
    "PROCESSED_DIR": "src/data/indicators_data/processed",

    # Strategy parameters (must match backtest)
    "MIN_ACCEPTED": 0.50,
    "STD_FACTOR": 1.0,
    "MC_DROPOUT_SAMPLES": 50,
    "WINDOW_SIZE": 90,

    # Risk management
    "MAX_CONCURRENT_POSITIONS": 3,       # Legacy: global cap (now use GLOBAL_MAX_SLOTS)
    "RISK_PER_TRADE_PCT": 1.0,         # Default % of equity risked per trade
    "DEFAULT_LOT_SIZE": 0.01,          # Fallback if sizing calc fails
    "MAX_LOT_SIZE": 1.0,               # Hard cap
    "ATR_SL_MULTIPLIER": 2.0,          # SL = ATR * multiplier
    "ATR_TP_MULTIPLIER": 3.0,          # TP = ATR * multiplier (1.5× reward/risk)
    "MAX_DRAWDOWN_PCT": 15.0,          # Halt trading if drawdown exceeds this

    # Per-instrument-type slot allocation
    "SLOTS_PER_TYPE": {"Index": 2, "Forex": 3, "Crypto": 1, "Commodity": 2},
    "RISK_PER_TYPE": {"Index": 1.0, "Forex": 0.75, "Crypto": 0.5, "Commodity": 1.0},
    "GLOBAL_MAX_SLOTS": 5,              # Max simultaneous trades across all types
    "GLOBAL_MAX_RISK_PCT": 5.0,         # Max total portfolio risk %

    # Correlation filter
    "CORRELATION_THRESHOLD": 0.7,       # Reject if corr > this with an open position
    "CORRELATION_LOOKBACK": 60,         # Rolling window (trading days) for correlation

    # Trailing stop
    "TRAILING_STOP_ENABLED": True,
    "BREAKEVEN_AFTER_R": 1.0,           # Move SL to breakeven after this many R of profit
    "TRAILING_ATR_MULTIPLIER": 1.5,     # Trail SL at this × ATR below/above current price

    # Spread / liquidity gate
    "MAX_SPREAD_ATR_RATIO": 0.15,      # Skip entry if spread > ATR * this ratio

    # Weekend gap-risk protection
    "CLOSE_BEFORE_WEEKEND": True,       # Flatten positions Friday afternoon
    "FRIDAY_CLOSE_HOUR_UTC": 20,        # Hour (UTC) on Friday to close all positions

    # Regime / volatility filter
    "VOLATILITY_FILTER_ENABLED": True,
    "ATR_PERCENTILE_LOW": 10,           # Skip if ATR percentile < this (dead market)
    "ATR_PERCENTILE_HIGH": 95,          # Skip if ATR percentile > this (crisis)
    "ATR_PERCENTILE_LOOKBACK": 252,     # Trading days for percentile calc

    # Horizon weighting (prefer shorter-horizon signals)
    "HORIZON_WEIGHTS": {"1d": 1.5, "1w": 1.2, "1m": 1.0, "6m": 0.6},

    # Walk-forward retrain
    "RETRAIN_ENABLED": True,
    "RETRAIN_INTERVAL_DAYS": 30,

    # Notifications (empty = disabled)
    "NOTIFY_WEBHOOK_URL": "",           # Discord / Slack / Telegram webhook URL
    "NOTIFY_ON_TRADE": True,
    "NOTIFY_ON_DRAWDOWN": True,
    "NOTIFY_ON_ERROR": True,

    # State persistence
    "STATE_FILE": "outputs/bot_state.json",

    # Scheduling
    "CHECK_INTERVAL_SECONDS": 300,     # 5 minutes between cycles
    "DAILY_REFRESH_HOUR": 0,           # Hour (UTC) to refresh candles & reprocess
    "POSITION_REVIEW_HOURS": 24,       # Close stale positions after N hours (0=disabled)
}


def load_config() -> Dict:
    """Load config from config/bot_config.json, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    config_path = PROJECT_ROOT / "config" / "bot_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        config.update(user_cfg)
        logger.info(f"Loaded bot config from {config_path}")
    else:
        logger.info("No bot_config.json found – using defaults")
    return config


# ---------------------------------------------------------------------------
# CORE BOT
# ---------------------------------------------------------------------------

class TradingBot:
    """Autonomous LSTM-based trading bot."""

    def __init__(self):
        self.config = load_config()
        validate_config_or_die(self.config)  # Fail fast on bad config
        self.running = False
        self.start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None
        self._cycle_cache: Dict[str, pd.DataFrame] = {}  # Per-cycle CSV cache
        self.last_refresh_date: Optional[str] = None

        # Per-instrument-type slot manager
        self.slot_manager = SlotManager.from_config(self.config)

        # Load bridge client
        self.mt5 = MT5Client(
            host=self.config["MT5_HOST"],
            port=self.config["MT5_PORT"],
            magic=self.config["MAGIC"],
            api_key=self.config.get("BRIDGE_API_KEY"),
        )

        # Load model artifacts
        self.model = None
        self.scaler = None
        self.feature_cols: List[str] = []
        self.symbols: List[Dict] = []          # From symbols.json
        self.symbol_name_map: Dict[str, str] = {}  # sanitized_name -> MT5 name

        self._load_artifacts()
        self._load_symbols()

        # Trade journal
        self.journal = TradeJournal()

        # Notifications
        self.notifier = Notifier(self.config.get("NOTIFY_WEBHOOK_URL", ""))

        # Restore persisted state (drawdown tracking, last refresh, etc.)
        self._load_state()

        # Compile MC prediction as tf.function for speed
        self._mc_predict_fn = tf.function(lambda x: self.model(x, training=True))

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------
    def _load_artifacts(self):
        """Load trained model, scaler, and feature column list."""
        model_path = PROJECT_ROOT / self.config["MODEL_PATH"]
        scaler_path = PROJECT_ROOT / self.config["SCALER_PATH"]
        features_path = PROJECT_ROOT / self.config["FEATURES_PATH"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python src/inference/run_forecast.py` first to train & save the model."
            )
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Feature cols not found at {features_path}")

        logger.info("Loading model...")
        self.model = load_model(model_path, custom_objects={"MCDropout": MCDropout})
        logger.info(f"Model loaded: {model_path}")

        # Try joblib first, fall back to pickle for legacy scaler.pkl files
        if scaler_path.suffix == ".joblib" or scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception:
                # Fall back to pickle for old artifacts
                import pickle
                pkl_path = scaler_path.with_suffix(".pkl")
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        self.scaler = pickle.load(f)
                else:
                    raise
        logger.info(f"Scaler loaded: {scaler_path}")

        with open(features_path, "r") as f:
            self.feature_cols = json.load(f)
        logger.info(f"Feature cols loaded: {len(self.feature_cols)} features")

        # Verify feature hash matches saved model artifacts (detect processor changes)
        feature_hash_path = PROJECT_ROOT / "outputs" / "models" / "feature_hash.txt"
        if feature_hash_path.exists():
            import hashlib
            expected_hash = feature_hash_path.read_text().strip()
            actual_hash = hashlib.sha256(
                json.dumps(sorted(self.feature_cols)).encode()
            ).hexdigest()[:16]
            if actual_hash != expected_hash:
                logger.warning(
                    f"FEATURE HASH MISMATCH: saved={expected_hash}, current={actual_hash}. "
                    f"The processor may have changed since the model was trained. "
                    f"Consider retraining with `python src/inference/run_forecast.py`."
                )
            else:
                logger.info(f"Feature hash verified: {actual_hash}")

    def _load_symbols(self):
        """Load symbol definitions from config/symbols.json."""
        symbols_path = PROJECT_ROOT / self.config["SYMBOLS_PATH"]
        with open(symbols_path, "r") as f:
            data = json.load(f)
        self.symbols = data.get("symbols", [])

        # Build mapping: sanitized filename stem -> MT5 symbol name
        for sym in self.symbols:
            mt5_name = sym["name"]
            safe_name = sanitize_filename(mt5_name)
            self.symbol_name_map[safe_name] = mt5_name

        logger.info(f"Loaded {len(self.symbols)} symbols")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _load_state(self):
        """Restore persisted bot state (equity tracking, last refresh date)."""
        state_path = PROJECT_ROOT / self.config.get("STATE_FILE", "outputs/bot_state.json")
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                self.start_equity = state.get("start_equity")
                self.peak_equity = state.get("peak_equity")
                self.last_refresh_date = state.get("last_refresh_date")
                logger.info(
                    f"Restored bot state: start_equity={self.start_equity}, "
                    f"peak_equity={self.peak_equity}, last_refresh={self.last_refresh_date}"
                )
            except Exception as e:
                logger.warning(f"Failed to load bot state: {e}")

    def _save_state(self):
        """Persist bot state to disk for safe restarts."""
        state_path = PROJECT_ROOT / self.config.get("STATE_FILE", "outputs/bot_state.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            state = {
                "start_equity": self.start_equity,
                "peak_equity": self.peak_equity,
                "last_refresh_date": self.last_refresh_date,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save bot state: {e}")

    # ------------------------------------------------------------------
    # Data refresh: download latest candles & reprocess
    # ------------------------------------------------------------------
    def refresh_data(self):
        """Download latest D1 candles from MT5 Bridge and reprocess all instruments."""
        logger.info("=" * 60)
        logger.info("REFRESHING MARKET DATA")
        logger.info("=" * 60)

        raw_base = PROJECT_ROOT / self.config["RAW_DIR"]
        processed_base = PROJECT_ROOT / self.config["PROCESSED_DIR"]

        category_map = CATEGORY_TYPE_TO_FOLDER

        for sym in self.symbols:
            mt5_name = sym["name"]
            safe_name = sanitize_filename(mt5_name)
            cat_folder = category_map.get(sym.get("type", ""), "other")
            raw_dir = raw_base / cat_folder
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{safe_name}_daily.csv"

            try:
                candles = self.mt5.candles(mt5_name, timeframe="D1", count=500)
                if not candles:
                    logger.warning(f"No candles for {mt5_name}")
                    continue

                df_new = pd.DataFrame([
                    {
                        "date": c["time"].split("T")[0],
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "volume": c.get("real_volume", c.get("tick_volume", 0)),
                    }
                    for c in candles
                ])
                df_new = df_new.drop_duplicates(subset=["date"]).sort_values("date")

                # Merge with existing raw data
                if raw_path.exists():
                    existing = pd.read_csv(raw_path)
                    df_combined = pd.concat([existing, df_new]).drop_duplicates(subset=["date"]).sort_values("date")
                else:
                    df_combined = df_new

                df_combined.to_csv(raw_path, index=False)

            except Exception as e:
                logger.error(f"Failed to download {mt5_name}: {e}")

        # Reprocess all instruments
        logger.info("Reprocessing all instruments...")
        invalidate_cross_asset_cache()  # Clear stale cross-asset features before reprocessing
        for subfolder in ["forex", "indices", "commodities", "crypto"]:
            raw_sub = raw_base / subfolder
            proc_sub = processed_base / subfolder
            proc_sub.mkdir(parents=True, exist_ok=True)
            if not raw_sub.exists():
                continue
            for csv_file in raw_sub.glob("*.csv"):
                out_path = proc_sub / f"{csv_file.stem}_processed.csv"
                try:
                    process_file(str(csv_file), str(out_path))
                except Exception as e:
                    logger.error(f"Failed to process {csv_file.name}: {e}")

        self.last_refresh_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Data refresh complete")

    # ------------------------------------------------------------------
    # Inference: generate predictions for all symbols
    # ------------------------------------------------------------------
    def _get_cached_df(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Return a cached DataFrame for the given CSV, reading from disk once per cycle."""
        key = str(csv_path)
        if key not in self._cycle_cache:
            if not csv_path.exists():
                self._cycle_cache[key] = None
                return None
            self._cycle_cache[key] = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        return self._cycle_cache.get(key)

    def _prepare_features(self, csv_path: Path) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
        """
        Read a processed CSV, apply the saved scaler, and return the
        latest window of features ready for model prediction.
        Returns (X_window, df) or None if insufficient data.
        """
        df = self._get_cached_df(csv_path)
        if df is None:
            return None

        df = df.copy()

        # Add target columns (needed by the pipeline even though we don't use them)
        df["Target_1d"] = np.log(df["close"].shift(-1) / df["close"])
        df["Target_1w"] = np.log(df["close"].shift(-5) / df["close"])
        df["Target_1m"] = np.log(df["close"].shift(-21) / df["close"])
        df["Target_6m"] = np.log(df["close"].shift(-126) / df["close"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Need at least WINDOW_SIZE + 1 rows
        window = self.config["WINDOW_SIZE"]
        if len(df) < window + 1:
            return None

        features = df[self.feature_cols].values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features_scaled = self.scaler.transform(features)

        # Take the most recent window
        X = features_scaled[-(window + 1):]
        X = X.reshape(1, window + 1, len(self.feature_cols))

        return X, df

    def _mc_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run MC Dropout inference and return (mean_probs, std_probs).
        
        Uses tf.function-compiled forward pass for speed.
        """
        n_samples = self.config["MC_DROPOUT_SAMPLES"]
        preds = np.array([self._mc_predict_fn(X).numpy() for _ in range(n_samples)])
        return preds.mean(axis=0), preds.std(axis=0)

    def generate_signals(self) -> List[Dict]:
        """
        Scan all instruments and return ranked trade candidates (BUY and SELL).
        Each candidate: {symbol, mt5_name, side, horizon, adj_prob, pred_prob, pred_std, atr, close}
        """
        processed_base = PROJECT_ROOT / self.config["PROCESSED_DIR"]
        candidates = []

        horizons = ["1d", "1w", "1m", "6m"]
        horizon_days = {"1d": 1, "1w": 5, "1m": 21, "6m": 126}
        n_horizons = len(horizons)

        for sym in self.symbols:
            mt5_name = sym["name"]
            safe_name = sanitize_filename(mt5_name)
            cat_map = CATEGORY_TYPE_TO_FOLDER
            cat_folder = cat_map.get(sym.get("type", ""), "other")
            csv_path = processed_base / cat_folder / f"{safe_name}_daily_processed.csv"

            result = self._prepare_features(csv_path)
            if result is None:
                continue

            X, df = result
            mean_probs, std_probs = self._mc_predict(X)

            # Get the latest close and ATR for position sizing
            latest_close = float(df["close"].iloc[-1])
            latest_atr = float(df.get("ATR_14", pd.Series([0])).iloc[-1]) if "ATR_14" in df.columns else 0

            # --- Volatility / regime filter ---
            if self.config.get("VOLATILITY_FILTER_ENABLED", False) and "ATR_14" in df.columns:
                lookback = self.config.get("ATR_PERCENTILE_LOOKBACK", 252)
                atr_series = df["ATR_14"].dropna().iloc[-lookback:]
                if len(atr_series) > 20:
                    pctile = float((atr_series < latest_atr).sum() / len(atr_series) * 100)
                    lo = self.config.get("ATR_PERCENTILE_LOW", 10)
                    hi = self.config.get("ATR_PERCENTILE_HIGH", 95)
                    if pctile < lo or pctile > hi:
                        logger.debug(
                            f"  Skipping {mt5_name}: ATR percentile {pctile:.0f}% "
                            f"outside [{lo}, {hi}]"
                        )
                        continue

            for i, h in enumerate(horizons):
                # --- BUY signal (upside probability, outputs 0..3) ---
                pred_prob = float(mean_probs[0, i])
                pred_std = float(std_probs[0, i])
                adj_prob = pred_prob - self.config["STD_FACTOR"] * pred_std

                if adj_prob > self.config["MIN_ACCEPTED"]:
                    candidates.append({
                        "symbol": safe_name,
                        "mt5_name": mt5_name,
                        "side": "BUY",
                        "horizon": h,
                        "horizon_days": horizon_days[h],
                        "pred_prob": pred_prob,
                        "pred_std": pred_std,
                        "adj_prob": adj_prob,
                        "close": latest_close,
                        "atr": latest_atr,
                        "type": sym.get("type", "Unknown"),
                    })

                # --- SELL signal (downside probability, outputs 4..7) ---
                pred_prob_down = float(mean_probs[0, n_horizons + i])
                pred_std_down = float(std_probs[0, n_horizons + i])
                adj_prob_down = pred_prob_down - self.config["STD_FACTOR"] * pred_std_down

                if adj_prob_down > self.config["MIN_ACCEPTED"]:
                    candidates.append({
                        "symbol": safe_name,
                        "mt5_name": mt5_name,
                        "side": "SELL",
                        "horizon": h,
                        "horizon_days": horizon_days[h],
                        "pred_prob": pred_prob_down,
                        "pred_std": pred_std_down,
                        "adj_prob": adj_prob_down,
                        "close": latest_close,
                        "atr": latest_atr,
                        "type": sym.get("type", "Unknown"),
                    })

        # Apply horizon weights to ranking score
        horizon_weights = self.config.get("HORIZON_WEIGHTS", {"1d": 1.5, "1w": 1.2, "1m": 1.0, "6m": 0.6})
        for c in candidates:
            c["weighted_score"] = c["adj_prob"] * horizon_weights.get(c["horizon"], 1.0)

        # Sort by weighted score descending (best signals first)
        candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Risk management & position sizing
    # ------------------------------------------------------------------
    def _get_profit_currency_rate(self, symbol_info: Dict, candidate_type: str) -> float:
        """
        Get the exchange rate to convert 1 unit of profit currency into account currency.

        For forex pairs the profit currency is the quote currency (last 3 chars).
        For indices/commodities/crypto the P&L is typically in the account currency
        already (USD-denominated instruments on a USD account).

        Returns the multiplier: loss_in_account = loss_in_profit_ccy / rate
        (i.e. rate = profit_ccy per 1 account_ccy, so we *divide* by it).
        Returns 1.0 as a safe fallback.
        """
        try:
            # Determine account currency
            acct = self.mt5.account_info()
            account_ccy = acct.get("currency", "USD").upper()

            # Determine profit (quote) currency
            profit_ccy = symbol_info.get("currency_profit", "").upper()

            # If broker reports a real currency (not "INT"), use it directly
            if not profit_ccy or profit_ccy == "INT":
                name = symbol_info.get("name", "")
                # Standard forex: 6-char alphabetic name → quote ccy is chars 3-6
                if len(name) == 6 and name.isalpha():
                    profit_ccy = name[3:6].upper()
                elif candidate_type == "Forex" and len(name) >= 6:
                    # Try extracting from longer names (e.g. with suffixes)
                    profit_ccy = name[3:6].upper()
                else:
                    # Non-forex (indices, commodities, crypto) – typically USD P&L
                    profit_ccy = "USD"

            if profit_ccy == account_ccy:
                return 1.0

            # Fetch conversion rate: we need profit_ccy → account_ccy.
            # Try direct pair: profit_ccy + account_ccy  (e.g. JPYUSD doesn't exist)
            # and inverse pair: account_ccy + profit_ccy  (e.g. USDJPY)
            for attempt_symbol, invert in [
                (f"{profit_ccy}{account_ccy}", False),
                (f"{account_ccy}{profit_ccy}", True),
            ]:
                try:
                    q = self.mt5.quote(attempt_symbol)
                    bid = q.get("bid", 0)
                    if bid and bid > 0:
                        return (1.0 / bid) if invert else bid
                except Exception:
                    continue

            logger.debug(
                f"Could not find conversion rate for {profit_ccy}→{account_ccy}, "
                f"assuming 1.0"
            )
            return 1.0

        except Exception as e:
            logger.debug(f"Profit currency rate lookup failed: {e}, assuming 1.0")
            return 1.0

    def _compute_lot_size(self, symbol_info: Dict, atr: float, close: float,
                          candidate_type: str = "Forex") -> float:
        """Compute position size based on risk % of equity and ATR-based stop.
        
        Converts the SL distance into account-currency risk using the profit
        currency exchange rate, so cross-currency pairs (e.g. GBPJPY on a USD
        account) are sized correctly.
        """
        try:
            acct = self.mt5.account_info()
            equity = acct["equity"]
            risk_amount = equity * (self.config["RISK_PER_TRADE_PCT"] / 100.0)

            sl_distance = atr * self.config["ATR_SL_MULTIPLIER"]
            if sl_distance <= 0:
                return self.config["DEFAULT_LOT_SIZE"]

            contract_size = symbol_info.get("trade_contract_size", 100000)

            # sl_distance is in price (quote) units → monetary loss per 1 lot
            # in the instrument's profit currency
            loss_per_lot_profit_ccy = sl_distance * contract_size

            # Convert to account currency
            profit_rate = self._get_profit_currency_rate(symbol_info, candidate_type)
            loss_per_lot_account = loss_per_lot_profit_ccy * profit_rate

            if loss_per_lot_account <= 0:
                return self.config["DEFAULT_LOT_SIZE"]

            lot_size = risk_amount / loss_per_lot_account

            # Clamp to broker limits
            vol_min = symbol_info.get("volume_min", 0.01)
            vol_max = symbol_info.get("volume_max", 100.0)
            vol_step = symbol_info.get("volume_step", 0.01)

            lot_size = max(vol_min, min(lot_size, vol_max, self.config["MAX_LOT_SIZE"]))
            # Round to vol_step
            if vol_step > 0:
                lot_size = round(lot_size / vol_step) * vol_step
            lot_size = round(lot_size, 2)

            return lot_size

        except Exception as e:
            logger.warning(f"Lot sizing failed: {e}, using default")
            return self.config["DEFAULT_LOT_SIZE"]

    def _compute_sl_tp(
        self, side: str, close: float, atr: float, digits: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute SL and TP based on ATR."""
        if atr <= 0:
            return None, None

        sl_dist = atr * self.config["ATR_SL_MULTIPLIER"]
        tp_dist = atr * self.config["ATR_TP_MULTIPLIER"]

        if side == "BUY":
            sl = round(close - sl_dist, digits)
            tp = round(close + tp_dist, digits)
        else:
            sl = round(close + sl_dist, digits)
            tp = round(close - tp_dist, digits)

        return sl, tp

    def _check_drawdown(self) -> bool:
        """Return True if drawdown exceeds max allowed → halt trading."""
        try:
            acct = self.mt5.account_info()
            equity = acct["equity"]

            if self.start_equity is None:
                self.start_equity = equity
                self.peak_equity = equity

            self.peak_equity = max(self.peak_equity, equity)

            if self.peak_equity > 0:
                drawdown_pct = ((self.peak_equity - equity) / self.peak_equity) * 100
                if drawdown_pct > self.config["MAX_DRAWDOWN_PCT"]:
                    msg = (
                        f"DRAWDOWN LIMIT HIT: {drawdown_pct:.1f}% "
                        f"(max {self.config['MAX_DRAWDOWN_PCT']}%). HALTING."
                    )
                    logger.critical(msg)
                    if self.config.get("NOTIFY_ON_DRAWDOWN", True):
                        self.notifier.send(f"🚨 {msg}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Drawdown check failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Correlation filter
    # ------------------------------------------------------------------
    def _load_close_series(self, mt5_name: str) -> Optional[pd.Series]:
        """Load the daily close price series for a symbol from processed data (cached)."""
        safe_name = sanitize_filename(mt5_name)
        processed_base = PROJECT_ROOT / self.config["PROCESSED_DIR"]

        # Find the symbol's type from our symbol list
        sym_entry = next((s for s in self.symbols if s["name"] == mt5_name), None)
        if sym_entry is None:
            return None

        cat_folder = CATEGORY_TYPE_TO_FOLDER.get(sym_entry.get("type", ""), "other")
        csv_path = processed_base / cat_folder / f"{safe_name}_daily_processed.csv"

        df = self._get_cached_df(csv_path)
        if df is None:
            return None

        try:
            return df.set_index("date")["close"]
        except Exception:
            return None

    def _is_too_correlated(
        self,
        candidate_name: str,
        candidate_side: str,
        open_positions_info: List[Dict],
    ) -> bool:
        """Direction-aware correlation filter.

        Rejects when the candidate would add concentrated exposure:
        - Same-direction trade on a positively correlated pair
        - Opposite-direction trade on a negatively correlated pair
        Both effectively double the same directional bet.
        """
        threshold = self.config.get("CORRELATION_THRESHOLD", 0.7)
        lookback = self.config.get("CORRELATION_LOOKBACK", 60)

        if not open_positions_info:
            return False

        cand_close = self._load_close_series(candidate_name)
        if cand_close is None or len(cand_close) < lookback:
            return False

        cand_ret = cand_close.pct_change().iloc[-lookback:]

        for open_pos in open_positions_info:
            open_name = open_pos["symbol"]
            open_side = open_pos.get("type", "BUY")

            open_close = self._load_close_series(open_name)
            if open_close is None or len(open_close) < lookback:
                continue

            open_ret = open_close.pct_change().iloc[-lookback:]

            # Align on common dates
            combined = pd.concat([cand_ret.rename("cand"), open_ret.rename("open")], axis=1).dropna()
            if len(combined) < 20:
                continue

            corr = combined["cand"].corr(combined["open"])

            # Determine if directions compound risk
            same_direction = (candidate_side == open_side)
            # Positive correlation + same direction = concentrated risk
            # Negative correlation + opposite direction = concentrated risk
            is_risky = (corr > threshold and same_direction) or \
                       (corr < -threshold and not same_direction)

            if is_risky:
                logger.info(
                    f"  Skipping {candidate_name} {candidate_side} — corr={corr:.2f} with "
                    f"open {open_name} {open_side} (threshold={threshold})"
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------
    def _manage_trailing_stops(self):
        """Adjust SL on open bot positions: breakeven after 1R, then ATR trail."""
        if not self.config.get("TRAILING_STOP_ENABLED", False):
            return

        try:
            positions = self.mt5.bot_positions()
        except Exception as e:
            logger.error(f"Trailing stop: failed to fetch positions: {e}")
            return

        for pos in positions:
            ticket = pos["ticket"]
            symbol = pos["symbol"]
            side = pos.get("type", "BUY")  # "BUY" or "SELL"
            price_open = pos.get("price_open", 0)
            current_sl = pos.get("sl", 0)

            if not price_open:
                continue

            # Get current price and symbol info
            try:
                quote = self.mt5.quote(symbol)
                sym_info = self.mt5.symbol_info(symbol)
            except Exception as e:
                logger.debug(f"Trailing stop: skip {symbol}, quote/info failed: {e}")
                continue

            digits = sym_info.get("digits", 5)
            current_price = quote.get("bid", 0) if side == "BUY" else quote.get("ask", 0)
            if not current_price:
                continue

            # Compute current ATR from processed data
            atr = self._get_current_atr(symbol)
            if atr <= 0:
                continue

            # Original risk (1R) = distance from entry to original SL
            sl_multiplier = self.config["ATR_SL_MULTIPLIER"]
            one_r = atr * sl_multiplier  # Approximate 1R from ATR

            breakeven_r = self.config.get("BREAKEVEN_AFTER_R", 1.0)
            trail_mult = self.config.get("TRAILING_ATR_MULTIPLIER", 1.5)
            trail_distance = atr * trail_mult
            stops_level = sym_info.get("trade_stops_level", 0) * sym_info.get("point", 0.00001)

            if side == "BUY":
                unrealized = current_price - price_open
                # Only act if in profit by at least breakeven_r × 1R
                if unrealized < breakeven_r * one_r:
                    continue

                # New trailing SL: current price minus trail distance
                new_sl = round(current_price - trail_distance, digits)

                # Must be at least at breakeven (entry price)
                new_sl = max(new_sl, round(price_open, digits))

                # Only move SL up, never down
                if current_sl and new_sl <= current_sl:
                    continue

                # Respect broker minimum stop distance
                if stops_level > 0 and (current_price - new_sl) < stops_level:
                    new_sl = round(current_price - stops_level, digits)

            else:  # SELL
                unrealized = price_open - current_price
                if unrealized < breakeven_r * one_r:
                    continue

                new_sl = round(current_price + trail_distance, digits)
                new_sl = min(new_sl, round(price_open, digits))

                if current_sl and current_sl != 0 and new_sl >= current_sl:
                    continue

                if stops_level > 0 and (new_sl - current_price) < stops_level:
                    new_sl = round(current_price + stops_level, digits)

            # Apply the new SL
            try:
                result = self.mt5.modify_position(ticket, sl=new_sl)
                logger.info(
                    f"  [TRAIL] {symbol} #{ticket} {side}: SL {current_sl} → {new_sl} "
                    f"(price={current_price}, entry={price_open}, ATR={atr:.{digits}f})"
                )
            except Exception as e:
                logger.warning(f"  [TRAIL] Failed to modify {symbol} #{ticket}: {e}")

    def _get_current_atr(self, mt5_name: str) -> float:
        """Get the latest ATR_14 value for a symbol from processed data (cached)."""
        safe_name = sanitize_filename(mt5_name)
        processed_base = PROJECT_ROOT / self.config["PROCESSED_DIR"]

        sym_entry = next((s for s in self.symbols if s["name"] == mt5_name), None)
        if sym_entry is None:
            return 0.0

        cat_folder = CATEGORY_TYPE_TO_FOLDER.get(sym_entry.get("type", ""), "other")
        csv_path = processed_base / cat_folder / f"{safe_name}_daily_processed.csv"

        df = self._get_cached_df(csv_path)
        if df is None:
            return 0.0

        try:
            if "ATR_14" in df.columns and len(df) > 0:
                return float(df["ATR_14"].iloc[-1])
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def _review_open_positions(self):
        """Close bot positions that have been open too long or hit conditions.
        Records exits in the trade journal."""
        review_hours = self.config.get("POSITION_REVIEW_HOURS", 0)
        if review_hours <= 0:
            return

        try:
            positions = self.mt5.bot_positions()
            now = datetime.now(timezone.utc)

            for pos in positions:
                open_time_str = pos.get("time", "")
                if not open_time_str:
                    continue
                try:
                    open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00").replace("+00:00", ""))
                except ValueError:
                    continue

                hours_open = (now - open_time).total_seconds() / 3600
                if hours_open > review_hours:
                    logger.info(
                        f"Position {pos['ticket']} ({pos['symbol']}) open {hours_open:.1f}h — closing (stale)"
                    )
                    try:
                        result = self.mt5.close_position(pos["ticket"], comment="bot:stale")
                        # Record exit in journal
                        self.journal.record_exit(
                            ticket=pos["ticket"],
                            exit_price=pos.get("price_current", 0),
                            realized_pnl=pos.get("profit", 0),
                            commission=pos.get("commission", 0),
                            swap=pos.get("swap", 0),
                        )
                    except Exception as e:
                        logger.error(f"Failed to close stale position {pos['ticket']}: {e}")
        except Exception as e:
            logger.error(f"Position review failed: {e}")

    def _close_all_positions(self, comment: str = "bot:close_all"):
        """Flatten all bot positions (e.g., before weekend)."""
        try:
            positions = self.mt5.bot_positions()
            if not positions:
                logger.info("No positions to close.")
                return
            logger.info(f"Closing all {len(positions)} bot positions ({comment})")
            for pos in positions:
                try:
                    self.mt5.close_position(pos["ticket"], comment=comment)
                    self.journal.record_exit(
                        ticket=pos["ticket"],
                        exit_price=pos.get("price_current", 0),
                        realized_pnl=pos.get("profit", 0),
                        commission=pos.get("commission", 0),
                        swap=pos.get("swap", 0),
                    )
                    logger.info(f"  Closed {pos['symbol']} #{pos['ticket']}")
                except Exception as e:
                    logger.error(f"  Failed to close {pos['ticket']}: {e}")
            if self.config.get("NOTIFY_ON_TRADE", True):
                self.notifier.send(f"🏁 Closed all {len(positions)} positions: {comment}")
        except Exception as e:
            logger.error(f"Close all positions failed: {e}")

    # ------------------------------------------------------------------
    # Main trade execution cycle
    # ------------------------------------------------------------------
    def execute_cycle(self):
        """One full scan → signal → trade cycle."""
        logger.info("-" * 60)
        logger.info("STARTING TRADE CYCLE")
        logger.info("-" * 60)

        # Clear per-cycle cache
        self._cycle_cache.clear()

        now = datetime.now(timezone.utc)
        weekday = now.weekday()

        # 0. Skip weekends (Sat=5, Sun=6) — most markets are closed
        if weekday >= 5:
            logger.info(f"Weekend (day={weekday}) — skipping trade cycle")
            return

        # 0b. Friday pre-weekend close
        if (self.config.get("CLOSE_BEFORE_WEEKEND", False)
                and weekday == 4
                and now.hour >= self.config.get("FRIDAY_CLOSE_HOUR_UTC", 20)):
            self._close_all_positions("bot:weekend")
            return

        # 1. Check drawdown
        if self._check_drawdown():
            return

        # 2. Review & close stale positions
        self._review_open_positions()

        # 2b. Manage trailing stops on open positions
        self._manage_trailing_stops()

        # 3. Count current bot positions & compute per-type slot usage
        try:
            open_positions = self.mt5.bot_positions()
            n_open = len(open_positions)
            open_symbols = {p["symbol"] for p in open_positions}
            open_by_type = SlotManager.count_open_by_type(open_positions, self.symbols)
            logger.info(f"Open bot positions: {n_open} {list(open_symbols)}")
            self.slot_manager.log_slot_status(open_by_type)
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return

        global_remaining = self.slot_manager.global_slots_remaining(open_by_type)
        if global_remaining <= 0:
            logger.info(
                f"Global slot cap ({self.slot_manager.global_max_slots}) reached "
                f"— skipping signal generation"
            )
            return

        # 4. Account snapshot + equity journal
        try:
            acct = self.mt5.account_info()
            logger.info(
                f"Account: balance={acct['balance']:.2f} equity={acct['equity']:.2f} "
                f"margin={acct['margin']:.2f} free_margin={acct['free_margin']:.2f}"
            )
            self.journal.record_equity_snapshot(
                equity=acct["equity"], balance=acct["balance"], open_positions=n_open
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return

        # 5. Generate signals
        try:
            candidates = self.generate_signals()
            logger.info(f"Signal scan complete: {len(candidates)} candidates above threshold")
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            traceback.print_exc()
            if self.config.get("NOTIFY_ON_ERROR", True):
                self.notifier.send(f"⚠️ Signal generation failed: {e}")
            return

        if not candidates:
            logger.info("No trade candidates today")
            return

        # Log top candidates
        for c in candidates[:10]:
            logger.info(
                f"  {c['mt5_name']:20s} {c['horizon']:3s} adj_prob={c['adj_prob']:.4f} "
                f"weighted={c.get('weighted_score', c['adj_prob']):.4f} "
                f"prob={c['pred_prob']:.4f} std={c['pred_std']:.4f} ATR={c['atr']:.6f}"
            )

        # 6. Execute best candidates (up to available slots)
        trades_opened = 0
        for candidate in candidates:
            # Re-check global slots (may have been consumed this cycle)
            open_by_type = SlotManager.count_open_by_type(open_positions, self.symbols)
            if self.slot_manager.global_slots_remaining(open_by_type) <= 0:
                logger.info("Global slot cap reached — stopping candidate evaluation")
                break

            mt5_name = candidate["mt5_name"]

            # Skip if we already have a position in this symbol
            if mt5_name.upper() in {s.upper() for s in open_symbols}:
                logger.info(f"  Skipping {mt5_name} — already have position")
                continue

            # Direction-aware correlation filter
            if self._is_too_correlated(mt5_name, candidate["side"], open_positions):
                continue

            # Per-instrument-type slot check
            candidate_type = candidate.get("type", "Unknown")
            type_remaining = self.slot_manager.type_slots_remaining(candidate_type, open_by_type)
            if type_remaining <= 0:
                logger.info(
                    f"  Skipping {mt5_name} — {candidate_type} slots full "
                    f"({self.slot_manager.slots_per_type.get(candidate_type, 0)} max)"
                )
                continue

            try:
                sym_info = self.mt5.symbol_info(mt5_name)
            except Exception as e:
                logger.warning(f"  Cannot get symbol info for {mt5_name}: {e}")
                continue

            digits = sym_info.get("digits", 5)
            atr = candidate["atr"]
            side = candidate["side"]

            # --- Spread / liquidity gate ---
            try:
                live_quote = self.mt5.quote(mt5_name)
                bid = live_quote.get("bid", 0)
                ask = live_quote.get("ask", 0)
                spread = ask - bid
                max_spread = atr * self.config.get("MAX_SPREAD_ATR_RATIO", 0.15)
                if atr > 0 and spread > max_spread:
                    logger.info(
                        f"  Skipping {mt5_name} — spread {spread:.{digits}f} > "
                        f"max {max_spread:.{digits}f} (ATR×{self.config.get('MAX_SPREAD_ATR_RATIO', 0.15)})"
                    )
                    continue
                # Use live price for SL/TP (not stale daily close)
                live_price = bid if side == "BUY" else ask
                if not live_price or live_price <= 0:
                    live_price = candidate["close"]
            except Exception as e:
                logger.debug(f"  Quote failed for {mt5_name}: {e}, using daily close")
                live_price = candidate["close"]

            # Compute SL/TP from live price
            sl, tp = self._compute_sl_tp(side, live_price, atr, digits)

            # Compute lot size
            lot_size = self._compute_lot_size(sym_info, atr, live_price,
                                              candidate_type=candidate.get("type", "Forex"))

            # Comment with signal info
            comment = f"bot:{candidate['horizon']}:{candidate['adj_prob']:.2f}"

            try:
                result = self.mt5.place_market_order(
                    symbol=mt5_name,
                    side=side,
                    volume=lot_size,
                    sl=sl,
                    tp=tp,
                    comment=comment,
                )

                if result.get("accepted"):
                    trades_opened += 1
                    open_symbols.add(mt5_name.upper())
                    # Track the new position for accurate per-type slot counting
                    open_positions.append({"symbol": mt5_name})

                    fill_price = result.get("price", live_price)
                    ticket = result.get("ticket", 0)

                    # Record in trade journal
                    self.journal.record_entry(
                        ticket=ticket,
                        symbol=mt5_name,
                        side=side,
                        volume=lot_size,
                        entry_price=fill_price,
                        sl=sl,
                        tp=tp,
                        horizon=candidate["horizon"],
                        adj_prob=candidate["adj_prob"],
                        comment=comment,
                    )

                    msg = (
                        f"[OK] OPENED {side} {lot_size} {mt5_name} | "
                        f"ticket={ticket} price={fill_price} "
                        f"SL={sl} TP={tp} | horizon={candidate['horizon']} "
                        f"adj_prob={candidate['adj_prob']:.4f}"
                    )
                    logger.info(f"  {msg}")
                    if self.config.get("NOTIFY_ON_TRADE", True):
                        self.notifier.send(f"📈 {msg}")
                else:
                    logger.warning(
                        f"  [REJECTED] {side} {mt5_name}: {result.get('message', 'unknown')}"
                    )
                    
            except Exception as e:
                logger.error(f"  Order failed for {mt5_name}: {e}")

        logger.info(f"Cycle complete: {trades_opened} new trades opened")

        # Persist state after every cycle
        self._save_state()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Main bot loop – runs until interrupted."""
        logger.info("=" * 60)
        logger.info("  LSTM TRADING BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Config: {json.dumps(self.config, indent=2, default=str)}")

        # Verify bridge connectivity
        if not self.mt5.is_healthy():
            logger.error("MT5 Bridge is not healthy. Exiting.")
            return

        acct = self.mt5.account_info()
        # Only overwrite equity if not restored from state
        if self.start_equity is None:
            self.start_equity = acct["equity"]
        if self.peak_equity is None:
            self.peak_equity = acct["equity"]
        logger.info(f"Starting equity: {self.start_equity:.2f}")

        self.running = True

        # Graceful shutdown handler
        def _shutdown(signum, frame):
            logger.info("Shutdown signal received – stopping bot...")
            self.running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Initial data refresh
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_refresh_date != today:
            try:
                self.refresh_data()
            except Exception as e:
                logger.error(f"Initial data refresh failed: {e}")
                traceback.print_exc()

        cycle_count = 0

        while self.running:
            try:
                # Daily data refresh check
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                current_hour = datetime.now(timezone.utc).hour

                if self.last_refresh_date != today and current_hour >= self.config["DAILY_REFRESH_HOUR"]:
                    logger.info("New trading day – refreshing data...")
                    try:
                        self.refresh_data()
                    except Exception as e:
                        logger.error(f"Data refresh failed: {e}")

                    # Walk-forward retrain check (once per day after data refresh)
                    if self.config.get("RETRAIN_ENABLED", False):
                        self._maybe_retrain()

                # Health check before each cycle
                if not self.mt5.is_healthy():
                    logger.warning("MT5 Bridge unhealthy – waiting...")
                    time.sleep(30)
                    continue

                # Execute trade cycle
                cycle_count += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE #{cycle_count} | {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
                logger.info(f"{'='*60}")

                self.execute_cycle()

                # Sleep until next cycle
                interval = self.config["CHECK_INTERVAL_SECONDS"]
                logger.info(f"Next cycle in {interval}s...")
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt – shutting down")
                self.running = False
            except Exception as e:
                logger.error(f"Unhandled error in main loop: {e}")
                traceback.print_exc()
                if self.config.get("NOTIFY_ON_ERROR", True):
                    self.notifier.send(f"⚠️ Bot error: {e}")
                time.sleep(60)  # Wait before retrying

        logger.info("Bot stopped.")

        # Final summary
        try:
            acct = self.mt5.account_info()
            final_equity = acct["equity"]
            pnl = final_equity - self.start_equity
            pnl_pct = (pnl / self.start_equity) * 100 if self.start_equity else 0
            logger.info(f"Session P&L: {pnl:+.2f} ({pnl_pct:+.2f}%)")
            logger.info(f"Final equity: {final_equity:.2f}")

            report = self.journal.format_stats_report()
            logger.info(f"\n{report}")
        except Exception:
            pass

        # Save final state and close journal
        self._save_state()
        self.journal.close()

    # ------------------------------------------------------------------
    # Walk-forward retrain integration
    # ------------------------------------------------------------------
    def _maybe_retrain(self):
        """Trigger walk-forward retraining if due, and hot-reload the model."""
        try:
            from src.inference.walk_forward_retrain import retrain, is_retrain_due

            interval = self.config.get("RETRAIN_INTERVAL_DAYS", 30)
            if not is_retrain_due(interval):
                logger.info(f"Walk-forward retrain not due (interval={interval}d)")
                return

            logger.info("Walk-forward retrain is DUE — starting retraining...")
            if self.config.get("NOTIFY_ON_TRADE", True):
                self.notifier.send("🔄 Walk-forward retraining started")

            deployed = retrain(force=False, interval_days=interval)

            if deployed:
                logger.info("New model deployed — reloading artifacts...")
                self._load_artifacts()
                # Re-compile tf.function with new model
                self._mc_predict_fn = tf.function(lambda x: self.model(x, training=True))
                if self.config.get("NOTIFY_ON_TRADE", True):
                    self.notifier.send("✅ New model deployed and loaded")
            else:
                logger.info("Retrain complete — existing model kept.")
        except Exception as e:
            logger.error(f"Walk-forward retrain failed: {e}")
            traceback.print_exc()
            if self.config.get("NOTIFY_ON_ERROR", True):
                self.notifier.send(f"⚠️ Retrain failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
