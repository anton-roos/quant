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
import pickle
import logging
import signal
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

from src.trading.mt5_client import MT5Client
from src.data.processor import process_file
from src.data.features.mt5_bridge_downloader import _sanitize_filename

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

# ---------------------------------------------------------------------------
# MCDropout (must match training definition for model loading)
# ---------------------------------------------------------------------------

class MCDropout(Dropout):
    """MC Dropout – keeps dropout active during inference."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


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
    "SCALER_PATH": "outputs/models/scaler.pkl",
    "FEATURES_PATH": "outputs/models/feature_cols.json",
    "SYMBOLS_PATH": "config/symbols.json",

    # Data paths
    "RAW_DIR": "src/data/indicators_data/raw",
    "PROCESSED_DIR": "src/data/indicators_data/processed",

    # Strategy parameters (must match backtest)
    "MIN_ACCEPTED": 0.10,
    "STD_FACTOR": 1.0,
    "MC_DROPOUT_SAMPLES": 50,
    "WINDOW_SIZE": 90,

    # Risk management
    "MAX_CONCURRENT_POSITIONS": 3,
    "RISK_PER_TRADE_PCT": 1.0,         # % of equity risked per trade
    "DEFAULT_LOT_SIZE": 0.01,          # Fallback if sizing calc fails
    "MAX_LOT_SIZE": 1.0,               # Hard cap
    "ATR_SL_MULTIPLIER": 2.0,          # SL = ATR * multiplier
    "ATR_TP_MULTIPLIER": 3.0,          # TP = ATR * multiplier (1.5× reward/risk)
    "MAX_DRAWDOWN_PCT": 15.0,          # Halt trading if drawdown exceeds this

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
        self.running = False
        self.start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None
        self.last_refresh_date: Optional[str] = None

        # Load bridge client
        self.mt5 = MT5Client(
            host=self.config["MT5_HOST"],
            port=self.config["MT5_PORT"],
            magic=self.config["MAGIC"],
        )

        # Load model artifacts
        self.model = None
        self.scaler = None
        self.feature_cols: List[str] = []
        self.symbols: List[Dict] = []          # From symbols.json
        self.symbol_name_map: Dict[str, str] = {}  # sanitized_name -> MT5 name

        self._load_artifacts()
        self._load_symbols()

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

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded: {scaler_path}")

        with open(features_path, "r") as f:
            self.feature_cols = json.load(f)
        logger.info(f"Feature cols loaded: {len(self.feature_cols)} features")

    def _load_symbols(self):
        """Load symbol definitions from config/symbols.json."""
        symbols_path = PROJECT_ROOT / self.config["SYMBOLS_PATH"]
        with open(symbols_path, "r") as f:
            data = json.load(f)
        self.symbols = data.get("symbols", [])

        # Build mapping: sanitized filename stem -> MT5 symbol name
        for sym in self.symbols:
            mt5_name = sym["name"]
            safe_name = _sanitize_filename(mt5_name)
            self.symbol_name_map[safe_name] = mt5_name

        logger.info(f"Loaded {len(self.symbols)} symbols")

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

        category_map = {"Forex": "forex", "Index": "indices", "Commodity": "commodities", "Crypto": "crypto"}

        for sym in self.symbols:
            mt5_name = sym["name"]
            safe_name = _sanitize_filename(mt5_name)
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

        self.last_refresh_date = datetime.utcnow().strftime("%Y-%m-%d")
        logger.info("Data refresh complete")

    # ------------------------------------------------------------------
    # Inference: generate predictions for all symbols
    # ------------------------------------------------------------------
    def _prepare_features(self, csv_path: Path) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
        """
        Read a processed CSV, apply the saved scaler, and return the
        latest window of features ready for model prediction.
        Returns (X_window, df) or None if insufficient data.
        """
        if not csv_path.exists():
            return None

        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

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
        """Run MC Dropout inference and return (mean_probs, std_probs)."""
        n_samples = self.config["MC_DROPOUT_SAMPLES"]
        preds = np.array([self.model(X, training=True).numpy() for _ in range(n_samples)])
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
            safe_name = _sanitize_filename(mt5_name)
            cat_map = {"Forex": "forex", "Index": "indices", "Commodity": "commodities", "Crypto": "crypto"}
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

        # Sort by adj_prob descending (best signals first)
        candidates.sort(key=lambda x: x["adj_prob"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Risk management & position sizing
    # ------------------------------------------------------------------
    def _compute_lot_size(self, symbol_info: Dict, atr: float, close: float) -> float:
        """Compute position size based on risk % of equity and ATR-based stop."""
        try:
            acct = self.mt5.account_info()
            equity = acct["equity"]
            risk_amount = equity * (self.config["RISK_PER_TRADE_PCT"] / 100.0)

            sl_distance = atr * self.config["ATR_SL_MULTIPLIER"]
            if sl_distance <= 0:
                return self.config["DEFAULT_LOT_SIZE"]

            # Approximate: lot_size = risk_amount / (sl_distance * contract_size)
            contract_size = symbol_info.get("trade_contract_size", 100000)
            point = symbol_info.get("point", 0.00001)
            digits = symbol_info.get("digits", 5)

            # sl_distance is in price units; convert to monetary loss per lot
            loss_per_lot = sl_distance * contract_size

            if loss_per_lot <= 0:
                return self.config["DEFAULT_LOT_SIZE"]

            lot_size = risk_amount / loss_per_lot

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
                    logger.critical(
                        f"DRAWDOWN LIMIT HIT: {drawdown_pct:.1f}% "
                        f"(max {self.config['MAX_DRAWDOWN_PCT']}%). HALTING."
                    )
                    return True
            return False
        except Exception as e:
            logger.error(f"Drawdown check failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def _review_open_positions(self):
        """Close bot positions that have been open too long or hit conditions."""
        review_hours = self.config.get("POSITION_REVIEW_HOURS", 0)
        if review_hours <= 0:
            return

        try:
            positions = self.mt5.bot_positions()
            now = datetime.utcnow()

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
                        self.mt5.close_position(pos["ticket"], comment="bot:stale")
                    except Exception as e:
                        logger.error(f"Failed to close stale position {pos['ticket']}: {e}")
        except Exception as e:
            logger.error(f"Position review failed: {e}")

    # ------------------------------------------------------------------
    # Main trade execution cycle
    # ------------------------------------------------------------------
    def execute_cycle(self):
        """One full scan → signal → trade cycle."""
        logger.info("-" * 60)
        logger.info("STARTING TRADE CYCLE")
        logger.info("-" * 60)

        # 0. Skip weekends (Sat=5, Sun=6) — most markets are closed
        weekday = datetime.utcnow().weekday()
        if weekday >= 5:
            logger.info(f"Weekend (day={weekday}) — skipping trade cycle")
            return

        # 1. Check drawdown
        if self._check_drawdown():
            return

        # 2. Review & close stale positions
        self._review_open_positions()

        # 3. Count current bot positions
        try:
            open_positions = self.mt5.bot_positions()
            n_open = len(open_positions)
            open_symbols = {p["symbol"] for p in open_positions}
            logger.info(f"Open bot positions: {n_open} {list(open_symbols)}")
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return

        max_pos = self.config["MAX_CONCURRENT_POSITIONS"]
        slots_available = max_pos - n_open
        if slots_available <= 0:
            logger.info(f"Max positions ({max_pos}) reached — skipping signal generation")
            return

        # 4. Account snapshot
        try:
            acct = self.mt5.account_info()
            logger.info(
                f"Account: balance={acct['balance']:.2f} equity={acct['equity']:.2f} "
                f"margin={acct['margin']:.2f} free_margin={acct['free_margin']:.2f}"
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
            return

        if not candidates:
            logger.info("No trade candidates today")
            return

        # Log top candidates
        for c in candidates[:10]:
            logger.info(
                f"  {c['mt5_name']:20s} {c['horizon']:3s} adj_prob={c['adj_prob']:.4f} "
                f"prob={c['pred_prob']:.4f} std={c['pred_std']:.4f} ATR={c['atr']:.6f}"
            )

        # 6. Execute best candidates (up to available slots)
        trades_opened = 0
        for candidate in candidates:
            if trades_opened >= slots_available:
                break

            mt5_name = candidate["mt5_name"]

            # Skip if we already have a position in this symbol
            if mt5_name.upper() in {s.upper() for s in open_symbols}:
                logger.info(f"  Skipping {mt5_name} — already have position")
                continue

            try:
                sym_info = self.mt5.symbol_info(mt5_name)
            except Exception as e:
                logger.warning(f"  Cannot get symbol info for {mt5_name}: {e}")
                continue

            digits = sym_info.get("digits", 5)
            atr = candidate["atr"]
            close = candidate["close"]
            side = candidate["side"]

            # Compute SL/TP
            sl, tp = self._compute_sl_tp(side, close, atr, digits)

            # Compute lot size
            lot_size = self._compute_lot_size(sym_info, atr, close)

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
                    logger.info(
                        f"  [OK] OPENED {side} {lot_size} {mt5_name} | "
                        f"ticket={result.get('ticket')} price={result.get('price')} "
                        f"SL={sl} TP={tp} | horizon={candidate['horizon']} adj_prob={candidate['adj_prob']:.4f}"
                    )
                else:
                    logger.warning(
                        f"  [REJECTED] {side} {mt5_name}: {result.get('message', 'unknown')}"
                    )
                    
            except Exception as e:
                logger.error(f"  Order failed for {mt5_name}: {e}")

        logger.info(f"Cycle complete: {trades_opened} new trades opened")

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
        self.start_equity = acct["equity"]
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
        today = datetime.utcnow().strftime("%Y-%m-%d")
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
                today = datetime.utcnow().strftime("%Y-%m-%d")
                current_hour = datetime.utcnow().hour

                if self.last_refresh_date != today and current_hour >= self.config["DAILY_REFRESH_HOUR"]:
                    logger.info("New trading day – refreshing data...")
                    try:
                        self.refresh_data()
                    except Exception as e:
                        logger.error(f"Data refresh failed: {e}")

                # Health check before each cycle
                if not self.mt5.is_healthy():
                    logger.warning("MT5 Bridge unhealthy – waiting...")
                    time.sleep(30)
                    continue

                # Execute trade cycle
                cycle_count += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE #{cycle_count} | {datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}")
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
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
