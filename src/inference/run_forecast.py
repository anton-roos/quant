#!/usr/bin/env python3
"""
Deterministic LSTM-based market forecasting model.
Supports Forex, Indices, Commodities, and Crypto.
Converted from run_forecast_.ipynb to a reproducible Python script.
"""

import os
import sys
import logging
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Set random seeds for reproducibility BEFORE importing TensorFlow/numpy
from src.utils.constants import RANDOM_SEED, EXCLUDED_COLS, HORIZONS, HORIZON_DAYS, CATEGORY_FOLDERS, sanitize_filename

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
np.random.seed(RANDOM_SEED)

import random
random.seed(RANDOM_SEED)

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Enable GPU memory growth so TF doesn't claim all VRAM on startup.
# Safe no-op if no GPU is present.
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)
if tf.config.list_physical_devices('GPU'):
    import logging as _log
    _log.getLogger(__name__).info(
        f"GPU enabled: {[g.name for g in tf.config.list_physical_devices('GPU')]}"
    )
else:
    import logging as _log
    _log.getLogger(__name__).warning("No GPU detected — training on CPU.")

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import AUC
from sklearn.preprocessing import StandardScaler
import glob
import shutil
import time
import joblib
import json
import hashlib

# Shared modules
from src.models.layers import MCDropout, Attention, binary_focal_loss
from src.models.lstm import build_model
from src.models.context import PipelineContext
from src.models.calibration import fit_calibrator, save_calibrator
from src.config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

# ---------------------------------------------------------------------------
# Build CONFIG dict from unified TrainingConfig
# ---------------------------------------------------------------------------
_training_cfg = TrainingConfig()
_paths = _training_cfg.resolve_paths()

CONFIG = {
    "DATA_DIR": _paths["DATA_DIR"],
    "FORECAST_DIR": _paths["FORECAST_DIR"],
    "MODEL_DIR": _paths["MODEL_DIR"],
    "CACHE_DIR": _paths["CACHE_DIR"],
    "WINDOW_SIZE": _training_cfg.WINDOW_SIZE,
    "TRAIN_VAL_FRAC": _training_cfg.TRAIN_VAL_FRAC,
    "VAL_FRAC_WITHIN_TRAIN": _training_cfg.VAL_FRAC_WITHIN_TRAIN,
    "MC_DROPOUT_SAMPLES": _training_cfg.MC_DROPOUT_SAMPLES,
    "EXCLUDED_COLS": EXCLUDED_COLS,
    "PROB_THRESHOLD": _training_cfg.PROB_THRESHOLD,
    "SIGMA_FACTOR": _training_cfg.SIGMA_FACTOR,
    "RANDOM_SEED": RANDOM_SEED,
    "MIN_VAL_AUC": _training_cfg.MIN_VAL_AUC,
    "BATCH_SIZE": _training_cfg.BATCH_SIZE,
    "MAX_EPOCHS": _training_cfg.MAX_EPOCHS,
    "EARLY_STOP_PATIENCE": _training_cfg.EARLY_STOP_PATIENCE,
    "REDUCE_LR_PATIENCE": _training_cfg.REDUCE_LR_PATIENCE,
    "AUGMENT_PROB": _training_cfg.AUGMENT_PROB,
    "N_ENSEMBLE": _training_cfg.N_ENSEMBLE,
    "PNL_LOSS_ENABLED": _training_cfg.PNL_LOSS_ENABLED,
    "PRETRAIN_ENABLED": _training_cfg.PRETRAIN_ENABLED,
    "PRETRAIN_EPOCHS": _training_cfg.PRETRAIN_EPOCHS,
    "PRETRAIN_N_FUTURE": _training_cfg.PRETRAIN_N_FUTURE,
    "MODEL_TYPE": _training_cfg.MODEL_TYPE,
}

horizons = HORIZONS
horizon_days = [HORIZON_DAYS[h] for h in HORIZONS]

# Ensure dirs exist
CONFIG["FORECAST_DIR"].mkdir(exist_ok=True)
CONFIG["MODEL_DIR"].mkdir(exist_ok=True)
CONFIG["CACHE_DIR"].mkdir(exist_ok=True)

# Module-level PipelineContext — populated during main() and consumed
# by process_instrument / process_instrument_for_inference.  The
# walk_forward_retrain module can supply its own context to avoid
# mutating these globals.
_ctx = PipelineContext()

# Legacy global aliases (kept for walk_forward_retrain backward compat)
scaler = _ctx.scaler
feature_cols = None
global_thresholds_up = None
global_thresholds_down = None
per_symbol_thresholds = None   # dict[symbol_stem, {"up": {...}, "down": {...}}]
symbol_ids = None               # dict[symbol_stem, int]

logger.info(f"Config: {CONFIG}")

# ============================================================================
# MC DROPOUT
# ============================================================================

# MCDropout is imported from src.models.layers (single definition)


def mc_dropout_predict(model, X, n_samples=50):
    """Generate predictions with MC Dropout sampling"""
    preds = np.array([model(X, training=True).numpy() for _ in range(n_samples)])
    return preds.mean(axis=0), preds.std(axis=0), preds


# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

def compute_horizon_thresholds(df, train_cutoff=None, sigma_factor=None):
    """Compute required return thresholds for each horizon.
    
    Uses ``sigma_factor × σ`` above/below the mean.  Lower values (e.g. 1.5)
    increase the positive label rate from ~2.3% to ~6.7%, giving the model
    more signal to learn from.

    If train_cutoff is provided, only uses data before the cutoff to prevent
    leaking future test-set statistics into the training labels.
    """
    if sigma_factor is None:
        sigma_factor = CONFIG.get("SIGMA_FACTOR", 1.5)

    if train_cutoff is not None:
        df_train = df[df["date"] < train_cutoff]
        if df_train.empty:
            df_train = df  # Fallback if instrument has no training data
    else:
        df_train = df

    thresholds_up = {}
    thresholds_down = {}
    for h in HORIZONS:
        mu = df_train[f"Target_{h}"].mean()
        sig = df_train[f"Target_{h}"].std()
        thresholds_up[h] = mu + sigma_factor * sig
        thresholds_down[h] = mu - sigma_factor * sig
    return thresholds_up, thresholds_down


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target features (future log returns) to dataframe"""
    df["Target_1d"] = np.log(df["close"].shift(-1) / df["close"])
    df["Target_1w"] = np.log(df["close"].shift(-5) / df["close"])  # 5 trading days
    df["Target_1m"] = np.log(df["close"].shift(-21) / df["close"])  # 21 trading days
    df["Target_6m"] = np.log(df["close"].shift(-126) / df["close"])  # 126 trading days
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def process_instrument(csv_path: Path, for_training=True, ctx: PipelineContext = None):
    """Process instrument data into sliding windows.

    Parameters
    ----------
    ctx : PipelineContext, optional
        Explicit context (scaler, feature_cols, thresholds).  When *None*
        the module-level ``_ctx`` / legacy globals are used.
    """
    # Resolve context
    if ctx is None:
        _sc = _ctx.scaler if _ctx.feature_cols else scaler
        _fc = _ctx.feature_cols or feature_cols
        _tu = _ctx.thresholds_up or global_thresholds_up
        _td = _ctx.thresholds_down or global_thresholds_down
        _pst = _ctx.per_symbol_thresholds or per_symbol_thresholds
    else:
        _sc = ctx.scaler
        _fc = ctx.feature_cols
        _tu = ctx.thresholds_up
        _td = ctx.thresholds_down
        _pst = ctx.per_symbol_thresholds

    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = add_features(df)

    # Prefer per-symbol thresholds so each instrument's binary labels
    # reflect its own volatility profile instead of the global average.
    symbol_stem = csv_path.stem
    if _pst and symbol_stem in _pst:
        thresholds_up = _pst[symbol_stem]["up"]
        thresholds_down = _pst[symbol_stem]["down"]
    elif _tu is not None and _td is not None:
        thresholds_up = _tu
        thresholds_down = _td
    else:
        thresholds_up, thresholds_down = compute_horizon_thresholds(df)

    # Create binary classification targets (upside and downside)
    for h in HORIZONS:
        df[f"Class_{h}"] = (df[f"Target_{h}"] > thresholds_up[h]).astype(int)
        df[f"Class_{h}_down"] = (df[f"Target_{h}"] < thresholds_down[h]).astype(int)
        
    if df.empty:
        return np.array([]), np.array([]), df, np.array([])

    for col in _fc:
        if col not in df.columns:
            df[col] = 0.0

    features_scaled = _sc.transform(df[_fc].values)
    dates = df["date"].values
    target = df[[
        *[f"Class_{h}" for h in HORIZONS],
        *[f"Class_{h}_down" for h in HORIZONS],
    ]].values if for_training else None

    X, y, y_dates = [], [], []
    window = CONFIG["WINDOW_SIZE"]
    stride = 1  # Use stride=1 to maximize training samples (was max_horizon=126)

    for i in range(window, len(features_scaled), stride):
        X.append(features_scaled[i - window:i + 1])
        if for_training:
            y.append(target[i])
        y_dates.append(dates[i])

    return (
        np.array(X),
        np.array(y) if for_training else None,
        df,
        np.array(y_dates, dtype="datetime64[ns]"),
    )


def process_instrument_for_inference(csv_path: Path, ctx: PipelineContext = None):
    """Process instrument data for daily inference predictions."""
    if ctx is None:
        _sc = _ctx.scaler if _ctx.feature_cols else scaler
        _fc = _ctx.feature_cols or feature_cols
    else:
        _sc = ctx.scaler
        _fc = ctx.feature_cols

    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = add_features(df)
    if df.empty:
        return np.array([]), np.array([]), df, np.array([])

    for col in _fc:
        if col not in df.columns:
            df[col] = 0.0

    features_scaled = _sc.transform(df[_fc].values)
    dates = df["date"].values
    target = df[["close"]].values

    X, x_dates = [], []
    window = CONFIG["WINDOW_SIZE"]
    max_horizon = 1 

    for i in range(window, len(features_scaled), max_horizon):
        X.append(features_scaled[i - window:i + 1])
        x_dates.append(dates[i])

    return (
        np.array(X),
        df.loc[window:, "close"].values[:len(X)],
        np.array(x_dates, dtype="datetime64[ns]"),
        df
    )


def cache_preprocessed(csv_path: Path, train_cutoff, train_val_cutoff, ctx: PipelineContext = None):
    """Cache preprocessed data split by train/val/test.

    Also saves ``{symbol}_regime_{split}.npy`` containing the regime_trend
    value at each window's label date, used by regime-filtered training.
    """
    symbol_name = csv_path.stem
    logger.info(f"Rebuilding cache for: {symbol_name}")

    X, y, df, y_dates = process_instrument(csv_path, for_training=True, ctx=ctx)
    if X.size == 0:
        for split in ["train", "val", "test"]:
            np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{split}.npy", np.array([]))
            np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_{split}.npy", np.array([]))
        return

    y_dates = pd.to_datetime(y_dates)

    # Extract regime_trend at each label date for regime-gated training
    regime_arr = np.full(len(y_dates), 0.5, dtype=np.float32)
    if df is not None and "regime_trend" in df.columns:
        df_idx = df.set_index("date")["regime_trend"]
        for j, d in enumerate(y_dates):
            try:
                regime_arr[j] = float(df_idx.loc[d])
            except (KeyError, TypeError):
                pass

    # Compute ATR-normalised weight for P&L loss: clip(ATR_14 / close, 0.5, 2.0)
    atr_weight_arr = np.full(len(y_dates), 1.0, dtype=np.float32)
    if df is not None and "ATR_14" in df.columns and "close" in df.columns:
        df_atr = df.set_index("date")
        for j, d in enumerate(y_dates):
            try:
                close_val = float(df_atr.loc[d, "close"])
                atr_val = float(df_atr.loc[d, "ATR_14"])
                if close_val > 0:
                    atr_weight_arr[j] = float(np.clip(atr_val / close_val, 0.5, 2.0))
            except (KeyError, TypeError):
                pass

    splits = {
        "train": y_dates < np.datetime64(train_cutoff),
        "val": (y_dates >= np.datetime64(train_cutoff)) & (y_dates < np.datetime64(train_val_cutoff)),
        "test": y_dates >= np.datetime64(train_val_cutoff),
    }
    for split, mask in splits.items():
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{split}.npy", X[mask])
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_{split}.npy", y[mask])
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_dates_{split}.npy", y_dates[mask])
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_regime_{split}.npy", regime_arr[mask])
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_atr_weight_{split}.npy", atr_weight_arr[mask])

    logger.info(
        f"Cached {symbol_name}: "
        + ", ".join([f"{s}={np.sum(m)}" for s, m in splits.items()])
    )


# ============================================================================
# DATA GENERATOR
# ============================================================================

class InstrumentDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of instrument data for training.

    When ``symbol_id_map`` is provided the generator yields
    ``([X_batch, sym_id_batch], y_batch)`` so the model receives
    per-symbol identity information through its embedding input.
    """
    def __init__(
        self,
        csv_paths,
        batch_size=512,
        split="train",
        shuffle=True,
        use_time_weights=True,
        decay_factor=0.001,
        symbol_id_map=None,
        p_augment=0.0,
        regime_filter="all",
        regime_bull_threshold=0.65,
        regime_bear_threshold=0.35,
        pnl_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.csv_paths = csv_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.use_time_weights = use_time_weights
        self.decay_factor = decay_factor
        self.symbol_id_map = symbol_id_map or {}
        self.p_augment = p_augment if split == "train" else 0.0
        self.regime_filter = regime_filter  # "all", "bull", or "bear"
        self.regime_bull_threshold = regime_bull_threshold
        self.regime_bear_threshold = regime_bear_threshold
        self.pnl_loss = pnl_loss  # append ATR weight as 9th label channel
        self.windows = []
        self.symbol_names = [Path(p).stem for p in csv_paths]
        self._prepare_indices()
        self.on_epoch_end()

    def _prepare_indices(self):
        """Prepare window indices for all instruments, applying regime filter if set."""
        self.windows = []
        self.lengths = {}
        self.date_arrays = {}
        self.atr_weight_arrays = {}
        for symbol_name in self.symbol_names:
            X_path = CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{self.split}.npy"
            date_path = CONFIG["CACHE_DIR"] / f"{symbol_name}_y_dates_{self.split}.npy"
            regime_path = CONFIG["CACHE_DIR"] / f"{symbol_name}_regime_{self.split}.npy"
            atr_path = CONFIG["CACHE_DIR"] / f"{symbol_name}_atr_weight_{self.split}.npy"
            if not X_path.exists():
                self.lengths[symbol_name] = 0
                continue

            X = np.load(X_path, mmap_mode="r")
            n_windows = len(X)
            if date_path.exists():
                self.date_arrays[symbol_name] = np.load(date_path, allow_pickle=True)
            if self.pnl_loss and atr_path.exists():
                self.atr_weight_arrays[symbol_name] = np.load(atr_path)

            # Regime filtering: include only windows whose regime_trend matches
            if self.regime_filter != "all" and regime_path.exists():
                regime_vals = np.load(regime_path)
                if self.regime_filter == "bull":
                    valid = np.where(regime_vals >= self.regime_bull_threshold)[0]
                elif self.regime_filter == "bear":
                    valid = np.where(regime_vals <= self.regime_bear_threshold)[0]
                else:
                    valid = np.arange(n_windows)
            else:
                valid = np.arange(n_windows)

            self.lengths[symbol_name] = len(valid)
            for i in valid:
                self.windows.append((symbol_name, int(i)))
        self.indices = np.arange(len(self.windows))

    def __len__(self):
        return int(np.ceil(len(self.windows) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, y_batch, sym_ids_batch = [], [], []
        weights_batch = [] if self.use_time_weights else None
        atr_weights_batch = [] if self.pnl_loss else None
        cache = {}
        for bi in batch_indices:
            symbol_name, win_idx = self.windows[bi]
            if symbol_name not in cache:
                X = np.load(CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{self.split}.npy", mmap_mode="r")
                y = np.load(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_{self.split}.npy", mmap_mode="r")
                cache[symbol_name] = (X, y)
            X_arr, y_arr = cache[symbol_name]
            x_single = X_arr[win_idx]
            if self.p_augment > 0 and np.random.random() < self.p_augment:
                x_single = self._augment_window(x_single.copy())
            X_batch.append(x_single)
            y_batch.append(y_arr[win_idx])
            sym_ids_batch.append(self.symbol_id_map.get(symbol_name, 0))

            if self.use_time_weights:
                if symbol_name in self.date_arrays:
                    dates = self.date_arrays[symbol_name]
                    date = pd.to_datetime(dates[win_idx])
                    date_ago = (pd.Timestamp.now() - date).days
                    weights_batch.append(np.exp(-self.decay_factor * date_ago))
                else:
                    weights_batch.append(1.0)

            if self.pnl_loss:
                if symbol_name in self.atr_weight_arrays:
                    atr_weights_batch.append(float(self.atr_weight_arrays[symbol_name][win_idx]))
                else:
                    atr_weights_batch.append(1.0)

        X_out = np.array(X_batch, dtype=np.float32)
        y_out = np.array(y_batch, dtype=np.float32)
        sym_ids_out = np.array(sym_ids_batch, dtype=np.int32)

        # Append ATR weight as 9th label channel for P&L-weighted loss
        if self.pnl_loss:
            atr_weights = np.array(atr_weights_batch, dtype=np.float32).reshape(-1, 1)
            y_out = np.concatenate([y_out, atr_weights], axis=1)

        # Build multi-input list when symbol IDs are available
        if self.symbol_id_map:
            x_inputs = (X_out, sym_ids_out)
        else:
            x_inputs = X_out

        if self.use_time_weights:
            return (x_inputs, y_out, np.array(weights_batch, dtype=np.float32))
        return (x_inputs, y_out)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment_window(self, X: np.ndarray) -> np.ndarray:
        """Apply a random synthetic crisis augmentation to a scaled window.

        Four regimes are sampled with equal probability:
        - Volatility shock: scale all features by 2.5–5×
        - Trend reversal: negate the last 20% of timesteps (×1.5–3)
        - Gap event: inject a single large spike (±4–8σ) at a random timestep
        - Liquidity drain: zero-out 20% of randomly selected timesteps

        The label is left unchanged — the model learns that extreme conditions
        can precede any outcome, improving generalisation to unseen crises.
        """
        aug_type = np.random.randint(0, 4)
        if aug_type == 0:
            X = X * np.random.uniform(2.5, 5.0)
        elif aug_type == 1:
            cutoff = max(1, int(0.8 * len(X)))
            X = X.copy()
            X[cutoff:] = -X[cutoff:] * np.random.uniform(1.5, 3.0)
        elif aug_type == 2:
            t = np.random.randint(0, len(X))
            spike = np.random.uniform(4.0, 8.0) * np.random.choice([-1, 1])
            X = X.copy()
            X[t] = X[t] + spike
        else:
            n_zero = max(1, int(0.2 * len(X)))
            X = X.copy()
            indices = np.random.choice(len(X), n_zero, replace=False)
            X[indices] = 0.0
        return X


# ============================================================================
# MODEL AND FORECASTING
# ============================================================================

def make_forecast(model, X, dates, closes, horizons, horizon_days, symbol_id=None):
    """Generate forecast dataframe with predictions and uncertainty.

    When ``symbol_id`` is provided the model is called with multi-input
    ``[X, sym_ids]`` to leverage the learned symbol embedding.
    """
    if symbol_id is not None:
        sym_ids = np.full((len(X), 1), symbol_id, dtype=np.int32)
        model_input = [X, sym_ids]
    else:
        model_input = X
    y_pred_mean, y_pred_std, _ = mc_dropout_predict(model, model_input, n_samples=CONFIG["MC_DROPOUT_SAMPLES"])
    df_dict = {"Date": dates, "Close": closes}

    n_horizons = len(horizons)
    for i, h in enumerate(horizons):
        # Upside probabilities (first 4 outputs)
        df_dict[f"Pred_Prob_{h}"] = y_pred_mean[:, i]
        df_dict[f"Pred_Prob_Std_{h}"] = y_pred_std[:, i]
        # Downside probabilities (last 4 outputs)
        df_dict[f"Pred_Prob_Down_{h}"] = y_pred_mean[:, n_horizons + i]
        df_dict[f"Pred_Prob_Down_Std_{h}"] = y_pred_std[:, n_horizons + i]

    forecasting_df = pd.DataFrame(df_dict)
    return forecasting_df


# Attention class is imported from src.models.layers (single definition)


def feature_importance(model, X_val, y_val, feature_names):
    """Compute feature importance via permutation.

    ``X_val`` may be a list ``[X_ts, sym_ids]`` for multi-input models.
    Only the time-series tensor is permuted; the symbol IDs stay fixed.
    """
    from sklearn.metrics import mean_squared_error

    # Separate time-series input from auxiliary inputs
    if isinstance(X_val, (list, tuple)):
        X_ts = X_val[0]
        aux_inputs = list(X_val[1:])
    else:
        X_ts = X_val
        aux_inputs = []

    def _predict(X_ts_in):
        if aux_inputs:
            return model.predict([X_ts_in] + aux_inputs, verbose=0)
        return model.predict(X_ts_in, verbose=0)
    
    base_preds = _predict(X_ts)
    base_loss = mean_squared_error(y_val, base_preds)
    importances = []

    for i, col in enumerate(feature_names):
        X_val_permuted = X_ts.copy()
        np.random.shuffle(X_val_permuted[:, :, i])
        preds = _predict(X_val_permuted)
        loss = mean_squared_error(y_val, preds)
        importances.append(loss - base_loss)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    return importance_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution flow"""
    logger.info("Starting LSTM forecasting pipeline...")
    
    # Load allowed symbols from symbols.json to avoid training on stale/
    # non-tradeable data (e.g. leftover USDZAR from a previous download).
    symbols_file = Path(__file__).parent.parent.parent / "config" / "symbols.json"
    allowed_stems: set[str] | None = None
    if symbols_file.exists():
        with open(symbols_file) as f:
            _sym_data = json.load(f)
        allowed_stems = {
            sanitize_filename(s["name"]) + "_daily_processed"
            for s in _sym_data.get("symbols", [])
        }
        logger.info(f"Loaded {len(allowed_stems)} allowed symbols from symbols.json")

    # Load all CSVs from category subdirectories
    all_csvs = []
    for category in ["forex", "indices", "commodities", "crypto"]:
        cat_dir = CONFIG["DATA_DIR"] / category
        if cat_dir.exists():
            all_csvs.extend(sorted(glob.glob(str(cat_dir / "*.csv"))))

    # Filter out any CSVs not in symbols.json
    if allowed_stems is not None:
        before = len(all_csvs)
        all_csvs = [p for p in all_csvs if Path(p).stem in allowed_stems]
        skipped = before - len(all_csvs)
        if skipped:
            logger.warning(f"Skipped {skipped} CSVs not in symbols.json")

    logger.info(f"Found {len(all_csvs)} instruments")
    
    if not all_csvs:
        logger.error(f"No CSV files found in {CONFIG['DATA_DIR']} subdirectories (forex/, indices/, commodities/, crypto/)")
        sys.exit(1)
    
    # ========================================================================
    # STEP 1: Determine global cutoffs & fit scaler
    # ========================================================================
    logger.info("Step 1: Determining global cutoffs and fitting scaler...")
    
    global global_min_date, global_max_date, feature_cols, scaler
    global global_thresholds_up, global_thresholds_down, _ctx
    
    global_min_date, global_max_date = None, None
    scaler_inputs = []
    feature_cols = None
    scaler = StandardScaler()

    for csv_path in all_csvs:
        df_tmp = pd.read_csv(csv_path, parse_dates=["date"])
        if global_min_date is None or df_tmp["date"].min() < global_min_date:
            global_min_date = df_tmp["date"].min()
        if global_max_date is None or df_tmp["date"].max() > global_max_date:
            global_max_date = df_tmp["date"].max()

    train_val_cutoff = global_min_date + (global_max_date - global_min_date) * CONFIG["TRAIN_VAL_FRAC"]
    train_cutoff = global_min_date + (train_val_cutoff - global_min_date) * (1 - CONFIG["VAL_FRAC_WITHIN_TRAIN"])

    logger.info(f"Train cutoff: {train_cutoff}")
    logger.info(f"Train-Val cutoff: {train_val_cutoff}")

    for csv_path in all_csvs:
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").dropna()
        df = add_features(df)
        if df.empty:
            continue
        feat_cols = [c for c in df.columns if c not in CONFIG["EXCLUDED_COLS"]]
        if feature_cols is None:
            feature_cols = feat_cols
        train_rows = df[df["date"] < train_cutoff]
        if len(train_rows) > 0:
            scaler_inputs.append(train_rows[feat_cols].values)

    X = np.vstack(scaler_inputs)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(X)
    logger.info(f"Fitted scaler. n_features: {len(feature_cols)}")

    # ========================================================================
    # STEP 1b: Compute classification thresholds PER SYMBOL from TRAINING data
    # ========================================================================
    logger.info("Step 1b: Computing per-symbol target thresholds from training data only...")
    
    global per_symbol_thresholds, symbol_ids

    # Per-asset-class sigma factors control what counts as a "positive" label.
    # Lower values → more positive labels → stronger learning signal.
    # σ=1.5 gave only ~5% positives (too sparse). σ=0.75-1.0 gives ~15-20%.
    _sf_default = CONFIG.get("SIGMA_FACTOR", 1.0)
    _sf_per_class = CONFIG.get("SIGMA_FACTOR_PER_CLASS", {
        "forex": 0.75, "indices": 1.0, "commodities": 1.0, "crypto": 1.5,
    })

    per_symbol_thresholds = {}
    symbol_ids = {}

    # Also accumulate global stats as a fallback for symbols with too few rows
    all_targets = {f"Target_{h}": [] for h in HORIZONS}

    for idx, csv_path in enumerate(all_csvs):
        symbol_stem = Path(csv_path).stem
        symbol_ids[symbol_stem] = idx

        # Infer asset class from parent folder name to select sigma factor
        asset_folder = Path(csv_path).parent.name
        _sf = _sf_per_class.get(asset_folder, _sf_default)

        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").dropna()
        df = add_features(df)
        if df.empty:
            continue
        train_rows = df[df["date"] < train_cutoff]
        if len(train_rows) == 0:
            continue

        # Accumulate for global fallback
        for h in HORIZONS:
            all_targets[f"Target_{h}"].append(train_rows[f"Target_{h}"].values)

        # Per-symbol thresholds — only if enough training rows for stable stats
        if len(train_rows) >= 50:
            sym_up, sym_down = {}, {}
            for h in HORIZONS:
                vals = train_rows[f"Target_{h}"].dropna().values
                mu = np.nanmean(vals)
                sig = np.nanstd(vals)
                sym_up[h] = float(mu + _sf * sig)
                sym_down[h] = float(mu - _sf * sig)
            per_symbol_thresholds[symbol_stem] = {"up": sym_up, "down": sym_down}
            logger.info(
                f"  {symbol_stem} [{asset_folder}, σ×{_sf}]: "
                f"up_1d={sym_up['1d']:.6f}, down_1d={sym_down['1d']:.6f} "
                f"(n_train={len(train_rows)})"
            )

    # Global fallback thresholds (for symbols with <50 train rows)
    global_thresholds_up = {}
    global_thresholds_down = {}
    for h in HORIZONS:
        vals = np.concatenate(all_targets[f"Target_{h}"])
        mu = np.nanmean(vals)
        sig = np.nanstd(vals)
        global_thresholds_up[h] = float(mu + _sf_default * sig)
        global_thresholds_down[h] = float(mu - _sf_default * sig)
        logger.info(f"  GLOBAL {h}: up={global_thresholds_up[h]:.6f}, down={global_thresholds_down[h]:.6f}")

    # Assign global thresholds to symbols without per-symbol stats
    for csv_path in all_csvs:
        symbol_stem = Path(csv_path).stem
        if symbol_stem not in per_symbol_thresholds:
            per_symbol_thresholds[symbol_stem] = {
                "up": global_thresholds_up,
                "down": global_thresholds_down,
            }
            logger.info(f"  {symbol_stem}: using GLOBAL thresholds (insufficient training data)")

    n_symbols = len(symbol_ids)
    logger.info(f"Per-symbol thresholds computed for {len(per_symbol_thresholds)} instruments, "
                f"{n_symbols} symbol IDs assigned")

    # Populate module-level PipelineContext for use by process_instrument
    _ctx = PipelineContext(
        scaler=scaler,
        feature_cols=feature_cols,
        thresholds_up=global_thresholds_up,
        thresholds_down=global_thresholds_down,
        per_symbol_thresholds=per_symbol_thresholds,
        symbol_ids=symbol_ids,
    )

    # ========================================================================
    # STEP 2: Cache preprocessed data
    # ========================================================================
    logger.info("Step 2: Caching preprocessed data...")
    time.sleep(1)
    
    if CONFIG["CACHE_DIR"].exists():
        shutil.rmtree(CONFIG["CACHE_DIR"])
    CONFIG["CACHE_DIR"].mkdir(exist_ok=True)

    for csv_path in all_csvs:
        cache_preprocessed(Path(csv_path), train_cutoff, train_val_cutoff)
    
    logger.info("Done caching.")

    # ========================================================================
    # STEP 3: Create data generators
    # ========================================================================
    logger.info("Step 3: Creating data generators...")
    
    _regime = CONFIG.get("_REGIME_FILTER", "all")
    _bull_thr = CONFIG.get("REGIME_BULL_THRESHOLD", 0.65)
    _bear_thr = CONFIG.get("REGIME_BEAR_THRESHOLD", 0.35)
    if _regime != "all":
        logger.info(f"Regime filter active: '{_regime}' (bull>={_bull_thr}, bear<={_bear_thr})")

    _use_pnl_loss = CONFIG.get("PNL_LOSS_ENABLED", False)
    if _use_pnl_loss:
        logger.info("P&L-weighted focal loss enabled (ATR weight channel appended to labels)")

    train_gen = InstrumentDataGenerator(
        all_csvs, batch_size=128, split="train", shuffle=True,
        use_time_weights=True, decay_factor=0.0005,
        symbol_id_map=symbol_ids,
        p_augment=CONFIG.get("AUGMENT_PROB", 0.15),
        regime_filter=_regime,
        regime_bull_threshold=_bull_thr,
        regime_bear_threshold=_bear_thr,
        pnl_loss=_use_pnl_loss,
    )
    val_gen = InstrumentDataGenerator(
        all_csvs, batch_size=128, split="val", shuffle=False,
        use_time_weights=False,
        symbol_id_map=symbol_ids,
        regime_filter=_regime,
        regime_bull_threshold=_bull_thr,
        regime_bear_threshold=_bear_thr,
        pnl_loss=_use_pnl_loss,
    )
    
    logger.info(f"Train generator length: {len(train_gen)}")
    logger.info(f"Val generator length: {len(val_gen)}")

    # ========================================================================
    # STEP 3b: Self-supervised pre-training (optional)
    # ========================================================================
    _pretrain_weights_path = CONFIG["MODEL_DIR"] / "encoder_pretrained_weights.npz"
    _pretrained_weights = None

    if CONFIG.get("PRETRAIN_ENABLED", True) and not _pretrain_weights_path.exists():
        logger.info("Step 3b: Self-supervised pre-training (next-candle prediction)...")
        try:
            from src.models.lstm import build_encoder_decoder

            n_future = CONFIG.get("PRETRAIN_N_FUTURE", 5)
            enc_dec = build_encoder_decoder(len(feature_cols), CONFIG["WINDOW_SIZE"], n_future=n_future)
            enc_dec.compile(optimizer="adam", loss="mse")

            # Build a generator that yields (X, X_future) pairs
            # X_future = the scaled features for the next n_future rows after window
            # We use the train split, unfiltered (all regimes)
            pretrain_gen = InstrumentDataGenerator(
                all_csvs, batch_size=128, split="train", shuffle=True,
                use_time_weights=False, symbol_id_map=symbol_ids,
            )

            class PretextGen(tf.keras.utils.Sequence):
                """Yield (X_past, X_future) pairs for next-candle pre-training."""
                def __init__(self, base_gen, fc_list):
                    self.base = base_gen
                    self.n_future = n_future
                    self.fc = fc_list

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, idx):
                    batch = self.base[idx]
                    x_b = batch[0] if not isinstance(batch[0], tuple) else batch[0][0]
                    # x_b shape: (batch, window+1, n_features)
                    # Target: next n_future rows — approximate with last n_future windows shifted
                    # Use the last n_future timesteps of the window as a proxy target
                    x_future = x_b[:, -n_future:, :]  # (batch, n_future, n_features)
                    x_past = x_b                       # full window as input
                    return x_past, x_future

                def on_epoch_end(self):
                    self.base.on_epoch_end()

            pretext_gen = PretextGen(pretrain_gen, feature_cols)
            pretrain_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            enc_dec.fit(
                pretext_gen,
                epochs=CONFIG.get("PRETRAIN_EPOCHS", 30),
                callbacks=[pretrain_stop],
                verbose=1,
            )

            # Extract encoder weights by layer name and save
            encoder_submodel = enc_dec.get_layer("encoder")
            weight_dict = {w.name: w.numpy() for w in encoder_submodel.weights}
            np.savez(_pretrain_weights_path, **{k.replace("/", "_"): v for k, v in weight_dict.items()})
            _pretrained_weights = weight_dict
            logger.info(f"Pre-training complete. Encoder weights saved to {_pretrain_weights_path}")
            del enc_dec, pretext_gen, pretrain_gen

        except Exception as e:
            logger.warning(f"Pre-training failed (non-fatal): {e} — proceeding with random init")
            import traceback; traceback.print_exc()

    elif _pretrain_weights_path.exists() and CONFIG.get("PRETRAIN_ENABLED", True):
        logger.info(f"Step 3b: Loading cached pre-trained encoder weights from {_pretrain_weights_path}")
        raw = np.load(_pretrain_weights_path, allow_pickle=True)
        _pretrained_weights = {k.replace("_", "/"): v for k, v in raw.items()}

    # ========================================================================
    # STEP 4: Build and train model(s)
    # ========================================================================
    logger.info("Step 4: Building and training model(s)...")

    n_features = len(feature_cols)

    def _train_one(seed: int):
        """Train one model with the given random seed. Returns (model, val_auc, history)."""
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model_type = CONFIG.get("MODEL_TYPE", "lstm").lower()
        if model_type == "tft":
            from src.models.tft import build_tft_model
            m = build_tft_model(n_features, CONFIG["WINDOW_SIZE"], n_symbols=n_symbols)
        else:
            m = build_model(n_features, CONFIG["WINDOW_SIZE"], n_symbols=n_symbols)

        # Load pre-trained encoder weights if available (LSTM only)
        if _pretrained_weights and model_type == "lstm":
            loaded = 0
            for w in m.weights:
                # Match by normalised name (replace "/" with "_" for npz compat)
                key = w.name.replace("/", "_")
                if key in _pretrained_weights:
                    try:
                        w.assign(_pretrained_weights[key])
                        loaded += 1
                    except Exception:
                        pass
            if loaded:
                logger.info(f"  Loaded {loaded} pre-trained encoder weights into model")

        if _use_pnl_loss:
            from src.models.layers import pnl_weighted_focal_loss
            loss_fn = pnl_weighted_focal_loss(gamma=2.0, alpha=0.75)
        else:
            loss_fn = binary_focal_loss(gamma=2.0, alpha=0.75)
        m.compile(optimizer="adam", loss=loss_fn, metrics=[AUC(name="auc")])

        early_stop = EarlyStopping(
            patience=CONFIG["EARLY_STOP_PATIENCE"],
            monitor="val_auc", restore_best_weights=True, mode="max",
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_auc", factor=0.5,
            patience=CONFIG["REDUCE_LR_PATIENCE"], min_lr=1e-5, mode="max", verbose=1,
        )

        hist = m.fit(train_gen, validation_data=val_gen, callbacks=[early_stop, reduce_lr],
                     epochs=CONFIG["MAX_EPOCHS"], verbose=1)
        best_epoch = int(np.argmax(hist.history.get("val_auc", [0])))
        auc = hist.history.get("val_auc", [0])[best_epoch]
        return m, auc, hist

    # Log estimated positive rate once
    try:
        sample_batch = train_gen[0]
        y_sample = sample_batch[1] if isinstance(sample_batch, tuple) else sample_batch
        pos_rate = float(np.mean(y_sample > 0.5))
        logger.info(f"Estimated positive label rate: {pos_rate:.4f} ({pos_rate*100:.1f}%)")
    except Exception:
        pass

    n_ensemble = CONFIG.get("N_ENSEMBLE", 1)
    best_model, best_val_auc, best_history = None, -1.0, None
    ensemble_paths = []

    for ens_i in range(n_ensemble):
        seed = RANDOM_SEED + ens_i * 100
        logger.info(f"Training model {ens_i + 1}/{n_ensemble} (seed={seed})...")
        m_i, auc_i, hist_i = _train_one(seed)
        save_path = CONFIG["MODEL_DIR"] / f"lstm_model_{ens_i}.keras"
        m_i.save(save_path)
        ensemble_paths.append(save_path)
        logger.info(f"  Model {ens_i}: val_AUC={auc_i:.4f} → saved to {save_path.name}")
        if auc_i > best_val_auc:
            best_model, best_val_auc, best_history = m_i, auc_i, hist_i

    model = best_model
    history = best_history
    final_val_auc = best_val_auc

    # ========================================================================
    # STEP 4b: Model quality gate — refuse to deploy a bad model
    # ========================================================================
    logger.info("Step 4b: Checking model quality before deployment...")

    MIN_VAL_AUC = CONFIG["MIN_VAL_AUC"]

    val_auc_hist = history.history.get("val_auc", [0])
    val_loss_hist = history.history.get("val_loss", [float('inf')])
    best_epoch = int(np.argmax(val_auc_hist))
    final_val_loss = val_loss_hist[best_epoch]
    logger.info(
        f"Best ensemble val_auc: {final_val_auc:.4f} (at epoch {best_epoch + 1}), "
        f"val loss: {final_val_loss:.4f}"
    )

    if final_val_auc < MIN_VAL_AUC:
        logger.error(
            f"MODEL QUALITY GATE FAILED: val AUC {final_val_auc:.4f} < minimum {MIN_VAL_AUC}. "
            f"Model is near-random and will NOT be saved. "
            f"Check data quality, class balance, and feature engineering."
        )
        sys.exit(1)

    logger.info(f"Model quality gate PASSED (val AUC {final_val_auc:.4f} >= {MIN_VAL_AUC})")

    # ========================================================================
    # STEP 4c: Save model, scaler, and feature_cols for live trading
    # ========================================================================
    logger.info("Step 4c: Saving model artifacts for live trading...")

    # Primary model (best of ensemble) — used as fallback by single-model bots.
    # If a regime filter was used, also save under the regime-specific name.
    primary_path = CONFIG["MODEL_DIR"] / "lstm_model.keras"
    model.save(primary_path)
    logger.info(f"Saved primary model to {primary_path}")

    _regime_filter = CONFIG.get("_REGIME_FILTER", "all")
    if _regime_filter in ("bull", "bear"):
        regime_path = CONFIG["MODEL_DIR"] / f"lstm_{_regime_filter}.keras"
        model.save(regime_path)
        logger.info(f"Saved regime model to {regime_path}")
    
    joblib.dump(scaler, CONFIG["MODEL_DIR"] / "scaler.joblib")
    logger.info(f"Saved scaler to {CONFIG['MODEL_DIR'] / 'scaler.joblib'}")
    
    with open(CONFIG["MODEL_DIR"] / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    logger.info(f"Saved feature_cols to {CONFIG['MODEL_DIR'] / 'feature_cols.json'}")
    
    # Save feature version hash (detects processor changes vs saved scaler)
    feature_hash = hashlib.sha256(json.dumps(sorted(feature_cols)).encode()).hexdigest()[:16]
    with open(CONFIG["MODEL_DIR"] / "feature_hash.txt", "w") as f:
        f.write(feature_hash)
    logger.info(f"Saved feature hash: {feature_hash}")
    
    thresholds_artifact = {
        "thresholds_up": global_thresholds_up,
        "thresholds_down": global_thresholds_down,
    }
    with open(CONFIG["MODEL_DIR"] / "thresholds.json", "w") as f:
        json.dump(thresholds_artifact, f, indent=2)
    logger.info(f"Saved thresholds to {CONFIG['MODEL_DIR'] / 'thresholds.json'}")

    # Save per-symbol thresholds (used by process_instrument for correct labels)
    with open(CONFIG["MODEL_DIR"] / "per_symbol_thresholds.json", "w") as f:
        json.dump(per_symbol_thresholds, f, indent=2)
    logger.info(f"Saved per-symbol thresholds for {len(per_symbol_thresholds)} instruments")

    # Save symbol-to-ID mapping (needed at inference time for the embedding input)
    with open(CONFIG["MODEL_DIR"] / "symbol_ids.json", "w") as f:
        json.dump(symbol_ids, f, indent=2)
    logger.info(f"Saved symbol_ids for {len(symbol_ids)} instruments")

    # ========================================================================
    # STEP 4d: Fit probability calibrator on validation predictions
    # ========================================================================
    logger.info("Step 4d: Fitting probability calibrator on validation set...")

    try:
        # Collect all validation predictions and labels
        y_val_all, y_prob_all = [], []
        for batch_idx in range(len(val_gen)):
            batch = val_gen[batch_idx]
            if isinstance(batch, tuple):
                x_batch, y_batch = batch[0], batch[1]
            else:
                raise ValueError("val_gen did not return a tuple.")
            preds = model.predict(x_batch, verbose=0)
            y_val_all.append(y_batch)
            y_prob_all.append(preds)

        y_val_concat = np.concatenate(y_val_all, axis=0)
        y_prob_concat = np.concatenate(y_prob_all, axis=0)

        calibrator = fit_calibrator(y_val_concat, y_prob_concat, method="isotonic")
        save_calibrator(calibrator)
        logger.info("Calibrator fitted and saved successfully")

        # Log calibration effect on a sample
        cal_probs = calibrator.transform(y_prob_concat[:100])
        logger.info(
            f"  Raw probs range:  [{y_prob_concat[:100].min():.4f}, {y_prob_concat[:100].max():.4f}]"
        )
        logger.info(
            f"  Cal probs range:  [{cal_probs.min():.4f}, {cal_probs.max():.4f}]"
        )
    except Exception as e:
        logger.warning(f"Calibrator fitting failed: {e} — raw probabilities will be used")

    # Save training history plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG["FORECAST_DIR"] / "training_history.png", dpi=150)
    plt.close()
    logger.info("Saved training history plot")

    # Save training history as JSON for programmatic comparison
    history_data = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    history_data["final_val_auc"] = float(final_val_auc)
    history_data["final_val_loss"] = float(final_val_loss)
    with open(CONFIG["MODEL_DIR"] / "training_history.json", "w") as f:
        json.dump(history_data, f, indent=2)
    logger.info("Saved training history JSON")

    # ========================================================================
    # STEP 5: Feature importance
    # ========================================================================
    logger.info("Step 5: Computing feature importance...")
    
    batch = val_gen[0]
    if isinstance(batch, tuple):
        if len(batch) == 2:
            X_val, y_val = batch
        elif len(batch) == 3:
            X_val, y_val, _ = batch
        else:
            raise ValueError(f"Unexpected number of outputs: {len(batch)}")
    else:
        raise ValueError("val_gen[0] did not return a tuple.")
    
    importance_df = feature_importance(model, X_val, y_val, feature_cols)

    # Plot feature importance
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.3)))

    ax.barh(importance_df["Feature"], importance_df["Importance"], color="white", linewidth=1.2)
    ax.set_xlabel("Increase in MSE After Feature Permutation", fontsize=12)
    ax.set_ylabel("Feature", color="lime", fontsize=12)
    ax.set_title("Feature Importance (Permutation Method)", fontsize=14, pad=15)

    ax.tick_params(axis='x', colors='lime')
    ax.tick_params(axis='y', colors='lime')
    ax.grid(color='lime', linestyle='--', linewidth=0.3, alpha=0.3)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    output_path = CONFIG["FORECAST_DIR"] / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path}")

    # ========================================================================
    # STEP 6: Generate forecasts
    # ========================================================================
    logger.info("Step 6: Generating forecasts...")
    
    for file in glob.glob(str(CONFIG["FORECAST_DIR"] / "*.csv")):
        os.remove(file)

    for csv_path in all_csvs:
        symbol_name = Path(csv_path).stem

        X_all, closes, pred_dates, df = process_instrument_for_inference(Path(csv_path), ctx=_ctx)

        if X_all.size == 0:
            continue

        sym_id = symbol_ids.get(symbol_name, 0)
        forecast_df = make_forecast(
            model=model,
            X=X_all,
            dates=pred_dates,
            closes=closes,
            horizons=horizons,
            horizon_days=horizon_days,
            symbol_id=sym_id,
        )

        forecast_df.to_csv(CONFIG["FORECAST_DIR"] / f"{symbol_name}_forecast.csv", index=False)
        logger.info(f"Saved forecast for {symbol_name}")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="LSTM forecasting pipeline")
    _parser.add_argument(
        "--regime", choices=["all", "bull", "bear"], default="all",
        help="Train on a regime-filtered subset: 'all' (default), 'bull' (regime_trend>=0.65), 'bear' (regime_trend<=0.35)"
    )
    _parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Skip self-supervised pre-training even if PRETRAIN_ENABLED=True"
    )
    _args = _parser.parse_args()
    CONFIG["_REGIME_FILTER"] = _args.regime
    if _args.skip_pretrain:
        CONFIG["PRETRAIN_ENABLED"] = False
    main()
