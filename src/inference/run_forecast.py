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
    """Cache preprocessed data split by train/val/test"""
    symbol_name = csv_path.stem
    logger.info(f"Rebuilding cache for: {symbol_name}")

    X, y, _, y_dates = process_instrument(csv_path, for_training=True, ctx=ctx)
    if X.size == 0:
        for split in ["train", "val", "test"]:
            np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{split}.npy", np.array([]))
            np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_{split}.npy", np.array([]))
        return

    y_dates = pd.to_datetime(y_dates)
    splits = {
        "train": y_dates < np.datetime64(train_cutoff),
        "val": (y_dates >= np.datetime64(train_cutoff)) & (y_dates < np.datetime64(train_val_cutoff)),
        "test": y_dates >= np.datetime64(train_val_cutoff),
    }
    for split, mask in splits.items():
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{split}.npy", X[mask])
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_{split}.npy", y[mask])
        np.save(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_dates_{split}.npy", y_dates[mask])

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
        self.windows = []
        self.symbol_names = [Path(p).stem for p in csv_paths]
        self._prepare_indices()
        self.on_epoch_end()

    def _prepare_indices(self):
        """Prepare window indices for all instruments"""
        self.windows = []
        self.lengths = {}
        self.date_arrays = {}
        for symbol_name in self.symbol_names:
            X_path = CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{self.split}.npy"
            y_path = CONFIG["CACHE_DIR"] / f"{symbol_name}_y_{self.split}.npy"
            date_path = CONFIG["CACHE_DIR"] / f"{symbol_name}_y_dates_{self.split}.npy"
            if not X_path.exists():
                n_windows = 0
            else:
                X = np.load(X_path, mmap_mode="r")
                n_windows = len(X)
                if date_path.exists():
                    self.date_arrays[symbol_name] = np.load(date_path, allow_pickle=True)
            self.lengths[symbol_name] = n_windows
            for i in range(n_windows):
                self.windows.append((symbol_name, i))
        self.indices = np.arange(len(self.windows))

    def __len__(self):
        return int(np.ceil(len(self.windows) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, y_batch, sym_ids_batch = [], [], []
        weights_batch = [] if self.use_time_weights else None
        cache = {}
        for bi in batch_indices:
            symbol_name, win_idx = self.windows[bi]
            if symbol_name not in cache:
                X = np.load(CONFIG["CACHE_DIR"] / f"{symbol_name}_X_{self.split}.npy", mmap_mode="r")
                y = np.load(CONFIG["CACHE_DIR"] / f"{symbol_name}_y_{self.split}.npy", mmap_mode="r")
                cache[symbol_name] = (X, y)
            X_arr, y_arr = cache[symbol_name]
            X_batch.append(X_arr[win_idx])
            y_batch.append(y_arr[win_idx])
            sym_ids_batch.append(self.symbol_id_map.get(symbol_name, 0))

            if self.use_time_weights:
                if symbol_name in self.date_arrays:
                    dates = self.date_arrays[symbol_name]
                    date = pd.to_datetime(dates[win_idx])
                    # Give more weight to recent dates
                    date_ago = (pd.Timestamp.now() - date).days
                    weights_batch.append(np.exp(-self.decay_factor * date_ago))
                else:
                    weights_batch.append(1.0)

        X_out = np.array(X_batch, dtype=np.float32)
        y_out = np.array(y_batch, dtype=np.float32)
        sym_ids_out = np.array(sym_ids_batch, dtype=np.int32)

        # Build multi-input list when symbol IDs are available
        if self.symbol_id_map:
            x_inputs = [X_out, sym_ids_out]
        else:
            x_inputs = X_out

        if self.use_time_weights:
            return (x_inputs, y_out, np.array(weights_batch, dtype=np.float32))
        return (x_inputs, y_out)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


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
    if isinstance(X_val, list):
        X_ts = X_val[0]
        aux_inputs = X_val[1:]
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

    _sf = CONFIG.get("SIGMA_FACTOR", 1.5)
    per_symbol_thresholds = {}
    symbol_ids = {}

    # Also accumulate global stats as a fallback for symbols with too few rows
    all_targets = {f"Target_{h}": [] for h in HORIZONS}

    for idx, csv_path in enumerate(all_csvs):
        symbol_stem = Path(csv_path).stem
        symbol_ids[symbol_stem] = idx

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
                f"  {symbol_stem}: up_1d={sym_up['1d']:.6f}, down_1d={sym_down['1d']:.6f} "
                f"(n_train={len(train_rows)})"
            )

    # Global fallback thresholds (for symbols with <50 train rows)
    global_thresholds_up = {}
    global_thresholds_down = {}
    for h in HORIZONS:
        vals = np.concatenate(all_targets[f"Target_{h}"])
        mu = np.nanmean(vals)
        sig = np.nanstd(vals)
        global_thresholds_up[h] = float(mu + _sf * sig)
        global_thresholds_down[h] = float(mu - _sf * sig)
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
    
    train_gen = InstrumentDataGenerator(
        all_csvs, batch_size=128, split="train", shuffle=True,
        use_time_weights=True, decay_factor=0.002,
        symbol_id_map=symbol_ids,
    )
    val_gen = InstrumentDataGenerator(
        all_csvs, batch_size=128, split="val", shuffle=False,
        use_time_weights=False,
        symbol_id_map=symbol_ids,
    )
    
    logger.info(f"Train generator length: {len(train_gen)}")
    logger.info(f"Val generator length: {len(val_gen)}")

    # ========================================================================
    # STEP 4: Build and train model
    # ========================================================================
    logger.info("Step 4: Building model...")
    
    n_features = len(feature_cols)
    model = build_model(n_features, CONFIG["WINDOW_SIZE"], n_symbols=n_symbols)

    early_stop = EarlyStopping(
        patience=CONFIG["EARLY_STOP_PATIENCE"],
        monitor="val_loss",
        restore_best_weights=True,
        mode="min",
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=CONFIG["REDUCE_LR_PATIENCE"],
        min_lr=1e-5,
        verbose=1,
    )

    # Use focal loss instead of binary_crossentropy to handle severe class
    # imbalance (~5-7% positive rate at 1.5σ).  Focal loss down-weights easy
    # negatives so the gradient signal comes from informative hard examples.
    focal_loss = binary_focal_loss(gamma=2.0, alpha=0.25)
    model.compile(optimizer="adam", loss=focal_loss, metrics=[AUC(name="auc")])

    # Log estimated positive rate for diagnostics
    try:
        sample_batch = train_gen[0]
        y_sample = sample_batch[1] if isinstance(sample_batch, tuple) else sample_batch
        pos_rate = float(np.mean(y_sample > 0.5))
        logger.info(f"Estimated positive label rate: {pos_rate:.4f} ({pos_rate*100:.1f}%)")
    except Exception:
        pass

    logger.info("Training model with focal loss (gamma=2.0, alpha=0.25)...")
    history = model.fit(train_gen, validation_data=val_gen, callbacks=[early_stop, reduce_lr], 
                        epochs=CONFIG["MAX_EPOCHS"], verbose=1)

    # ========================================================================
    # STEP 4b: Model quality gate — refuse to deploy a bad model
    # ========================================================================
    logger.info("Step 4b: Checking model quality before deployment...")
    
    MIN_VAL_AUC = CONFIG["MIN_VAL_AUC"]

    # Use the best-epoch metrics (matching restore_best_weights=True),
    # not the last epoch which may be worse.
    val_auc_hist = history.history.get("val_auc", [0])
    val_loss_hist = history.history.get("val_loss", [float('inf')])
    best_epoch = int(np.argmin(val_loss_hist))  # EarlyStopping monitors val_loss
    final_val_auc = val_auc_hist[best_epoch]
    final_val_loss = val_loss_hist[best_epoch]
    logger.info(
        f"Best val_loss at epoch {best_epoch + 1}: "
        f"val AUC: {final_val_auc:.4f}, val loss: {final_val_loss:.4f}"
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
    
    model.save(CONFIG["MODEL_DIR"] / "lstm_model.keras")
    logger.info(f"Saved model to {CONFIG['MODEL_DIR'] / 'lstm_model.keras'}")
    
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
    main()
