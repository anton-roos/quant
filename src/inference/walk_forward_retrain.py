#!/usr/bin/env python3
"""
Walk-forward retraining for the LSTM trading model.

Periodically retrains using an expanding window of data.  The new model
is deployed *only* if it beats the current one on out-of-sample validation.

Usage:
    # Standalone (monthly cron or manual):
    python -m src.inference.walk_forward_retrain

    # Or called from the bot (see TradingBot.maybe_retrain).
"""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import AUC
from sklearn.preprocessing import StandardScaler

from src.models.layers import MCDropout
from src.models.lstm import build_model
from src.models.context import PipelineContext
from src.utils.constants import HORIZONS, EXCLUDED_COLS
from src.inference.run_forecast import (
    CONFIG,
    add_features,
    InstrumentDataGenerator,
    mc_dropout_predict,
    cache_preprocessed,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

RETRAIN_STATE_PATH = PROJECT_ROOT / "outputs" / "models" / "retrain_state.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_retrain_state() -> dict:
    if RETRAIN_STATE_PATH.exists():
        with open(RETRAIN_STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_retrain_state(state: dict):
    with open(RETRAIN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def is_retrain_due(interval_days: int = 30) -> bool:
    """Return True if enough time has passed since the last retrain."""
    state = _load_retrain_state()
    last = state.get("last_retrain_date")
    if last is None:
        return True
    last_dt = datetime.fromisoformat(last)
    return (datetime.utcnow() - last_dt).days >= interval_days


def _collect_csvs() -> list:
    all_csvs = []
    for category in ["forex", "indices", "commodities", "crypto"]:
        cat_dir = CONFIG["DATA_DIR"] / category
        if cat_dir.exists():
            all_csvs.extend(sorted(cat_dir.glob("*.csv")))
    return [str(p) for p in all_csvs]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model_on_val(model, val_gen) -> float:
    """Return average validation loss (binary cross-entropy)."""
    losses = []
    for i in range(len(val_gen)):
        batch = val_gen[i]
        if len(batch) == 3:
            X_b, y_b, _ = batch
        else:
            X_b, y_b = batch
        preds = model.predict(X_b, verbose=0)
        # binary cross-entropy per sample
        eps = 1e-7
        bce = -np.mean(
            y_b * np.log(preds + eps) + (1 - y_b) * np.log(1 - preds + eps)
        )
        losses.append(bce)
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main retrain logic
# ---------------------------------------------------------------------------

def retrain(force: bool = False, interval_days: int = 30) -> bool:
    """
    Run walk-forward retraining.

    1. Fit scaler on *all* available training data (expanding window).
    2. Train a new model.
    3. Compare new vs old on the most recent validation slice.
    4. Deploy if new model is better; otherwise discard.

    Returns True if a new model was deployed.
    """
    if not force and not is_retrain_due(interval_days):
        logger.info("Retrain not due yet — skipping.")
        return False

    logger.info("=" * 60)
    logger.info("WALK-FORWARD RETRAINING")
    logger.info("=" * 60)

    all_csvs = _collect_csvs()
    if not all_csvs:
        logger.error("No processed CSVs found — cannot retrain.")
        return False

    # ------------------------------------------------------------------
    # 1. Compute expanding-window cutoffs
    # ------------------------------------------------------------------
    global_min_date, global_max_date = None, None
    for csv_path in all_csvs:
        df_tmp = pd.read_csv(csv_path, parse_dates=["date"])
        if global_min_date is None or df_tmp["date"].min() < global_min_date:
            global_min_date = df_tmp["date"].min()
        if global_max_date is None or df_tmp["date"].max() > global_max_date:
            global_max_date = df_tmp["date"].max()

    train_val_cutoff = global_min_date + (global_max_date - global_min_date) * CONFIG["TRAIN_VAL_FRAC"]
    train_cutoff = global_min_date + (train_val_cutoff - global_min_date) * (1 - CONFIG["VAL_FRAC_WITHIN_TRAIN"])

    logger.info(f"Expanding window: {global_min_date.date()} → {global_max_date.date()}")
    logger.info(f"Train cutoff: {train_cutoff.date()}, Val cutoff: {train_val_cutoff.date()}")

    # ------------------------------------------------------------------
    # 2. Fit scaler on training data
    # ------------------------------------------------------------------
    feature_cols = None
    scaler_inputs = []

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

    if not scaler_inputs:
        logger.error("No training data — aborting retrain.")
        return False

    new_scaler = StandardScaler()
    X_all = np.nan_to_num(np.vstack(scaler_inputs), nan=0.0, posinf=1e6, neginf=-1e6)
    new_scaler.fit(X_all)
    n_features = len(feature_cols)
    logger.info(f"Scaler fitted on {X_all.shape[0]} rows, {n_features} features")

    # ------------------------------------------------------------------
    # 3. Compute thresholds from training data
    # ------------------------------------------------------------------
    all_targets = {f"Target_{h}": [] for h in HORIZONS}
    for csv_path in all_csvs:
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").dropna()
        df = add_features(df)
        if df.empty:
            continue
        train_rows = df[df["date"] < train_cutoff]
        if len(train_rows) == 0:
            continue
        for h in HORIZONS:
            all_targets[f"Target_{h}"].append(train_rows[f"Target_{h}"].values)

    thresholds_up, thresholds_down = {}, {}
    for h in HORIZONS:
        vals = np.concatenate(all_targets[f"Target_{h}"])
        mu, sig = np.nanmean(vals), np.nanstd(vals)
        thresholds_up[h] = mu + 2 * sig
        thresholds_down[h] = mu - 2 * sig

    # Build a PipelineContext for this retrain run (no global mutation)
    retrain_ctx = PipelineContext(
        scaler=new_scaler,
        feature_cols=feature_cols,
        thresholds_up=thresholds_up,
        thresholds_down=thresholds_down,
    )

    # ------------------------------------------------------------------
    # 4. Cache preprocessed data
    # ------------------------------------------------------------------
    cache_bak = CONFIG["CACHE_DIR"].parent / "cache_retrain"
    if cache_bak.exists():
        shutil.rmtree(cache_bak)
    cache_bak.mkdir(exist_ok=True)

    old_cache = CONFIG["CACHE_DIR"]
    # Use a temp cache dir for retraining
    CONFIG["CACHE_DIR"] = cache_bak
    for csv_path in all_csvs:
        cache_preprocessed(Path(csv_path), train_cutoff, train_val_cutoff, ctx=retrain_ctx)
    CONFIG["CACHE_DIR"] = old_cache  # restore

    # ------------------------------------------------------------------
    # 5. Build & train new model
    # ------------------------------------------------------------------
    # Temporarily swap cache dir for generators
    orig_cache = CONFIG["CACHE_DIR"]
    CONFIG["CACHE_DIR"] = cache_bak

    train_gen = InstrumentDataGenerator(all_csvs, batch_size=128, split="train", shuffle=True, use_time_weights=False)
    val_gen = InstrumentDataGenerator(all_csvs, batch_size=128, split="val", shuffle=False)

    logger.info(f"Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")

    new_model = build_model(n_features, CONFIG["WINDOW_SIZE"])

    new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[AUC(name="auc")])

    early_stop = EarlyStopping(patience=CONFIG.get("EARLY_STOP_PATIENCE", 30), monitor="val_loss", restore_best_weights=True, mode="min")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=CONFIG.get("REDUCE_LR_PATIENCE", 10), min_lr=1e-6, verbose=1)

    new_model.fit(
        train_gen,
        validation_data=val_gen,
        callbacks=[early_stop, reduce_lr],
        epochs=CONFIG.get("MAX_EPOCHS", 500),
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 6. Compare new vs old model
    # ------------------------------------------------------------------
    new_val_loss = evaluate_model_on_val(new_model, val_gen)
    logger.info(f"New model val loss: {new_val_loss:.6f}")

    model_path = CONFIG["MODEL_DIR"] / "lstm_model.keras"
    deployed = False

    if model_path.exists():
        try:
            old_model = load_model(model_path, custom_objects={"MCDropout": MCDropout})
            # Need to use the new scaler for fair comparison
            old_val_loss = evaluate_model_on_val(old_model, val_gen)
            logger.info(f"Old model val loss: {old_val_loss:.6f}")

            if new_val_loss < old_val_loss:
                improvement = ((old_val_loss - new_val_loss) / old_val_loss) * 100
                logger.info(f"New model is BETTER by {improvement:.2f}% — deploying.")
                deployed = True
            else:
                logger.info("New model is NOT better — keeping old model.")
        except Exception as e:
            logger.warning(f"Could not load old model for comparison: {e}. Deploying new model.")
            deployed = True
    else:
        logger.info("No existing model found — deploying new model.")
        deployed = True

    CONFIG["CACHE_DIR"] = orig_cache  # restore

    # ------------------------------------------------------------------
    # 7. Deploy if better
    # ------------------------------------------------------------------
    if deployed:
        # Archive old model
        archive_dir = CONFIG["MODEL_DIR"] / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        for artifact in ["lstm_model.keras", "scaler.joblib", "scaler.pkl", "feature_cols.json", "feature_hash.txt", "thresholds.json"]:
            src = CONFIG["MODEL_DIR"] / artifact
            if src.exists():
                shutil.copy2(src, archive_dir / f"{ts}_{artifact}")

        # Save new artifacts
        new_model.save(model_path)
        joblib.dump(new_scaler, CONFIG["MODEL_DIR"] / "scaler.joblib")
        with open(CONFIG["MODEL_DIR"] / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f)
        feature_hash = hashlib.sha256(json.dumps(sorted(feature_cols)).encode()).hexdigest()[:16]
        with open(CONFIG["MODEL_DIR"] / "feature_hash.txt", "w") as f:
            f.write(feature_hash)
        with open(CONFIG["MODEL_DIR"] / "thresholds.json", "w") as f:
            json.dump({"thresholds_up": thresholds_up, "thresholds_down": thresholds_down}, f, indent=2)

        logger.info(f"New model deployed. Feature hash: {feature_hash}")

    # Clean up temp cache
    if cache_bak.exists():
        shutil.rmtree(cache_bak)

    # Update retrain state
    _save_retrain_state({
        "last_retrain_date": datetime.utcnow().isoformat(),
        "deployed": deployed,
        "new_val_loss": new_val_loss,
    })

    logger.info("Walk-forward retrain complete.")
    return deployed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward retraining")
    parser.add_argument("--force", action="store_true", help="Force retrain even if not due")
    parser.add_argument("--interval-days", type=int, default=30, help="Days between retrains")
    args = parser.parse_args()
    retrain(force=args.force, interval_days=args.interval_days)
