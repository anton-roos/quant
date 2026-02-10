"""
Diagnostic script to analyze model prediction distribution and potential directional bias.

Loads the trained model and generates predictions for all symbols to check if there's
a systematic bias toward BUY or SELL signals.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.models.layers import MCDropout
from src.utils.constants import sanitize_filename, CATEGORY_TYPE_TO_FOLDER

# Load configuration
def load_config() -> Dict:
    defaults = {
        "MODEL_PATH": "outputs/models/lstm_model.keras",
        "SCALER_PATH": "outputs/models/scaler.joblib",
        "FEATURES_PATH": "outputs/models/feature_cols.json",
        "SYMBOLS_PATH": "config/symbols.json",
        "PROCESSED_DIR": "src/data/indicators_data/processed",
        "WINDOW_SIZE": 90,
        "MC_DROPOUT_SAMPLES": 50,
        "STD_FACTOR": 1.0,
        "MIN_ACCEPTED": 0.10,
        "MIN_ACCEPTED_BUY": 0.10,
        "MIN_ACCEPTED_SELL": 0.05,
    }
    config_path = PROJECT_ROOT / "config" / "bot_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = json.load(f)
            defaults.update(user_config)
    return defaults

config = load_config()

# Load model artifacts
print("Loading model artifacts...")
model_path = PROJECT_ROOT / config["MODEL_PATH"]
scaler_path = PROJECT_ROOT / config["SCALER_PATH"]
features_path = PROJECT_ROOT / config["FEATURES_PATH"]
symbols_path = PROJECT_ROOT / config["SYMBOLS_PATH"]

model = load_model(model_path, custom_objects={"MCDropout": MCDropout})
scaler = joblib.load(scaler_path)

with open(features_path, "r") as f:
    feature_cols = json.load(f)

with open(symbols_path, "r") as f:
    symbols_data = json.load(f)
    symbols = symbols_data.get("symbols", [])

print(f"Model loaded: {len(feature_cols)} features, {len(symbols)} symbols")

# Prepare MC Dropout prediction function
mc_predict_fn = tf.function(lambda x: model(x, training=True))

def prepare_features(csv_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Read processed CSV and prepare feature window."""
    if not csv_path.exists():
        return None, None
    
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    
    # Add target columns (required by scaler)
    df["Target_1d"] = np.log(df["close"].shift(-1) / df["close"])
    df["Target_1w"] = np.log(df["close"].shift(-5) / df["close"])
    df["Target_1m"] = np.log(df["close"].shift(-21) / df["close"])
    df["Target_6m"] = np.log(df["close"].shift(-126) / df["close"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    window = config["WINDOW_SIZE"]
    if len(df) < window + 1:
        return None, None
    
    # Scale and prepare window
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    # Get latest window
    X_window = df_scaled[feature_cols].iloc[-window:].values
    
    return X_window[np.newaxis, :, :], df

# Analyze all symbols
print("\n" + "="*80)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("="*80)

results = []
buy_signals = []
sell_signals = []

horizons = ["1d", "1w", "1m", "6m"]
n_horizons = len(horizons)

processed_base = PROJECT_ROOT / config["PROCESSED_DIR"]
category_map = CATEGORY_TYPE_TO_FOLDER

for sym in symbols:
    mt5_name = sym["name"]
    safe_name = sanitize_filename(mt5_name)
    cat_folder = category_map.get(sym.get("type", ""), "other")
    csv_path = processed_base / cat_folder / f"{safe_name}_daily_processed.csv"
    
    try:
        X_window, df = prepare_features(csv_path)
        if X_window is None:
            continue
        
        # MC Dropout predictions
        samples = []
        for _ in range(config["MC_DROPOUT_SAMPLES"]):
            pred = mc_predict_fn(X_window)
            samples.append(pred.numpy())
        
        samples = np.array(samples)
        mean_probs = samples.mean(axis=0)
        std_probs = samples.std(axis=0)
        
        # Analyze each horizon
        for i, h in enumerate(horizons):
            # BUY signal (upside probability)
            pred_prob_buy = float(mean_probs[0, i])
            pred_std_buy = float(std_probs[0, i])
            adj_prob_buy = pred_prob_buy - config["STD_FACTOR"] * pred_std_buy
            
            # SELL signal (downside probability)
            pred_prob_sell = float(mean_probs[0, n_horizons + i])
            pred_std_sell = float(std_probs[0, n_horizons + i])
            adj_prob_sell = pred_prob_sell - config["STD_FACTOR"] * pred_std_sell
            
            results.append({
                "symbol": mt5_name,
                "type": sym.get("type", "Unknown"),
                "horizon": h,
                "buy_prob": pred_prob_buy,
                "buy_std": pred_std_buy,
                "buy_adj": adj_prob_buy,
                "sell_prob": pred_prob_sell,
                "sell_std": pred_std_sell,
                "sell_adj": adj_prob_sell,
                "diff": adj_prob_buy - adj_prob_sell,
            })
            
            if adj_prob_buy > config.get("MIN_ACCEPTED_BUY", config["MIN_ACCEPTED"]):
                buy_signals.append(adj_prob_buy)
            if adj_prob_sell > config.get("MIN_ACCEPTED_SELL", config["MIN_ACCEPTED"]):
                sell_signals.append(adj_prob_sell)
    
    except Exception as e:
        print(f"Error processing {mt5_name}: {e}")
        continue

# Convert to DataFrame for analysis
df_results = pd.DataFrame(results)

print(f"\nTotal predictions analyzed: {len(df_results)}")
print(f"Symbols with data: {df_results['symbol'].nunique()}")
print(f"Horizons analyzed: {horizons}")

print("\n" + "-"*80)
print("SIGNAL COUNT (above threshold)")
print("-"*80)
print(f"MIN_ACCEPTED_BUY threshold: {config.get('MIN_ACCEPTED_BUY', config['MIN_ACCEPTED']):.2%}")
print(f"MIN_ACCEPTED_SELL threshold: {config.get('MIN_ACCEPTED_SELL', config['MIN_ACCEPTED']):.2%}")
print(f"BUY signals above threshold: {len(buy_signals)}")
print(f"SELL signals above threshold: {len(sell_signals)}")
print(f"Ratio (BUY/SELL): {len(buy_signals)/max(1, len(sell_signals)):.2f}x")

print("\n" + "-"*80)
print("ADJUSTED PROBABILITY STATISTICS")
print("-"*80)
print("\nBUY (upside) probabilities:")
print(f"  Mean: {df_results['buy_adj'].mean():.4f}")
print(f"  Median: {df_results['buy_adj'].median():.4f}")
print(f"  Std: {df_results['buy_adj'].std():.4f}")
print(f"  Min: {df_results['buy_adj'].min():.4f}")
print(f"  Max: {df_results['buy_adj'].max():.4f}")

print("\nSELL (downside) probabilities:")
print(f"  Mean: {df_results['sell_adj'].mean():.4f}")
print(f"  Median: {df_results['sell_adj'].median():.4f}")
print(f"  Std: {df_results['sell_adj'].std():.4f}")
print(f"  Min: {df_results['sell_adj'].min():.4f}")
print(f"  Max: {df_results['sell_adj'].max():.4f}")

print("\nDifference (BUY - SELL):")
print(f"  Mean: {df_results['diff'].mean():.4f}")
print(f"  Median: {df_results['diff'].median():.4f}")

print("\n" + "-"*80)
print("TOP 20 BUY SIGNALS")
print("-"*80)
top_buys = df_results.nlargest(20, 'buy_adj')[['symbol', 'horizon', 'buy_adj', 'sell_adj', 'diff']]
print(top_buys.to_string(index=False))

print("\n" + "-"*80)
print("TOP 20 SELL SIGNALS")
print("-"*80)
top_sells = df_results.nlargest(20, 'sell_adj')[['symbol', 'horizon', 'sell_adj', 'buy_adj', 'diff']]
print(top_sells.to_string(index=False))

print("\n" + "-"*80)
print("SIGNALS BY INSTRUMENT TYPE")
print("-"*80)
min_buy_thresh = config.get('MIN_ACCEPTED_BUY', config['MIN_ACCEPTED'])
min_sell_thresh = config.get('MIN_ACCEPTED_SELL', config['MIN_ACCEPTED'])
type_summary = df_results.groupby('type').agg({
    'buy_adj': ['mean', 'max', lambda x: (x > min_buy_thresh).sum()],
    'sell_adj': ['mean', 'max', lambda x: (x > min_sell_thresh).sum()],
}).round(4)
type_summary.columns = ['BUY_mean', 'BUY_max', 'BUY_count', 'SELL_mean', 'SELL_max', 'SELL_count']
print(type_summary)

print("\n" + "-"*80)
print("SIGNALS BY HORIZON")
print("-"*80)
horizon_summary = df_results.groupby('horizon').agg({
    'buy_adj': ['mean', lambda x: (x > min_buy_thresh).sum()],
    'sell_adj': ['mean', lambda x: (x > min_sell_thresh).sum()],
}).round(4)
horizon_summary.columns = ['BUY_mean', 'BUY_count', 'SELL_mean', 'SELL_count']
print(horizon_summary)

# Distribution histogram
print("\n" + "-"*80)
print("PROBABILITY DISTRIBUTION (histogram)")
print("-"*80)

def print_histogram(values, label, bins=20):
    hist, bin_edges = np.histogram(values, bins=bins, range=(0, 0.3))
    print(f"\n{label}:")
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / hist.max() * 50) if hist.max() > 0 else ""
        print(f"  {bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}: {hist[i]:4d} {bar}")

print_histogram(df_results['buy_adj'].values, "BUY adjusted probabilities")
print_histogram(df_results['sell_adj'].values, "SELL adjusted probabilities")

# Check if there's systematic bias
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

buy_mean = df_results['buy_adj'].mean()
sell_mean = df_results['sell_adj'].mean()
ratio = buy_mean / sell_mean if sell_mean > 0 else float('inf')

if ratio > 2.0:
    print("⚠️  STRONG UPWARD BIAS DETECTED")
    print(f"   BUY probabilities are {ratio:.2f}x higher than SELL on average")
    print("   Possible causes:")
    print("   - Model trained primarily on bull market data")
    print("   - Feature engineering favors upside detection")
    print("   - Target labeling imbalance")
elif ratio < 0.5:
    print("⚠️  STRONG DOWNWARD BIAS DETECTED")
    print(f"   SELL probabilities are {1/ratio:.2f}x higher than BUY on average")
elif 0.8 < ratio < 1.2:
    print("✅ MODEL APPEARS BALANCED")
    print(f"   BUY/SELL ratio: {ratio:.2f}")
else:
    print("ℹ️  MODERATE DIRECTIONAL BIAS")
    print(f"   BUY/SELL ratio: {ratio:.2f}")

if len(sell_signals) == 0 and len(buy_signals) > 0:
    print("\n⚠️  CRITICAL: No SELL signals above threshold!")
    print(f"   Consider lowering MIN_ACCEPTED_SELL below {config.get('MIN_ACCEPTED_SELL', config['MIN_ACCEPTED'])}")
    print(f"   Max SELL adj_prob found: {df_results['sell_adj'].max():.4f}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if ratio > 1.5:
    print("1. Review training data for temporal bias")
    print("2. Consider separate thresholds: MIN_ACCEPTED_BUY and MIN_ACCEPTED_SELL")
    print("3. Analyze feature importances for asymmetry")
    print(f"4. Test with lower threshold for SELL: MIN_ACCEPTED = {sell_mean:.3f}")
elif len(sell_signals) < 5 and len(buy_signals) > 20:
    print("1. SELL signals are too weak to clear threshold")
    print(f"2. Temporary fix: Lower MIN_ACCEPTED to {max(0.05, sell_mean):.3f}")
    print("3. Long-term: Retrain model with balanced up/down examples")
else:
    print("Model predictions appear reasonable for current market conditions")
    print("The BUY bias may be reflecting genuine market bullishness")

print("\n" + "="*80)

# Export detailed results
output_path = PROJECT_ROOT / "outputs" / "model_bias_analysis.csv"
df_results.to_csv(output_path, index=False)
print(f"\nDetailed results saved to: {output_path}")
