"""
Process raw market data from indicators_data/raw folder
and save to indicators_data/processed folder.

Supports: Forex pairs, Indices, Commodities, and Crypto.

Features include:
  - Basic OHLCV-derived (log returns, MAs, EMA, RSI, MACD, Bollinger, etc.)
  - Cross-asset / inter-market features (Gold, DXY-proxy, S&P500 returns)
  - Market regime detection (rolling Sharpe, volatility regime, trend regime)
  - Calendar & seasonality features
"""

import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Suppress harmless divide-by-zero warnings from log operations on zero volume
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')

# Use paths relative to this file's location
SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "indicators_data" / "raw"
PROCESSED_DIR = SCRIPT_DIR / "indicators_data" / "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================================
# CROSS-ASSET REFERENCE SERIES
# ============================================================================
# These are loaded once and merged into every instrument for inter-market context.
# We use date-aligned daily close returns so they work even if date ranges differ.

_cross_asset_cache = {}  # symbol -> DataFrame with 'date' + return cols

# Set of MT5 symbol names that map to cross-asset references (for refresh ordering)
CROSS_ASSET_MT5_NAMES = {"EURUSD", "Gold", "SP_500", "Bitcoin (BTCUSD)", "Nymex_Light_Crude"}

CROSS_ASSET_REFS = {
    # Gold as risk-off proxy
    "Gold":         ("commodities", "Gold_daily"),
    # US Dollar proxy via EURUSD inverse
    "DXY_proxy":    ("forex", "EURUSD_daily"),
    # US equity proxy
    "SP500":        ("indices", "SP_500_daily"),
    # Bitcoin as crypto regime proxy
    "BTC":          ("crypto", "BTCUSD_daily"),
    # Oil as macro proxy
    "Oil":          ("commodities", "Nymex_Light_Crude_daily"),
}


def invalidate_cross_asset_cache():
    """Clear the cross-asset cache so it's reloaded from fresh files on next use."""
    global _cross_asset_cache
    _cross_asset_cache = {}


def _load_cross_asset_series():
    """Load reference instruments and compute their rolling features."""
    global _cross_asset_cache
    if _cross_asset_cache:
        return _cross_asset_cache

    for label, (category, stem) in CROSS_ASSET_REFS.items():
        raw_path = RAW_DIR / category / f"{stem}.csv"
        if not raw_path.exists():
            print(f"[WARN] Cross-asset ref not found: {raw_path}")
            continue

        df = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            continue

        prefix = f"XA_{label}"  # XA = cross-asset
        out = pd.DataFrame({"date": df["date"]})

        # Daily log return
        out[f"{prefix}_ret_1d"] = np.log(df["close"] / df["close"].shift(1))
        # 5-day momentum
        out[f"{prefix}_mom_5d"] = df["close"] / df["close"].shift(5) - 1
        # 20-day momentum
        out[f"{prefix}_mom_20d"] = df["close"] / df["close"].shift(20) - 1
        # 20-day rolling volatility
        out[f"{prefix}_vol_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

        # For DXY proxy, invert (EURUSD up = USD down)
        if label == "DXY_proxy":
            for col in out.columns:
                if col != "date":
                    out[col] = -out[col]

        out.dropna(inplace=True)
        _cross_asset_cache[label] = out

    return _cross_asset_cache


def add_cross_asset_features(df, cross_assets):
    """Merge cross-asset features onto instrument dataframe by date.
    Uses forward-fill for date mismatches (holidays, weekends differ by market)."""
    for label, ref_df in cross_assets.items():
        df = pd.merge(df, ref_df, on="date", how="left")
    # Forward-fill then back-fill cross-asset columns (different trading calendars)
    xa_cols = [c for c in df.columns if c.startswith("XA_")]
    df[xa_cols] = df[xa_cols].ffill().bfill().fillna(0)
    return df


# ============================================================================
# REGIME DETECTION FEATURES
# ============================================================================

def add_regime_features(df):
    """Add market regime indicators derived from the instrument's own price."""

    ret = df["close"].pct_change()

    # --- Volatility regime (rolling 20d vol, bucketed) ---
    vol_20 = ret.rolling(20).std() * np.sqrt(252)
    vol_median = vol_20.expanding().median()
    df["regime_high_vol"] = (vol_20 > vol_median).astype(float)

    # --- Trend regime (EMA crossover) ---
    ema_fast = df["close"].ewm(span=10, adjust=False).mean()
    ema_slow = df["close"].ewm(span=50, adjust=False).mean()
    df["regime_trend"] = np.where(ema_fast > ema_slow, 1.0, -1.0)

    # --- Rolling Sharpe (20d and 60d) ---
    df["rolling_sharpe_20d"] = (ret.rolling(20).mean() / ret.rolling(20).std()) * np.sqrt(252)
    df["rolling_sharpe_60d"] = (ret.rolling(60).mean() / ret.rolling(60).std()) * np.sqrt(252)

    # --- Mean reversion indicator (distance from 50d mean, normalized) ---
    ma50 = df["close"].rolling(50).mean()
    std50 = df["close"].rolling(50).std()
    df["mean_reversion_zscore"] = (df["close"] - ma50) / std50

    # --- Autocorrelation of returns (5d lag) ---
    df["ret_autocorr_5d"] = ret.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)

    # --- ATR (Average True Range, normalized) ---
    high = df.get("high")
    low = df.get("low")
    if high is not None and low is not None:
        tr = pd.DataFrame({
            "hl": high - low,
            "hc": (high - df["close"].shift(1)).abs(),
            "lc": (low - df["close"].shift(1)).abs()
        }).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()
        df["ATR_norm"] = df["ATR_14"] / df["close"]
    else:
        df["ATR_14"] = 0
        df["ATR_norm"] = 0

    return df

def process_file(csv_path, output_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["YesterdayClose"] = df["close"].shift(1)
    df["YesterdayOpenLogR"]  = np.log(df["open"] / df["open"].shift(1))
    df["YesterdayHighLogR"]  = np.log(df["high"] / df["high"].shift(1))
    df["YesterdayLowLogR"]   = np.log(df["low"]  / df["low"].shift(1))
    
    # Handle missing volume (common for forex/commodities/crypto)
    has_volume = "volume" in df.columns and df["volume"].notna().any() and (df["volume"] != 0).any()
    if not has_volume:
        df["volume"] = 0  # Set to 0 if missing
    
    if has_volume:
        df["YesterdayVolumeLogR"] = np.log(df["volume"] / df["volume"].shift(1))
    else:
        df["YesterdayVolumeLogR"] = 0
    df["YesterdayCloseLogR"] = np.log(df["close"] / df["YesterdayClose"])

    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA30"] = df["close"].rolling(window=30).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()

    df["DayOfWeek"] = df["date"].dt.weekday         # 0 = Monday, 6 = Sunday
    df["DayOfMonth"] = df["date"].dt.day            # 1 to 31
    df["MonthNumber"] = df["date"].dt.month         # 1 = January, 12 = December
    # Cyclical encoding (so Jan~Dec and Mon~Fri are close in feature space)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 5)
    df["Month_sin"] = np.sin(2 * np.pi * df["MonthNumber"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["MonthNumber"] / 12)

    df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["close"].ewm(span=30, adjust=False).mean()

    #Relative strength index (RSI) calculation — Wilder's exponential smoothing
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    #Moving average convergence divergence (MACD) calculation
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    #Bollinger Bands - Volitility indicator
    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["BollingerUpper"] = ma20 + 2 * std20
    df["BollingerLower"] = ma20 - 2 * std20
    df["BollingerWidth"] = (df["BollingerUpper"] - df["BollingerLower"]) / ma20
    df["BollingerPctB"] = (df["close"] - df["BollingerLower"]) / (df["BollingerUpper"] - df["BollingerLower"])

    #Rolling Volatility
    df["Volatility_10"] = df["close"].pct_change().rolling(window=10).std()
    df["Volatility_20"] = df["close"].pct_change().rolling(window=20).std()
    df["Volatility_30"] = df["close"].pct_change().rolling(window=30).std()

    #On-Balance Volume (OBV) - Volume indicator
    # Only compute if volume data is present (not zeros)
    if has_volume:
        df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    else:
        df["OBV"] = 0  # Set to 0 if no volume data

    #Z-score of close
    mean = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["ZScore"] = (df["close"] - mean) / std
    
    #Overnight gap
    # Overnight gap % (predicts t+1 move)
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Abnormal volume z-score (handle when volume is 0 for forex/indices)
    if has_volume:
        rolling_vol = df['volume'].rolling(20)
        df['abnormal_vol'] = (df['volume'] - rolling_vol.mean()) / rolling_vol.std()
    else:
        df['abnormal_vol'] = 0
    
    #Short term realized volatility
    df['volatility_5d'] = df['close'].pct_change().rolling(5).std() * np.sqrt(252)
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    #Momentum 
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    #Skewness
    df['skew_5d'] = df['close'].pct_change().rolling(5).skew()
    #Intraday change
    df['intraday_range'] = (df['high'] - df['low']) / df['close']

    # ==== REGIME DETECTION FEATURES ====
    df = add_regime_features(df)

    # ==== CROSS-ASSET / INTER-MARKET FEATURES ====
    cross_assets = _load_cross_asset_series()
    df = add_cross_asset_features(df, cross_assets)

    # ==== ASSET CLASS ONE-HOT ENCODING ====
    # Infer asset class from the parent directory of the raw CSV
    category_folder = Path(csv_path).parent.name.lower()
    category_map = {"forex": "forex", "indices": "index", "commodities": "commodity", "crypto": "crypto"}
    asset_class = category_map.get(category_folder, "other")
    for cls in ["forex", "index", "commodity", "crypto"]:
        df[f"asset_{cls}"] = 1.0 if asset_class == cls else 0.0

    df.dropna(inplace=True)
    
    # Replace any remaining inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Drop raw OHLCV columns (keep processed features)
    cols_to_drop = ['open', 'high', 'low', 'volume']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"[OK] {Path(csv_path).stem} -> {Path(output_path).name} ({len(df)} rows, {len(df.columns)} features)")

def check_missing_today():
    today = pd.Timestamp(datetime.today().date())
    print("\n[INFO] Checking which files are missing today's data...\n")
    missing = []

    # Check all subfolders for market data
    for subfolder in ["forex", "indices", "commodities", "crypto"]:
        processed_subdir = PROCESSED_DIR / subfolder
        if not processed_subdir.exists():
            continue
        for file in processed_subdir.iterdir():
            if not file.name.endswith("_processed.csv"):
                continue
            try:
                df = pd.read_csv(file, parse_dates=["date"])
                if df.empty:
                    missing.append((file.name, "EMPTY"))
                    continue
                last_date = df["date"].max()
                if last_date != today:
                    missing.append((file.name, last_date.date()))
            except Exception as e:
                print(f"[ERROR] Failed to check {file}: {e}")

    if missing:
        print("❌ The following files are missing today's data:")
        for filename, last_date in missing:
            print(f" - {filename}: Last date = {last_date}")
    else:
        print("✅ All files contain today's data.")

def main():
    print(f"Processing market data from: {RAW_DIR}")
    print(f"Output directory: {PROCESSED_DIR}\n")
    
    # Process market data by category
    for subfolder in ["forex", "indices", "commodities", "crypto"]:
        raw_subdir = RAW_DIR / subfolder
        
        # Skip if directory doesn't exist
        if not raw_subdir.exists():
            print(f"[INFO] Skipping {subfolder} - directory not found")
            continue
        
        processed_subdir = PROCESSED_DIR / subfolder
        processed_subdir.mkdir(parents=True, exist_ok=True)

        for file in raw_subdir.iterdir():
            if file.name.startswith("._"):
                print(f"[Skipping] macOS metadata: {file.name}")
                continue
            if file.suffix == ".csv":
                processed_file_path = processed_subdir / f"{file.stem}_processed.csv"
                process_file(str(file), str(processed_file_path))

if __name__ == "__main__":
    main()
    check_missing_today()