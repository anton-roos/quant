"""
Process raw market data from indicators_data/raw folder
and save to indicators_data/processed folder.

Supports: Forex pairs, Indices, Commodities, and Crypto
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

    df["DayOfWeek"] = df["date"].dt.weekday         # 0 = Monday, 6 = Sunday
    df["DayOfMonth"] = df["date"].dt.day            # 1 to 31
    df["MonthNumber"] = df["date"].dt.month         # 1 = January, 12 = December

    df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["close"].ewm(span=30, adjust=False).mean()

    #Relative strength index (RSI) calculation
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    #Moving average convergence divergence (MACD) calculation
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    #Bollinger Bands - Volitility indicator
    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["BollingerUpper"] = ma20 + 2 * std20
    df["BollingerLower"] = ma20 - 2 * std20

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

    df.dropna(inplace=True)
    
    # Drop raw OHLCV columns (keep processed features)
    cols_to_drop = ['open', 'high', 'low', 'volume']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"[OK] {Path(csv_path).stem} -> {Path(output_path).name} ({len(df)} rows)")

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