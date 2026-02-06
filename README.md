# AI Market Predictor

A deep learning pipeline that forecasts short-term market movements in **Forex, Indices, Commodities, and Crypto** using technical indicators, Monte Carlo dropout uncertainty estimation, and walk-forward validation.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Markets](#supported-markets)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Step 1 — Start the MT5 Bridge](#step-1--start-the-mt5-bridge)
6. [Step 2 — Download Market Data](#step-2--download-market-data)
7. [Step 3 — Process Data (Compute Indicators)](#step-3--process-data-compute-indicators)
8. [Step 4 — Train Model & Generate Forecasts](#step-4--train-model--generate-forecasts)
9. [Step 5 — Run Backtests](#step-5--run-backtests)
10. [Project Structure](#project-structure)
11. [Configuration Reference](#configuration-reference)
12. [Customizing Symbols](#customizing-symbols)
13. [Technical Indicators](#technical-indicators)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline follows four stages executed in order:

```
Download (MT5 Bridge) → Process (indicators) → Forecast (LSTM) → Backtest (walk-forward)
```

| Feature | Description |
|---------|-------------|
| **Conv1D + LSTM Architecture** | Learns short- and long-term dependencies in market data |
| **Monte Carlo Dropout** | Produces uncertainty estimates for every forecast |
| **Walk-Forward Validation** | Prevents data leakage; simulates real-time trading |
| **Multi-Asset Support** | 65 instruments across Forex, Indices, Commodities, Crypto |
| **MT5 Bridge** | REST API bridge to MetaTrader 5 (FastAPI / uvicorn) |
| **Efficient Data Generator** | Loads multiple instruments from NumPy cache in batches |
| **Confidence Threshold** | Only trades when model confidence > 0.7 (configurable) |

---

## Supported Markets

**65 instruments across 4 asset classes:**

| Class | Count | Examples |
|-------|-------|----------|
| Forex | 44 | EURUSD, GBPUSD, USDJPY, AUDUSD, NZDJPY … |
| Indices | 10 | S&P 500, Nasdaq 100, DAX 30, FTSE 100, Nikkei 225 |
| Commodities | 5 | Gold, Silver, Crude Oil (Brent & WTI), Natural Gas |
| Crypto | 4 | Bitcoin, Ethereum, Litecoin, Bitcoin Cash |

The full list lives in `config/symbols.json`.

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| **Python** | 3.10+ | 3.13 tested on Windows |
| **MetaTrader 5** | Latest | Must be installed, open, and logged in to your broker |
| **pip** | 23+ | Comes with Python |
| **FFmpeg** *(optional)* | Any | Only needed to export backtest videos (`.mp4`) |

> **OS:** This guide uses **PowerShell on Windows**. Adjust commands for macOS/Linux (use `source .venv/bin/activate` instead of `.\\.venv\\Scripts\\Activate.ps1`, etc.).

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/<your-user>/LSTM_AI_Stock_Predictor.git
cd LSTM_AI_Stock_Predictor
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install main dependencies

```powershell
pip install -r requirements.txt
```

This installs: TensorFlow, Keras, scikit-learn, NumPy, pandas, SciPy, Matplotlib, Pillow, requests, python-dotenv, tqdm.

### 4. Install MT5 Bridge dependencies

The bridge has its own requirements (FastAPI, uvicorn, MetaTrader5 package, etc.):

```powershell
pip install -r src/mt5_bridge/requirements.txt
```

### 5. Create your config file

```powershell
Copy-Item config/config.example.json config/config.json
```

The default config points to the bridge at `127.0.0.1:8787` — no changes needed unless you run the bridge on a different host/port.

```json
{
    "MT5_BRIDGE_HOST": "127.0.0.1",
    "MT5_BRIDGE_PORT": 8787
}
```

### 6. Set up the MT5 Bridge `.env`

```powershell
Copy-Item src/mt5_bridge/.env.example src/mt5_bridge/.env
```

Edit `src/mt5_bridge/.env` and set your API key (and optionally MT5 credentials):

```ini
API_KEY=your_strong_secret_key

# Only needed if MT5 isn't already logged-in in the terminal
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
```

If MetaTrader 5 is already open and logged in, you can leave the `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER` fields empty — the bridge will attach to the running terminal.

---

## Step 1 — Start the MT5 Bridge

> **Prerequisite:** MetaTrader 5 must be open and logged in to your broker account.

Open a **separate terminal** (this process stays running) and start the bridge:

```powershell
cd src/mt5_bridge
..\..\..\.venv\Scripts\Activate.ps1      # or wherever your venv is
python -m uvicorn src.main:app --host 127.0.0.1 --port 8787
```

You should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     MT5 initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8787
```

**Quick health check** — open a new terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health
```

You should get a JSON response with `"status": "ok"`.

> **Leave this terminal running.** All subsequent steps talk to this bridge.

---

## Step 2 — Download Market Data

In your main terminal (with `.venv` activated):

```powershell
python src/data/downloader.py
```

This calls the MT5 Bridge to download ~2 years of daily OHLCV data for all 65 instruments defined in `config/symbols.json`. Data is saved to:

```
src/data/indicators_data/raw/
    ├── forex/          # e.g. EURUSD.csv, GBPUSD.csv
    ├── indices/        # e.g. US500.csv, NAS100.csv
    ├── commodities/    # e.g. XAUUSD.csv, XTIUSD.csv
    └── crypto/         # e.g. BTCUSD.csv, ETHUSD.csv
```

**Expected runtime:** ~2–5 minutes depending on broker response time.

---

## Step 3 — Process Data (Compute Indicators)

```powershell
python src/data/processor.py
```

This reads every raw CSV, computes 33 technical indicators, and saves the result:

```
src/data/indicators_data/processed/
    ├── forex/          # e.g. EURUSD_processed.csv
    ├── indices/
    ├── commodities/
    └── crypto/
```

At the end it prints a check confirming whether each file has today's date.

---

## Step 4 — Train Model & Generate Forecasts

```powershell
python src/inference/run_forecast.py
```

**What it does (takes 10–60 min depending on GPU/CPU):**

1. Loads all processed CSVs across forex/indices/commodities/crypto subdirectories
2. Fits a `StandardScaler` on the feature columns
3. Caches preprocessed NumPy arrays (train/val/test splits) to `outputs/cache/`
4. Builds a **Conv1D → LSTM → Dense** model with Monte Carlo Dropout
5. Trains with early stopping and learning-rate reduction
6. Runs MC Dropout inference (25 forward passes) to estimate prediction uncertainty
7. Saves one forecast CSV per instrument to `outputs/forecasts/`

Each forecast CSV contains columns like:

| Column | Description |
|--------|-------------|
| `Date` | Prediction date |
| `Close` | Closing price on that date |
| `Pred_Prob_1d` | Probability of up-move over 1 day |
| `Pred_Prob_Std_1d` | Uncertainty (std of MC samples) |
| `Actual_LogR_1d` | Actual log-return (for backtesting) |
| … | Same for `1w`, `1m`, `6m` horizons |

**Tip:** The model checkpoint is saved to `outputs/models/` so you can re-run inference without retraining.

---

## Step 5 — Run Backtests

```powershell
python src/inference/run_backtest.py
```

This reads the forecast CSVs and simulates a probability-based trading strategy:

- Buys instruments where adjusted probability > threshold
- Compares strategy equity curve against a random baseline (N=10 runs)
- Saves a trade summary CSV and performance plot to `outputs/videos/`

---

## Project Structure

```
LSTM_AI_Stock_Predictor/
│
├── src/                              # All source code
│   ├── data/                         # Data pipeline
│   │   ├── downloader.py             # Orchestrates MT5 download
│   │   ├── processor.py              # Computes 33 technical indicators
│   │   ├── features/
│   │   │   └── mt5_bridge_downloader.py  # MT5 Bridge HTTP client
│   │   ├── indicators_data/
│   │   │   ├── raw/                  # Raw OHLCV (forex/, indices/, ...)
│   │   │   └── processed/           # Processed features
│   │   ├── symbols.csv              # Flat symbol list
│   │   └── indicators_list.txt      # Feature documentation
│   │
│   ├── inference/                    # ML pipeline
│   │   ├── run_forecast.py           # Train + predict (Conv1D + LSTM + MC Dropout)
│   │   └── run_backtest.py           # Walk-forward backtesting
│   │
│   ├── mt5_bridge/                   # FastAPI bridge to MetaTrader 5
│   │   ├── src/
│   │   │   ├── main.py              # FastAPI app (endpoints)
│   │   │   ├── mt5_client.py        # MT5 connection wrapper
│   │   │   ├── models.py            # Pydantic models
│   │   │   ├── storage.py           # Logging / persistence
│   │   │   └── indicators.py        # Server-side indicator calc
│   │   ├── tests/
│   │   ├── requirements.txt         # Bridge-specific deps
│   │   ├── .env.example             # Environment template
│   │   ├── README.md                # Bridge docs
│   │   └── API_REFERENCE.md         # Full API reference
│   │
│   ├── models/                       # (reserved for model definitions)
│   └── utils/                        # Shared utilities
│
├── config/
│   ├── config.json                   # Runtime config (host, port)
│   ├── config.example.json           # Template (committed to git)
│   └── symbols.json                  # 65 trading instruments
│
├── outputs/                          # All generated outputs
│   ├── forecasts/                    # Forecast CSVs (one per instrument)
│   ├── cache/                        # Preprocessed NumPy cache
│   ├── models/                       # Saved Keras model checkpoints
│   └── videos/                       # Backtest plots & videos
│
├── tests/                            # Test suite
├── notebooks/                        # Jupyter notebooks
├── experiments/                      # Experimental code
├── docs/                             # Extra documentation
│
├── requirements.txt                  # Python dependencies (main)
├── PROJECT_STRUCTURE.md              # Detailed structure docs
├── .gitignore
└── README.md                         # ← you are here
```

---

## Configuration Reference

### `config/config.json`

| Key | Default | Description |
|-----|---------|-------------|
| `MT5_BRIDGE_HOST` | `127.0.0.1` | Host where MT5 Bridge is running |
| `MT5_BRIDGE_PORT` | `8787` | Port for MT5 Bridge |

### `src/mt5_bridge/.env`

| Key | Default | Description |
|-----|---------|-------------|
| `API_KEY` | `change_me` | Auth token for bridge endpoints |
| `BRIDGE_HOST` | `127.0.0.1` | Bind address for bridge server |
| `BRIDGE_PORT` | `8787` | Bind port for bridge server |
| `MT5_LOGIN` | *(empty)* | MT5 account number (optional if terminal is logged in) |
| `MT5_PASSWORD` | *(empty)* | MT5 password (optional) |
| `MT5_SERVER` | *(empty)* | MT5 broker server (optional) |
| `ALLOW_LIVE_TRADING` | `false` | Enable live trade execution |

### Forecast pipeline settings (in `run_forecast.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `WINDOW_SIZE` | 90 | Number of trading days per input window |
| `MC_DROPOUT_SAMPLES` | 25 | Monte Carlo forward passes for uncertainty |
| `PROB_THRESHOLD` | 0.7 | Minimum probability to signal a buy |
| `TRAIN_VAL_FRAC` | 0.8 | Fraction of data used for train+val |

---

## Customizing Symbols

Edit `config/symbols.json` to add or remove instruments. Each entry looks like:

```json
{
  "name": "EURUSD",
  "description": "EURUSD",
  "type": "Forex",
  "path": "CFDs\\International Markets\\Currency\\PP\\EURUSD",
  "currency_base": "INT",
  "currency_profit": "INT"
}
```

- `name` — the MetaTrader 5 symbol name (must match your broker's naming)
- `type` — one of `Forex`, `Index`, `Commodity`, `Crypto` (used for folder routing)
- `path` — the full MT5 symbol path (visible in MT5 → Market Watch → right-click → Show All)

After editing, re-run from **Step 2** (download → process → forecast).

---

## Technical Indicators

All 33 features are computed automatically in `src/data/processor.py`:

| Category | Features |
|----------|----------|
| **Price** | `close`, `YesterdayClose`, `YesterdayOpenLogR`, `YesterdayHighLogR`, `YesterdayLowLogR`, `YesterdayVolumeLogR`, `YesterdayCloseLogR` |
| **Moving Averages** | `MA10`, `MA20`, `MA30`, `EMA10`, `EMA30` |
| **Momentum** | `RSI`, `MACD`, `MACD_Signal`, `momentum_5d`, `momentum_20d` |
| **Volatility** | `BollingerUpper`, `BollingerLower`, `Volatility_10`, `Volatility_20`, `Volatility_30`, `volatility_5d`, `volatility_20d`, `intraday_range` |
| **Volume** | `OBV`, `abnormal_vol` |
| **Statistical** | `ZScore`, `skew_5d`, `overnight_gap` |
| **Temporal** | `DayOfWeek`, `DayOfMonth`, `MonthNumber` |

---

## Troubleshooting

### MT5 Bridge won't start

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: MetaTrader5` | `pip install MetaTrader5` — only works on Windows |
| `MT5 initialization failed` | Make sure MetaTrader 5 is open and logged in |
| Port already in use | Change `BRIDGE_PORT` in `.env` and `config.json` |

### Downloader fails

| Symptom | Fix |
|---------|-----|
| `Connection refused` | MT5 Bridge isn't running — see [Step 1](#step-1--start-the-mt5-bridge) |
| Symbol not found | Check that the symbol name in `symbols.json` matches your broker (e.g. `US500` vs `SPX500`) |
| Empty CSVs | Broker may not have history for that symbol — check MT5 terminal manually |

### Forecast errors

| Symptom | Fix |
|---------|-----|
| `No CSV files found` | Run downloader and processor first (Steps 2–3) |
| Out-of-memory | Reduce `batch_size` in `run_forecast.py` (default 128) |
| Very slow training | Install GPU-enabled TensorFlow: `pip install tensorflow[and-cuda]` |

### General

```powershell
# Verify your virtual environment is active
python --version           # should show 3.10+
pip list | Select-String tensorflow  # should show tensorflow

# Check bridge health
Invoke-RestMethod http://127.0.0.1:8787/health

# Check if raw data was downloaded
Get-ChildItem src/data/indicators_data/raw/forex/*.csv | Measure-Object

# Check if processed data exists
Get-ChildItem src/data/indicators_data/processed/forex/*.csv | Measure-Object
```