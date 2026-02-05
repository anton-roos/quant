# AI Market Predictor

A deep learning pipeline for forecasting short-term market movements in **Forex, Indices, Commodities, and Crypto** using price indicators, Monte Carlo dropout uncertainty estimation, and walk-forward validation.

# Overview

This project builds an end-to-end machine learning pipeline that:

- Loads and processes market data for 65+ trading instruments from MT5 Bridge
- Supports Forex pairs, major indices, commodities, and cryptocurrencies
- Generates normalized training windows
- Trains a Conv1D + LSTM model with Monte Carlo dropout
- Uses uncertainty to determine confidence in predictions
- Outputs buy/hold forecasts to the outputs/forecasts/ folder

The design emphasizes realistic backtesting, uncertainty-aware predictions, and ease of extension — ideal for research, education, or algorithmic-trading experimentation with multiple asset classes.
```
├── src/
│   ├── inference/              # Forecasting & backtesting
│   │   ├── run_forecast.py     # Main forecasting script
│   │   └── run_backtest.py     # Backtesting script
│   ├── data/                   # Data processing pipeline
│   │   ├── processor.py        # Main feature pipeline
│   │   ├── downloader.py       # Data download orchestrator
│   │   ├── features/           # Feature generation scripts
│   │   │   └── mt5_bridge_downloader.py
│   │   └── indicators_data/
│   │       ├── raw/            # Raw data (forex, indices, commodities, crypto)
│   │       └── processed/      # Processed features
│   ├── models/                 # Model definitions
│   ├── utils/                  # Shared utilities
│   └── mt5_bridge/             # MT5 API integration
│
├── config/
│   ├── symbols.json            # 65 tradeable instruments
│   └── config.json             # Runtime configuration
│
├── outputs/
│   ├── forecasts/              # Model forecast CSV outputs
│   ├── videos/                 # Backtest visualizations
│   └── models/                 # Saved model checkpoints
│
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Test suite
└── README.md
```

## **Supported Markets**

**65+ Trading Instruments across 4 asset classes:**

- 🪙 **Forex (44 pairs)**: EURUSD, GBPUSD, USDJPY, AUDUSD, and 40 more
- 📊 **Indices (10)**: S&P 500, Nasdaq 100, DAX 30, FTSE 100, Nikkei 225, etc.
- ⛏️ **Commodities (5)**: Gold, Silver, Crude Oil (Brent & WTI), Natural Gas
- ₿ **Crypto (4)**: Bitcoin, Ethereum, Litecoin, Bitcoin Cash

All data is sourced from **MT5 Bridge API** which connects to MetaTrader 5.

## **Available Features (Indicators)**

Technical indicators computed for each instrument:

`close, YesterdayClose, YesterdayOpenLogR, YesterdayHighLogR, YesterdayLowLogR, YesterdayVolumeLogR, YesterdayCloseLogR, MA10, MA20, MA30, DayOfWeek, DayOfMonth, MonthNumber, EMA10, EMA30, RSI, MACD, MACD_Signal, BollingerUpper, BollingerLower, Volatility_10, Volatility_20, Volatility_30, OBV, ZScore, overnight_gap, abnormal_vol, volatility_5d, volatility_20d, momentum_5d, momentum_20d, skew_5d, intraday_range`

Each is automatically computed, cleaned, and normalized during preprocessing.

---

## Key Features

| Feature | Description |
|----------|-------------|
| **Conv1D + LSTM Architecture** | Learns short- and long-term dependencies in market data |
| **Monte Carlo Dropout** | Produces uncertainty estimates for each forecast |
| **Walk-Forward Validation** | Prevents data leakage, simulates real-time trading |
| **Multi-Asset Support** | Forex, Indices, Commodities, and Crypto |
| **MT5 Integration** | Direct connection to MetaTrader 5 via REST API |
| **Batch Data Generator** | Loads multiple instruments efficiently from cache |
| **Forecast Confidence Threshold** | Trades only when confidence > threshold (default 0.7) |

## Getting Started

### 1. Prerequisites

- Python 3.9+
- MetaTrader 5 terminal (for live data)
- MT5 Bridge API running (see `src/mt5_bridge/README.md`)

### 2. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Setup MT5 Bridge

Start the MT5 Bridge API server:

```bash
cd src/mt5_bridge
python -m uvicorn src.main:app --host 127.0.0.1 --port 8787
```

See [src/mt5_bridge/README.md](src/mt5_bridge/README.md) for detailed setup instructions.

### 4. Download Market Data

Download historical data for all 65 instruments:

```bash
python src/data/downloader.py
```

This will fetch 2 years of daily data for:
- 44 Forex pairs
- 10 Major indices
- 5 Commodities
- 4 Cryptocurrencies

### 5. Process Data

Process raw data and compute technical indicators:

```bash
python src/data/processor.py
```

### 6. Run Forecasts

Generate predictions for all instruments:

```bash
python src/inference/run_forecast.py
```

### 7. Run Backtests

Visualize and analyze backtest performance:

```bash
python src/inference/run_backtest.py
```

## Customizing Symbols

Edit `config/symbols.json` to add/remove trading instruments. The file is organized by asset type (Forex, Index, Commodity, Crypto) and includes MetaTrader 5 symbol names and paths.

## Features & Indicators

All technical indicators are computed automatically during data processing:

- **Price Action**: Moving averages (MA10, MA20, MA30), EMAs (10, 30)
- **Momentum**: RSI, MACD, Momentum (5d, 20d)
- **Volatility**: Bollinger Bands, Realized volatility (5d, 20d, 30d), Intraday range
- **Volume**: OBV (On-Balance Volume), Abnormal volume z-score
- **Statistical**: Z-Score, Skewness (5d)
- **Temporal**: Day of week, Day of month, Month number

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.