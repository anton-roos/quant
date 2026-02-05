# Project Structure

This document describes the organization of the AI Market Predictor project.

## Directory Layout

```
LSTM_AI_Stock_Predictor/
│
├── src/                          # Main source code
│   ├── __init__.py
│   ├── models/                   # Model definitions
│   │   └── __init__.py
│   ├── inference/                # Inference and forecasting
│   │   ├── __init__.py
│   │   ├── run_forecast.py       # Main LSTM forecasting pipeline
│   │   └── run_backtest.py       # Walk-forward backtesting
│   ├── data/                     # Data pipeline and processing
│   │   ├── __init__.py
│   │   ├── processor.py          # Technical indicator computation
│   │   ├── downloader.py         # Data download orchestrator
│   │   ├── symbols.csv           # Symbol list (65 instruments)
│   │   ├── indicators_list.txt   # Feature documentation
│   │   ├── features/             # Feature download modules
│   │   │   ├── __init__.py
│   │   │   └── mt5_bridge_downloader.py
│   │   └── indicators_data/      # Downloaded & processed data
│   │       ├── raw/              # Raw OHLCV data by category
│   │       │   ├── forex/
│   │       │   ├── indices/
│   │       │   ├── commodities/
│   │       │   └── crypto/
│   │       └── processed/        # Processed features by category
│   │           ├── forex/
│   │           ├── indices/
│   │           ├── commodities/
│   │           └── crypto/
│   ├── utils/                    # Shared utilities
│   │   └── __init__.py
│   └── mt5_bridge/               # MT5 Trading Platform Integration
│       ├── src/
│       │   ├── main.py           # FastAPI application
│       │   ├── mt5_client.py     # MT5 connection & trading
│       │   ├── models.py         # Data models
│       │   ├── storage.py        # Persistence layer
│       │   └── indicators.py     # Technical indicators
│       ├── tests/
│       ├── logs/
│       ├── requirements.txt
│       ├── README.md
│       └── API_REFERENCE.md
│
├── config/                       # Configuration
│   ├── config.json               # Runtime config (MT5 Bridge settings)
│   ├── config.example.json       # Template config
│   └── symbols.json              # 65 tradeable instruments
│
├── outputs/                      # Generated outputs
│   ├── forecasts/                # Model forecast CSVs
│   ├── videos/                   # Backtest visualizations
│   ├── models/                   # Saved model checkpoints
│   └── cache/                    # Preprocessed data cache
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   └── test_risk.py
│
├── notebooks/                    # Jupyter notebooks
├── experiments/                  # Experimental code
├── docs/                         # Documentation
│
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## Pipeline Overview

1. **Download**: `python src/data/downloader.py` — fetches OHLCV data from MT5 Bridge
2. **Process**: `python src/data/processor.py` — computes 33 technical indicators
3. **Forecast**: `python src/inference/run_forecast.py` — trains Conv1D+LSTM model, generates predictions
4. **Backtest**: `python src/inference/run_backtest.py` — evaluates strategy vs random baseline

