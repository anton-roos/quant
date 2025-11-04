
# AI Stock Predictor

A deep learning pipeline for forecasting short-term stock market movements using price indicators, Monte Carlo dropout uncertainty estimation, and walk-forward validation.

# Overview

This project builds an end-to-end machine learning pipeline that:

- Loads and processes stock indicator data from CSV files
- Generates normalized training windows
- Trains a Conv1D + LSTM model with Monte Carlo dropout
- Uses uncertainty to determine confidence in predictions
- Outputs buy/hold forecasts to the /forecasts folder

The design emphasizes realistic backtesting, uncertainty-aware predictions, and ease of extension — ideal for research, education, or algorithmic-trading experimentation.
```
├── forecasts/                  # Model forecast CSV outputs
├── cache/                      # Cached preprocessed numpy arrays
├── TrainingData/
│   ├── indicators_data/
│   │   ├── raw/                # Raw scraped data (price, sentiment, insider)
│   │   └── processed/
│   │       ├── SPY-VIX/        # Market indicators
│   │       └── stocksData/     # Stock CSVs (one per ticker)
│   ├── featuresPy/             # Feature-generation scripts
│   ├── processor.py            # Main feature pipeline
│   └── downloader.py           # Data download helpers
├── forecast.ipynb              # Jupyter notebook for running forecasts
├── forecasting_backtest_Predictor.py  # Main training & backtest script
├── forecasting_backtest_Predictor_v2.py # Newer version with attention/uncertainty
├── videos/                     # Rendered content for visualization or YouTube
└── README.md
```

Key Features
FeatureDescriptionConv1D + LSTM ArchitectureLearns short- and long-term dependencies in stock dataMonte Carlo DropoutProduces uncertainty estimates for each forecastWalk-Forward ValidationPrevents data leakage, simulates real-time tradingRegime and Volatility AwarenessDetects market conditions for more robust signalsBatch Data GeneratorLoads multiple stock datasets efficiently from cacheForecast Confidence ThresholdTrades only when confidence > threshold (default 0.7)

## Key Features

| Feature | Description |
|----------|-------------|
| **Conv1D + LSTM Architecture** | Learns short- and long-term dependencies in stock data |
| **Monte Carlo Dropout** | Produces uncertainty estimates for each forecast |
| **Walk-Forward Validation** | Prevents data leakage, simulates real-time trading |
| **Regime and Volatility Awareness** | Detects market conditions for more robust signals |
| **Batch Data Generator** | Loads multiple stock datasets efficiently from cache |
| **Forecast Confidence Threshold** | Trades only when confidence > threshold (default 0.7) |

## Getting Started

### 1. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```
Work in progress... 