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

## **Available Features (Indicators)**

`close, YesterdayClose, YesterdayOpenLogR, YesterdayHighLogR, YesterdayLowLogR, YesterdayVolumeLogR, YesterdayCloseLogR, MA10, MA20, MA30, DayOfWeek, DayOfMonth, MonthNumber, EMA10, EMA30, RSI, MACD, MACD_Signal, BollingerUpper, BollingerLower, Volatility_10, Volatility_20, Volatility_30, OBV, ZScore, insider_shares, insider_amount, insider_buy_flag, sentiment, num_articles, overnight_gap, abnormal_vol, volatility_5d, volatility_20d, momentum_5d, momentum_20d, skew_5d, intraday_range, sentiment_change`

Each is automatically merged, cleaned, and normalized during preprocessing.

---

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
### 2. Training & Generating Predictions

The *forecast.ipynb* file is used to train the model, based on the list of processed stocks in */TrainingData/indicators_data/processed/stocksData/*. There is a sample of limited stocks already included in the package, however, more data could be added. 

### 3. Creating Historical Datasets (Optional)

If you want to generate your own datasets from scratch:

#### Step 1: Configure AlphaVantage API Key

The `downloader.py` script uses **AlphaVantage** as the primary data source for historical stock prices and market data. To access this data, you'll need a free API key:

1. Sign up for a free API key at [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Open `TrainingData/downloader.py` in a text editor
3. Locate the API key configuration section near the top of the file
4. Replace the placeholder with your API key:
   ```python
   ALPHAVANTAGE_API_KEY = "your_api_key_here"
   ```
5. Save the file

**Note:** AlphaVantage's free tier has rate limits (typically 5 API calls per minute and 500 calls per day). The downloader script includes built-in rate limiting to respect these constraints.

#### Step 2: Run `TrainingData/downloader.py`

```bash
python TrainingData/downloader.py
```

This script will:
- Download historical price data from AlphaVantage
- Fetch sentiment data and insider trading information from SEC EDGAR and other sources
- Store raw data in `/TrainingData/indicators_data/raw/`

#### Step 3: Run `TrainingData/processor.py`

```bash
python TrainingData/processor.py
```

This script will:
- Clean and merge raw data
- Compute all technical indicators
- Output final processed CSVs to: `/TrainingData/indicators_data/processed/stocksData/`

---

![Backtest Example](images/image1.PNG)