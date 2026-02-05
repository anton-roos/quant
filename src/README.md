# Source Code

Core application code for the LSTM AI Stock Predictor.

## Directories

- **models/** - Model training and architecture
- **inference/** - Forecasting and backtesting scripts  
- **data/** - Data processing pipeline and features
- **utils/** - Shared utilities and helpers
- **mt5_bridge/** - MetaTrader 5 trading integration API

## Quick Start

```bash
# Run forecasts
python src/inference/run_forecast.py

# Run backtests
python src/inference/run_backtest.py

# Start MT5 API server
cd src/mt5_bridge && python -m uvicorn src.main:app --host 127.0.0.1 --port 8787

# Process data
python src/data/processor.py
```

See individual directories' README files for more details.
