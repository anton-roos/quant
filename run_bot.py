#!/usr/bin/env python3
"""
Launch the LSTM trading bot (live trading).

Pre-requisites:
    1. python run_data_pipeline.py   – train model & generate forecasts
    2. python run_bridge.py          – start the MT5 bridge server

Usage (from project root):
    python run_bot.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.trading.bot import main

if __name__ == "__main__":
    main()
