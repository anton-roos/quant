#!/usr/bin/env python3
"""
Launch the LSTM trading bot.

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
