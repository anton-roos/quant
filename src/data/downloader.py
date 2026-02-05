"""
Download historical data for tradeable instruments via MT5 Bridge.

Supports:
- Forex pairs (EURUSD, GBPUSD, etc.)
- Indices (S&P 500, DAX 30, FTSE 100, etc.)
- Commodities (Gold, Silver, Oil, etc.)
- Crypto (Bitcoin, Ethereum, etc.)

All data is sourced from MT5 Bridge API.
"""

import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("📊 MARKET DATA DOWNLOADER - MT5 Bridge")
print("=" * 70)

print("\n🪙 Downloading: Forex, Indices, Commodities, and Crypto...")
print("   Source: MT5 Bridge API (127.0.0.1:8787)")
print("   Symbols: 65 instruments from config/symbols.json\n")

# Update paths based on new structure
script_dir = Path(__file__).parent
mt5_downloader = script_dir / "features" / "mt5_bridge_downloader.py"

# Run MT5 Bridge downloader for all instruments
result = subprocess.run([sys.executable, str(mt5_downloader)])

if result.returncode != 0:
    print("\n⚠️  MT5 Bridge downloader failed!")
    print("   Make sure:")
    print("   1. MT5 Bridge is running: cd src/mt5_bridge && python -m uvicorn src.main:app --host 127.0.0.1 --port 8787")
    print("   2. MT5 terminal is open and logged in")
    print("   3. config/symbols.json exists with your trading symbols")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL DATA DOWNLOADS COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("  1. Run processor: python src/data/processor.py")
print("  2. Train model: python src/inference/run_forecast.py")