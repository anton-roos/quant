"""
The purpose of this script is to download the stock and market data
for training the model.
"""

import subprocess
import sys

# Option to use MT5 Bridge instead of Alpha Vantage
USE_MT5_BRIDGE = True  # Set False to use AlphaVantage instead

print("=" * 60)
print("STOCK & MARKET DATA DOWNLOADER OPTIONS")
print("=" * 60)

if USE_MT5_BRIDGE:
    print("\n🪙 Using MT5 Bridge (Forex & Commodities)...")
    print("   Downloading: EURUSD, GBPUSD, XAUUSD, XTIUSD, etc.\n")
    
    # Run MT5 Bridge downloader for forex/commodities
    result = subprocess.run(["python", "training_data/featuresPy/mt5_bridge_downloader.py"])
    
    if result.returncode != 0:
        print("\n⚠️  MT5 Bridge downloader failed!")
        print("   Make sure MT5 Bridge is running on 127.0.0.1:8787")
        sys.exit(1)
else:
    print("\n📈 Using Alpha Vantage (Stocks)...")
    print("   Downloading: Stocks from stockList.csv\n")
    
    # Run the stockScrapper.py script to download stock data for:
    # Close, Open, High, Low, Volume
    subprocess.run(["python", "training_data/featuresPy/stockScrapper.py"])

# Run the markets.py script to download market data for SPY and VIX
print("\n📊 Downloading market indices (SPY, VIX)...")
subprocess.run(["python", "training_data/featuresPy/markets.py"])

# Run the insiderbuying.py script to download insider buying data
print("\n👤 Downloading insider buying data...")
subprocess.run(["python", "training_data/featuresPy/insiderbuying.py"])

# Run the sentiment.py script to download sentiment data
print("\n💭 Downloading sentiment data...")
subprocess.run(["python", "training_data/featuresPy/sentiment.py"])

print("\n" + "=" * 60)
print("✅ ALL DATA DOWNLOADS COMPLETE")
print("=" * 60)