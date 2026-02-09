"""
Download historical OHLCV data from MT5 Bridge API.

Supports:
  - Forex pairs (EURUSD, GBPUSD, etc.)
  - Indices (S&P 500, DAX 30, FTSE 100, etc.)
  - Commodities (Gold, Silver, Oil, etc.)
  - Crypto (Bitcoin, Ethereum, etc.)

Symbols are loaded from config/symbols.json

Usage:
  python src/data/features/mt5_bridge_downloader.py
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import json
from typing import List, Optional, Dict
from pathlib import Path

from src.utils.constants import sanitize_filename


def _sanitize_filename(symbol: str) -> str:
    """Backward-compatible alias — delegates to ``src.utils.constants.sanitize_filename``."""
    return sanitize_filename(symbol)

class MT5BridgeDownloader:
    """Download historical data from MT5 Bridge API."""
    
    def __init__(
        self,
        bridge_host: str = "127.0.0.1",
        bridge_port: int = 8787,
        save_folder: str = "src/data/indicators_data/raw"
    ):
        """
        Initialize MT5 Bridge downloader.
        
        Args:
            bridge_host: MT5 Bridge host (default: 127.0.0.1)
            bridge_port: MT5 Bridge port (default: 8787)
            save_folder: Where to save CSV files
        """
        self.base_url = f"http://{bridge_host}:{bridge_port}"
        self.save_folder = save_folder
    
    def health_check(self) -> bool:
        """Check if MT5 Bridge is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ MT5 Bridge is UP: {self.base_url}")
                return True
            else:
                print(f"❌ MT5 Bridge returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to MT5 Bridge at {self.base_url}")
            print(f"   Make sure the bridge is running: python -m uvicorn main:app --host 127.0.0.1 --port 8787")
            return False
        except Exception as e:
            print(f"❌ Error checking MT5 Bridge: {e}")
            return False
    
    def get_available_symbols(self, group: str = "*") -> List[str]:
        """
        Get list of available symbols in MT5.
        
        Args:
            group: Filter pattern (e.g., "*USD*", "Forex*", "*" for all)
        
        Returns:
            List of symbol names
        """
        try:
            print(f"📊 Fetching available symbols (group={group})...")
            response = requests.get(
                f"{self.base_url}/symbols",
                params={"group": group},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                symbols = data.get("symbols", [])
                print(f"   Found {len(symbols)} symbols")
                return symbols
            else:
                print(f"❌ Error fetching symbols: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"❌ Exception fetching symbols: {e}")
            return []
    
    def download_daily_data(
        self,
        symbols: List[str],
        save_folder: Optional[str] = None,
        days_back: int = 365,
    ) -> None:
        """
        Download daily (D1) candle data for multiple symbols.
        
        Args:
            symbols: List of symbols to download
            save_folder: Where to save. Defaults to self.save_folder
            days_back: How many days of history to retrieve
        """
        save_folder = save_folder or self.save_folder
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        print(f"\n📥 Downloading daily data for {len(symbols)} symbols...")
        print(f"   History: {days_back} days")
        print(f"   Output: {save_folder}\n")
        
        for i, symbol in enumerate(symbols, 1):
            self._download_symbol(symbol, save_folder, days_back)
            
            if i < len(symbols):
                percent = (i / len(symbols)) * 100
                print(f"   Progress: {percent:.1f}% ({i}/{len(symbols)})")
        
        print(f"\n✅ Download complete!")
    
    def _download_symbol(
        self,
        symbol: str,
        save_folder: str,
        days_back: int
    ) -> None:
        """Download data for a single symbol."""
        safe_name = _sanitize_filename(symbol)
        filepath = os.path.join(save_folder, f"{safe_name}_daily.csv")
        
        try:
            # Check if file exists and is up-to-date
            if os.path.exists(filepath):
                existing_data = pd.read_csv(filepath, parse_dates=["date"])
                last_date = existing_data["date"].max()
                
                # If latest date is today, skip
                if pd.to_datetime(last_date).date() >= pd.Timestamp.now().date():
                    print(f"⏭️  {safe_name:20} already up-to-date (latest: {last_date.date()})")
                    return
                
                # Fetch from the last known date (not +1 day) to avoid
                # weekend/holiday gaps causing "no data" errors. We de-dup on merge.
                from_date = pd.to_datetime(last_date).to_pydatetime().isoformat()
            else:
                # Fetch historical data
                from_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # Fetch candles from MT5 Bridge
            print(f"📥 {safe_name:20} fetching from {from_date.split('T')[0]}...", end=" ")
            
            response = requests.get(
                f"{self.base_url}/candles",
                params={
                    "symbol": symbol,
                    "timeframe": "D1",
                    "count": 10000,  # Max candles
                    "from_date": from_date
                },
                timeout=30
            )
            
            if response.status_code != 200:
                body = (response.text or "").strip().replace("\n", " ")
                body_preview = (body[:200] + "…") if len(body) > 200 else body
                suffix = f" – {body_preview}" if body_preview else ""
                print(f"❌ HTTP {response.status_code}{suffix}")
                return
            
            data = response.json()
            candles = data.get("candles", [])
            
            if not candles:
                print(f"⚠️  no data")
                return
            
            # Convert candles to DataFrame
            df = pd.DataFrame([
                {
                    "date": c["time"].split("T")[0],  # Extract date only (YYYY-MM-DD)
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["real_volume"]  # Use real_volume, not tick_volume
                }
                for c in candles
            ])
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
            
            # Merge with existing data if it exists
            if os.path.exists(filepath):
                existing = pd.read_csv(filepath)
                df = pd.concat([existing, df])
                df = df.drop_duplicates(subset=["date"]).sort_values("date")
                print(f"✅ appended {len(candles)} new candles ({len(df)} total)")
            else:
                print(f"✅ saved {len(df)} candles")
            
            # Save to CSV
            df.to_csv(filepath, index=False)
        
        except requests.exceptions.Timeout:
            print(f"⏱️  TIMEOUT")
        except json.JSONDecodeError:
            print(f"❌ invalid JSON response")
        except Exception as e:
            print(f"❌ {str(e)}")


def load_symbols_from_json(json_path: str) -> Dict[str, List[str]]:
    """
    Load symbols from config/symbols.json, organized by type.
    
    Returns:
        Dict with categories: {"Forex": [...], "Index": [...], "Commodity": [...], "Crypto": [...]}
    """
    if not os.path.exists(json_path):
        print(f"❌ Symbol file not found: {json_path}")
        return {}
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Group symbol names by type
    grouped: Dict[str, List[str]] = {"Forex": [], "Index": [], "Commodity": [], "Crypto": []}
    
    for symbol in data.get("symbols", []):
        sym_type = symbol.get("type", "")
        sym_name = symbol.get("name", "")
        if sym_type in grouped and sym_name:
            grouped[sym_type].append(sym_name)
    
    return grouped


def download_by_category(
    downloader: MT5BridgeDownloader,
    category: str,
    symbols: List[str],
    base_dir: str,
    days_back: int = 730,
) -> None:
    """
    Download data for a single category into its own subfolder.
    
    Args:
        downloader: MT5BridgeDownloader instance
        category: Category name (Forex, Index, Commodity, Crypto)
        symbols: List of symbol names
        base_dir: Base directory for raw data
        days_back: Number of days of history to fetch
    """
    # Map category to folder name
    folder_map = {
        "Forex": "forex",
        "Index": "indices",
        "Commodity": "commodities",
        "Crypto": "crypto",
    }
    folder_name = folder_map.get(category, category.lower())
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'─'*60}")
    print(f"  {category}: {len(symbols)} symbols → {save_dir}")
    print(f"{'─'*60}")
    
    downloader.download_daily_data(symbols, save_folder=save_dir, days_back=days_back)


if __name__ == "__main__":
    print("=" * 70)
    print("MT5 BRIDGE DATA DOWNLOADER")
    print("=" * 70)
    
    # Find project root (where config/ folder is)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent  # Go up to project root
    symbols_path = project_root / "config" / "symbols.json"
    raw_data_dir = project_root / "src" / "data" / "indicators_data" / "raw"
    
    print(f"\n📂 Project root: {project_root}")
    print(f"📄 Symbols file: {symbols_path}")
    print(f"💾 Data directory: {raw_data_dir}")
    
    # Load symbols from JSON
    if not symbols_path.exists():
        print(f"\n❌ ERROR: Symbol file not found!")
        print(f"   Expected: {symbols_path}")
        print(f"   Please create config/symbols.json with your trading symbols")
        exit(1)
    
    print(f"\n📊 Loading symbols from {symbols_path}...")
    symbol_groups = load_symbols_from_json(str(symbols_path))
    
    total_symbols = sum(len(symbols) for symbols in symbol_groups.values())
    print(f"\n✅ Loaded {total_symbols} symbols:")
    for category, symbols in symbol_groups.items():
        print(f"   - {category}: {len(symbols)} symbols")
    
    # Initialize downloader
    print(f"\n🔗 Connecting to MT5 Bridge...")
    downloader = MT5BridgeDownloader(save_folder=str(raw_data_dir))
    
    # Check bridge health
    if not downloader.health_check():
        print("\n⚠️  MT5 Bridge is not running or not accessible.")
        print("\nTo start the bridge:")
        print("  cd src/mt5_bridge")
        print("  python -m uvicorn main:app --host 127.0.0.1 --port 8787")
        exit(1)
    
    # Download data for each category
    print(f"\n{'='*70}")
    print("Starting downloads...")
    print(f"{'='*70}")
    
    for category in ["Forex", "Index", "Commodity", "Crypto"]:
        symbols = symbol_groups.get(category, [])
        if symbols:
            download_by_category(
                downloader=downloader,
                category=category,
                symbols=symbols,
                base_dir=str(raw_data_dir),
                days_back=730,  # 2 years of history
            )
    
    print(f"\n{'='*70}")
    print("✅ ALL DOWNLOADS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nData saved to: {raw_data_dir}")
    print("\nNext steps:")
    print("  1. Process data: python src/data/processor.py")
    print("  2. Train model:  python src/inference/run_forecast.py")
