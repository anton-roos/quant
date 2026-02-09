"""
Shared constants used across the AI Market Predictor project.

Centralizes magic strings, category mappings, and configuration defaults
so they are defined once and imported everywhere.
"""

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Forecast horizons
# ---------------------------------------------------------------------------
HORIZONS: List[str] = ["1d", "1w", "1m", "6m"]
HORIZON_DAYS: Dict[str, int] = {"1d": 1, "1w": 5, "1m": 21, "6m": 126}

# ---------------------------------------------------------------------------
# Columns excluded from feature set (targets + date)
# ---------------------------------------------------------------------------
EXCLUDED_COLS: List[str] = ["date", "Target_1d", "Target_1w", "Target_1m", "Target_6m"]

# ---------------------------------------------------------------------------
# Category mapping: symbol type -> filesystem subfolder
# ---------------------------------------------------------------------------
# keys = values from symbols.json "type" field
# values = subfolder names under indicators_data/raw and indicators_data/processed
CATEGORY_TYPE_TO_FOLDER: Dict[str, str] = {
    "Forex": "forex",
    "Index": "indices",
    "Commodity": "commodities",
    "Crypto": "crypto",
}

# Folder names used when iterating over all categories
CATEGORY_FOLDERS: List[str] = ["forex", "indices", "commodities", "crypto"]

# ---------------------------------------------------------------------------
# Symbol name sanitisation
# ---------------------------------------------------------------------------

def sanitize_filename(symbol: str) -> str:
    """
    Convert an MT5 symbol name to a safe, clean filename.

    Examples:
        "EURUSD"              -> "EURUSD"
        "Bitcoin (BTCUSD)"    -> "BTCUSD"
        "Brent Crude Oil"     -> "Brent_Crude_Oil"
        "S&P 500"             -> "SP500"
        "DJ Euro Stoxx 50"    -> "DJ_Euro_Stoxx_50"
    """
    # If the name contains a parenthesized ticker like "Bitcoin (BTCUSD)", extract it
    match = re.search(r'\(([A-Z0-9]+)\)', symbol)
    if match:
        return match.group(1)

    # Otherwise sanitize: replace & with nothing, spaces with _, strip non-alphanumeric
    name = symbol.replace("&", "").strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^A-Za-z0-9_]', '', name)
    return name
