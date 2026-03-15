"""
Cointegration pairs trading overlay.

Scans the universe of instruments for statistically cointegrated pairs
using the Engle-Granger two-step procedure (statsmodels).  When the spread
between a cointegrated pair deviates significantly from its mean (|z| > entry
threshold), generates a mean-reversion signal.

This strategy is orthogonal to the directional LSTM signals — it profits
from relative mispricing rather than absolute trend, providing portfolio
diversification.

The scan is expensive (O(n²)) and only needs to run weekly.  Results are
cached to ``outputs/pairs_cache.json``.

Usage (in TradingBot):
    self.pairs_trader = PairsTrader(config, mt5_client)
    # Weekly: scan for cointegrated pairs
    signals = self.pairs_trader.scan_and_signal(processed_base, symbols)
    # Each signal: {sym1, sym2, side1, side2, hedge_ratio, zscore, adj_prob}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PairsTrader:
    """Detect cointegrated pairs and generate mean-reversion signals."""

    def __init__(self, config: dict, mt5_client=None):
        """
        Parameters
        ----------
        config : dict
            Bot config.  Keys consumed:
            - PAIRS_TRADING_ENABLED (bool, default False)
            - PAIRS_COINT_PVAL (float, default 0.05)
            - PAIRS_ZSCORE_ENTRY (float, default 2.0)
            - PAIRS_ZSCORE_EXIT (float, default 0.5)
            - PAIRS_LOOKBACK (int, default 252) — rows for spread stats
        mt5_client : MT5Client, optional
        """
        self.enabled = config.get("PAIRS_TRADING_ENABLED", False)
        self.coint_pval = config.get("PAIRS_COINT_PVAL", 0.05)
        self.entry_z = config.get("PAIRS_ZSCORE_ENTRY", 2.0)
        self.exit_z = config.get("PAIRS_ZSCORE_EXIT", 0.5)
        self.lookback = config.get("PAIRS_LOOKBACK", 252)
        self.mt5 = mt5_client
        self._cache_path = PROJECT_ROOT / "outputs" / "pairs_cache.json"
        self._pairs: List[Dict] = []  # [{sym1, sym2, hedge_ratio, last_updated}]
        self._load_cache()

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, "r") as f:
                data = json.load(f)
            self._pairs = data.get("pairs", [])
            logger.info(f"Pairs cache loaded: {len(self._pairs)} cointegrated pairs")
        except Exception as e:
            logger.warning(f"Pairs cache load failed: {e}")

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._cache_path, "w") as f:
                json.dump(
                    {"last_updated": datetime.utcnow().isoformat(), "pairs": self._pairs},
                    f, indent=2,
                )
        except Exception as e:
            logger.warning(f"Pairs cache save failed: {e}")

    # ------------------------------------------------------------------
    # Cointegration scan
    # ------------------------------------------------------------------

    def scan_pairs(self, processed_base: Path, symbols: List[Dict]) -> List[Dict]:
        """Run Engle-Granger cointegration on all symbol pairs.

        Returns a list of cointegrated pairs with hedge ratios.
        Results are cached; call this at most weekly.
        """
        if not self.enabled:
            return []

        try:
            from statsmodels.tsa.stattools import coint
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm
        except ImportError:
            logger.warning("statsmodels not installed — pairs trading disabled")
            return []

        from src.utils.constants import CATEGORY_TYPE_TO_FOLDER, sanitize_filename

        # Load close price series for each symbol
        price_series: Dict[str, pd.Series] = {}
        for sym in symbols:
            name = sym["name"]
            safe = sanitize_filename(name)
            cat = CATEGORY_TYPE_TO_FOLDER.get(sym.get("type", ""), "other")
            csv_path = processed_base / cat / f"{safe}_daily_processed.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
                price_series[name] = df.set_index("date")["close"].tail(self.lookback)
            except Exception:
                pass

        names = list(price_series.keys())
        n = len(names)
        pairs: List[Dict] = []

        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = names[i], names[j]
                p1, p2 = price_series[s1].dropna(), price_series[s2].dropna()

                # Align
                aligned = pd.concat([p1, p2], axis=1, join="inner").dropna()
                if len(aligned) < 60:
                    continue

                y1, y2 = aligned.iloc[:, 0].values, aligned.iloc[:, 1].values

                try:
                    _, pval, _ = coint(y1, y2)
                except Exception:
                    continue

                if pval > self.coint_pval:
                    continue

                # Estimate hedge ratio via OLS
                try:
                    result = OLS(y1, sm.add_constant(y2)).fit()
                    hedge = float(result.params[-1])
                except Exception:
                    hedge = 1.0

                pairs.append({"sym1": s1, "sym2": s2, "hedge_ratio": hedge, "pval": pval})
                logger.info(f"Cointegrated pair: {s1}/{s2} p={pval:.4f} hedge={hedge:.4f}")

        self._pairs = pairs
        self._save_cache()
        logger.info(f"Pairs scan complete: {len(pairs)} pairs found")
        return pairs

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def get_signals(
        self,
        processed_base: Path,
        symbols: List[Dict],
    ) -> List[Dict]:
        """Compute spread z-scores for all cached pairs and return signals.

        A signal is generated when |z| > entry_threshold.
        Signal format: {sym1, sym2, side1, side2, zscore, hedge_ratio, adj_prob}
        """
        if not self.enabled or not self._pairs:
            return []

        from src.utils.constants import CATEGORY_TYPE_TO_FOLDER, sanitize_filename

        sym_type_map = {s["name"]: s.get("type", "Unknown") for s in symbols}
        signals: List[Dict] = []

        for pair in self._pairs:
            s1, s2 = pair["sym1"], pair["sym2"]
            hedge = pair.get("hedge_ratio", 1.0)

            p1 = self._load_close(processed_base, s1, sym_type_map)
            p2 = self._load_close(processed_base, s2, sym_type_map)
            if p1 is None or p2 is None:
                continue

            aligned = pd.concat([p1, p2], axis=1, join="inner").dropna().tail(self.lookback)
            if len(aligned) < 30:
                continue

            spread = aligned.iloc[:, 0] - hedge * aligned.iloc[:, 1]
            mu, sigma = spread.mean(), spread.std()
            if sigma == 0:
                continue

            z = (spread.iloc[-1] - mu) / sigma

            if abs(z) < self.entry_z:
                continue  # Not far enough from mean

            # Long spread (long s1, short s2) when z < -entry (spread too low)
            # Short spread (short s1, long s2) when z > +entry (spread too high)
            if z > self.entry_z:
                side1, side2 = "SELL", "BUY"   # sell expensive s1, buy cheap s2
            else:
                side1, side2 = "BUY", "SELL"

            # Normalised confidence: how far beyond the threshold in σ units
            excess = (abs(z) - self.entry_z) / max(abs(z), 1e-9)
            adj_prob = float(np.clip(0.5 + 0.5 * excess, 0.5, 0.95))

            signals.append({
                "sym1": s1, "sym2": s2,
                "side1": side1, "side2": side2,
                "zscore": float(z),
                "hedge_ratio": hedge,
                "adj_prob": adj_prob,
                "type": "Pairs",
            })
            logger.info(
                f"Pairs signal: {s1} {side1} / {s2} {side2}  z={z:.2f}  adj_prob={adj_prob:.3f}"
            )

        return signals

    def _load_close(
        self, processed_base: Path, name: str, sym_type_map: Dict[str, str]
    ) -> Optional[pd.Series]:
        from src.utils.constants import CATEGORY_TYPE_TO_FOLDER, sanitize_filename
        safe = sanitize_filename(name)
        cat = CATEGORY_TYPE_TO_FOLDER.get(sym_type_map.get(name, ""), "other")
        csv_path = processed_base / cat / f"{safe}_daily_processed.csv"
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
            return df.set_index("date")["close"].tail(self.lookback)
        except Exception:
            return None
