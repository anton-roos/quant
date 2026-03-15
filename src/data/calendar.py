"""
Economic calendar filter — blocks new trades around high-impact macro events.

Events are loaded from a local JSON cache file.  The cache can be refreshed
from a user-supplied URL (e.g. Forex Factory's unofficial JSON feed or any
service that returns the schema below).

Cache schema  (outputs/calendar_cache.json):
    {
        "last_updated": "2026-03-05T12:00:00",
        "events": [
            {
                "datetime": "2026-03-07T13:30:00",   # UTC ISO-8601
                "currency": "USD",                    # affected currency
                "impact": "High",                     # "High" | "Medium" | "Low"
                "title": "Non-Farm Payrolls"
            },
            ...
        ]
    }

Integration:
    1.  Instantiate once in TradingBot.__init__:
            self.calendar = CalendarFilter(config)
    2.  Refresh daily (in _daily_refresh):
            self.calendar.try_refresh()
    3.  Gate each trade entry (in _open_position):
            if self.calendar.is_blocked(symbol_entry, datetime.utcnow()):
                continue
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CalendarFilter:
    """Block new trade entries around scheduled high-impact macro events."""

    def __init__(self, config: dict, symbols: Optional[List[Dict]] = None):
        """
        Parameters
        ----------
        config : dict
            Bot configuration dict.  Keys consumed:
            - CALENDAR_ENABLED (bool, default True)
            - CALENDAR_BLOCK_HOURS (int, default 4)
            - CALENDAR_CACHE_PATH (str, default 'outputs/calendar_cache.json')
            - CALENDAR_REFRESH_URL (str, default '')
        symbols : list of dict, optional
            Symbol definitions (from symbols.json).  Used to build a mapping
            from symbol name → set of currencies (base + profit).
        """
        self.enabled = config.get("CALENDAR_ENABLED", True)
        self.block_hours = config.get("CALENDAR_BLOCK_HOURS", 4)
        self.cache_path = PROJECT_ROOT / config.get(
            "CALENDAR_CACHE_PATH", "outputs/calendar_cache.json"
        )
        self.refresh_url = config.get("CALENDAR_REFRESH_URL", "")
        self._events: List[Dict] = []
        self._symbol_currencies: Dict[str, Set[str]] = {}

        if symbols:
            self._build_currency_map(symbols)

        self._load_cache()

    # ------------------------------------------------------------------
    # Currency mapping
    # ------------------------------------------------------------------

    def _build_currency_map(self, symbols: List[Dict]) -> None:
        """Build symbol_name → {currencies} from symbols.json entries."""
        for sym in symbols:
            name = sym.get("name", "")
            currencies: Set[str] = set()
            base = sym.get("currency_base", "")
            profit = sym.get("currency_profit", "")
            if base:
                currencies.add(base.upper())
            if profit:
                currencies.add(profit.upper())
            # Index / commodity symbols have no currency fields; derive from name
            if not currencies and len(name) >= 3:
                currencies.add(name[:3].upper())
            if name:
                self._symbol_currencies[name.upper()] = currencies

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load the local calendar cache if it exists."""
        if not self.cache_path.exists():
            logger.debug("Calendar cache not found — no events loaded")
            return
        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)
            self._events = data.get("events", [])
            logger.info(
                f"Calendar cache loaded: {len(self._events)} events "
                f"(last updated: {data.get('last_updated', 'unknown')})"
            )
        except Exception as e:
            logger.warning(f"Failed to load calendar cache: {e}")
            self._events = []

    def _save_cache(self, events: List[Dict]) -> None:
        """Persist events to the local JSON cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, "w") as f:
                json.dump(
                    {
                        "last_updated": datetime.utcnow().isoformat(),
                        "events": events,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Calendar cache saved: {len(events)} events → {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save calendar cache: {e}")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def try_refresh(self) -> bool:
        """Refresh the calendar from the configured URL if available.

        Returns True if the cache was successfully updated.
        """
        if not self.refresh_url:
            logger.debug("No CALENDAR_REFRESH_URL configured — skipping refresh")
            return False

        try:
            import urllib.request
            with urllib.request.urlopen(self.refresh_url, timeout=10) as resp:
                raw = resp.read()
            data = json.loads(raw)

            # Normalise: accept a list of events OR a dict with an 'events' key
            if isinstance(data, list):
                events = data
            else:
                events = data.get("events", data)

            # Keep only high-impact events; normalise field names
            high_impact: List[Dict] = []
            for ev in events:
                impact = str(ev.get("impact", ev.get("Impact", ""))).strip()
                if impact.lower() not in ("high", "red", "3"):
                    continue
                dt_raw = ev.get("datetime", ev.get("date", ev.get("Date", "")))
                currency = str(ev.get("currency", ev.get("Currency", ev.get("country", "")))).upper()
                title = ev.get("title", ev.get("Title", ev.get("event", "")))
                if dt_raw and currency:
                    high_impact.append({
                        "datetime": dt_raw,
                        "currency": currency,
                        "impact": "High",
                        "title": title,
                    })

            self._events = high_impact
            self._save_cache(high_impact)
            return True

        except Exception as e:
            logger.warning(f"Calendar refresh failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    def is_blocked(self, symbol_name: str, check_time: Optional[datetime] = None) -> bool:
        """Return True if the symbol should be blocked due to a nearby event.

        Parameters
        ----------
        symbol_name : str
            MT5 instrument name, e.g. 'EURUSD'.
        check_time : datetime, optional
            Time to check (defaults to utcnow).
        """
        if not self.enabled or not self._events:
            return False

        now = check_time or datetime.utcnow()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        window = timedelta(hours=self.block_hours)
        currencies = self._symbol_currencies.get(symbol_name.upper(), set())

        # Fallback: derive currencies from symbol name if not in map
        if not currencies:
            name = symbol_name.upper()
            if len(name) >= 6:
                currencies = {name[:3], name[3:6]}
            elif len(name) >= 3:
                currencies = {name[:3]}

        for ev in self._events:
            if str(ev.get("impact", "")).lower() != "high":
                continue
            ev_currency = str(ev.get("currency", "")).upper()
            if ev_currency not in currencies:
                continue
            try:
                ev_dt_raw = ev.get("datetime", "")
                # Strip timezone info for naive comparison
                if "T" in ev_dt_raw:
                    ev_dt = datetime.fromisoformat(ev_dt_raw[:19])
                else:
                    ev_dt = datetime.strptime(ev_dt_raw[:16], "%Y-%m-%d %H:%M")
            except Exception:
                continue

            if abs((now - ev_dt).total_seconds()) <= window.total_seconds():
                logger.info(
                    f"  Calendar block: {symbol_name} — "
                    f"'{ev.get('title', '?')}' ({ev_currency}) "
                    f"at {ev.get('datetime', '?')} UTC "
                    f"(window ±{self.block_hours}h)"
                )
                return True

        return False
