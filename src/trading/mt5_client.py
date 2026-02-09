"""
MT5 Bridge REST API client for the live trading bot.
Wraps all HTTP calls to the MT5 Bridge FastAPI server.
Includes exponential backoff with retries on transient failures.
"""

import logging
import time
from typing import Optional, Dict, List, Any
from functools import wraps

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry decorator for transient HTTP failures
# ---------------------------------------------------------------------------
TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _is_retryable_status(status_code: int) -> bool:
    """Return True for 5xx server errors and 429 Too Many Requests."""
    return status_code >= 500 or status_code == 429


def with_retries(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Decorator: retry on transient network errors and 5xx responses.

    Uses exponential backoff: delay = base_delay * 2^attempt (capped at max_delay).
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    result = fn(*args, **kwargs)
                    return result
                except TRANSIENT_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"[Retry {attempt+1}/{max_retries}] {fn.__name__} "
                            f"transient error: {exc}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[Retry exhausted] {fn.__name__} failed after "
                            f"{max_retries} retries: {exc}"
                        )
                        raise
                except requests.exceptions.HTTPError as exc:
                    # Retry on 5xx / 429, raise immediately on 4xx client errors
                    if exc.response is not None and _is_retryable_status(exc.response.status_code):
                        last_exc = exc
                        if attempt < max_retries:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            logger.warning(
                                f"[Retry {attempt+1}/{max_retries}] {fn.__name__} "
                                f"server error {exc.response.status_code}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                            time.sleep(delay)
                        else:
                            logger.error(
                                f"[Retry exhausted] {fn.__name__} failed after "
                                f"{max_retries} retries: {exc}"
                            )
                            raise
                    else:
                        raise
            raise last_exc  # Should not reach here, but safety net
        return wrapper
    return decorator


class MT5Client:
    """Thin REST wrapper around the MT5 Bridge at localhost:8787."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8787, magic: int = 24001,
                 api_key: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        self.magic = magic
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    @with_retries(max_retries=3, base_delay=1.0)
    def health(self) -> Dict:
        """GET /health – check bridge + MT5 terminal status."""
        r = self.session.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def is_healthy(self) -> bool:
        try:
            h = self.health()
            return h.get("ok", False) and h.get("mt5_initialized", False)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    @with_retries(max_retries=3, base_delay=1.0)
    def account_info(self) -> Dict:
        """GET /account – balance, equity, margin, etc."""
        r = self.session.get(f"{self.base_url}/account", timeout=10)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Quotes & candles
    # ------------------------------------------------------------------
    @with_retries(max_retries=3, base_delay=1.0)
    def quote(self, symbol: str) -> Dict:
        """GET /quote – current bid/ask for symbol."""
        r = self.session.get(f"{self.base_url}/quote", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return r.json()

    @with_retries(max_retries=3, base_delay=2.0)
    def candles(
        self,
        symbol: str,
        timeframe: str = "D1",
        count: int = 500,
        from_date: Optional[str] = None,
    ) -> List[Dict]:
        """GET /candles – historical OHLCV bars."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count,
        }
        if from_date:
            params["from_date"] = from_date
        r = self.session.get(f"{self.base_url}/candles", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("candles", [])

    @with_retries(max_retries=3, base_delay=1.0)
    def symbol_info(self, symbol: str) -> Dict:
        """GET /symbol/{symbol} – digits, point, volumes, spread, etc."""
        r = self.session.get(f"{self.base_url}/symbol/{symbol}", timeout=10)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    @with_retries(max_retries=3, base_delay=1.0)
    def positions(self) -> List[Dict]:
        """GET /positions – all open positions."""
        r = self.session.get(f"{self.base_url}/positions", timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("positions", [])

    def bot_positions(self) -> List[Dict]:
        """Return only positions opened by this bot (matching magic number)."""
        return [p for p in self.positions() if p.get("magic") == self.magic]

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    # NOTE: No retries on order placement to prevent duplicate positions.
    # If the first attempt succeeded but the HTTP response was lost, a retry
    # would create a second position with double exposure.
    def place_market_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
    ) -> Dict:
        """POST /orders – send a market order."""
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "volume": volume,
            "deviation": 20,
            "magic": self.magic,
            "comment": comment[:31],
        }
        if sl is not None:
            payload["sl"] = sl
        if tp is not None:
            payload["tp"] = tp

        logger.info(f"Placing {side} {volume} {symbol} | SL={sl} TP={tp} | {comment}")
        r = self.session.post(f"{self.base_url}/orders", json=payload, timeout=15)
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            logger.error(f"Order rejected ({r.status_code}): {detail}")
            return {"accepted": False, "message": str(detail)}
        resp = r.json()
        logger.info(f"Order response: {resp}")
        return resp

    @with_retries(max_retries=2, base_delay=1.0)
    def close_position(self, ticket: int, comment: str = "") -> Dict:
        """POST /orders/close – close an open position by ticket."""
        payload = {
            "ticket": ticket,
            "deviation": 20,
            "comment": comment[:31],
        }
        logger.info(f"Closing position ticket={ticket}")
        r = self.session.post(f"{self.base_url}/orders/close", json=payload, timeout=15)
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            logger.error(f"Close rejected ({r.status_code}): {detail}")
            return {"accepted": False, "message": str(detail)}
        resp = r.json()
        logger.info(f"Close response: {resp}")
        return resp

    @with_retries(max_retries=2, base_delay=1.0)
    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict:
        """POST /orders/modify – change SL/TP on an open position."""
        payload: Dict[str, Any] = {"ticket": ticket}
        if sl is not None:
            payload["sl"] = sl
        if tp is not None:
            payload["tp"] = tp
        r = self.session.post(f"{self.base_url}/orders/modify", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Analytics (optional helpers)
    # ------------------------------------------------------------------
    @with_retries(max_retries=3, base_delay=1.0)
    def trade_analytics(self, days: int = 30) -> Dict:
        """GET /analytics/trades."""
        r = self.session.get(
            f"{self.base_url}/analytics/trades",
            params={"days": days, "magic": self.magic},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
