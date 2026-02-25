"""
Sentiment data ingestion using Brave Search + Google Gemini LLM.

Architecture
------------
1. **Brave Search** — queries recent news headlines for each symbol
   (free tier: 2 000 queries/month).
2. **Built-in heuristic scorer** — fast keyword/phrase sentiment without
   any external LLM dependency (always available).
3. **Gemini LLM analysis** — if a Gemini API key is provided, headlines are
   sent to Gemini for structured sentiment scoring.

Scores are saved as a daily CSV in ``outputs/sentiment/`` and appended to
``sentiment_history.csv`` for time-series feature engineering.

Usage
-----
    # Heuristic-only (no LLM)
    python -m src.data.sentiment_ingest --brave_key=YOUR_BRAVE_KEY

    # With Gemini analysis
    python -m src.data.sentiment_ingest --brave_key=KEY --gemini_key=GEMINI_KEY

    # Override symbols
    python -m src.data.sentiment_ingest --brave_key=KEY --symbols EURUSD,Gold

Keys can also be set via environment variables or bot_config.json:
    BRAVE_API_KEY, GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SENTIMENT_DIR = PROJECT_ROOT / "outputs" / "sentiment"
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = SENTIMENT_DIR / "sentiment_history.csv"

logger = logging.getLogger("sentiment_ingest")

# ---------------------------------------------------------------------------
# HTTP session with retry / exponential back-off
# ---------------------------------------------------------------------------
_http_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return a shared ``requests.Session`` with automatic retry on transient errors."""
    global _http_session
    if _http_session is not None:
        return _http_session

    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,                     # waits 1s, 2s, 4s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _http_session = session
    return session


# ---------------------------------------------------------------------------
# Brave Search API
# ---------------------------------------------------------------------------
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Mapping from internal symbol names to descriptive search queries.
SYMBOL_SEARCH_MAP = {
    # Forex
    "EURUSD": "EURUSD euro dollar forex",
    "GBPUSD": "GBPUSD pound dollar forex",
    "USDJPY": "USDJPY dollar yen forex",
    "AUDUSD": "AUDUSD australian dollar forex",
    "USDCAD": "USDCAD dollar canadian forex",
    "USDCHF": "USDCHF dollar swiss franc forex",
    "NZDUSD": "NZDUSD new zealand dollar forex",
    "EURGBP": "EURGBP euro pound forex",
    "EURJPY": "EURJPY euro yen forex",
    "GBPJPY": "GBPJPY pound yen forex",
    "AUDCAD": "AUDCAD australian canadian dollar forex",
    "AUDCHF": "AUDCHF australian swiss franc forex",
    "AUDJPY": "AUDJPY australian yen forex",
    "AUDNZD": "AUDNZD australian new zealand forex",
    "CADCHF": "CADCHF canadian swiss franc forex",
    "CADJPY": "CADJPY canadian yen forex",
    "CHFJPY": "CHFJPY swiss franc yen forex",
    "EURAUD": "EURAUD euro australian forex",
    "EURCAD": "EURCAD euro canadian forex",
    "EURCHF": "EURCHF euro swiss franc forex",
    "EURNZD": "EURNZD euro new zealand forex",
    "GBPAUD": "GBPAUD pound australian forex",
    "GBPCAD": "GBPCAD pound canadian forex",
    "GBPCHF": "GBPCHF pound swiss franc forex",
    "GBPNZD": "GBPNZD pound new zealand forex",
    "NZDCAD": "NZDCAD new zealand canadian forex",
    "NZDCHF": "NZDCHF new zealand swiss franc forex",
    "NZDJPY": "NZDJPY new zealand yen forex",
    # Indices
    "SP_500": "S&P 500 index stock market",
    "Australia 200": "ASX 200 australia index",
    "DAX 30": "DAX 30 germany index",
    "DJI 30": "Dow Jones industrial index",
    "FTSE 100": "FTSE 100 UK index",
    "Nikkei 225": "Nikkei 225 japan index",
    "NASDAQ 100": "NASDAQ 100 tech index",
    # Commodities
    "Gold": "gold price commodity XAUUSD",
    "Silver": "silver price commodity XAGUSD",
    "Nymex_Light_Crude": "crude oil WTI price commodity",
    # Crypto
    "Bitcoin (BTCUSD)": "bitcoin BTC price crypto",
    "Ethereum (ETHUSD)": "ethereum ETH price crypto",
}


def _build_search_query(symbol: str) -> str:
    """Return a Brave-friendly search query for *symbol*."""
    if symbol in SYMBOL_SEARCH_MAP:
        return SYMBOL_SEARCH_MAP[symbol] + " news today"
    clean = symbol.replace("_", " ").replace("(", "").replace(")", "")
    return f"{clean} financial news today"


def brave_search(query: str, api_key: str, count: int = 10) -> List[Dict[str, str]]:
    """Query Brave Web Search and return ``[{title, description, url}]``."""
    session = _get_session()
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "count": count,
        "freshness": "pd",  # past day
        "text_decorations": False,
    }
    try:
        resp = session.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
            }
            for item in data.get("web", {}).get("results", [])
        ]
    except requests.exceptions.HTTPError as e:
        logger.warning("Brave search HTTP error for '%s': %s", query, e)
    except Exception as e:
        logger.warning("Brave search failed for '%s': %s", query, e)
    return []


# ---------------------------------------------------------------------------
# Heuristic sentiment scorer (always available, zero external deps)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "rally": 2, "surge": 2, "soar": 2, "jump": 1.5, "gain": 1, "rise": 1,
    "bullish": 2, "upbeat": 1, "strong": 1, "beat": 1, "upgrade": 1.5,
    "recovery": 1, "optimism": 1.5, "boom": 2, "breakout": 1.5, "record high": 2,
    "positive": 1, "improve": 1, "growth": 1, "advance": 1, "outperform": 1.5,
    "buy": 1, "long": 0.5, "hawkish": 1, "tighten": 0.5,
}
_NEGATIVE_WORDS = {
    "crash": 2, "plunge": 2, "tumble": 2, "drop": 1.5, "fall": 1, "decline": 1,
    "bearish": 2, "slump": 1.5, "weak": 1, "miss": 1, "downgrade": 1.5,
    "sell-off": 2, "selloff": 2, "fear": 1, "recession": 2, "crisis": 2,
    "negative": 1, "risk-off": 1.5, "collapse": 2, "loss": 1, "underperform": 1.5,
    "sell": 1, "short": 0.5, "dovish": 1, "cut": 0.5, "warning": 1,
    "concern": 1, "uncertainty": 1, "volatile": 1,
}


def heuristic_sentiment(text: str) -> float:
    """Return a sentiment score in ``[-1, 1]`` from keyword matching."""
    text_lower = text.lower()
    score = 0.0
    total_weight = 0.0
    for word, weight in _POSITIVE_WORDS.items():
        if word in text_lower:
            score += weight
            total_weight += weight
    for word, weight in _NEGATIVE_WORDS.items():
        if word in text_lower:
            score -= weight
            total_weight += weight
    if total_weight == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(total_weight, 1.0)))


def score_headlines_heuristic(results: List[Dict[str, str]]) -> Tuple[float, int]:
    """Score a list of Brave results — returns ``(avg_sentiment, n_headlines)``."""
    if not results:
        return 0.0, 0
    scores = [
        heuristic_sentiment(f"{r.get('title', '')} {r.get('description', '')}")
        for r in results
    ]
    return float(sum(scores) / len(scores)), len(scores)


# ---------------------------------------------------------------------------
# Google Gemini sentiment scorer
# ---------------------------------------------------------------------------

_GEMINI_SYSTEM_PROMPT = """\
You are a financial sentiment analyst. You will receive news headlines \
about a financial instrument.

Assess the aggregate sentiment impact on the instrument's price direction \
over the next 1-5 trading days.

Return a JSON object with exactly these fields:
- "sentiment": float in [-1.0, 1.0] (bearish to bullish)
- "confidence": float in [0.0, 1.0]
- "reasoning": 1-2 sentences
- "key_themes": list of 2-4 themes (e.g. "central bank policy")

0.0 = truly neutral, not uncertain.  Respond ONLY with valid JSON."""

# Default Gemini URL — matches the working aiplatform endpoint.
_GOOGLE_AI_URL = (
    "https://aiplatform.googleapis.com/v1/publishers/google/models"
    "/{model}:generateContent?key={api_key}"
)


def _build_gemini_url(
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    endpoint: str = "",
) -> str:
    """Build the ``generateContent`` URL.

    If *endpoint* is set (e.g. a Vertex AI URL), substitute ``${API_KEY}``
    and ``${MODEL}`` placeholders and switch from streaming to unary.
    Otherwise use the standard Google AI ``generativelanguage`` URL which
    is the correct endpoint for plain API-key authentication.
    """
    if endpoint:
        url = endpoint
        url = url.replace("${API_KEY}", api_key).replace("{API_KEY}", api_key)
        url = url.replace("${MODEL}", model).replace("{MODEL}", model)
        url = url.replace("streamGenerateContent", "generateContent")
        return url
    return _GOOGLE_AI_URL.format(model=model, api_key=api_key)


def gemini_sentiment(
    symbol: str,
    headlines: List[Dict[str, str]],
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
    endpoint: str = "",
) -> Dict[str, Any]:
    """Call Gemini for structured sentiment analysis.

    Returns ``{sentiment, confidence, reasoning, key_themes}``.
    Falls back to heuristic scoring on any failure.
    """
    if not headlines:
        return {"sentiment": 0.0, "confidence": 0.0, "reasoning": "No headlines",
                "key_themes": []}

    headline_text = "\n".join(
        f"- {h.get('title', '')} \u2014 {h.get('description', '')}"
        for h in headlines[:15]
    )
    user_prompt = f"Instrument: {symbol}\n\nHeadlines:\n{headline_text}"

    url = _build_gemini_url(api_key, model, endpoint)
    session = _get_session()

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": _GEMINI_SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 400,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # streamGenerateContent wraps output in a JSON array — handle both
        if isinstance(data, list):
            data = data[-1] if data else {}

        candidates = data.get("candidates", [])
        if not candidates:
            err_detail = data.get("error", {}).get("message", str(data)[:200])
            raise ValueError(f"Gemini returned no candidates: {err_detail}")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            finish = candidates[0].get("finishReason", "UNKNOWN")
            raise ValueError(f"Empty parts (finishReason={finish})")

        raw_text = parts[0].get("text", "")
        # Strip markdown fences the model may add despite responseMimeType
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        raw_text = re.sub(r"\s*```$", "", raw_text.strip())

        parsed = json.loads(raw_text)

        # Clamp to valid ranges
        sentiment = max(-1.0, min(1.0, float(parsed.get("sentiment", 0.0))))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "reasoning": str(parsed.get("reasoning", ""))[:500],
            "key_themes": list(parsed.get("key_themes", []))[:6],
        }

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        logger.warning("Gemini HTTP %s for %s: %s", status, symbol, e)
    except json.JSONDecodeError as e:
        logger.warning("Gemini returned non-JSON for %s: %s", symbol, e)
    except Exception as e:
        logger.warning("Gemini sentiment failed for %s: %s", symbol, e)

    # Fallback — heuristic
    score, _ = score_headlines_heuristic(headlines)
    return {
        "sentiment": score,
        "confidence": 0.3,
        "reasoning": "Gemini unavailable — heuristic fallback",
        "key_themes": [],
    }


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def load_symbols_from_config() -> List[str]:
    """Load symbol names from ``config/symbols.json``."""
    path = PROJECT_ROOT / "config" / "symbols.json"
    if not path.exists():
        logger.warning("symbols.json not found at %s", path)
        return []
    with open(path) as f:
        data = json.load(f)
    return [s["name"] for s in data.get("symbols", [])]


def _load_bot_config() -> Dict[str, Any]:
    """Load bot_config.json for API keys (used by CLI only)."""
    path = PROJECT_ROOT / "config" / "bot_config.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def ingest_sentiment(
    symbols: List[str],
    brave_key: str,
    gemini_key: Optional[str] = None,
    gemini_model: str = "gemini-2.5-flash-lite",
    gemini_endpoint: str = "",
    *,
    delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch sentiment for all *symbols* and return a DataFrame.

    Parameters
    ----------
    delay : float
        Seconds to sleep between Gemini calls (rate-limit guard).
        Default ``1.0`` is safe for the free Gemini tier (15 RPM).

    Returns
    -------
    DataFrame
        Columns: date, symbol, heuristic_sentiment,
        llm_sentiment, llm_confidence, llm_reasoning,
        key_themes, headline_count, combined_sentiment.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(symbols):
        # ---- Brave news search ----
        query = _build_search_query(symbol)
        logger.info("  [%d/%d] %s", idx + 1, len(symbols), symbol)
        logger.info("    Brave: '%s'", query)
        headlines = brave_search(query, brave_key, count=10)

        # Heuristic scores (always)
        h_score, n_headlines = score_headlines_heuristic(headlines)

        # Gemini (optional)
        llm_result: Dict[str, Any] = {
            "sentiment": 0.0, "confidence": 0.0,
            "reasoning": "", "key_themes": [],
        }
        if gemini_key and headlines:
            llm_result = gemini_sentiment(
                symbol, headlines, gemini_key, gemini_model, gemini_endpoint,
            )
            if delay > 0:
                time.sleep(delay)

        # ---- Combine: confidence-weighted blend ----
        if gemini_key and llm_result["confidence"] > 0.2:
            w = llm_result["confidence"]
            combined = (1.0 - w) * h_score + w * llm_result["sentiment"]
        else:
            combined = h_score

        records.append({
            "date": today,
            "symbol": symbol,
            "heuristic_sentiment": round(h_score, 4),
            "llm_sentiment": round(llm_result["sentiment"], 4),
            "llm_confidence": round(llm_result["confidence"], 4),
            "llm_reasoning": llm_result["reasoning"],
            "key_themes": json.dumps(llm_result["key_themes"]),
            "headline_count": n_headlines,
            "combined_sentiment": round(combined, 4),
        })
        logger.info(
            "    heuristic=%+.3f  gemini=%+.3f(conf=%.2f)  combined=%+.3f",
            h_score,
            llm_result["sentiment"], llm_result["confidence"],
            combined,
        )

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No sentiment records produced")
        return df

    # Save daily snapshot
    daily_path = SENTIMENT_DIR / f"sentiment_{today}.csv"
    df.to_csv(daily_path, index=False)
    logger.info("Saved daily snapshot: %s", daily_path)

    # Append to history (de-duplicate today's rows)
    if HISTORY_PATH.exists():
        try:
            existing = pd.read_csv(HISTORY_PATH)
            existing = existing[existing["date"] != today]
            combined_df = pd.concat([existing, df], ignore_index=True)
        except Exception as e:
            logger.warning("Could not read existing history, overwriting: %s", e)
            combined_df = df
    else:
        combined_df = df
    combined_df.to_csv(HISTORY_PATH, index=False)
    logger.info("Updated history: %s (%d rows)", HISTORY_PATH, len(combined_df))

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest financial sentiment via Brave Search + Gemini",
    )
    parser.add_argument(
        "--brave_key", default=os.environ.get("BRAVE_API_KEY", ""),
        help="Brave Search API key (or BRAVE_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini_key", default=os.environ.get("GEMINI_API_KEY", ""),
        help="Google Gemini API key (or GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini_model", default="",
        help="Gemini model (default from config or gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--gemini_endpoint", default="",
        help="Custom Gemini endpoint URL (optional, for Vertex AI)",
    )
    parser.add_argument(
        "--symbols", default="",
        help="Comma-separated symbols (default: all from config/symbols.json)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between Gemini calls (default: 1.0)",
    )
    args = parser.parse_args()

    # Fall back to bot_config.json for keys not passed via CLI / env
    bot_cfg = _load_bot_config()
    brave_key = args.brave_key or bot_cfg.get("BRAVE_API_KEY", "")
    gemini_key = args.gemini_key or bot_cfg.get("GEMINI_API_KEY", "")
    gemini_model = args.gemini_model or bot_cfg.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    gemini_endpoint = args.gemini_endpoint or bot_cfg.get("GEMINI_API_ENDPOINT", "")

    if not brave_key:
        logger.error(
            "Brave API key required. Pass --brave_key=KEY or set BRAVE_API_KEY env var.\n"
            "Get a free key at https://brave.com/search/api/"
        )
        sys.exit(1)

    symbols: List[str]
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols_from_config()

    if not symbols:
        logger.error("No symbols to process.")
        sys.exit(1)

    # Configure root logger for CLI usage only
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("Ingesting sentiment for %d symbols...", len(symbols))
    df = ingest_sentiment(
        symbols, brave_key,
        gemini_key=gemini_key or None,
        gemini_model=gemini_model,
        gemini_endpoint=gemini_endpoint,
        delay=args.delay,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Sentiment Summary \u2014 {datetime.now(timezone.utc):%Y-%m-%d}")
    print(f"{'=' * 60}")
    for _, row in df.iterrows():
        s = row["combined_sentiment"]
        tag = "+" if s > 0.1 else ("-" if s < -0.1 else "~")
        print(
            f"  [{tag}] {row['symbol']:25s}  combined={s:+.3f}  "
            f"(heur={row['heuristic_sentiment']:+.3f}, "
            f"gemini={row['llm_sentiment']:+.3f})"
        )


if __name__ == "__main__":
    main()
