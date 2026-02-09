#!/usr/bin/env python3
"""
Start the MT5 Bridge (FastAPI/Uvicorn server).

Usage (from project root):
    python run_bridge.py                       # default host/port from config
    python run_bridge.py --host 0.0.0.0        # custom host
    python run_bridge.py --port 8787           # custom port
    python run_bridge.py --reload              # auto-reload during dev
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "bot_config.json"


def _load_defaults() -> dict:
    """Pull host/port defaults from bot_config.json."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        return {
            "host": cfg.get("MT5_HOST", "127.0.0.1"),
            "port": cfg.get("MT5_PORT", 8787),
        }
    return {"host": "127.0.0.1", "port": 8787}


def parse_args() -> argparse.Namespace:
    defaults = _load_defaults()
    p = argparse.ArgumentParser(description="MT5 Bridge server")
    p.add_argument("--host", default=defaults["host"], help="Bind address")
    p.add_argument("--port", type=int, default=defaults["port"], help="Bind port")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    return p.parse_args()


def main():
    args = parse_args()

    import uvicorn

    print(f"Starting MT5 Bridge on {args.host}:{args.port}")
    uvicorn.run(
        "src.mt5_bridge.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
