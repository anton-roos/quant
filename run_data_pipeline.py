#!/usr/bin/env python3
"""
Run the full data + ML pipeline: download data, train model, generate
forecasts, and backtest.

Usage (from project root):
    python run_data_pipeline.py                # full pipeline
    python run_data_pipeline.py --skip-download # skip data download
    python run_data_pipeline.py --forecast-only # forecast + backtest only (skip training)
    python run_data_pipeline.py --backtest-only # backtest only
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM trading data pipeline")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip the data download step (use cached CSVs)")
    p.add_argument("--forecast-only", action="store_true",
                   help="Only generate forecasts and run backtest (skip training)")
    p.add_argument("--backtest-only", action="store_true",
                   help="Only run the backtest (assumes forecasts exist)")
    return p.parse_args()


def run_download():
    """Download market data via the MT5 bridge."""
    print("\n" + "=" * 60)
    print(" STEP 0: Downloading market data")
    print("=" * 60)
    from src.data.downloader import main as download_main
    download_main()


def run_training():
    """Train model and generate forecasts."""
    print("\n" + "=" * 60)
    print(" STEP 1: Training model + generating forecasts")
    print("=" * 60)

    # Fail fast with a helpful message if user runs the wrong interpreter.
    try:
        import tensorflow as _tf  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "tensorflow":
            raise
        project_root = Path(__file__).parent
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
        hint = (
            f"TensorFlow is not available in this Python interpreter: {sys.executable}\n"
            f"Activate the project's virtual environment or run using:\n  {venv_python} run_data_pipeline.py\n"
        )
        if not venv_python.exists():
            hint += "(Couldn't find .venv at project root.)\n"
        raise SystemExit(hint) from None

    from src.inference.run_forecast import main as forecast_main
    forecast_main()


def run_backtest():
    """Run backtest over existing forecasts."""
    print("\n" + "=" * 60)
    print(" STEP 2: Running backtest")
    print("=" * 60)
    from src.inference.run_backtest import main as backtest_main
    backtest_main()


def main():
    args = parse_args()

    if args.backtest_only:
        run_backtest()
        return

    if args.forecast_only:
        run_training()
        run_backtest()
        return

    if not args.skip_download:
        run_download()

    run_training()
    run_backtest()

    print("\n" + "=" * 60)
    print(" Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
