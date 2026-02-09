#!/usr/bin/env python3
"""
Backtest forecasting model performance across multiple trading instruments.
Supports Forex, Indices, Commodities, and Crypto from MT5 Bridge.

Usage:
    python -m src.inference.run_backtest          # run with defaults
    bt = Backtester(); bt.run(); bt.print_summary()  # programmatic
"""
import os
import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.facecolor'] = 'black'
plt.rcParams['savefig.edgecolor'] = 'black'
from PIL import Image


def force_png_rgb(path):
    img = Image.open(path).convert("RGB")
    img.save(path)


# Default backtest horizons
DEFAULT_PERIODS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
}


@dataclass
class BacktestConfig:
    """All tuneable knobs for the backtester, matching live bot defaults."""
    initial_value: float = 1.0
    random_runs: int = 50
    min_accepted: float = 0.50
    std_factor: float = 1.0
    max_concurrent: int = 3
    slippage_bps: float = 5.0
    swap_bps_per_day: float = 1.5
    periods: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PERIODS))
    project_root: Path = field(default_factory=lambda: Path(os.getcwd()))


class Backtester:
    """
    Walk-forward backtest engine that mirrors the live bot's position-sizing
    and signal-filtering rules.

    Designed to be:
      - testable (instantiate with custom config / data)
      - composable (call individual methods instead of monolithic script)
      - deterministic (pass seed for reproducibility)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg = config or BacktestConfig()
        self.video_dir = self.cfg.project_root / "outputs" / "videos"
        self.forecast_dir = self.cfg.project_root / "outputs" / "forecasts"
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.forecast_dir, exist_ok=True)

        # Data
        self.all_forecasts: Dict[str, pd.DataFrame] = {}
        self.common_dates: List = []

        # Strategy state
        self.strategy_history: List[float] = []
        self.trade_log: List[dict] = []
        self.buy_points: List[dict] = []
        self.sell_points: List[dict] = []
        self.successful_trades: int = 0
        self.total_trades: int = 0

        # Random baseline
        self.random_results: Optional[np.ndarray] = None
        self.random_mean: Optional[np.ndarray] = None
        self.random_std: Optional[np.ndarray] = None

        # Computed metrics
        self.metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_forecasts(self) -> int:
        """Load all *_forecast.csv files from the forecast directory.

        Returns the number of symbols loaded.
        """
        self.all_forecasts.clear()
        for filename in os.listdir(self.forecast_dir):
            if not filename.endswith("_forecast.csv"):
                continue
            filepath = self.forecast_dir / filename
            symbol = filename.split("_forecast")[0]
            df = pd.read_csv(filepath, parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            df["Symbol"] = symbol
            if "Close" in df.columns:
                for label, days in self.cfg.periods.items():
                    df[f"Actual_LogR_{label}"] = np.log(
                        df["Close"].shift(-days) / df["Close"]
                    )
            self.all_forecasts[symbol] = df

        print(f"\nLoaded {len(self.all_forecasts)} symbols (no spike filter applied)")
        self._build_common_dates()
        return len(self.all_forecasts)

    def _build_common_dates(self):
        if not self.all_forecasts:
            self.common_dates = []
            return
        all_dates = [set(df.index) for df in self.all_forecasts.values()]
        self.common_dates = sorted(set.union(*all_dates))
        print(f"\nUsing {len(self.common_dates)} dates across {len(self.all_forecasts)} symbols")

    # ------------------------------------------------------------------
    # Strategy simulation
    # ------------------------------------------------------------------
    def run_strategy(self):
        """Run the main probability-based strategy over common_dates."""
        cfg = self.cfg
        strategy_value = cfg.initial_value
        self.strategy_history.clear()
        self.trade_log.clear()
        self.buy_points.clear()
        self.sell_points.clear()
        self.successful_trades = 0
        self.total_trades = 0

        active_positions: List[dict] = []
        slippage_mult = 1.0 - (cfg.slippage_bps / 10000.0)

        for i, date in enumerate(self.common_dates):
            # ------- Check for position exits -------
            positions_to_close = []
            for pos_idx, pos in enumerate(active_positions):
                if i >= pos["exit_idx"]:
                    positions_to_close.append(pos_idx)

            for pos_idx in sorted(positions_to_close, reverse=True):
                pos = active_positions.pop(pos_idx)
                realized_logr = pos["Actual_LogR"]
                side = pos["Side"]
                effective_logr = -realized_logr if side == "SELL" else realized_logr
                swap_cost = (cfg.swap_bps_per_day / 10000.0) * pos["Days"]
                effective_return = (
                    (np.exp(effective_logr) * slippage_mult - 1.0 - swap_cost)
                    if not np.isnan(effective_logr) else np.nan
                )
                alloc = pos["Allocation"]

                self.trade_log.append({
                    "BuyDate":           pos["BuyDate"],
                    "SellDate":          date,
                    "Symbol":            pos["Symbol"],
                    "Side":              side,
                    "Horizon":           pos["Period"],
                    "DaysHeld":          pos["Days"],
                    "Pred_Prob":         pos["Pred_Prob"],
                    "Pred_Prob_Std":     pos["Pred_Prob_Std"],
                    "Adj_Prob":          pos["Adj_Prob"],
                    "Actual_LogR":       realized_logr,
                    "Effective_Return%": effective_return * 100 if not np.isnan(effective_return) else np.nan,
                    "Allocation":        alloc,
                })

                if not np.isnan(effective_return):
                    strategy_value *= (1.0 + effective_return * alloc)
                    if effective_return > 0:
                        self.successful_trades += 1
                self.total_trades += 1

            # ------- Check for new entries -------
            slots_available = cfg.max_concurrent - len(active_positions)
            if slots_available <= 0:
                self.strategy_history.append(strategy_value)
                continue

            open_symbols = {pos["Symbol"] for pos in active_positions}
            candidates = self._build_candidates(date, open_symbols)

            if not candidates:
                self.strategy_history.append(strategy_value)
                continue

            candidates.sort(key=lambda x: x["Adj_Prob"], reverse=True)

            filled = 0
            for best in candidates:
                if filled >= slots_available:
                    break
                if best["Symbol"] in open_symbols:
                    continue

                exit_idx = min(i + best["Days"], len(self.common_dates) - 1)
                alloc = 1.0 / cfg.max_concurrent

                position = {
                    **best,
                    "entry_idx":  i,
                    "exit_idx":   exit_idx,
                    "BuyDate":    date,
                    "Allocation": alloc,
                }
                active_positions.append(position)
                open_symbols.add(best["Symbol"])
                filled += 1

                entry_point = {
                    "Date":      date,
                    "Value":     strategy_value,
                    "Symbol":    best["Symbol"],
                    "Horizon":   best["Period"],
                    "Pred_Prob": best["Pred_Prob"],
                    "Adj_Prob":  best["Adj_Prob"],
                }
                if best["Side"] == "BUY":
                    self.buy_points.append(entry_point)
                else:
                    self.sell_points.append(entry_point)

            self.strategy_history.append(strategy_value)

    def _build_candidates(self, date, open_symbols: set) -> List[dict]:
        """Build BUY/SELL candidate list for a single date."""
        cfg = self.cfg
        candidates = []
        for symbol, df in self.all_forecasts.items():
            if date not in df.index:
                continue
            if symbol in open_symbols:
                continue
            row = df.loc[date]
            for label, days in cfg.periods.items():
                actual_logr = float(row.get(f"Actual_LogR_{label}", np.nan))

                # BUY
                prob_col = f"Pred_Prob_{label}"
                std_col = f"Pred_Prob_Std_{label}"
                if prob_col in row and not pd.isna(row[prob_col]):
                    pred_prob = float(row[prob_col])
                    pred_std = max(float(row.get(std_col, 0.0)), 1e-6)
                    adj_prob = pred_prob - cfg.std_factor * pred_std
                    if adj_prob > cfg.min_accepted:
                        candidates.append({
                            "Symbol": symbol, "Side": "BUY", "Period": label,
                            "Days": days, "Pred_Prob": pred_prob,
                            "Pred_Prob_Std": pred_std, "Adj_Prob": adj_prob,
                            "Actual_LogR": actual_logr,
                        })

                # SELL
                prob_down_col = f"Pred_Prob_Down_{label}"
                std_down_col = f"Pred_Prob_Down_Std_{label}"
                if prob_down_col in row and not pd.isna(row[prob_down_col]):
                    pred_prob_down = float(row[prob_down_col])
                    pred_std_down = max(float(row.get(std_down_col, 0.0)), 1e-6)
                    adj_prob_down = pred_prob_down - cfg.std_factor * pred_std_down
                    if adj_prob_down > cfg.min_accepted:
                        candidates.append({
                            "Symbol": symbol, "Side": "SELL", "Period": label,
                            "Days": days, "Pred_Prob": pred_prob_down,
                            "Pred_Prob_Std": pred_std_down, "Adj_Prob": adj_prob_down,
                            "Actual_LogR": actual_logr,
                        })
        return candidates

    # ------------------------------------------------------------------
    # Random baseline
    # ------------------------------------------------------------------
    def run_random_baseline(self, seed: Optional[int] = None):
        """Fair random baseline using the same entry schedule as the strategy."""
        cfg = self.cfg
        if seed is not None:
            random.seed(seed)

        print("\nBuilding fair random baseline...")
        available_returns = self._precompute_available_returns()

        date_to_idx = {d: i for i, d in enumerate(self.common_dates)}
        strategy_entries: Dict[int, list] = {}
        for bp in self.buy_points:
            idx = date_to_idx[bp["Date"]]
            strategy_entries.setdefault(idx, []).append({"horizon": bp["Horizon"], "side": "BUY"})
        for sp in self.sell_points:
            idx = date_to_idx[sp["Date"]]
            strategy_entries.setdefault(idx, []).append({"horizon": sp["Horizon"], "side": "SELL"})

        slippage_mult = 1.0 - (cfg.slippage_bps / 10000.0)
        self.random_results = np.zeros((len(self.common_dates), cfg.random_runs))

        for run in range(cfg.random_runs):
            value = cfg.initial_value
            rand_positions: List[dict] = []
            for i, date in enumerate(self.common_dates):
                to_close = [p for p in rand_positions if i >= p["exit_idx"]]
                for p in to_close:
                    rand_positions.remove(p)
                    if not np.isnan(p["ret"]):
                        value *= (1.0 + p["ret"] * p["alloc"])

                if i in strategy_entries:
                    for entry_info in strategy_entries[i]:
                        if len(rand_positions) >= cfg.max_concurrent:
                            break
                        horizon = entry_info["horizon"]
                        days = cfg.periods[horizon]
                        exit_idx = min(i + days, len(self.common_dates) - 1)
                        rets = available_returns[date][horizon]
                        if rets:
                            raw_ret = random.choice(rets)
                            rand_side = random.choice(["BUY", "SELL"])
                            effective = raw_ret if rand_side == "BUY" else -raw_ret
                            effective_return = np.exp(effective) * slippage_mult - 1.0
                        else:
                            effective_return = np.nan
                        rand_positions.append({
                            "exit_idx": exit_idx,
                            "ret": effective_return,
                            "alloc": 1.0 / cfg.max_concurrent,
                        })
                self.random_results[i, run] = value

        self.random_mean = np.mean(self.random_results, axis=1)
        self.random_std = np.std(self.random_results, axis=1)

    def _precompute_available_returns(self) -> dict:
        available = {}
        for date in self.common_dates:
            available[date] = {}
            for label in self.cfg.periods:
                act_col = f"Actual_LogR_{label}"
                rets = []
                for symbol, df in self.all_forecasts.items():
                    if date in df.index and act_col in df.columns:
                        v = df.loc[date, act_col]
                        if not pd.isna(v):
                            rets.append(v)
                available[date][label] = rets
        return available

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all performance metrics and store in self.metrics."""
        arr = np.array(self.strategy_history)
        strat_final = arr[-1] if len(arr) else 1.0
        rand_final = self.random_mean[-1] if self.random_mean is not None else 1.0

        n_years = len(self.common_dates) / 252
        strat_ann = (strat_final ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        rand_ann = (rand_final ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        strat_peak = np.maximum.accumulate(arr)
        strat_dd = (arr - strat_peak) / strat_peak
        max_dd = strat_dd.min() * 100

        strat_log_rets = np.diff(np.log(np.maximum(arr, 1e-10)))
        if len(strat_log_rets) > 1 and np.std(strat_log_rets) > 0:
            sharpe = (np.mean(strat_log_rets) / np.std(strat_log_rets)) * np.sqrt(252)
        else:
            sharpe = 0.0

        neg_rets = strat_log_rets[strat_log_rets < 0]
        if len(neg_rets) > 1 and np.std(neg_rets) > 0:
            sortino = (np.mean(strat_log_rets) / np.std(neg_rets)) * np.sqrt(252)
        else:
            sortino = 0.0

        calmar = abs(strat_ann / max_dd) if max_dd != 0 else 0.0

        if self.total_trades > 0:
            trade_returns = [
                t["Effective_Return%"] for t in self.trade_log
                if not np.isnan(t.get("Effective_Return%", np.nan))
            ]
            gross_wins = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        else:
            profit_factor = 0.0

        self.metrics = {
            "strat_final":   strat_final,
            "rand_final":    rand_final,
            "n_years":       n_years,
            "strat_ann":     strat_ann,
            "rand_ann":      rand_ann,
            "max_dd":        max_dd,
            "sharpe":        sharpe,
            "sortino":       sortino,
            "calmar":        calmar,
            "profit_factor": profit_factor,
        }
        return self.metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def save_trade_summary(self) -> Path:
        summary_df = pd.DataFrame(self.trade_log)
        summary_path = self.video_dir / "trade_summary_prob_strategy.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved trade summary to {summary_path}")
        print(summary_df.head())
        return summary_path

    def print_summary(self):
        m = self.metrics
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f" BACKTEST PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f" Max Concurrent Positions: {cfg.max_concurrent}")
        print(f" Slippage (round-trip):    {cfg.slippage_bps} bps")
        print(f" Strategy Final Value:    {m['strat_final']:.4f}  ({(m['strat_final']-1)*100:+.1f}%)")
        print(f" Random Mean Final Value: {m['rand_final']:.4f}  ({(m['rand_final']-1)*100:+.1f}%)")
        print(f" Strategy Annualized:     {m['strat_ann']:+.2f}%")
        print(f" Random Mean Annualized:  {m['rand_ann']:+.2f}%")
        print(f" Max Drawdown:            {m['max_dd']:.1f}%")
        print(f" Sharpe Ratio (approx):   {m['sharpe']:.2f}")
        print(f" Sortino Ratio:           {m['sortino']:.2f}")
        print(f" Calmar Ratio:            {m['calmar']:.2f}")
        print(f" Profit Factor:           {m['profit_factor']:.2f}")
        print(f" Swap Cost (bps/day):     {cfg.swap_bps_per_day}")
        print(f" Total Trades:            {self.total_trades}")
        if self.total_trades > 0:
            print(f" Win Rate:                {self.successful_trades}/{self.total_trades} = {(self.successful_trades/self.total_trades):.1%}")
        else:
            print(f" Win Rate:                N/A")
        print(f" Timespan:                {m['n_years']:.1f} years ({len(self.common_dates)} trading days)")
        if self.total_trades > 0:
            buy_count = sum(1 for t in self.trade_log if t["Side"] == "BUY")
            sell_count = sum(1 for t in self.trade_log if t["Side"] == "SELL")
            print(f" BUY trades:              {buy_count}")
            print(f" SELL trades:             {sell_count}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def save_plots(self):
        strategy_arr = np.array(self.strategy_history)
        self._save_final_plot(
            self.common_dates, self.random_results, strategy_arr,
            show_uncertainty=False,
            filename="random_vs_prob_strategy_clean.png",
        )
        self._save_final_plot(
            self.common_dates, self.random_results, strategy_arr,
            random_mean=self.random_mean, random_std=self.random_std,
            show_uncertainty=True,
            filename="random_vs_prob_strategy_uncertainty.png",
        )

    def _save_final_plot(
        self, dates, random_results, strat_values,
        random_mean=None, random_std=None,
        show_uncertainty=False, filename="final_plot.png",
    ):
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        for r in range(random_results.shape[1]):
            ax.plot(dates, random_results[:, r], alpha=0.15, lw=1, color="white")

        if show_uncertainty and random_mean is not None and random_std is not None:
            upper_3s = random_mean + 3 * random_std
            lower_3s = np.maximum(random_mean - 3 * random_std, 1e-6)
            upper_1s = random_mean + random_std
            lower_1s = np.maximum(random_mean - random_std, 1e-6)
            ax.fill_between(dates, lower_3s, upper_3s, color="gray", alpha=0.08)
            ax.fill_between(dates, lower_1s, upper_1s, color="gray", alpha=0.20)

        if random_mean is not None:
            ax.plot(dates, random_mean, color="white", lw=2, alpha=0.7, label="Random Mean")

        ax.plot(dates, strat_values, color="#39FF14", lw=3, label="AI Strategy")

        if self.buy_points:
            bp_dates = [bp["Date"] for bp in self.buy_points]
            bp_vals = [bp["Value"] for bp in self.buy_points]
            ax.scatter(bp_dates, bp_vals, color="#39FF14", s=12, zorder=5, alpha=0.4, label="BUY")

        if self.sell_points:
            sp_dates = [sp["Date"] for sp in self.sell_points]
            sp_vals = [sp["Value"] for sp in self.sell_points]
            ax.scatter(sp_dates, sp_vals, color="#FF3939", s=12, zorder=5, alpha=0.4, marker="v", label="SELL")

        ax.set_yscale("log")
        ax.set_xlim(dates[0], dates[-1])
        ax.set_title("AI Strategy vs Random Baseline (Fair Comparison)", color="white", fontsize=22)
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Portfolio Value (log scale)", color="white")
        ax.tick_params(colors="white")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.axhline(y=1.0, color="gray", lw=1, ls="--", alpha=0.4)

        legend = ax.legend(facecolor="black", edgecolor="white", fontsize=12, loc="upper left")
        for text in legend.get_texts():
            text.set_color("white")

        png_path = self.video_dir / filename
        plt.savefig(png_path, dpi=200, facecolor="black")
        force_png_rgb(png_path)
        plt.close(fig)
        print(f"Saved: {png_path}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def run(self):
        """Full pipeline: load → strategy → random baseline → metrics → save."""
        n = self.load_forecasts()
        if n == 0 or not self.common_dates:
            print(f"\nNo forecast files found in: {self.forecast_dir}")
            print("Expected files like: <SYMBOL>_forecast.csv")
            return
        self.run_strategy()
        self.save_trade_summary()
        self.run_random_baseline()
        self.compute_metrics()
        self.print_summary()
        self.save_plots()


# ======================================================================
# CLI entry point
# ======================================================================
def main():
    bt = Backtester()
    bt.run()


if __name__ == "__main__":
    main()
