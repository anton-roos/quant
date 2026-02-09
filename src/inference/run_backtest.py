#!/usr/bin/env python3
"""
Backtest forecasting model performance across multiple trading instruments.
Supports Forex, Indices, Commodities, and Crypto from MT5 Bridge.
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
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.facecolor'] = 'black'
plt.rcParams['savefig.edgecolor'] = 'black'
from PIL import Image

def force_png_rgb(path):
    img = Image.open(path).convert("RGB")
    img.save(path)

PROJECT_ROOT = Path(os.getcwd())
video_dir    = PROJECT_ROOT / "outputs" / "videos"
forecast_dir = PROJECT_ROOT / "outputs" / "forecasts"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(forecast_dir, exist_ok=True)

# Config — aligned with live bot (config/bot_config.json)
initial_value   = 1.0
random_runs     = 50
MIN_ACCEPTED    = 0.10      # Only trade when model is highly confident
STD_FACTOR      = 1.0       # AdjustedProb = PredProb - STD_FACTOR * StdDev (penalize uncertainty)
MAX_CONCURRENT  = 3         # Max simultaneous positions (must match bot)
SLIPPAGE_BPS    = 5         # Round-trip spread + slippage cost in basis points

# Backtest horizons
PERIODS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
}

all_forecasts = {}
rejected_symbols = []

for filename in os.listdir(forecast_dir):

    if not filename.endswith("_forecast.csv"):
        continue

    filepath = forecast_dir / filename
    symbol   = filename.split("_forecast")[0]

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df["Symbol"] = symbol

    # Compute realized log returns for horizons (no spike filter — keep all instruments)
    if "Close" in df.columns:
        for label, days in PERIODS.items():
            df[f"Actual_LogR_{label}"] = np.log(df["Close"].shift(-days) / df["Close"])

    all_forecasts[symbol] = df

print(f"\nLoaded {len(all_forecasts)} symbols (no spike filter applied)")


# Get common dates across all symbols

if not all_forecasts:
    print(f"\nNo forecast files found in: {forecast_dir}")
    print("Expected files like: <SYMBOL>_forecast.csv")
    sys.exit(0)

all_dates     = [set(df.index) for df in all_forecasts.values()]
common_dates  = sorted(set.union(*all_dates))

print(f"\nUsing {len(common_dates)} dates across {len(all_forecasts)} symbols")

if not common_dates:
    print("No dates available for backtesting.")
    sys.exit(0)


# Backtesting strategy (aligned with live bot logic):
# Rules:
#   For each date, compute adj_prob = Pred_Prob - STD_FACTOR * Pred_Prob_Std
#   Reject candidates with adj_prob <= MIN_ACCEPTED
#   Hold up to MAX_CONCURRENT positions simultaneously
#   Skip symbols that already have an open position
#   Apply spread/slippage cost to each trade
#   Each position gets an equal share of capital (1/MAX_CONCURRENT)

strategy_value  = initial_value
strategy_history = []
trade_log        = []
buy_points       = []
sell_points      = []

# Active positions: list of dicts with exit_idx, allocated capital fraction, etc.
active_positions = []
successful_trades = 0
total_trades      = 0

# Slippage cost as a multiplier (applied once at entry — covers round-trip spread)
slippage_mult = 1.0 - (SLIPPAGE_BPS / 10000.0)

for i, date in enumerate(common_dates):
    # ------- Check for position exits -------
    positions_to_close = []
    for pos_idx, pos in enumerate(active_positions):
        if i >= pos["exit_idx"]:
            positions_to_close.append(pos_idx)

    # Close expired positions (iterate in reverse to preserve indices)
    for pos_idx in sorted(positions_to_close, reverse=True):
        pos = active_positions.pop(pos_idx)

        realized_logr = pos["Actual_LogR"]
        side = pos["Side"]

        # For SELL trades, profit is the inverse of the log return
        effective_logr = -realized_logr if side == "SELL" else realized_logr

        # Apply slippage cost
        effective_return = np.exp(effective_logr) * slippage_mult - 1.0 if not np.isnan(effective_logr) else np.nan

        # This position's P&L in portfolio terms (weighted by allocation)
        alloc = pos["Allocation"]

        trade_log.append({
            "BuyDate"        : pos["BuyDate"],
            "SellDate"       : date,
            "Symbol"         : pos["Symbol"],
            "Side"           : side,
            "Horizon"        : pos["Period"],
            "DaysHeld"       : pos["Days"],
            "Pred_Prob"      : pos["Pred_Prob"],
            "Pred_Prob_Std"  : pos["Pred_Prob_Std"],
            "Adj_Prob"       : pos["Adj_Prob"],
            "Actual_LogR"    : realized_logr,
            "Effective_Return%" : effective_return * 100 if not np.isnan(effective_return) else np.nan,
            "Allocation"     : alloc,
        })

        if not np.isnan(effective_return):
            # Apply weighted return to portfolio
            strategy_value *= (1.0 + effective_return * alloc)
            if effective_return > 0:
                successful_trades += 1

        total_trades += 1

    # ------- Check for new entries -------
    slots_available = MAX_CONCURRENT - len(active_positions)
    if slots_available <= 0:
        strategy_history.append(strategy_value)
        continue

    # Symbols currently in open positions (no duplicates, matching live bot)
    open_symbols = {pos["Symbol"] for pos in active_positions}

    # Build candidates for today (both BUY and SELL)
    candidates = []

    for symbol, df in all_forecasts.items():

        if date not in df.index:
            continue

        # Skip if already holding this symbol
        if symbol in open_symbols:
            continue

        row = df.loc[date]

        for label, days in PERIODS.items():

            act_col  = f"Actual_LogR_{label}"
            actual_logr = float(row.get(act_col, np.nan))

            # --- BUY candidate (upside probability) ---
            prob_col = f"Pred_Prob_{label}"
            std_col  = f"Pred_Prob_Std_{label}"

            if prob_col in row and not pd.isna(row[prob_col]):
                pred_prob = float(row[prob_col])
                pred_std  = float(row.get(std_col, 0.0))
                pred_std  = max(pred_std, 1e-6)
                adj_prob = pred_prob - STD_FACTOR * pred_std

                if adj_prob > MIN_ACCEPTED:
                    candidates.append({
                        "Symbol"        : symbol,
                        "Side"          : "BUY",
                        "Period"        : label,
                        "Days"          : days,
                        "Pred_Prob"     : pred_prob,
                        "Pred_Prob_Std" : pred_std,
                        "Adj_Prob"      : adj_prob,
                        "Actual_LogR"   : actual_logr,
                    })

            # --- SELL candidate (downside probability) ---
            prob_down_col = f"Pred_Prob_Down_{label}"
            std_down_col  = f"Pred_Prob_Down_Std_{label}"

            if prob_down_col in row and not pd.isna(row[prob_down_col]):
                pred_prob_down = float(row[prob_down_col])
                pred_std_down  = float(row.get(std_down_col, 0.0))
                pred_std_down  = max(pred_std_down, 1e-6)
                adj_prob_down = pred_prob_down - STD_FACTOR * pred_std_down

                if adj_prob_down > MIN_ACCEPTED:
                    candidates.append({
                        "Symbol"        : symbol,
                        "Side"          : "SELL",
                        "Period"        : label,
                        "Days"          : days,
                        "Pred_Prob"     : pred_prob_down,
                        "Pred_Prob_Std" : pred_std_down,
                        "Adj_Prob"      : adj_prob_down,
                        "Actual_LogR"   : actual_logr,
                    })

    # No candidates, then nothing to trade
    if not candidates:
        strategy_history.append(strategy_value)
        continue

    # Sort by adj_prob descending (best signals first), fill available slots
    candidates.sort(key=lambda x: x["Adj_Prob"], reverse=True)

    filled = 0
    for best in candidates:
        if filled >= slots_available:
            break

        # Skip if this symbol was just added in this same loop iteration
        if best["Symbol"] in open_symbols:
            continue

        entry_idx = i
        exit_idx  = min(i + best["Days"], len(common_dates) - 1)

        # Each position gets equal allocation (1/MAX_CONCURRENT of portfolio)
        alloc = 1.0 / MAX_CONCURRENT

        position = {
            **best,
            "entry_idx"  : entry_idx,
            "exit_idx"   : exit_idx,
            "BuyDate"    : date,
            "Allocation" : alloc,
        }

        active_positions.append(position)
        open_symbols.add(best["Symbol"])
        filled += 1

        entry_point = {
            "Date"        : date,
            "Value"       : strategy_value,
            "Symbol"      : best["Symbol"],
            "Horizon"     : best["Period"],
            "Pred_Prob"   : best["Pred_Prob"],
            "Adj_Prob"    : best["Adj_Prob"],
        }

        if best["Side"] == "BUY":
            buy_points.append(entry_point)
        else:
            sell_points.append(entry_point)

    strategy_history.append(strategy_value)


# save trade summary
summary_df = pd.DataFrame(trade_log)
summary_path = video_dir / "trade_summary_prob_strategy.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nSaved trade summary to {summary_path}")
print(summary_df.head())

if total_trades > 0:
    print(f"\nTrade success rate: {successful_trades}/{total_trades} = {(successful_trades/total_trades):.2%}")
    buy_count = sum(1 for t in trade_log if t["Side"] == "BUY")
    sell_count = sum(1 for t in trade_log if t["Side"] == "SELL")
    print(f"BUY trades: {buy_count}, SELL trades: {sell_count}")
else:
    print("\nNo completed trades.")


# ==========================================================================
# Fair Random Baseline: mirrors strategy trade timing & horizons,
# but picks a random symbol instead of the model's pick.
# This is an apples-to-apples "skill vs luck" comparison.
# ==========================================================================

# Pre-compute available returns per (date, horizon) for efficiency
print("\nBuilding fair random baseline...")
available_returns = {}
for date in common_dates:
    available_returns[date] = {}
    for label in PERIODS:
        act_col = f"Actual_LogR_{label}"
        rets = []
        for symbol, df in all_forecasts.items():
            if date in df.index and act_col in df.columns:
                v = df.loc[date, act_col]
                if not pd.isna(v):
                    rets.append(v)
        available_returns[date][label] = rets

# Build strategy entry schedule from buy_points + sell_points
date_to_idx = {d: i for i, d in enumerate(common_dates)}
strategy_entries = {}   # idx -> list of {horizon, side}
for bp in buy_points:
    idx = date_to_idx[bp["Date"]]
    strategy_entries.setdefault(idx, []).append({"horizon": bp["Horizon"], "side": "BUY"})
for sp in sell_points:
    idx = date_to_idx[sp["Date"]]
    strategy_entries.setdefault(idx, []).append({"horizon": sp["Horizon"], "side": "SELL"})

random_results = np.zeros((len(common_dates), random_runs))

for run in range(random_runs):
    value = initial_value
    # Track multiple concurrent positions: list of (exit_idx, pending_ret, allocation)
    rand_positions = []

    for i, date in enumerate(common_dates):
        # Exit expired positions
        to_close = [p for p in rand_positions if i >= p["exit_idx"]]
        for p in to_close:
            rand_positions.remove(p)
            if not np.isnan(p["ret"]):
                value *= (1.0 + p["ret"] * p["alloc"])

        # Enter new positions when strategy would (same timing, concurrent slots)
        if i in strategy_entries:
            for entry_info in strategy_entries[i]:
                if len(rand_positions) >= MAX_CONCURRENT:
                    break

                horizon = entry_info["horizon"]
                days = PERIODS[horizon]
                exit_idx = min(i + days, len(common_dates) - 1)

                rets = available_returns[date][horizon]
                if rets:
                    raw_ret = random.choice(rets)
                    rand_side = random.choice(["BUY", "SELL"])
                    effective = raw_ret if rand_side == "BUY" else -raw_ret
                    # Apply slippage
                    effective_return = (np.exp(effective) * slippage_mult - 1.0)
                else:
                    effective_return = np.nan

                rand_positions.append({
                    "exit_idx": exit_idx,
                    "ret": effective_return,
                    "alloc": 1.0 / MAX_CONCURRENT,
                })

        random_results[i, run] = value

random_mean = np.mean(random_results, axis=1)
random_std  = np.std(random_results, axis=1)

# ==========================================================================
# Performance Metrics
# ==========================================================================
strategy_history_arr = np.array(strategy_history)
strat_final = strategy_history_arr[-1]
rand_final  = random_mean[-1]

# Annualized return (approximate trading days)
n_years = len(common_dates) / 252
strat_ann = (strat_final ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
rand_ann  = (rand_final  ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

# Max drawdown
strat_peak = np.maximum.accumulate(strategy_history_arr)
strat_dd   = ((strategy_history_arr - strat_peak) / strat_peak)
max_dd     = strat_dd.min() * 100

# Sharpe-like ratio (daily log returns of strategy)
strat_log_rets = np.diff(np.log(np.maximum(strategy_history_arr, 1e-10)))
if len(strat_log_rets) > 1 and np.std(strat_log_rets) > 0:
    sharpe = (np.mean(strat_log_rets) / np.std(strat_log_rets)) * np.sqrt(252)
else:
    sharpe = 0.0

print(f"\n{'='*60}")
print(f" BACKTEST PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f" Max Concurrent Positions: {MAX_CONCURRENT}")
print(f" Slippage (round-trip):    {SLIPPAGE_BPS} bps")
print(f" Strategy Final Value:    {strat_final:.4f}  ({(strat_final-1)*100:+.1f}%)")
print(f" Random Mean Final Value: {rand_final:.4f}  ({(rand_final-1)*100:+.1f}%)")
print(f" Strategy Annualized:     {strat_ann:+.2f}%")
print(f" Random Mean Annualized:  {rand_ann:+.2f}%")
print(f" Max Drawdown:            {max_dd:.1f}%")
print(f" Sharpe Ratio (approx):   {sharpe:.2f}")
print(f" Total Trades:            {total_trades}")
print(f" Win Rate:                {successful_trades}/{total_trades} = {(successful_trades/total_trades):.1%}" if total_trades > 0 else " Win Rate:                N/A")
print(f" Timespan:                {n_years:.1f} years ({len(common_dates)} trading days)")
if total_trades > 0:
    buy_count = sum(1 for t in trade_log if t["Side"] == "BUY")
    sell_count = sum(1 for t in trade_log if t["Side"] == "SELL")
    print(f" BUY trades:              {buy_count}")
    print(f" SELL trades:             {sell_count}")
print(f"{'='*60}")


# ==========================================================================
# Plotting
# ==========================================================================

def save_final_plot(
    dates,
    random_results,
    strat_values,
    random_mean=None,
    random_std=None,
    show_uncertainty=False,
    filename="final_plot.png"
):
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Plot random runs faintly
    for r in range(random_results.shape[1]):
        ax.plot(dates, random_results[:, r], alpha=0.15, lw=1, color="white")

    # Plot uncertainty shading (if enabled)
    if show_uncertainty and random_mean is not None and random_std is not None:
        upper_3s = random_mean + 3 * random_std
        lower_3s = np.maximum(random_mean - 3 * random_std, 1e-6)
        upper_1s = random_mean + random_std
        lower_1s = np.maximum(random_mean - random_std, 1e-6)
        ax.fill_between(dates, lower_3s, upper_3s, color="gray", alpha=0.08)
        ax.fill_between(dates, lower_1s, upper_1s, color="gray", alpha=0.20)

    # Random baseline mean
    if random_mean is not None:
        ax.plot(dates, random_mean, color="white", lw=2, alpha=0.7, label="Random Mean")

    # Strategy line
    ax.plot(dates, strat_values, color="#39FF14", lw=3, label="AI Strategy")

    # Buy markers (green)
    if buy_points:
        bp_dates = [bp["Date"] for bp in buy_points]
        bp_vals  = [bp["Value"] for bp in buy_points]
        ax.scatter(bp_dates, bp_vals, color="#39FF14", s=12, zorder=5, alpha=0.4, label="BUY")

    # Sell markers (red)
    if sell_points:
        sp_dates = [sp["Date"] for sp in sell_points]
        sp_vals  = [sp["Value"] for sp in sell_points]
        ax.scatter(sp_dates, sp_vals, color="#FF3939", s=12, zorder=5, alpha=0.4, marker="v", label="SELL")

    # Log scale for long timespan
    ax.set_yscale("log")
    ax.set_xlim(dates[0], dates[-1])

    ax.set_title("AI Strategy vs Random Baseline (Fair Comparison)", color="white", fontsize=22)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Portfolio Value (log scale)", color="white")
    ax.tick_params(colors="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    # 1.0 reference line
    ax.axhline(y=1.0, color="gray", lw=1, ls="--", alpha=0.4)

    legend = ax.legend(facecolor="black", edgecolor="white", fontsize=12, loc="upper left")
    for text in legend.get_texts():
        text.set_color("white")

    # Save PNG
    png_path = video_dir / filename
    plt.savefig(png_path, dpi=200, facecolor="black")
    force_png_rgb(png_path)
    plt.close(fig)

    print(f"Saved: {png_path}")


save_final_plot(
    common_dates,
    random_results,
    strategy_history_arr,
    show_uncertainty=False,
    filename="random_vs_prob_strategy_clean.png"
)

save_final_plot(
    common_dates,
    random_results,
    strategy_history_arr,
    random_mean=random_mean,
    random_std=random_std,
    show_uncertainty=True,
    filename="random_vs_prob_strategy_uncertainty.png"
)
