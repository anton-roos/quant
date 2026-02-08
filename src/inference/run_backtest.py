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

# Config
initial_value   = 1.0
random_runs     = 50
MIN_ACCEPTED    = 0.10      # Only trade when model is highly confident
STD_FACTOR      = 1.0       # AdjustedProb = PredProb - STD_FACTOR * StdDev (penalize uncertainty)

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


# Backtesting strategy:
# Rules:
#   For each date, compute adj_prob = Pred_Prob - STD_FACTOR * Pred_Prob_Std
#   Reject candidates with adj_prob <= MIN_ACCEPTED
#   Out of all candidates, choose the one with the highest adj_prob
#   Only hold 1 position at a time

strategy_value  = initial_value
strategy_history = []
trade_log        = []
buy_points       = []
sell_points      = []

current_hold = None
successful_trades = 0
total_trades      = 0

for i, date in enumerate(common_dates):
    # Exit if horizon has expired
    if current_hold is not None and i == current_hold["exit_idx"]:

        realized_logr = current_hold["Actual_LogR"]
        side = current_hold["Side"]

        # For SELL trades, profit is the inverse of the log return
        effective_logr = -realized_logr if side == "SELL" else realized_logr
        realized_pct  = np.exp(effective_logr) - 1 if not np.isnan(effective_logr) else np.nan

        trade_log.append({
            "BuyDate"        : current_hold["BuyDate"],
            "SellDate"       : date,
            "Symbol"         : current_hold["Symbol"],
            "Side"           : side,
            "Horizon"        : current_hold["Period"],
            "DaysHeld"       : current_hold["Days"],
            "Pred_Prob"      : current_hold["Pred_Prob"],
            "Pred_Prob_Std"  : current_hold["Pred_Prob_Std"],
            "Adj_Prob"       : current_hold["Adj_Prob"],
            "Actual_LogR"    : realized_logr,
            "Effective_LogR"  : effective_logr,
            "Actual_Return%" : realized_pct * 100 if not np.isnan(realized_pct) else np.nan,
        })

        if not np.isnan(effective_logr):
            strategy_value *= np.exp(effective_logr)
            if effective_logr > 0:
                successful_trades += 1

        current_hold = None

    # If still holding a position, skip new entries
    if current_hold is not None:
        strategy_history.append(strategy_value)
        continue

    # Build candidates for today (both BUY and SELL)
    candidates = []

    for symbol, df in all_forecasts.items():

        if date not in df.index:
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

    # Selecting best candidate (highest adj_prob regardless of side)
    best = max(candidates, key=lambda x: x["Adj_Prob"])

    entry_idx = i
    exit_idx  = min(i + best["Days"], len(common_dates) - 1)

    best["entry_idx"] = entry_idx
    best["exit_idx"]  = exit_idx
    best["BuyDate"]   = date

    current_hold = best
    total_trades  += 1

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
strategy_entries = {}
for bp in buy_points:
    idx = date_to_idx[bp["Date"]]
    strategy_entries[idx] = {"horizon": bp["Horizon"], "side": "BUY"}
for sp in sell_points:
    idx = date_to_idx[sp["Date"]]
    strategy_entries[idx] = {"horizon": sp["Horizon"], "side": "SELL"}

random_results = np.zeros((len(common_dates), random_runs))

for run in range(random_runs):
    value = initial_value
    hold_exit_idx = -1
    pending_ret = np.nan

    for i, date in enumerate(common_dates):
        # Exit current position if horizon expired
        if i == hold_exit_idx:
            if not np.isnan(pending_ret):
                value *= np.exp(pending_ret)
            hold_exit_idx = -1
            pending_ret = np.nan

        # Enter new position when strategy would (same timing, random symbol & side)
        if i in strategy_entries and hold_exit_idx == -1:
            entry_info = strategy_entries[i]
            horizon = entry_info["horizon"]
            days = PERIODS[horizon]
            hold_exit_idx = min(i + days, len(common_dates) - 1)

            rets = available_returns[date][horizon]
            if rets:
                raw_ret = random.choice(rets)
                # Random baseline also randomly picks BUY or SELL direction
                rand_side = random.choice(["BUY", "SELL"])
                pending_ret = raw_ret if rand_side == "BUY" else -raw_ret
            else:
                pending_ret = np.nan

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
