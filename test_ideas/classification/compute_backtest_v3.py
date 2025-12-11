import os
import pandas as pd
import numpy as np
import random
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# CONFIG
video_dir = "videos"
forecast_dir = "forecasts"
os.makedirs(video_dir, exist_ok=True)

# Parameters
initial_value = 1.0
random_runs = 100
CONFIDENCE_THRESHOLD = 0.05 
CONFIDENCE_Z = 0.5           # For subtract_std adjustment
UNCERTAINTY_MODE = "subtract_std"
SPIKE_THRESHOLD = 0.3        # Reject stocks with daily abs change > 50%

# Load Forecast Data
all_forecasts = {}
rejected_tickers = []

for filename in os.listdir(forecast_dir):
    if filename.endswith("_forecast.csv"):
        filepath = os.path.join(forecast_dir, filename)
        ticker = filename.split("_forecast")[0]
        df = pd.read_csv(filepath, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df["Ticker"] = ticker

        # Spike filter
        if "Close" in df.columns:
            df["pct_change"] = df["Close"].pct_change()
            max_change = df["pct_change"].abs().max()

            if max_change > SPIKE_THRESHOLD:
                rejected_tickers.append((ticker, max_change))
                continue  # reject this ticker entirely

            # Compute actual returns
            df["Actual_LogR_1d"] = np.log(df["Close"].shift(-1) / df["Close"])
            df["Actual_LogR_1w"] = np.log(df["Close"].shift(-5) / df["Close"])
            df["Actual_LogR_1m"] = np.log(df["Close"].shift(-21) / df["Close"])
            df["Actual_LogR_6m"] = np.log(df["Close"].shift(-126) / df["Close"])

        all_forecasts[ticker] = df

# Report rejected tickers
if rejected_tickers:
    print("\nRejected tickers due to unrealistic spikes (>50% daily change):")
    for t, m in rejected_tickers:
        print(f"  - {t}: max daily change = {m:.2%}")
else:
    print("\nNo stocks rejected for excessive daily change.")

# Common Dates
if not all_forecasts:
    raise ValueError("No valid forecast files found after filtering.")
all_dates = [set(df.index) for df in all_forecasts.values()]
common_dates = sorted(set.intersection(*all_dates))
print(f"\nUsing {len(common_dates)} common dates across {len(all_forecasts)} valid stocks")

# Uncertainty adjustment helper
def adjust_prediction(pred, std, mode=UNCERTAINTY_MODE):
    if np.isnan(pred):
        return np.nan
    std = 0.0 if (std is None or np.isnan(std)) else std
    if mode == "subtract_std":
        return pred - CONFIDENCE_Z * std
    elif mode == "sharpe":
        return pred / (1 + std / (abs(pred) + 1e-9))
    elif mode == "expected_positive":
        if std == 0:
            return pred
        p_up = 1 - norm.cdf(0, loc=pred, scale=std)
        return p_up * pred
    return pred

# Multi-horizon Adaptive Strategy
strategy_value = initial_value
strategy_history = []
buy_points = []
successful_buys = 0
total_buys = 0

periods = {"1d": 1, "1w": 5, "1m": 21, "6m": 126}
hold_days_remaining = 0
current_hold = None
trade_log = []

for i, date in enumerate(common_dates):

    # If we're holding a position, check if we exit today
    if current_hold is not None and i == current_hold["exit_idx"]:

        realized_logr = current_hold["Actual_LogR"]
        realized_pct = np.exp(realized_logr) - 1 if not np.isnan(realized_logr) else np.nan

        # Log completed trade
        trade_log.append({
            "BuyDate": current_hold["BuyDate"],
            "SellDate": date,
            "Ticker": current_hold["Ticker"],
            "Horizon": current_hold["Period"],
            "DaysHeld": current_hold["Days"],
            "Predicted_LogR": current_hold["Predicted"],
            "Predicted_Std": current_hold["Std"],
            "AdjPred_LogR": current_hold["AdjPred"],
            "Confidence": current_hold["Confidence"],
            "Actual_LogR": realized_logr,
            "Actual_ReturnPct": realized_pct * 100,
        })

        # Portfolio impact
        if not np.isnan(realized_logr):
            strategy_value *= np.exp(realized_logr)

        current_hold = None

    # If still holding after exit check, skip new buys
    if current_hold is not None:
        strategy_history.append(strategy_value)
        continue

    # Build candidates
    candidates = []
    for ticker, df in all_forecasts.items():
        if date not in df.index:
            continue

        row = df.loc[date]

        for label, days in periods.items():
            
            pred_col = f"Pred_LogR_{label}"
            std_col = f"Pred_LogR_PerDayStd_{label}"
            act_col = f"Actual_LogR_{label}"

            if pred_col not in row or np.isnan(row[pred_col]):
                continue

            pred = float(row[pred_col])
            std = float(row.get(std_col, 0.0))
            std = max(std, 1e-4)

            adj_pred = adjust_prediction(pred, std)
            act = float(row.get(act_col, np.nan))
            p_up = 1 - norm.cdf(0, loc=pred, scale=std)

            candidates.append({
                "Ticker": ticker,
                "Period": label,
                "Days": days,
                "Predicted": pred,
                "Std": std,
                "AdjPred": adj_pred,
                "Confidence": p_up,
                "Actual_LogR": act,
                "NormAdj": np.exp(adj_pred / days) - 1
            })

    if not candidates:
        strategy_history.append(strategy_value)
        continue

    best = max(candidates, key=lambda x: x["NormAdj"])

    # Buy under confidence rule
    if best["AdjPred"] > 0 and best["Confidence"] >= CONFIDENCE_THRESHOLD:

        entry_idx = i
        exit_idx = min(i + best["Days"], len(common_dates) - 1)

        best["entry_idx"] = entry_idx
        best["exit_idx"] = exit_idx
        best["BuyDate"] = date

        current_hold = best
        hold_days_remaining = best["Days"]

        buy_points.append({
            "Date": date,
            "Value": strategy_value,
            "Ticker": best["Ticker"],
            "Confidence": best["Confidence"],
            "Predicted": best["Predicted"]
        })

    strategy_history.append(strategy_value)

summary_df = pd.DataFrame(trade_log)
summary_df.to_csv("trade_summary.csv", index=False)

print("\nSaved trade summary to trade_summary.csv")
print(summary_df.head())

# Report Buy Success Rate
if total_buys > 0:
    success_fraction = successful_buys / total_buys
    print(f"\nBuy success rate: {successful_buys}/{total_buys} = {success_fraction:.2%}")
else:
    print("\nNo completed trades to evaluate.")


# Random Baseline
returns_by_date = {}
for date in common_dates:
    daily_data = []
    for ticker, df in all_forecasts.items():
        if date in df.index and "Actual_LogR_1d" in df.columns:
            val = df.loc[date, "Actual_LogR_1d"]
            if not pd.isna(val):
                daily_data.append(val)
    returns_by_date[date] = daily_data

random_results = np.zeros((len(common_dates), random_runs))
for run in range(random_runs):
    value = initial_value
    for i, date in enumerate(common_dates):
        if returns_by_date[date]:
            pick = random.choice(returns_by_date[date])
            value *= np.exp(pick)
        random_results[i, run] = value

random_mean = np.mean(random_results, axis=1)
random_std = np.std(random_results, axis=1)

# SPY Buy & Hold
spy_path = "TrainingData/indicators_data/processed/SPY-VIX/SPY_daily_processed.csv"
spy_df = pd.read_csv(spy_path, parse_dates=["date"])
spy_df = spy_df.rename(columns={"date": "Date", "close": "Close"})
spy_df = spy_df[spy_df["Date"].isin(common_dates)].sort_values("Date").reset_index(drop=True)
spy_df["PortfolioValue"] = initial_value * (spy_df["Close"] / spy_df["Close"].iloc[0])

# Animation
def animate_plot(dates, random_results, strat_values, spy_values,
                 random_mean=None, random_std=None, show_uncertainty=False, filename="out.mp4"):
    dates = np.asarray(dates)
    step = max(1, len(dates) // 300)
    frames = range(0, len(dates), step)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    random_lines = [ax.plot([], [], color="white", alpha=0.10, lw=1)[0] for _ in range(random_results.shape[1])]
    spy_line, = ax.plot([], [], color="white", lw=3, label="SPY Buy & Hold")
    strat_line, = ax.plot([], [], color="#39FF14", lw=3, label="AI Strategy")

    if show_uncertainty and random_mean is not None and random_std is not None:
        ax.fill_between(dates, random_mean - 3*random_std, random_mean + 3*random_std,
                        color="gray", alpha=0.08, label="±3σ Random Range")
        ax.fill_between(dates, random_mean - random_std, random_mean + random_std,
                        color="gray", alpha=0.2, label="±1σ Random Range")

    ymin, ymax = np.nanmin(random_results), np.nanmax(np.concatenate([random_results.flatten(), strat_values, spy_values]))
    margin = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_title("AI Strategy vs Random vs SPY", color="white", fontsize=22)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Portfolio Value", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black", edgecolor="white", fontsize=12)
    for text in ax.get_legend().get_texts():
        text.set_color("white")

    def update(i):
        for r, line in enumerate(random_lines):
            line.set_data(dates[:i], random_results[:i, r])
        spy_line.set_data(dates[:i], spy_values[:i])
        strat_line.set_data(dates[:i], strat_values[:i])
        return [spy_line, strat_line]
        

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    out_path = os.path.join(video_dir, filename)
    writer = FFMpegWriter(fps=30, bitrate=1800)
    anim.save(out_path, writer=writer, dpi=200)
    print(f"Saved animation: {out_path}")
    plt.close(fig)

# Run animations
animate_plot(common_dates, random_results, strategy_history, spy_df["PortfolioValue"].values,
             show_uncertainty=False, filename="random_vs_strategy_clean.mp4")

animate_plot(common_dates, random_results, strategy_history, spy_df["PortfolioValue"].values,
             random_mean, random_std, show_uncertainty=True, filename="random_vs_strategy_uncertainty.mp4")
