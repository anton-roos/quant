import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# === CONFIG ===
FORECAST_DIR = "forecasts"
CONFIDENCE_Z = 0.5  # smaller = more trades; larger = more conservative
UNCERTAINTY_MODE = "subtract_std"  # "sharpe", "expected_positive", or "subtract_std"

# === Load Forecast CSVs ===
df_container = []
for file in os.listdir(FORECAST_DIR):
    if file.endswith(".csv"):
        data = pd.read_csv(os.path.join(FORECAST_DIR, file))
        ticker_name = file.replace("_daily_processed_forecast.csv", "")
        df_container.append({"ticker": ticker_name, "data": data})

if not df_container:
    raise ValueError("No CSV files found in forecast directory!")

# === Compute actual log returns if missing ===
for entry in df_container:
    df = entry["data"]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "Close" not in df.columns:
        raise ValueError(f"Missing 'Close' column in {entry['ticker']}!")

    # Compute actual log returns
    df["Actual_LogR_1d"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["Actual_LogR_1w"] = np.log(df["Close"].shift(-5) / df["Close"])
    df["Actual_LogR_1m"] = np.log(df["Close"].shift(-21) / df["Close"])
    df["Actual_LogR_6m"] = np.log(df["Close"].shift(-126) / df["Close"])
    entry["data"] = df

# === Create unified trading calendar ===
all_dates = pd.concat([entry["data"]["Date"] for entry in df_container])
full_calendar = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq="B")

# === Initialize portfolio ===
account_value = 1.0
history = []
buy_signals = []
hold_days_remaining = 0
current_hold = None

periods = {"1d": 1, "1w": 5, "1m": 21, "6m": 126}

# === Helper: Adjust prediction for uncertainty ===
def adjust_prediction(pred, std, mode=UNCERTAINTY_MODE):
    if std is None or np.isnan(std):
        std = 0.0

    if mode == "subtract_std":
        # conservative subtraction approach
        return pred - CONFIDENCE_Z * std

    elif mode == "sharpe":
        # signal-to-noise ratio weighting
        if std == 0:
            return pred
        return pred / (1 + std / (abs(pred) + 1e-9))

    elif mode == "expected_positive":
        # expected return given uncertainty
        if std == 0:
            return pred
        p_positive = 1 - norm.cdf(0, loc=pred, scale=std)
        return p_positive * pred

    else:
        return pred

# === Run backtest ===
trade_count = 0

for current_date in full_calendar:
    if hold_days_remaining > 0:
        hold_days_remaining -= 1
        continue

    # Realize previous trade return
    if current_hold is not None:
        realized_return = current_hold["Actual_LogR"]
        account_value *= np.exp(realized_return)
        current_hold = None

    # Collect candidate predictions
    candidates = []

    for entry in df_container:
        df = entry["data"]
        row = df[df["Date"] == current_date]

        if row.empty:
            continue

        ticker = entry["ticker"]

        for label, days in periods.items():
            pred_col = f"Pred_LogR_{label}"
            std_col = f"Pred_Std_LogR_{label}"
            act_col = f"Actual_LogR_{label}"

            if pred_col in df.columns and not pd.isna(row[pred_col].iloc[0]):
                pred = row[pred_col].iloc[0]
                std = row[std_col].iloc[0] if std_col in df.columns else 0
                adj_pred = adjust_prediction(pred, std)
                act_logr = row[act_col].iloc[0] if act_col in df.columns else np.nan

                candidates.append({
                    "ticker": ticker,
                    "period": label,
                    "days": days,
                    "Predicted_LogR": pred,
                    "Adjusted_Pred": adj_pred,
                    "Actual_LogR": act_logr,
                    "Normalized_Pred": adj_pred / days
                })

    if not candidates:
        history.append({"Date": current_date, "Account_Value": account_value})
        continue

    # Pick the stock with the highest adjusted normalized return
    best = max(candidates, key=lambda x: x["Normalized_Pred"])

    if best["Adjusted_Pred"] > 0:
        current_hold = best
        hold_days_remaining = best["days"]
        buy_signals.append({"Date": current_date, "Value": account_value})
        trade_count += 1

    history.append({"Date": current_date, "Account_Value": account_value})

# === Benchmark: Random 1d strategy ===
random_account = 1.0
random_history = []

for current_date in full_calendar:
    valid_entries = []
    for entry in df_container:
        df = entry["data"]
        row = df[df["Date"] == current_date]
        if not row.empty and "Actual_LogR_1d" in row.columns:
            valid_entries.append(row["Actual_LogR_1d"].iloc[0])

    if valid_entries:
        chosen = np.random.choice(valid_entries)
        random_account *= np.exp(chosen)

    random_history.append({"Date": current_date, "Account_Value": random_account})

random_df = pd.DataFrame(random_history)
history_df = pd.DataFrame(history)
buy_df = pd.DataFrame(buy_signals)

# === Results ===
print(f"Total Trades: {trade_count}")
print(f"Final Portfolio Value: {account_value:.2f}")
print(f"Random Benchmark Value: {random_account:.2f}")

# === Plot ===
plt.style.use("dark_background")
plt.figure(figsize=(14, 7))
plt.plot(history_df["Date"], history_df["Account_Value"], color="white", lw=2, label="Uncertainty-Aware Strategy")
plt.plot(random_df["Date"], random_df["Account_Value"], color="cyan", lw=1.5, alpha=0.7, label="Random Benchmark")

if not buy_df.empty:
    plt.scatter(buy_df["Date"], buy_df["Value"], color="lime", s=30, label="Buy Signal", zorder=5)

plt.title(f"Backtest with Uncertainty Adjustment ({UNCERTAINTY_MODE} mode)", color="white")
plt.xlabel("Date", color="white")
plt.ylabel("Portfolio Value ($)", color="white")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
