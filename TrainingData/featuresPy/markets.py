from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
import time
from datetime import datetime
import json
def update_or_create_csv(ticker, ts, save_folder, delay_sec=1):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filepath = os.path.join(save_folder, f"{ticker}_daily.csv")

    try:
        print(f" Downloading full daily data for {ticker}...")
        new_data, _ = ts.get_daily(symbol=ticker, outputsize='full')

        new_data.reset_index(inplace=True)
        new_data.rename(columns={
            'date': 'date',
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }, inplace=True)

        new_data.sort_values('date', inplace=True)
        new_data.reset_index(drop=True, inplace=True)
        new_data['date'] = pd.to_datetime(new_data['date'])

        if os.path.exists(filepath):
            existing_data = pd.read_csv(filepath, parse_dates=['date'])
            all_data = pd.concat([existing_data, new_data])
            all_data = all_data.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
            print(f"📝 Updating existing file for {ticker} with new dates...")
        else:
            all_data = new_data
            print(f"📁 Creating new file for {ticker}...")

        all_data.to_csv(filepath, index=False)
        print(f" Data for {ticker} saved to {filepath}")
        time.sleep(delay_sec)

    except Exception as e:
        print(f" Failed to download {ticker}: {e}")

def load_api_key():
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
            key = cfg.get("ALPHA_VANTAGE_KEY")
            
            if key is None or key.strip() == "":
                raise ValueError(
                    "\nERROR: Your AlphaVantage API key is missing.\n"
                    "Please open config.json and set:\n"
                    '{ "ALPHA_VANTAGE_KEY": "YOUR_KEY_HERE" }\n'
                )
            return key

    except FileNotFoundError:
        raise FileNotFoundError(
            "\nERROR: config.json is missing.\n"
            "Create a config.json file in the project folder with:\n"
            '{ "ALPHA_VANTAGE_KEY": "YOUR_KEY_HERE" }\n'
        )
    
if __name__ == "__main__":
    api_key = load_api_key()#""  # Replace with your actual key
    tickers = ["SPY", "VIXY"]  # S&P 500 and VIX
    save_folder = "TrainingData/indicators_data/raw/SPY-VIX"
    delay_seconds = 0.8  # Alpha Vantage free tier = 5 API calls/min

    ts = TimeSeries(key=api_key, output_format='pandas')

    for ticker in tickers:
        update_or_create_csv(ticker, ts, save_folder, delay_seconds)
