import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

RAW_STOCKS_DIR = 'TrainingData/indicators_data/raw/stocksData'
SENTIMENT_DIR = 'TrainingData/indicators_data/raw/sentiment'
os.makedirs(SENTIMENT_DIR, exist_ok=True)

def load_api_key():
    #Load AlphaVantage API key from config.json
    import json, os

    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
            key = cfg.get("ALPHA_VANTAGE_KEY")

            if key is None or key.strip() == "":
                raise ValueError(
                    "\nERROR: Your AlphaVantage API key is missing.\n"
                    "Open config.json and add:\n"
                    '{ "ALPHA_VANTAGE_KEY": "YOUR_KEY_HERE" }\n'
                )
            return key

    except FileNotFoundError:
        raise FileNotFoundError(
            "\nERROR: config.json is missing.\n"
            "Create it (or copy config.example.json) and add your API key.\n"
        )
    
def get_date_range_from_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date')
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    return start_date, end_date

def get_last_sentiment_date(sentiment_path):
    if not os.path.exists(sentiment_path):
        return None
    df = pd.read_csv(sentiment_path, parse_dates=['date'])
    if df.empty:
        return None
    return df['date'].max()

def fetch_sentiment_for_range(ticker, start_date, end_date, sentiment_path, API_KEY):
    # Load existing sentiment if present
    if os.path.exists(sentiment_path):
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=['date'])

        # Force all dates to YYYY-MM-DD format (removes any 00:00:00 time parts)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Overwrite the CSV to clean old entries
        sentiment_df.to_csv(sentiment_path, index=False)

        rows = sentiment_df.to_dict('records')
        existing_dates = set(sentiment_df['date'])  # already in string form like '2025-06-01'

    else:
        rows = []
        existing_dates = set()

    date = start_date
    no_news_streak = 0
    while date <= end_date:
        date_str = date.strftime('%Y-%m-%d')
        if date_str in existing_dates:
            date += timedelta(days=1)
            continue

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': API_KEY,
            'time_from': date.strftime('%Y%m%dT0000'),
            'time_to': date.strftime('%Y%m%dT2359'),
            'sort': 'LATEST',
            'limit': 100
        }
        try:
            r = requests.get('https://www.alphavantage.co/query', params=params)
            data = r.json()
            if 'feed' in data and data['feed']:
                # News found
                no_news_streak = 0
                feed = data['feed']
                num_articles = len(feed)
                scores = [item['overall_sentiment_score'] for item in feed if 'overall_sentiment_score' in item]
                avg_score = sum(scores) / len(scores) if scores else None
                rows.append({'date': date_str, 'sentiment': avg_score, 'num_articles': num_articles})
            elif 'Information' in data and 'No articles found' in data['Information']:
                no_news_streak += 1
                if no_news_streak > 30:
                    print(f"Skipping ahead by 365 days from {date.strftime('%Y-%m-%d')}")
                    date += timedelta(days=365)
                    continue
                # Optionally append None/0 for sentiment and num_articles
                rows.append({'date': date_str, 'sentiment': None, 'num_articles': 0})
            else:
                print(f"API error or limit reached on {date_str} for {ticker}: {data}")
                rows.append({'date': date_str, 'sentiment': None, 'num_articles': 0})
        except Exception as e:
            print(f"Error on {date_str} for {ticker}: {e}")
            rows.append({'date': date_str, 'sentiment': None, 'num_articles': 0})
        finally:
            # Format all dates consistently as strings before saving
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df.to_csv(sentiment_path, index=False)

            date += timedelta(days=1)
            time.sleep(0.85)


    print(f"Saved sentiment data to {sentiment_path}")

def fetch_data_with_retry(api_url, max_retries=10, retry_delay=60):
    retries = 0
    while retries < max_retries:
        response = requests.get(api_url)
        data = response.json()
        if 'Information' in data and 'Thank you for using Alpha Vantage' in data['Information']:
            print(f"API limit reached, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retries += 1
            continue
        if 'feed' in data and not data['feed']:
            print("No data returned, retrying in 30 seconds...")
            time.sleep(1)
            retries += 1
            continue
        return data
    print("Max retries reached. Could not fetch valid data.")
    return None

def main():
    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    total = len(os.listdir(RAW_STOCKS_DIR))
    for idx, filename in enumerate(os.listdir(RAW_STOCKS_DIR), 1):
        if not filename.endswith('_daily.csv'):
            continue
        ticker = filename.split('_')[0]
        stock_csv_path = os.path.join(RAW_STOCKS_DIR, filename)
        sentiment_csv_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment_daily.csv")

        # Get date range from stock data
        stock_start, stock_end = get_date_range_from_csv(stock_csv_path)

        # Check for existing sentiment data
        # Normalize existing sentiment file if it exists
        if os.path.exists(sentiment_csv_path):
            df_existing = pd.read_csv(sentiment_csv_path, parse_dates=['date'])
            df_existing['date'] = pd.to_datetime(df_existing['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df_existing.to_csv(sentiment_csv_path, index=False)

        # Now get the last sentiment date
        last_sentiment_date = get_last_sentiment_date(sentiment_csv_path)
        if last_sentiment_date is not None:
            last_sentiment_date = pd.to_datetime(last_sentiment_date)
            if last_sentiment_date >= stock_end:
                print(f"Sentiment data for {ticker} is already up to date.")
                continue

        # If sentiment exists, start from the day after last sentiment date
        if last_sentiment_date is not None:
            fetch_start = last_sentiment_date + timedelta(days=1)
        else:
            fetch_start = stock_start


        percent = (idx / total) * 100
        print(f"[{idx}/{total}] ({percent:.1f}%) Fetching sentiment for {ticker} from {fetch_start.date()} to {stock_end.date()}")
        fetch_sentiment_for_range(ticker, fetch_start, stock_end, sentiment_csv_path, API_KEY)
        print(f"Completed fetching sentiment for {ticker}")
        

    tickers = [filename.split('_')[0] for filename in os.listdir(RAW_STOCKS_DIR) if filename.endswith('_daily.csv')]
    for ticker in tickers:
        print(f"Starting {ticker}...")
        try:
            api_url = f"https://www.alphavantage.co/query?...&symbol={ticker}"
            data = fetch_data_with_retry(api_url)
            if data is not None:
                # write_data_to_file(ticker, data)
                print(f"Saved sentiment data to ...{ticker}_sentiment_daily.csv")
            else:
                print(f"Failed to fetch data for {ticker} after max retries.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
        print(f"Finished {ticker}")

    print("All tickers processed.")

if __name__ == "__main__":
    API_KEY = load_api_key() 
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Script ran in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")