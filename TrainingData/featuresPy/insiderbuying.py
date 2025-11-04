import os
import csv
import time
import xml.etree.ElementTree as ET
import pandas as pd
from sec_edgar_downloader import Downloader

def get_cache_file_path(cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "insiderbuying_mostRecentDates.txt")

def read_cache(cache_dir="cache"):
    path = get_cache_file_path(cache_dir)
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        lines = f.readlines()
    return {line.split(',')[0]: line.strip().split(',')[1] for line in lines if ',' in line}

def write_cache(cache_dict, cache_dir="cache"):
    path = get_cache_file_path(cache_dir)
    with open(path, 'w') as f:
        for ticker, date in cache_dict.items():
            f.write(f"{ticker},{date}\n")

def download_form4s(ticker, after, before, outdir, user_agent="user agent"):
    dl = Downloader(user_agent, outdir)
    dl.get("4", ticker, after=after, before=before)
    return os.path.join(outdir, ticker.lower(), "4")

def extract_xml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.find('<?xml')
    end = content.rfind('</ownershipDocument>') + len('</ownershipDocument>')
    return content[start:end] if start != -1 and end != -1 else None

def parse_f4(xml_str):
    try:
        root = ET.fromstring(xml_str)
    except Exception as e:
        print(f" XML parsing failed: {e}")
        return []

    ns = {'ns': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    data = []

    for txn in root.findall('.//nonDerivativeTransaction', ns):
        try:
            code_elem = txn.find('.//transactionCoding/transactionCode', ns)
            if code_elem is None:
                continue
            code = code_elem.text.upper()
            if code not in ['P', 'S']:
                continue

            buy_flag = 1 if code == 'P' else 0
            date = txn.find('.//transactionDate/value', ns).text
            shares = float(txn.find('.//transactionShares/value', ns).text)
            price_el = txn.find('.//transactionPricePerShare/value', ns)
            price = float(price_el.text) if price_el is not None else 0.0
            amount = shares * price
            data.append((date, shares, amount, buy_flag))
        except Exception as e:
            print(f" Error extracting transaction: {e}")
    return data

def aggregate_by_day(transactions):
    daily = {}
    for date, shares, amount, flag in transactions:
        key = (date, flag)
        if key not in daily:
            daily[key] = {"shares": 0.0, "amount": 0.0}
        daily[key]["shares"] += shares
        daily[key]["amount"] += amount
    return sorted([(d, s["shares"], s["amount"], b) for (d, b), s in daily.items()])

def save_csv(rows, outpath):
    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["date", "shares", "amount", "buy_flag"])
        w.writerows(rows)
    print(f" CSV saved to: {outpath}")

def update_insider_buying(ticker, output_dir, fetch_insider_func):
    """
    ticker: str, e.g. 'AAPL'
    output_dir: str, path to output folder
    fetch_insider_func: function(ticker, start_date) -> pd.DataFrame with columns ['date', 'shares', 'amount', 'buy_flag']
    """
    output_path = os.path.join(output_dir, f"{ticker}_insider_trades_daily.csv")
    if os.path.exists(output_path):
        # Load existing file and get the latest date
        existing = pd.read_csv(output_path, parse_dates=['date'])
        if not existing.empty:
            last_date = existing['date'].max()
            # Fetch only new data after last_date
            new_data = fetch_insider_func(ticker, start_date=last_date)
            # Remove any overlap (in case of duplicate last_date)
            new_data = new_data[new_data['date'] > last_date]
            # Append new data if any
            if not new_data.empty:
                updated = pd.concat([existing, new_data], ignore_index=True)
                updated.to_csv(output_path, index=False)
                print(f"Updated {ticker} insider file with {len(new_data)} new rows.")
            else:
                print(f"No new insider data for {ticker}.")
        else:
            # File exists but is empty, fetch all data
            all_data = fetch_insider_func(ticker, start_date=None)
            all_data.to_csv(output_path, index=False)
            print(f"Downloaded all insider data for {ticker}.")
    else:
        # File does not exist, fetch all data
        all_data = fetch_insider_func(ticker, start_date=None)
        all_data.to_csv(output_path, index=False)
        print(f"Downloaded all insider data for {ticker}.")

# Example fetch_insider_func (replace with your actual data fetch logic)
def fetch_insider_func(ticker, start_date=None):
    # Replace this with your actual data fetching logic
    # For demonstration, returns an empty DataFrame
    return pd.DataFrame(columns=['date', 'shares', 'amount', 'buy_flag'])

# Usage:
# update_insider_buying('AAPL', 'TrainingData/indicators_data/raw/insiderBuying', fetch_insider_func)

def main():
    stock_data_dir = "TrainingData/indicators_data/raw/stocksData"
    insider_out_dir = "TrainingData/indicators_data/raw/insiderBuying"
    filings_dir = "./sec-edgar-filings"
    cache_dir = "./cache"

    os.makedirs(insider_out_dir, exist_ok=True)

    cache_dict = read_cache(cache_dir)

    for file in os.listdir(stock_data_dir):
        if not file.endswith("_daily.csv"):
            continue
        ticker = file.replace("_daily.csv", "")
        file_path = os.path.join(stock_data_dir, file)

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values("date")
            after = df['date'].iloc[0].strftime('%Y-%m-%d')
            before = df['date'].iloc[-1].strftime('%Y-%m-%d')
        except Exception as e:
            print(f" Failed reading {file}: {e}")
            continue

        cached_date = cache_dict.get(ticker)
        if cached_date == before:
            print(f" Skipping {ticker}, already up to date ({cached_date})")
            continue

        print(f"\n {ticker}: Downloading Form 4s from {after} to {before}")
        try:
            dir4 = download_form4s(ticker, after, before, filings_dir)
        except Exception as e:
            print(f" Error downloading filings for {ticker}: {e}")
            continue

        all_tx = []
        if not os.path.exists(dir4):
            print(f" No Form 4 data found for {ticker}")
            continue

        for sub in os.listdir(dir4):
            fp = os.path.join(dir4, sub, "full-submission.txt")
            if not os.path.exists(fp):
                continue
            xml = extract_xml(fp)
            if not xml:
                continue
            all_tx.extend(parse_f4(xml))

        if not all_tx:
            print(f" No insider transactions found for {ticker}")
            continue

        daily = aggregate_by_day(all_tx)
        out_csv_path = os.path.join(insider_out_dir, f"{ticker}_insider_trades_daily.csv")

        if os.path.exists(out_csv_path):
            old_df = pd.read_csv(out_csv_path)
            new_df = pd.DataFrame(daily, columns=["date", "shares", "amount", "buy_flag"])
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["date", "buy_flag"])
            combined.to_csv(out_csv_path, index=False)
        else:
            save_csv(daily, out_csv_path)

        cache_dict[ticker] = before
        write_cache(cache_dict, cache_dir)  # flush to disk right after processing
        print(f" Cache updated for {ticker} to {before}")

        time.sleep(1)  # Respect SEC rate limits

    write_cache(cache_dict, cache_dir)

if __name__ == "__main__":
    main()
