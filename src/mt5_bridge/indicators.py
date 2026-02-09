"""
Technical indicator calculations using numpy and pandas.
"""
import numpy as np
from typing import List, Tuple


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return []
    
    prices_arr = np.array(prices)
    deltas = np.diff(prices_arr)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Use Wilder's smoothing
    avg_gain = np.zeros(len(gains))
    avg_loss = np.zeros(len(losses))
    
    avg_gain[period-1] = np.mean(gains[:period])
    avg_loss[period-1] = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Prepend NaN for first period values
    result = [np.nan] * period + list(rsi[period-1:])
    return result


def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return []
    
    prices_arr = np.array(prices)
    sma = np.convolve(prices_arr, np.ones(period)/period, mode='valid')
    
    # Prepend NaN for first period-1 values
    result = [np.nan] * (period - 1) + list(sma)
    return result


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return []
    
    prices_arr = np.array(prices)
    ema = np.zeros(len(prices_arr))
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    ema[period-1] = np.mean(prices_arr[:period])
    
    for i in range(period, len(prices_arr)):
        ema[i] = (prices_arr[i] - ema[i-1]) * multiplier + ema[i-1]
    
    result = [np.nan] * (period - 1) + list(ema[period-1:])
    return result


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate MACD indicator.
    Returns: (macd_line, signal_line, histogram)
    """
    if len(prices) < slow + signal:
        return [], [], []
    
    fast_ema = calculate_ema(prices, fast)
    slow_ema = calculate_ema(prices, slow)
    
    # Calculate MACD line
    macd_line = [f - s if not np.isnan(f) and not np.isnan(s) else np.nan 
                 for f, s in zip(fast_ema, slow_ema)]
    
    # Calculate signal line (EMA of MACD)
    valid_macd = [x for x in macd_line if not np.isnan(x)]
    signal_ema = calculate_ema(valid_macd, signal)
    
    # Align signal line with macd line
    signal_line = [np.nan] * (len(macd_line) - len(signal_ema)) + signal_ema
    
    # Calculate histogram
    histogram = [m - s if not np.isnan(m) and not np.isnan(s) else np.nan 
                 for m, s in zip(macd_line, signal_line)]
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: List[float], period: int = 20, 
                               deviation: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands.
    Returns: (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        return [], [], []
    
    middle = calculate_sma(prices, period)
    
    prices_arr = np.array(prices)
    std_dev = []
    
    for i in range(len(prices_arr)):
        if i < period - 1:
            std_dev.append(np.nan)
        else:
            std_dev.append(np.std(prices_arr[i-period+1:i+1]))
    
    upper = [m + (s * deviation) if not np.isnan(m) else np.nan 
             for m, s in zip(middle, std_dev)]
    lower = [m - (s * deviation) if not np.isnan(m) else np.nan 
             for m, s in zip(middle, std_dev)]
    
    return upper, middle, lower


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], 
                  period: int = 14) -> List[float]:
    """Calculate Average True Range."""
    if len(highs) < period + 1:
        return []
    
    # Calculate True Range
    tr = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr.append(max(high_low, high_close, low_close))
    
    # Calculate ATR using Wilder's smoothing
    atr = np.zeros(len(tr))
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    result = [np.nan] * (period) + list(atr[period-1:])
    return result
