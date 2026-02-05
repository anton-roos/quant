# MT5 Trading Bridge - Complete API Reference

## 🎯 NEW FEATURES ADDED

### 1. Technical Indicators (Server-Side Calculation)

#### RSI (Relative Strength Index)
```http
GET /indicators/rsi?symbol=EURUSD&period=14&timeframe=H1&count=100
```
- **period**: RSI period (default: 14)
- **timeframe**: M1, M5, M15, M30, H1, H4, D1, W1, MN1
- **count**: Number of data points

#### MACD (Moving Average Convergence Divergence)
```http
GET /indicators/macd?symbol=EURUSD&fast=12&slow=26&signal=9&timeframe=H1
```
Returns MACD line, signal line, and histogram

#### Bollinger Bands
```http
GET /indicators/bollinger?symbol=EURUSD&period=20&deviation=2.0&timeframe=H1
```
Returns upper, middle, and lower bands

#### Moving Averages (SMA/EMA)
```http
GET /indicators/ma?symbol=EURUSD&period=50&ma_type=SMA&timeframe=H1
```
- **ma_type**: SMA or EMA

#### ATR (Average True Range)
```http
GET /indicators/atr?symbol=EURUSD&period=14&timeframe=H1
```

---

### 2. Real-Time Tick Data Streaming

#### WebSocket for Live Ticks
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8787/ws/ticks/EURUSD');

ws.onmessage = (event) => {
    const tick = JSON.parse(event.data);
    console.log(tick);
    // {
    //   symbol: "EURUSD",
    //   time: "2026-02-04T10:30:45.123Z",
    //   bid: 1.08245,
    //   ask: 1.08257,
    //   last: 1.08251,
    //   volume: 100,
    //   spread: 1.2
    // }
};
```

#### Historical Ticks (REST)
```http
GET /ticks?symbol=EURUSD&count=1000&from_date=2026-02-04T00:00:00
```
Get up to 100,000 historical ticks

---

### 3. Trade Analytics & Performance Metrics

#### Trading Performance Analytics
```http
GET /analytics/trades?from=2026-02-01&to=2026-02-04
```
Returns:
- Total trades, win rate, profit factor
- Average win/loss, largest win/loss
- Max consecutive wins/losses
- Expectancy

**Example Response:**
```json
{
  "total_trades": 45,
  "winning_trades": 28,
  "losing_trades": 17,
  "win_rate": 0.6222,
  "total_profit": 5420.50,
  "total_loss": 2310.25,
  "net_profit": 3110.25,
  "profit_factor": 2.35,
  "average_win": 193.59,
  "average_loss": -135.90,
  "largest_win": 850.00,
  "largest_loss": -420.00,
  "average_trade": 69.12,
  "max_consecutive_wins": 6,
  "max_consecutive_losses": 4,
  "expectancy": 69.12
}
```

#### Equity Curve
```http
GET /analytics/equity-curve?from=2026-02-01&to=2026-02-04
```
Returns:
- Historical equity points (time-series)
- Initial/final balance
- Maximum drawdown (absolute and percentage)

---

### 4. Advanced Order Types

#### Pending Orders (Limit/Stop)
```http
POST /orders/pending
```

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "side": "BUY",
  "order_type": "LIMIT",
  "volume": 0.1,
  "price": 1.08000,
  "sl": 1.07500,
  "tp": 1.09000,
  "expiration": "2026-02-05T23:59:59",
  "magic": 24001,
  "comment": "Buy limit order"
}
```

**Order Types:**
- `LIMIT` - Buy below market / Sell above market
- `STOP` - Buy above market / Sell below market

---

## 📊 Complete Endpoint List

### Market Data
- ✅ `GET /quote` - Current price
- ✅ `GET /candles` - Historical OHLC data
- ✅ `GET /ticks` - Historical tick data
- ✅ `GET /symbols` - Available symbols list
- ✅ `GET /symbol/{symbol}` - Detailed symbol info
- ✅ `WS /ws/ticks/{symbol}` - Real-time tick stream

### Technical Indicators
- ✅ `GET /indicators/rsi` - RSI
- ✅ `GET /indicators/macd` - MACD
- ✅ `GET /indicators/bollinger` - Bollinger Bands
- ✅ `GET /indicators/ma` - Moving Averages (SMA/EMA)
- ✅ `GET /indicators/atr` - ATR

### Account & Positions
- ✅ `GET /health` - System health
- ✅ `GET /account` - Account info
- ✅ `GET /positions` - Open positions

### Trading
- ✅ `POST /orders` - Market order
- ✅ `POST /orders/pending` - Pending order (limit/stop)
- ✅ `POST /orders/modify` - Modify SL/TP
- ✅ `POST /orders/close` - Close position
- ✅ `POST /orders/cancel` - Cancel pending order

### Analytics
- ✅ `GET /analytics/trades` - Performance metrics
- ✅ `GET /analytics/equity-curve` - Equity curve

### History
- ✅ `GET /history/deals` - Trade history
- ✅ `GET /history/orders` - Order history

---

## 🚀 Usage Examples

### Example 1: Get RSI and Make Trading Decision
```python
import requests

# Get RSI
rsi = requests.get('http://localhost:8787/indicators/rsi?symbol=EURUSD&period=14').json()
current_rsi = rsi['data'][-1]['value']

# Trading logic
if current_rsi < 30:  # Oversold
    # Place buy order
    requests.post('http://localhost:8787/orders', json={
        'symbol': 'EURUSD',
        'side': 'BUY',
        'volume': 0.1,
        'sl': 1.07500,
        'tp': 1.09000
    })
```

### Example 2: Monitor Performance
```python
# Get today's performance
analytics = requests.get(
    'http://localhost:8787/analytics/trades',
    params={'from': '2026-02-04T00:00:00', 'to': '2026-02-04T23:59:59'}
).json()

print(f"Win Rate: {analytics['win_rate']*100:.1f}%")
print(f"Profit Factor: {analytics['profit_factor']}")
print(f"Net Profit: ${analytics['net_profit']}")
```

### Example 3: Stream Live Prices
```python
import websocket
import json

def on_message(ws, message):
    tick = json.loads(message)
    print(f"{tick['symbol']}: Bid={tick['bid']}, Ask={tick['ask']}, Spread={tick['spread']}")

ws = websocket.WebSocketApp(
    'ws://localhost:8787/ws/ticks/EURUSD',
    on_message=on_message
)
ws.run_forever()
```

### Example 4: Place Limit Order with Bollinger Bands
```python
# Get Bollinger Bands
bb = requests.get('http://localhost:8787/indicators/bollinger?symbol=EURUSD').json()
lower_band = bb['data'][-1]['values']['lower']

# Place buy limit at lower band
requests.post('http://localhost:8787/orders/pending', json={
    'symbol': 'EURUSD',
    'side': 'BUY',
    'order_type': 'LIMIT',
    'volume': 0.1,
    'price': lower_band,
    'tp': lower_band + 0.00500
})
```

---

## 🔧 Installation

Update dependencies:
```bash
pip install -r requirements.txt
```

New dependencies added:
- `numpy>=1.24.0` - For indicator calculations
- `pandas>=2.0.0` - For data processing
- `websockets>=12.0` - For WebSocket support

---

## 📈 Technical Indicator Details

### RSI
- Range: 0-100
- Overbought: > 70
- Oversold: < 30
- Useful for: Identifying reversal points

### MACD
- Components: MACD line, Signal line, Histogram
- Buy Signal: MACD crosses above Signal
- Sell Signal: MACD crosses below Signal
- Useful for: Trend following

### Bollinger Bands
- Components: Upper, Middle (SMA), Lower
- Price bounces between bands
- Squeeze: Low volatility, potential breakout
- Useful for: Mean reversion strategies

### ATR
- Measures market volatility
- Higher values = more volatility
- Useful for: Position sizing, stop loss placement

---

## 🎯 Trading Strategy Ideas

### Mean Reversion with Bollinger Bands
1. Wait for price to touch lower band (oversold)
2. Confirm with RSI < 30
3. Place buy order
4. Take profit at middle band
5. Stop loss below recent low

### Trend Following with MACD + MA
1. Use 50-period EMA for trend direction
2. Wait for MACD bullish crossover when price > EMA
3. Enter long position
4. Trail stop using ATR (2x ATR below price)
5. Exit when MACD bearish crossover

### Breakout Trading
1. Monitor ATR for low volatility (consolidation)
2. Place pending stop orders above/below recent high/low
3. Use ATR for stop loss distance
4. Target 2-3x ATR for take profit

---

## 🔍 Performance Monitoring

Track your strategy with:
- **Win Rate**: Should be > 40% for trend following, > 60% for mean reversion
- **Profit Factor**: Should be > 1.5 (ideally > 2.0)
- **Expectancy**: Average $ per trade (should be positive)
- **Max Drawdown**: Track risk exposure
- **Consecutive Losses**: Prepare for worst-case scenarios

Use `/analytics/equity-curve` to visualize account growth and drawdowns.

---

## 📝 Notes

- All indicators are calculated server-side for better performance
- WebSocket connections stream data in real-time (~10 updates/second)
- Historical data limits: 50,000 candles, 100,000 ticks
- All times are in UTC/ISO 8601 format
- Pure bridge mode - no built-in risk management (add your own!)
