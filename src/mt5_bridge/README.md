# MT5 Trading Bridge (Pure Bridge Mode)

**⚠️ WARNING: This is a PURE BRIDGE with NO risk management, validation, or safety checks. All risk management must be implemented in the .NET AI Agent.**

## Overview

This Python bridge provides direct HTTP API access to MetaTrader 5 trading functions. It operates as a pass-through layer with minimal logic:

- ✅ **No risk management** - All risk logic handled by .NET Agent
- ✅ **No validation** - Trades executed as requested
- ✅ **No kill switch** - Full control in agent
- ✅ **Logging only** - Audit trail maintained
- ✅ **Simple REST API** - Clean HTTP endpoints

## Architecture

```
.NET AI Agent (Risk Management)
        ↓ HTTP
Python Bridge (Pass-through)
        ↓ Python API
MT5 Terminal
        ↓
Broker
```

## Quick Start

### 1. Install Dependencies

```powershell
cd trader
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

```powershell
cp .env.example .env
notepad .env
```

Set your API key:
```ini
API_KEY=new_strong_key
```

### 3. Start Bridge

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8787
```

Or from the root directory:
```powershell
.\run.ps1
```

## API Endpoints

All endpoints require `Authorization: Bearer <API_KEY>` header.

### Health Check
```http
GET /health
```

### Account Info
```http
GET /account
```

### Get Quote
```http
GET /quote?symbol=EURUSD
```

### Get Positions
```http
GET /positions
GET /positions?symbol=EURUSD
```

### Place Order
```http
POST /orders
Content-Type: application/json

{
  "symbol": "EURUSD",
  "side": "BUY",
  "volume": 0.01,
  "sl": 1.0800,
  "tp": 1.0900,
  "comment": "AI Agent Order"
}
```

### Modify Position
```http
POST /orders/modify
Content-Type: application/json

{
  "ticket": 123456,
  "sl": 1.0810,
  "tp": 1.0910
}
```

### Close Position
```http
POST /orders/close
Content-Type: application/json

{
  "ticket": 123456,
  "comment": "Close by AI"
}
```

### Cancel Pending Order
```http
POST /orders/cancel
Content-Type: application/json

{
  "ticket": 123456
}
```

## Logging

All requests are logged to:
- `logs/audit_YYYYMMDD.jsonl` - JSONL audit trail
- Console output - Real-time activity

## Important Notes

1. **No Safety Checks** - This bridge executes all valid requests immediately
2. **Risk Management Required** - Implement all risk logic in the .NET Agent
3. **No Order Validation** - Invalid orders will be rejected by MT5, not the bridge
4. **Authentication Only** - API key is the only security layer
5. **MT5 Must Be Running** - Terminal must be open and logged in

## Configuration

Environment variables in `.env`:

```ini
# API Security
API_KEY=your_secure_api_key_here

# Logging
LOG_DIR=./logs

# MT5 Connection (optional - uses logged-in terminal by default)
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
```

## Error Handling

The bridge returns standard HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid parameters)
- `401` - Unauthorized (invalid API key)
- `404` - Symbol not found
- `500` - MT5 error or internal error
- `503` - MT5 not initialized

Error response format:
```json
{
  "error": {
    "code": "ORDER_SEND_FAILED",
    "message": "Descriptive error message"
  }
}
```

## Version

**Version 2.0.0** - Pure Bridge Mode (No Risk Management)
