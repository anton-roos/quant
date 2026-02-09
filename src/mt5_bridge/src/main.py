from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import Optional
import numpy as np
import asyncio
import json
import secrets
import os

from .models import (
    PlaceOrderRequest, ModifyOrderRequest, CloseOrderRequest, CancelOrderRequest,
    OrderResponse, ModifyResponse, CloseResponse, PositionsResponse,
    HealthResponse, AccountInfo, QuoteInfo,
    ErrorResponse, ErrorDetail, ErrorCode, OrderStatus,
    CandlesResponse, CandleInfo, DealsResponse, OrdersHistoryResponse, SymbolInfo,
    PendingOrderRequest, TrailingStopRequest, TicksResponse,
    RSIResponse, MACDResponse, BollingerBandsResponse, MovingAverageResponse, ATRResponse,
    TradeAnalytics, EquityCurveResponse
)
from .mt5_client import MT5Client
from .storage import setup_logging, JSONLLogger
from .indicators import (
    calculate_rsi, calculate_sma, calculate_ema, calculate_macd,
    calculate_bollinger_bands, calculate_atr
)


class Settings(BaseSettings):
    """Application settings from environment variables."""
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 8787
    log_dir: str = "./logs"
    
    # API key for authentication (set via BRIDGE_API_KEY env var)
    # If empty/None, authentication is disabled (local dev mode)
    bridge_api_key: Optional[str] = None
    
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    
    @field_validator('mt5_login', 'mt5_password', 'mt5_server', 'bridge_api_key', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        """Convert empty strings to None."""
        if v == '' or v is None:
            return None
        return v


settings = Settings()

class AppState:
    """Global application state."""
    mt5_client: Optional[MT5Client] = None
    jsonl_logger: Optional[JSONLLogger] = None
    logger = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    state.logger = setup_logging(settings.log_dir)
    state.logger.info("Starting MT5 Bridge (Pure Bridge Mode - No Risk Management)...")
    state.mt5_client = MT5Client(
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server
    )
    state.jsonl_logger = JSONLLogger(settings.log_dir)
    result = state.mt5_client.initialize()
    if result.success:
        state.logger.info(f"MT5 initialized successfully: {result.data}")
    else:
        state.logger.error(f"MT5 initialization failed: {result.error_message}")
    state.logger.info(f"Bridge listening on {settings.bridge_host}:{settings.bridge_port}")
    yield
    state.logger.info("Shutting down MT5 Bridge...")
    if state.mt5_client:
        state.mt5_client.shutdown()
    state.logger.info("Bridge stopped.")

app = FastAPI(
    title="MT5 Trading Bridge",
    description="MT5 Trading Bridge - Local execution bridge for MetaTrader 5",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# API Key Authentication Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    """Verify API key on every request if BRIDGE_API_KEY is configured."""
    api_key = settings.bridge_api_key
    if api_key is not None:
        # Allow health check without auth
        if request.url.path not in ("/", "/health"):
            provided_key = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(provided_key, api_key):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"code": "UNAUTHORIZED", "message": "Invalid or missing X-API-Key header"}}
                )
    response = await call_next(request)
    return response


def error_response(code: ErrorCode, message: str, status_code: int = 400) -> JSONResponse:
    """Create standardized error response."""
    error = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump()
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MT5 Trading Bridge",
        "version": "2.0.0",
        "status": "running",
        "mode": "pure_bridge",
        "description": "Pure bridge with no risk management or validation",
        "endpoints": {
            "health": "GET /health",
            "account": "GET /account",
            "quote": "GET /quote?symbol=EURUSD",
            "symbols": "GET /symbols?group=*USD*",
            "symbol_info": "GET /symbol/EURUSD",
            "positions": "GET /positions?symbol=EURUSD",
            "candles": "GET /candles?symbol=EURUSD&timeframe=M15&count=100",
            "ticks": "GET /ticks?symbol=EURUSD&count=100",
            "indicators": {
                "rsi": "GET /indicators/rsi?symbol=EURUSD&period=14",
                "macd": "GET /indicators/macd?symbol=EURUSD",
                "bollinger": "GET /indicators/bollinger?symbol=EURUSD&period=20",
                "ma": "GET /indicators/ma?symbol=EURUSD&period=50&type=SMA",
                "atr": "GET /indicators/atr?symbol=EURUSD&period=14"
            },
            "analytics": {
                "trades": "GET /analytics/trades?from=2026-02-01&to=2026-02-04",
                "equity_curve": "GET /analytics/equity-curve?from=2026-02-01&to=2026-02-04"
            },
            "history_deals": "GET /history/deals?from=2026-02-01&to=2026-02-04",
            "history_orders": "GET /history/orders?from=2026-02-01&to=2026-02-04",
            "place_order": "POST /orders",
            "place_pending": "POST /orders/pending",
            "modify_order": "POST /orders/modify",
            "close_order": "POST /orders/close",
            "cancel_order": "POST /orders/cancel"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns bridge status, MT5 connection status, and terminal info.
    """
    if not state.mt5_client:
        return HealthResponse(
            ok=False,
            mt5_initialized=False
        )
    
    is_initialized = state.mt5_client.is_initialized()
    terminal_info = None
    server_time = None
    
    if is_initialized:
        result = state.mt5_client.get_account_info()
        if result.success:
            import MetaTrader5 as mt5
            term_info = mt5.terminal_info()
            if term_info:
                terminal_info = {
                    "name": term_info.name,
                    "version": term_info.build,
                    "company": term_info.company,
                    "connected": term_info.connected
                }
            
            from datetime import datetime
            server_time = datetime.utcnow().isoformat() + "Z"
    
    return HealthResponse(
        ok=is_initialized,
        mt5_initialized=is_initialized,
        terminal=terminal_info,
        server_time=server_time
    )


@app.get("/account", response_model=AccountInfo)
async def get_account():
    """
    Get account information.
    Returns balance, equity, margin, and open positions count.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.get_account_info()
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint="/account",
            method="GET",
            request_data={},
            response_data={},
            status_code=500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint="/account",
        method="GET",
        request_data={},
        response_data=result.data,
        status_code=200
    )
    
    return AccountInfo(**result.data)


@app.get("/quote", response_model=QuoteInfo)
async def get_quote(symbol: str):
    """
    Get current price quote for a symbol.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.get_quote(symbol)
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint="/quote",
            method="GET",
            request_data={"symbol": symbol},
            response_data={},
            status_code=404 if result.error_code == ErrorCode.SYMBOL_NOT_FOUND else 500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=404 if result.error_code == ErrorCode.SYMBOL_NOT_FOUND else 500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint="/quote",
        method="GET",
        request_data={"symbol": symbol},
        response_data=result.data,
        status_code=200
    )
    
    return QuoteInfo(**result.data)


@app.get("/positions", response_model=PositionsResponse)
async def get_positions(symbol: Optional[str] = None):
    """
    Get open positions, optionally filtered by symbol.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.get_positions(symbol if symbol else None)
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint="/positions",
            method="GET",
            request_data={"symbol": symbol},
            response_data={},
            status_code=500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint="/positions",
        method="GET",
        request_data={"symbol": symbol},
        response_data=result.data,
        status_code=200
    )
    
    return PositionsResponse(**result.data)


@app.get("/symbols")
async def get_symbols(group: str = "*"):
    """
    Get list of available trading symbols.
    Useful for discovering what symbols are available in MT5.
    
    Args:
        group: Filter pattern (e.g., "*USD*" for all USD pairs, "Forex*" for forex, "*" for all)
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.get_symbols(group)
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint="/symbols",
            method="GET",
            request_data={"group": group},
            response_data={},
            status_code=500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint="/symbols",
        method="GET",
        request_data={"group": group},
        response_data=result.data,
        status_code=200
    )
    
    return result.data


@app.post("/orders", response_model=OrderResponse)
async def place_order(order: PlaceOrderRequest):
    """
    Place a market order.
    Pure bridge - no risk checks or validation.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        state.jsonl_logger.log_event(
            action="PLACE_ORDER",
            request_data=order.model_dump(),
            response_data={},
            result="ERROR",
            error_code=ErrorCode.MT5_NOT_INITIALIZED.value,
            error_message="MT5 not initialized"
        )
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    # Place order directly - no checks
    result = state.mt5_client.place_market_order(
        symbol=order.symbol,
        side=order.side,
        volume=order.volume,
        sl=order.sl,
        tp=order.tp,
        deviation=order.deviation,
        magic=order.magic,
        comment=order.comment
    )
    
    if not result.success:
        state.jsonl_logger.log_event(
            action="PLACE_ORDER",
            request_data=order.model_dump(),
            response_data={},
            result="ERROR",
            error_code=result.error_code.value,
            error_message=result.error_message
        )
        raise HTTPException(
            status_code=400,
            detail=result.error_message
        )
    
    # Log success
    response = OrderResponse(
        accepted=True,
        ticket=result.data['ticket'],
        status=OrderStatus.FILLED,
        price=result.data['price'],
        time=result.data['time'],
        volume=result.data['volume']
    )
    
    state.jsonl_logger.log_event(
        action="PLACE_ORDER",
        request_data=order.model_dump(),
        response_data=response.model_dump(),
        result="SUCCESS"
    )
    
    return response


@app.post("/orders/modify", response_model=ModifyResponse)
async def modify_order(request: ModifyOrderRequest):
    """
    Modify SL/TP of an existing position.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.modify_position_sl_tp(
        ticket=request.ticket,
        sl=request.sl,
        tp=request.tp
    )
    
    if not result.success:
        state.jsonl_logger.log_event(
            action="MODIFY_ORDER",
            request_data=request.model_dump(),
            response_data={},
            result="ERROR",
            error_code=result.error_code.value,
            error_message=result.error_message
        )
        raise HTTPException(
            status_code=400,
            detail=result.error_message
        )
    
    response = ModifyResponse(
        success=True,
        ticket=request.ticket,
        message="Position modified successfully"
    )
    
    state.jsonl_logger.log_event(
        action="MODIFY_ORDER",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        result="SUCCESS"
    )
    
    return response


@app.post("/orders/close", response_model=CloseResponse)
async def close_order(request: CloseOrderRequest):
    """
    Close an open position.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.close_position(
        ticket=request.ticket,
        deviation=request.deviation,
        comment=request.comment
    )
    
    if not result.success:
        state.jsonl_logger.log_event(
            action="CLOSE_ORDER",
            request_data=request.model_dump(),
            response_data={},
            result="ERROR",
            error_code=result.error_code.value,
            error_message=result.error_message
        )
        raise HTTPException(
            status_code=400,
            detail=result.error_message
        )
    
    response = CloseResponse(
        success=True,
        ticket=request.ticket,
        close_price=result.data.get('close_price'),
        profit=result.data.get('profit'),
        message="Position closed successfully"
    )
    
    state.jsonl_logger.log_event(
        action="CLOSE_ORDER",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        result="SUCCESS"
    )
    
    return response


@app.post("/orders/cancel", response_model=CloseResponse)
async def cancel_order(request: CancelOrderRequest):
    """
    Cancel a pending order.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.cancel_pending_order(ticket=request.ticket)
    
    if not result.success:
        state.jsonl_logger.log_event(
            action="CANCEL_ORDER",
            request_data=request.model_dump(),
            response_data={},
            result="ERROR",
            error_code=result.error_code.value,
            error_message=result.error_message
        )
        raise HTTPException(
            status_code=400,
            detail=result.error_message
        )
    
    response = CloseResponse(
        success=True,
        ticket=request.ticket,
        message="Order cancelled successfully"
    )
    
    state.jsonl_logger.log_event(
        action="CANCEL_ORDER",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        result="SUCCESS"
    )
    
    return response


@app.get("/candles", response_model=CandlesResponse)
async def get_candles(
    symbol: str,
    timeframe: str = "H1",
    count: int = 100,
    from_date: Optional[str] = None
):
    """
    Get historical candle/OHLC data for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., EURUSD)
        timeframe: Timeframe - M1, M5, M15, M30, H1, H4, D1, W1, MN1
        count: Number of candles (default 100, max 50000)
        from_date: Start date in ISO format (e.g., 2026-02-01T00:00:00)
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    # Parse from_date if provided
    parsed_date = None
    if from_date:
        try:
            from datetime import datetime
            parsed_date = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {from_date}. Use ISO format like 2026-02-01T00:00:00"
            )
    
    result = state.mt5_client.get_candles(
        symbol=symbol,
        timeframe=timeframe.upper(),
        count=min(count, 50000),  # Cap at 50000
        from_date=parsed_date
    )
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint="/candles",
            method="GET",
            request_data={"symbol": symbol, "timeframe": timeframe, "count": count},
            response_data={},
            status_code=400 if result.error_code == ErrorCode.INVALID_PARAMETER else 500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=400 if result.error_code == ErrorCode.INVALID_PARAMETER else 500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint="/candles",
        method="GET",
        request_data={"symbol": symbol, "timeframe": timeframe, "count": count},
        response_data={"count": result.data['count']},
        status_code=200
    )
    
    return CandlesResponse(**result.data)


@app.get("/symbol/{symbol}", response_model=SymbolInfo)
async def get_symbol_info(symbol: str):
    """
    Get detailed information about a trading symbol.
    
    Args:
        symbol: Trading symbol (e.g., EURUSD)
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    result = state.mt5_client.get_symbol_info(symbol)
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint=f"/symbol/{symbol}",
            method="GET",
            request_data={"symbol": symbol},
            response_data={},
            status_code=404 if result.error_code == ErrorCode.SYMBOL_NOT_FOUND else 500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=404 if result.error_code == ErrorCode.SYMBOL_NOT_FOUND else 500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint=f"/symbol/{symbol}",
        method="GET",
        request_data={"symbol": symbol},
        response_data=result.data,
        status_code=200
    )
    
    return SymbolInfo(**result.data)


@app.get("/history/deals", response_model=DealsResponse)
async def get_deals_history(
    from_date: str,
    to_date: str,
    symbol: Optional[str] = None
):
    """
    Get deal history for a date range.
    
    Args:
        from_date: Start date in ISO format (e.g., 2026-02-01 or 2026-02-01T00:00:00)
        to_date: End date in ISO format
        symbol: Optional symbol filter
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    try:
        from datetime import datetime
        parsed_from = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
        parsed_to = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {str(e)}. Use ISO format like 2026-02-01T00:00:00"
        )
    
    result = state.mt5_client.get_deals_history(
        from_date=parsed_from,
        to_date=parsed_to,
        symbol=symbol if symbol else None
    )
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint="/history/deals",
            method="GET",
            request_data={"from": from_date, "to": to_date, "symbol": symbol},
            response_data={},
            status_code=500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint="/history/deals",
        method="GET",
        request_data={"from": from_date, "to": to_date, "symbol": symbol},
        response_data={"count": result.data['count']},
        status_code=200
    )
    
    return DealsResponse(**result.data)


@app.get("/history/orders", response_model=OrdersHistoryResponse)
async def get_orders_history(
    from_date: str,
    to_date: str,
    symbol: Optional[str] = None
):
    """
    Get order history for a date range.
    
    Args:
        from_date: Start date in ISO format (e.g., 2026-02-01 or 2026-02-01T00:00:00)
        to_date: End date in ISO format
        symbol: Optional symbol filter
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    try:
        from datetime import datetime
        parsed_from = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
        parsed_to = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {str(e)}. Use ISO format like 2026-02-01T00:00:00"
        )
    
    result = state.mt5_client.get_orders_history(
        from_date=parsed_from,
        to_date=parsed_to,
        symbol=symbol if symbol else None
    )
    
    if not result.success:
        state.jsonl_logger.log_request(
            endpoint="/history/orders",
            method="GET",
            request_data={"from": from_date, "to": to_date, "symbol": symbol},
            response_data={},
            status_code=500,
            error=result.error_message
        )
        raise HTTPException(
            status_code=500,
            detail=result.error_message
        )
    
    state.jsonl_logger.log_request(
        endpoint="/history/orders",
        method="GET",
        request_data={"from": from_date, "to": to_date, "symbol": symbol},
        response_data={"count": result.data['count']},
        status_code=200
    )
    
    return OrdersHistoryResponse(**result.data)


@app.get("/ticks", response_model=TicksResponse)
async def get_ticks(
    symbol: str,
    count: int = 100,
    from_date: Optional[str] = None
):
    """
    Get historical tick data (every price change).
    
    Args:
        symbol: Trading symbol (e.g., EURUSD)
        count: Number of ticks (default 100, max 100000)
        from_date: Start date in ISO format (optional)
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="MT5 not initialized"
        )
    
    parsed_date = None
    if from_date:
        try:
            from datetime import datetime
            parsed_date = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {from_date}"
            )
    
    result = state.mt5_client.get_ticks(
        symbol=symbol,
        count=min(count, 100000),
        from_date=parsed_date
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)
    
    return TicksResponse(**result.data)


@app.get("/indicators/rsi", response_model=RSIResponse)
async def get_rsi(symbol: str, period: int = 14, timeframe: str = "H1", count: int = 100):
    """Calculate RSI indicator for a symbol."""
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    # Get candles
    candles_result = state.mt5_client.get_candles(symbol, timeframe.upper(), count + period)
    if not candles_result.success:
        raise HTTPException(status_code=500, detail=candles_result.error_message)
    
    closes = [c['close'] for c in candles_result.data['candles']]
    times = [c['time'] for c in candles_result.data['candles']]
    
    rsi_values = calculate_rsi(closes, period)
    
    data = []
    for i, (time, value) in enumerate(zip(times, rsi_values)):
        if not np.isnan(value):
            data.append({"time": time, "value": round(value, 4)})
    
    return RSIResponse(symbol=symbol, period=period, data=data)


@app.get("/indicators/macd", response_model=MACDResponse)
async def get_macd(symbol: str, fast: int = 12, slow: int = 26, signal: int = 9, 
                   timeframe: str = "H1", count: int = 100):
    """Calculate MACD indicator for a symbol."""
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    candles_result = state.mt5_client.get_candles(symbol, timeframe.upper(), count + slow + signal)
    if not candles_result.success:
        raise HTTPException(status_code=500, detail=candles_result.error_message)
    
    closes = [c['close'] for c in candles_result.data['candles']]
    times = [c['time'] for c in candles_result.data['candles']]
    
    macd_line, signal_line, histogram = calculate_macd(closes, fast, slow, signal)
    
    data = []
    for time, macd, sig, hist in zip(times, macd_line, signal_line, histogram):
        if not np.isnan(macd) and not np.isnan(sig):
            data.append({
                "time": time,
                "values": {
                    "macd": round(macd, 5),
                    "signal": round(sig, 5),
                    "histogram": round(hist, 5)
                }
            })
    
    return MACDResponse(symbol=symbol, fast_period=fast, slow_period=slow, 
                       signal_period=signal, data=data)


@app.get("/indicators/bollinger", response_model=BollingerBandsResponse)
async def get_bollinger_bands(symbol: str, period: int = 20, deviation: float = 2.0,
                              timeframe: str = "H1", count: int = 100):
    """Calculate Bollinger Bands for a symbol."""
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    candles_result = state.mt5_client.get_candles(symbol, timeframe.upper(), count + period)
    if not candles_result.success:
        raise HTTPException(status_code=500, detail=candles_result.error_message)
    
    closes = [c['close'] for c in candles_result.data['candles']]
    times = [c['time'] for c in candles_result.data['candles']]
    
    upper, middle, lower = calculate_bollinger_bands(closes, period, deviation)
    
    data = []
    for time, u, m, l in zip(times, upper, middle, lower):
        if not np.isnan(u):
            data.append({
                "time": time,
                "values": {
                    "upper": round(u, 5),
                    "middle": round(m, 5),
                    "lower": round(l, 5)
                }
            })
    
    return BollingerBandsResponse(symbol=symbol, period=period, deviation=deviation, data=data)


@app.get("/indicators/ma", response_model=MovingAverageResponse)
async def get_moving_average(symbol: str, period: int = 50, ma_type: str = "SMA",
                            timeframe: str = "H1", count: int = 100):
    """Calculate Moving Average (SMA or EMA) for a symbol."""
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    if ma_type.upper() not in ["SMA", "EMA"]:
        raise HTTPException(status_code=400, detail="ma_type must be SMA or EMA")
    
    candles_result = state.mt5_client.get_candles(symbol, timeframe.upper(), count + period)
    if not candles_result.success:
        raise HTTPException(status_code=500, detail=candles_result.error_message)
    
    closes = [c['close'] for c in candles_result.data['candles']]
    times = [c['time'] for c in candles_result.data['candles']]
    
    if ma_type.upper() == "SMA":
        ma_values = calculate_sma(closes, period)
    else:
        ma_values = calculate_ema(closes, period)
    
    data = []
    for time, value in zip(times, ma_values):
        if not np.isnan(value):
            data.append({"time": time, "value": round(value, 5)})
    
    return MovingAverageResponse(symbol=symbol, period=period, ma_type=ma_type.upper(), data=data)


@app.get("/indicators/atr", response_model=ATRResponse)
async def get_atr(symbol: str, period: int = 14, timeframe: str = "H1", count: int = 100):
    """Calculate ATR (Average True Range) indicator for a symbol."""
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    candles_result = state.mt5_client.get_candles(symbol, timeframe.upper(), count + period)
    if not candles_result.success:
        raise HTTPException(status_code=500, detail=candles_result.error_message)
    
    highs = [c['high'] for c in candles_result.data['candles']]
    lows = [c['low'] for c in candles_result.data['candles']]
    closes = [c['close'] for c in candles_result.data['candles']]
    times = [c['time'] for c in candles_result.data['candles']]
    
    atr_values = calculate_atr(highs, lows, closes, period)
    
    data = []
    for time, value in zip(times, atr_values):
        if not np.isnan(value):
            data.append({"time": time, "value": round(value, 5)})
    
    return ATRResponse(symbol=symbol, period=period, data=data)


@app.get("/analytics/trades", response_model=TradeAnalytics)
async def get_trade_analytics(from_date: str, to_date: str):
    """
    Get trading performance analytics.
    
    Args:
        from_date: Start date in ISO format
        to_date: End date in ISO format
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    try:
        from datetime import datetime
        parsed_from = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
        parsed_to = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    
    result = state.mt5_client.calculate_trade_analytics(parsed_from, parsed_to)
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)
    
    return TradeAnalytics(**result.data)


@app.get("/analytics/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(from_date: str, to_date: str):
    """
    Get equity curve from trade history.
    
    Args:
        from_date: Start date in ISO format
        to_date: End date in ISO format
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    try:
        from datetime import datetime
        parsed_from = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
        parsed_to = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    
    result = state.mt5_client.get_equity_curve(parsed_from, parsed_to)
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)
    
    return EquityCurveResponse(**result.data)


@app.post("/orders/pending", response_model=OrderResponse)
async def place_pending_order(order: PendingOrderRequest):
    """
    Place a pending order (limit or stop).
    
    Args:
        order: Pending order request with type (LIMIT/STOP), price, etc.
    """
    if not state.mt5_client or not state.mt5_client.is_initialized():
        raise HTTPException(status_code=503, detail="MT5 not initialized")
    
    # Parse expiration if provided
    expiration = None
    if order.expiration:
        try:
            from datetime import datetime
            expiration = datetime.fromisoformat(order.expiration.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expiration date format")
    
    result = state.mt5_client.place_pending_order(
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        volume=order.volume,
        price=order.price,
        sl=order.sl,
        tp=order.tp,
        expiration=expiration,
        magic=order.magic,
        comment=order.comment
    )
    
    if not result.success:
        state.jsonl_logger.log_event(
            action="PLACE_PENDING_ORDER",
            request_data=order.model_dump(),
            response_data={},
            result="ERROR",
            error_code=result.error_code.value,
            error_message=result.error_message
        )
        raise HTTPException(status_code=400, detail=result.error_message)
    
    response = OrderResponse(
        accepted=True,
        ticket=result.data['ticket'],
        status=OrderStatus.PENDING,
        price=result.data['price'],
        time=result.data['time'],
        volume=result.data['volume']
    )
    
    state.jsonl_logger.log_event(
        action="PLACE_PENDING_ORDER",
        request_data=order.model_dump(),
        response_data=response.model_dump(),
        result="SUCCESS"
    )
    
    return response


@app.websocket("/ws/ticks/{symbol}")
async def websocket_ticks(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time tick streaming.
    Streams live price updates for a symbol.
    
    Usage:
        ws://localhost:8787/ws/ticks/EURUSD
    """
    await websocket.accept()
    
    if not state.mt5_client or not state.mt5_client.is_initialized():
        await websocket.send_json({
            "error": "MT5 not initialized"
        })
        await websocket.close()
        return
    
    # Ensure symbol is selected
    select_result = state.mt5_client.symbol_select(symbol)
    if not select_result.success:
        await websocket.send_json({
            "error": f"Symbol {symbol} not found"
        })
        await websocket.close()
        return
    
    try:
        import MetaTrader5 as mt5
        import pytz
        last_tick_time = 0
        
        while True:
            # Get latest tick
            tick = mt5.symbol_info_tick(symbol)
            
            if tick and tick.time > last_tick_time:
                last_tick_time = tick.time
                
                # Send tick data
                tick_data = {
                    "symbol": symbol,
                    "time": datetime.fromtimestamp(tick.time, tz=pytz.UTC).isoformat(),
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last": tick.last,
                    "volume": tick.volume,
                    "spread": round((tick.ask - tick.bid) / tick.bid * 10000, 2) if tick.bid > 0 else 0
                }
                
                await websocket.send_json(tick_data)
            
            # Sleep briefly to avoid overwhelming the client
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        state.logger.info(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        state.logger.error(f"WebSocket error for {symbol}: {str(e)}")
        await websocket.close()
