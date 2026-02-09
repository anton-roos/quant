"""
Pydantic models for request/response validation and data structures.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order execution status."""
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"


class ErrorCode(str, Enum):
    """Standard error codes."""
    AUTH_FAILED = "AUTH_FAILED"
    MT5_NOT_INITIALIZED = "MT5_NOT_INITIALIZED"
    LIVE_TRADING_DISABLED = "LIVE_TRADING_DISABLED"
    SYMBOL_NOT_FOUND = "SYMBOL_NOT_FOUND"
    ORDER_SEND_FAILED = "ORDER_SEND_FAILED"
    ORDER_MODIFY_FAILED = "ORDER_MODIFY_FAILED"
    ORDER_CLOSE_FAILED = "ORDER_CLOSE_FAILED"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # Risk management codes
    RISK_LIMIT_OPEN_TRADES = "RISK_LIMIT_OPEN_TRADES"
    RISK_LIMIT_DAILY_LOSS = "RISK_LIMIT_DAILY_LOSS"
    RISK_LIMIT_TRADE_RISK = "RISK_LIMIT_TRADE_RISK"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


# ============================================================================
# Request Models
# ============================================================================

class PlaceOrderRequest(BaseModel):
    """Request to place a market order."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    side: OrderSide = Field(..., description="Order side: BUY or SELL")
    volume: float = Field(..., gt=0, description="Lot size (must be positive)")
    sl: Optional[float] = Field(None, description="Stop loss price")
    tp: Optional[float] = Field(None, description="Take profit price")
    deviation: int = Field(20, ge=0, description="Maximum price deviation in points")
    magic: int = Field(24001, description="Magic number for order identification")
    comment: str = Field("", max_length=31, description="Order comment (max 31 chars)")

    @field_validator('symbol')
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()


class ModifyOrderRequest(BaseModel):
    """Request to modify SL/TP of an existing position."""
    ticket: int = Field(..., gt=0, description="Position ticket number")
    sl: Optional[float] = Field(None, description="New stop loss price")
    tp: Optional[float] = Field(None, description="New take profit price")


class CloseOrderRequest(BaseModel):
    """Request to close an existing position."""
    ticket: int = Field(..., gt=0, description="Position ticket number")
    deviation: int = Field(20, ge=0, description="Maximum price deviation in points")
    comment: str = Field("", max_length=31, description="Close comment")


class CancelOrderRequest(BaseModel):
    """Request to cancel a pending order."""
    ticket: int = Field(..., gt=0, description="Order ticket number")


class PendingOrderRequest(BaseModel):
    """Request to place a pending order (limit/stop)."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    side: OrderSide = Field(..., description="Order side: BUY or SELL")
    order_type: Literal["LIMIT", "STOP"] = Field(..., description="Pending order type")
    volume: float = Field(..., gt=0, description="Lot size")
    price: float = Field(..., gt=0, description="Order execution price")
    sl: Optional[float] = Field(None, description="Stop loss price")
    tp: Optional[float] = Field(None, description="Take profit price")
    expiration: Optional[str] = Field(None, description="Order expiration time (ISO format)")
    magic: int = Field(24001, description="Magic number")
    comment: str = Field("", max_length=31, description="Order comment")

    @field_validator('symbol')
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.upper()


class TrailingStopRequest(BaseModel):
    """Request to set trailing stop on a position."""
    ticket: int = Field(..., gt=0, description="Position ticket")
    distance: float = Field(..., gt=0, description="Trailing distance in points")
    step: float = Field(10.0, gt=0, description="Minimum price movement to trigger update")


class ErrorDetail(BaseModel):
    """Standard error response structure."""
    code: ErrorCode
    message: str


class ErrorResponse(BaseModel):
    """Wrapper for error responses."""
    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response."""
    ok: bool
    mt5_initialized: bool
    terminal: Optional[dict] = None
    server_time: Optional[str] = None


class AccountInfo(BaseModel):
    """Account information response."""
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str
    open_positions_count: int
    leverage: Optional[int] = None
    profit: float = 0.0


class QuoteInfo(BaseModel):
    """Price quote response."""
    symbol: str
    bid: float
    ask: float
    spread: float
    time: str


class PositionInfo(BaseModel):
    """Position information."""
    ticket: int
    symbol: str
    type: str  # "BUY" or "SELL"
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    swap: float = 0.0
    magic: int = 0
    comment: str = ""
    time: str = ""


class PositionsResponse(BaseModel):
    """List of positions response."""
    positions: list[PositionInfo]
    count: int


class CandleInfo(BaseModel):
    """OHLC candle/bar information."""
    time: str  # ISO timestamp
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    real_volume: int


class CandlesResponse(BaseModel):
    """Historical candles response."""
    symbol: str
    timeframe: str
    candles: list[CandleInfo]
    count: int


class DealInfo(BaseModel):
    """Deal/trade history information."""
    ticket: int
    order: int
    time: str
    type: str  # "BUY" or "SELL"
    entry: str  # "IN" or "OUT"
    symbol: str
    volume: float
    price: float
    commission: float
    swap: float
    profit: float
    magic: int
    comment: str


class DealsResponse(BaseModel):
    """Historical deals response."""
    deals: list[DealInfo]
    count: int


class OrderHistoryInfo(BaseModel):
    """Historical order information."""
    ticket: int
    time_setup: str
    time_done: str
    type: str
    state: str
    symbol: str
    volume_initial: float
    volume_current: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    magic: int
    comment: str


class OrdersHistoryResponse(BaseModel):
    """Historical orders response."""
    orders: list[OrderHistoryInfo]
    count: int


class SymbolInfo(BaseModel):
    """Detailed symbol information."""
    name: str
    description: str
    path: str
    currency_base: str
    currency_profit: str
    currency_margin: str
    digits: int
    point: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    spread: float
    bid: float
    ask: float
    trade_mode: str
    trade_stops_level: int
    trade_freeze_level: int

class TickInfo(BaseModel):
    """Real-time tick data."""
    symbol: str
    time: str
    bid: float
    ask: float
    last: float
    volume: int
    flags: int


class TicksResponse(BaseModel):
    """Historical ticks response."""
    symbol: str
    ticks: list[TickInfo]
    count: int


class IndicatorValue(BaseModel):
    """Single indicator value with timestamp."""
    time: str
    value: float


class IndicatorMultiValue(BaseModel):
    """Indicator with multiple values (e.g., MACD with signal and histogram)."""
    time: str
    values: dict[str, float]


class RSIResponse(BaseModel):
    """RSI indicator response."""
    symbol: str
    period: int
    data: list[IndicatorValue]


class MACDResponse(BaseModel):
    """MACD indicator response."""
    symbol: str
    fast_period: int
    slow_period: int
    signal_period: int
    data: list[IndicatorMultiValue]


class BollingerBandsResponse(BaseModel):
    """Bollinger Bands response."""
    symbol: str
    period: int
    deviation: float
    data: list[IndicatorMultiValue]


class MovingAverageResponse(BaseModel):
    """Moving average response."""
    symbol: str
    period: int
    ma_type: str
    data: list[IndicatorValue]


class ATRResponse(BaseModel):
    """ATR indicator response."""
    symbol: str
    period: int
    data: list[IndicatorValue]


class TradeAnalytics(BaseModel):
    """Trade performance analytics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    expectancy: float


class EquityPoint(BaseModel):
    """Single point in equity curve."""
    time: str
    balance: float
    equity: float
    profit: float


class EquityCurveResponse(BaseModel):
    """Equity curve response."""
    points: list[EquityPoint]
    initial_balance: float
    final_balance: float
    max_drawdown: float
    max_drawdown_percent: float


class OrderResponse(BaseModel):
    """Order placement response."""
    accepted: bool
    ticket: Optional[int] = None
    status: OrderStatus
    price: Optional[float] = None
    time: Optional[str] = None
    volume: Optional[float] = None
    message: str = ""


class ModifyResponse(BaseModel):
    """Order modification response."""
    success: bool
    ticket: int
    message: str = ""


class CloseResponse(BaseModel):
    """Order close response."""
    success: bool
    ticket: int
    close_price: Optional[float] = None
    profit: Optional[float] = None
    message: str = ""


# ============================================================================
# Internal Data Models
# ============================================================================

class JournalEntry(BaseModel):
    """Journal/log entry for trade actions."""
    timestamp: datetime
    action: str
    request_data: dict
    response_data: dict
    result: str  # "SUCCESS" or "ERROR"
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class MT5Result(BaseModel):
    """Wrapper for MT5 operation results."""
    success: bool
    data: Optional[dict] = None
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None
