"""
Tests for models validation.
"""
import pytest
from pydantic import ValidationError

from src.models import (
    PlaceOrderRequest, ModifyOrderRequest, CloseOrderRequest,
    OrderSide, ErrorCode
)


def test_place_order_request_valid():
    """Test valid order request."""
    order = PlaceOrderRequest(
        symbol="EURUSD",
        side=OrderSide.BUY,
        volume=0.01,
        sl=1.0832,
        tp=1.0860,
        deviation=20,
        magic=24001,
        comment="test order"
    )
    assert order.symbol == "EURUSD"
    assert order.side == OrderSide.BUY
    assert order.volume == 0.01


def test_place_order_request_symbol_uppercase():
    """Test that symbol is converted to uppercase."""
    order = PlaceOrderRequest(
        symbol="eurusd",
        side=OrderSide.BUY,
        volume=0.01
    )
    assert order.symbol == "EURUSD"


def test_place_order_request_invalid_volume():
    """Test that negative volume is rejected."""
    with pytest.raises(ValidationError):
        PlaceOrderRequest(
            symbol="EURUSD",
            side=OrderSide.BUY,
            volume=-0.01  # Invalid
        )


def test_place_order_request_comment_truncation():
    """Test comment length validation rejects strings exceeding max_length."""
    import pydantic
    long_comment = "x" * 100
    with pytest.raises(pydantic.ValidationError):
        PlaceOrderRequest(
            symbol="EURUSD",
            side=OrderSide.BUY,
            volume=0.01,
            comment=long_comment
        )


def test_modify_order_request_valid():
    """Test valid modify request."""
    modify = ModifyOrderRequest(
        ticket=12345678,
        sl=1.0835,
        tp=1.0862
    )
    assert modify.ticket == 12345678
    assert modify.sl == 1.0835


def test_modify_order_request_invalid_ticket():
    """Test that zero/negative ticket is rejected."""
    with pytest.raises(ValidationError):
        ModifyOrderRequest(
            ticket=0,  # Invalid
            sl=1.0835
        )


def test_close_order_request_valid():
    """Test valid close request."""
    close = CloseOrderRequest(
        ticket=12345678,
        deviation=20,
        comment="close"
    )
    assert close.ticket == 12345678
    assert close.deviation == 20


def test_order_side_enum():
    """Test OrderSide enum values."""
    assert OrderSide.BUY.value == "BUY"
    assert OrderSide.SELL.value == "SELL"


def test_error_code_enum():
    """Test ErrorCode enum has expected values."""
    assert ErrorCode.MT5_NOT_INITIALIZED == "MT5_NOT_INITIALIZED"
    assert ErrorCode.LIVE_TRADING_DISABLED == "LIVE_TRADING_DISABLED"
    assert ErrorCode.SYMBOL_NOT_FOUND == "SYMBOL_NOT_FOUND"
    assert ErrorCode.RISK_LIMIT_OPEN_TRADES == "RISK_LIMIT_OPEN_TRADES"
