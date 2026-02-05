"""
Tests for risk management checks.
"""
import pytest
from unittest.mock import Mock, patch

from src.risk import RiskManager, KillSwitch
from src.models import OrderSide, ErrorCode


def test_risk_manager_max_open_trades():
    """Test max open trades limit."""
    rm = RiskManager(max_open_trades=3)
    
    # Should pass with 2 open
    result = rm.check_max_open_trades(2)
    assert result.success is True
    
    # Should fail with 3 open (would be 4th trade)
    result = rm.check_max_open_trades(3)
    assert result.success is False
    assert result.error_code == ErrorCode.RISK_LIMIT_OPEN_TRADES


def test_risk_manager_daily_loss_limit():
    """Test daily loss limit."""
    rm = RiskManager(max_daily_loss_pct=3.0)
    
    # Initialize with starting balance
    starting_balance = 10000.0
    rm.reset_daily_tracking(starting_balance)
    
    # Current equity at 9800 (200 loss = 2%)
    result = rm.check_daily_loss_limit(9800.0)
    assert result.success is True
    
    # Current equity at 9600 (400 loss = 4%, exceeds 3%)
    result = rm.check_daily_loss_limit(9600.0)
    assert result.success is False
    assert result.error_code == ErrorCode.RISK_LIMIT_DAILY_LOSS


def test_risk_manager_no_stop_loss():
    """Test that orders without SL are rejected."""
    rm = RiskManager()
    
    result = rm.check_trade_risk(
        symbol="EURUSD",
        volume=0.01,
        entry_price=1.0850,
        sl=None,  # No SL
        side=OrderSide.BUY,
        account_equity=10000.0
    )
    
    assert result.success is False
    assert result.error_code == ErrorCode.RISK_LIMIT_TRADE_RISK
    assert "Stop loss is required" in result.error_message


@patch('src.risk.mt5.symbol_info')
def test_risk_manager_trade_risk_calculation(mock_symbol_info):
    """Test trade risk calculation."""
    # Mock symbol info
    mock_info = Mock()
    mock_info.point = 0.00001
    mock_info.trade_tick_value = 1.0
    mock_info.trade_tick_size = 0.00001
    mock_symbol_info.return_value = mock_info
    
    rm = RiskManager(max_risk_per_trade_pct=1.0)
    
    # Buy at 1.0850 with SL at 1.0830 (20 pips)
    result = rm.check_trade_risk(
        symbol="EURUSD",
        volume=0.01,
        entry_price=1.0850,
        sl=1.0830,
        side=OrderSide.BUY,
        account_equity=10000.0
    )
    
    # Should pass (risk is small)
    assert result.success is True


def test_kill_switch_default_inactive():
    """Test kill switch is inactive by default."""
    ks = KillSwitch(state_file="./test_kill_switch.state")
    assert ks.is_active() is False


def test_kill_switch_activate():
    """Test kill switch activation."""
    ks = KillSwitch(state_file="./test_kill_switch.state")
    ks.activate()
    
    assert ks.is_active() is True
    
    result = ks.check()
    assert result.success is False
    assert result.error_code == ErrorCode.KILL_SWITCH_ACTIVE


def test_kill_switch_deactivate():
    """Test kill switch deactivation."""
    ks = KillSwitch(state_file="./test_kill_switch.state")
    ks.activate()
    assert ks.is_active() is True
    
    ks.deactivate()
    assert ks.is_active() is False
    
    result = ks.check()
    assert result.success is True


def test_validate_order_full_checks():
    """Test full order validation with all checks."""
    rm = RiskManager(
        max_open_trades=3,
        max_daily_loss_pct=3.0,
        max_risk_per_trade_pct=1.0
    )
    
    # Mock MT5 symbol info
    with patch('src.risk.mt5.symbol_info') as mock_symbol_info:
        mock_info = Mock()
        mock_info.point = 0.00001
        mock_info.trade_tick_value = 1.0
        mock_info.trade_tick_size = 0.00001
        mock_symbol_info.return_value = mock_info
        
        # Valid order
        result = rm.validate_order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            volume=0.01,
            sl=1.0830,
            tp=1.0870,
            current_open_count=1,
            account_equity=10000.0,
            entry_price=1.0850
        )
        
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
