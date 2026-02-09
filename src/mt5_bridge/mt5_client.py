"""
MetaTrader 5 client wrapper.
Handles all direct interactions with the MT5 terminal.
"""
import MetaTrader5 as mt5
from datetime import datetime
from typing import Optional
import pytz

from .models import (
    MT5Result, ErrorCode, OrderSide
)


class MT5Client:
    """
    Wrapper for MetaTrader5 operations.
    All methods return MT5Result objects with structured success/error data.
    """
    
    def __init__(self):
        """
        Initialize MT5 client.
        
        Relies on the MT5 terminal already being logged in.
        """
        self._initialized = False
        self._timezone = pytz.UTC
    
    def initialize(self) -> MT5Result:
        """
        Initialize connection to MT5 terminal.
        
        Returns:
            MT5Result with success status and terminal info or error details
        """
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message=f"MT5 initialization failed: {error}"
                )
            
            self._initialized = True
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            account_info = mt5.account_info()
            
            data = {
                "terminal": {
                    "name": terminal_info.name if terminal_info else "Unknown",
                    "version": terminal_info.build if terminal_info else 0,
                    "company": terminal_info.company if terminal_info else "",
                    "connected": terminal_info.connected if terminal_info else False
                },
                "account": {
                    "login": account_info.login if account_info else 0,
                    "server": account_info.server if account_info else "",
                    "currency": account_info.currency if account_info else ""
                }
            }
            
            return MT5Result(success=True, data=data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception during MT5 initialization: {str(e)}"
            )
    
    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self._initialized:
            mt5.shutdown()
            self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if MT5 is initialized."""
        return self._initialized
    
    def symbol_select(self, symbol: str) -> MT5Result:
        """
        Ensure symbol is available in Market Watch.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MT5Result with success status
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Check if symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.SYMBOL_NOT_FOUND,
                    error_message=f"Symbol {symbol} not found"
                )
            
            # Enable symbol in Market Watch if not already visible
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return MT5Result(
                        success=False,
                        error_code=ErrorCode.SYMBOL_NOT_FOUND,
                        error_message=f"Failed to select symbol {symbol}"
                    )
            
            return MT5Result(success=True, data={"symbol": symbol})
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in symbol_select: {str(e)}"
            )
    
    def get_quote(self, symbol: str) -> MT5Result:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MT5Result with quote data
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Ensure symbol is selected
            select_result = self.symbol_select(symbol)
            if not select_result.success:
                return select_result
            
            # Get tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.SYMBOL_NOT_FOUND,
                    error_message=f"Failed to get tick for {symbol}"
                )
            
            # Check if tick has valid data (not zeros)
            if tick.bid == 0 and tick.ask == 0:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.SYMBOL_NOT_FOUND,
                    error_message=f"Symbol {symbol} has no market data. Check symbol name or add to Market Watch in MT5."
                )
            
            quote_data = {
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": (tick.ask - tick.bid),
                "time": datetime.fromtimestamp(tick.time, tz=self._timezone).isoformat()
            }
            
            return MT5Result(success=True, data=quote_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_quote: {str(e)}"
            )
    
    def get_account_info(self) -> MT5Result:
        """
        Get account information.
        
        Returns:
            MT5Result with account data
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            account = mt5.account_info()
            if account is None:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message="Failed to get account info"
                )
            
            # Count open positions
            positions = mt5.positions_get()
            open_positions_count = len(positions) if positions else 0
            
            account_data = {
                "balance": account.balance,
                "equity": account.equity,
                "margin": account.margin,
                "free_margin": account.margin_free,
                "currency": account.currency,
                "open_positions_count": open_positions_count,
                "leverage": account.leverage,
                "profit": account.profit
            }
            
            return MT5Result(success=True, data=account_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_account_info: {str(e)}"
            )
    
    def get_positions(self, symbol: Optional[str] = None) -> MT5Result:
        """
        Get open positions, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            MT5Result with list of positions
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get positions
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                positions = []
            
            positions_list = []
            for pos in positions:
                position_data = {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "magic": pos.magic,
                    "comment": pos.comment,
                    "time": datetime.fromtimestamp(pos.time, tz=self._timezone).isoformat()
                }
                positions_list.append(position_data)
            
            return MT5Result(
                success=True, 
                data={"positions": positions_list, "count": len(positions_list)}
            )
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_positions: {str(e)}"
            )
    
    def place_market_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        magic: int = 24001,
        comment: str = ""
    ) -> MT5Result:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY or SELL)
            volume: Lot size
            sl: Stop loss price
            tp: Take profit price
            deviation: Maximum price deviation in points
            magic: Magic number
            comment: Order comment
            
        Returns:
            MT5Result with order execution data
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Ensure symbol is selected
            select_result = self.symbol_select(symbol)
            if not select_result.success:
                return select_result
            
            # Get symbol info for precision
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.SYMBOL_NOT_FOUND,
                    error_message=f"Symbol info not available for {symbol}"
                )
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).ask if side == OrderSide.BUY else mt5.symbol_info_tick(symbol).bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": deviation,
                "magic": magic,
                "comment": comment[:31],  # Truncate to 31 chars
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL/TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_SEND_FAILED,
                    error_message=f"order_send failed: {error}"
                )
            
            # Check result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_SEND_FAILED,
                    error_message=f"Order rejected: retcode={result.retcode}, {result.comment}"
                )
            
            order_data = {
                "ticket": result.order,
                "status": "FILLED",
                "price": result.price,
                "volume": result.volume,
                "time": datetime.now(tz=self._timezone).isoformat(),
                "retcode": result.retcode,
                "comment": result.comment
            }
            
            return MT5Result(success=True, data=order_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in place_market_order: {str(e)}"
            )
    
    def modify_position_sl_tp(self, ticket: int, sl: Optional[float] = None, 
                              tp: Optional[float] = None) -> MT5Result:
        """
        Modify stop loss and/or take profit of an existing position.
        
        Args:
            ticket: Position ticket
            sl: New stop loss (None to keep current)
            tp: New take profit (None to keep current)
            
        Returns:
            MT5Result with modification status
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get position
            positions = mt5.positions_get(ticket=ticket)
            if not positions or len(positions) == 0:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_MODIFY_FAILED,
                    error_message=f"Position {ticket} not found"
                )
            
            position = positions[0]
            
            # Use current values if not specified
            new_sl = sl if sl is not None else position.sl
            new_tp = tp if tp is not None else position.tp
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": new_tp,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_MODIFY_FAILED,
                    error_message=f"Modify failed: {error}"
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_MODIFY_FAILED,
                    error_message=f"Modify rejected: retcode={result.retcode}, {result.comment}"
                )
            
            modify_data = {
                "ticket": ticket,
                "sl": new_sl,
                "tp": new_tp,
                "retcode": result.retcode,
                "comment": result.comment
            }
            
            return MT5Result(success=True, data=modify_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in modify_position_sl_tp: {str(e)}"
            )
    
    def close_position(self, ticket: int, deviation: int = 20, comment: str = "") -> MT5Result:
        """
        Close an open position.
        
        Args:
            ticket: Position ticket
            deviation: Maximum price deviation in points
            comment: Close comment
            
        Returns:
            MT5Result with close execution data
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get position
            positions = mt5.positions_get(ticket=ticket)
            if not positions or len(positions) == 0:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_CLOSE_FAILED,
                    error_message=f"Position {ticket} not found"
                )
            
            position = positions[0]
            
            # Determine close order type (opposite of position type)
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": deviation,
                "magic": position.magic,
                "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_CLOSE_FAILED,
                    error_message=f"Close failed: {error}"
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_CLOSE_FAILED,
                    error_message=f"Close rejected: retcode={result.retcode}, {result.comment}"
                )
            
            close_data = {
                "ticket": ticket,
                "close_price": result.price,
                "volume": result.volume,
                "profit": position.profit,  # Profit at time of close
                "retcode": result.retcode,
                "comment": result.comment
            }
            
            return MT5Result(success=True, data=close_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in close_position: {str(e)}"
            )
    
    def cancel_pending_order(self, ticket: int) -> MT5Result:
        """
        Cancel a pending order.
        
        Args:
            ticket: Order ticket
            
        Returns:
            MT5Result with cancel status
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Prepare cancel request
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_CLOSE_FAILED,
                    error_message=f"Cancel failed: {error}"
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_CLOSE_FAILED,
                    error_message=f"Cancel rejected: retcode={result.retcode}, {result.comment}"
                )
            
            cancel_data = {
                "ticket": ticket,
                "retcode": result.retcode,
                "comment": result.comment
            }
            
            return MT5Result(success=True, data=cancel_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in cancel_pending_order: {str(e)}"
            )
    
    def get_symbols(self, group: str = "*") -> MT5Result:
        """
        Get list of available trading symbols.
        
        Args:
            group: Symbol group filter (e.g., "*USD*", "Forex*", "*")
            
        Returns:
            MT5Result with list of symbols
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get symbols
            symbols = mt5.symbols_get(group=group)
            
            if symbols is None:
                symbols = []
            
            # Build symbol list with basic info
            symbol_list = []
            for sym in symbols:
                if sym.visible:  # Only show visible symbols
                    symbol_list.append({
                        "name": sym.name,
                        "description": sym.description if hasattr(sym, 'description') else "",
                        "path": sym.path if hasattr(sym, 'path') else "",
                        "currency_base": sym.currency_base if hasattr(sym, 'currency_base') else "",
                        "currency_profit": sym.currency_profit if hasattr(sym, 'currency_profit') else ""
                    })
            
            return MT5Result(
                success=True,
                data={"symbols": symbol_list, "count": len(symbol_list)}
            )
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_symbols: {str(e)}"
            )
    
    def get_candles(self, symbol: str, timeframe: str, count: int = 100, 
                    from_date: Optional[datetime] = None) -> MT5Result:
        """
        Get historical candle/bar data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "M1", "M5", "M15", "M30", "H1", "H4", "D1")
            count: Number of candles to retrieve (default 100, max 50000)
            from_date: Start date (if None, gets most recent candles)
            
        Returns:
            MT5Result with list of candles
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Ensure symbol is selected
            select_result = self.symbol_select(symbol)
            if not select_result.success:
                return select_result
            
            # Map timeframe string to MT5 constant
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M2": mt5.TIMEFRAME_M2,
                "M3": mt5.TIMEFRAME_M3,
                "M4": mt5.TIMEFRAME_M4,
                "M5": mt5.TIMEFRAME_M5,
                "M6": mt5.TIMEFRAME_M6,
                "M10": mt5.TIMEFRAME_M10,
                "M12": mt5.TIMEFRAME_M12,
                "M15": mt5.TIMEFRAME_M15,
                "M20": mt5.TIMEFRAME_M20,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H2": mt5.TIMEFRAME_H2,
                "H3": mt5.TIMEFRAME_H3,
                "H4": mt5.TIMEFRAME_H4,
                "H6": mt5.TIMEFRAME_H6,
                "H8": mt5.TIMEFRAME_H8,
                "H12": mt5.TIMEFRAME_H12,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1,
            }
            
            if timeframe not in timeframe_map:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.INVALID_PARAMETER,
                    error_message=f"Invalid timeframe: {timeframe}. Valid values: {list(timeframe_map.keys())}"
                )
            
            tf = timeframe_map[timeframe]
            
            # Get candles
            if from_date:
                rates = mt5.copy_rates_from(symbol, tf, from_date, count)
            else:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            
            if rates is None or len(rates) == 0:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.SYMBOL_NOT_FOUND,
                    error_message=f"No candle data available for {symbol} {timeframe}"
                )
            
            # Convert to list of dicts
            candles = []
            for rate in rates:
                candle = {
                    "time": datetime.fromtimestamp(rate['time'], tz=self._timezone).isoformat(),
                    "open": float(rate['open']),
                    "high": float(rate['high']),
                    "low": float(rate['low']),
                    "close": float(rate['close']),
                    "tick_volume": int(rate['tick_volume']),
                    "spread": int(rate['spread']),
                    "real_volume": int(rate['real_volume'])
                }
                candles.append(candle)
            
            return MT5Result(
                success=True,
                data={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles": candles,
                    "count": len(candles)
                }
            )
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_candles: {str(e)}"
            )
    
    def get_deals_history(self, from_date: datetime, to_date: datetime, 
                          symbol: Optional[str] = None) -> MT5Result:
        """
        Get deal history for a date range.
        
        Args:
            from_date: Start date
            to_date: End date
            symbol: Optional symbol filter
            
        Returns:
            MT5Result with list of deals
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get deals
            if symbol:
                deals = mt5.history_deals_get(from_date, to_date, group=symbol)
            else:
                deals = mt5.history_deals_get(from_date, to_date)
            
            if deals is None:
                deals = []
            
            deals_list = []
            for deal in deals:
                deal_type = "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL"
                entry_type = "IN" if deal.entry == mt5.DEAL_ENTRY_IN else "OUT"
                
                deal_info = {
                    "ticket": deal.ticket,
                    "order": deal.order,
                    "time": datetime.fromtimestamp(deal.time, tz=self._timezone).isoformat(),
                    "type": deal_type,
                    "entry": entry_type,
                    "symbol": deal.symbol,
                    "volume": deal.volume,
                    "price": deal.price,
                    "commission": deal.commission,
                    "swap": deal.swap,
                    "profit": deal.profit,
                    "magic": deal.magic,
                    "comment": deal.comment
                }
                deals_list.append(deal_info)
            
            return MT5Result(
                success=True,
                data={"deals": deals_list, "count": len(deals_list)}
            )
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_deals_history: {str(e)}"
            )
    
    def get_orders_history(self, from_date: datetime, to_date: datetime,
                           symbol: Optional[str] = None) -> MT5Result:
        """
        Get order history for a date range.
        
        Args:
            from_date: Start date
            to_date: End date
            symbol: Optional symbol filter
            
        Returns:
            MT5Result with list of historical orders
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get orders
            if symbol:
                orders = mt5.history_orders_get(from_date, to_date, group=symbol)
            else:
                orders = mt5.history_orders_get(from_date, to_date)
            
            if orders is None:
                orders = []
            
            orders_list = []
            for order in orders:
                order_type_map = {
                    mt5.ORDER_TYPE_BUY: "BUY",
                    mt5.ORDER_TYPE_SELL: "SELL",
                    mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
                    mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
                    mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
                    mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP",
                }
                
                state_map = {
                    mt5.ORDER_STATE_STARTED: "STARTED",
                    mt5.ORDER_STATE_PLACED: "PLACED",
                    mt5.ORDER_STATE_CANCELED: "CANCELED",
                    mt5.ORDER_STATE_PARTIAL: "PARTIAL",
                    mt5.ORDER_STATE_FILLED: "FILLED",
                    mt5.ORDER_STATE_REJECTED: "REJECTED",
                    mt5.ORDER_STATE_EXPIRED: "EXPIRED",
                }
                
                order_info = {
                    "ticket": order.ticket,
                    "time_setup": datetime.fromtimestamp(order.time_setup, tz=self._timezone).isoformat(),
                    "time_done": datetime.fromtimestamp(order.time_done, tz=self._timezone).isoformat(),
                    "type": order_type_map.get(order.type, "UNKNOWN"),
                    "state": state_map.get(order.state, "UNKNOWN"),
                    "symbol": order.symbol,
                    "volume_initial": order.volume_initial,
                    "volume_current": order.volume_current,
                    "price_open": order.price_open,
                    "price_current": order.price_current,
                    "sl": order.sl,
                    "tp": order.tp,
                    "magic": order.magic,
                    "comment": order.comment
                }
                orders_list.append(order_info)
            
            return MT5Result(
                success=True,
                data={"orders": orders_list, "count": len(orders_list)}
            )
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_orders_history: {str(e)}"
            )
    
    def get_symbol_info(self, symbol: str) -> MT5Result:
        """
        Get detailed information about a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MT5Result with symbol details
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Ensure symbol is selected
            select_result = self.symbol_select(symbol)
            if not select_result.success:
                return select_result
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.SYMBOL_NOT_FOUND,
                    error_message=f"Symbol {symbol} not found"
                )
            
            # Get current tick for bid/ask
            tick = mt5.symbol_info_tick(symbol)
            
            trade_mode_map = {
                mt5.SYMBOL_TRADE_MODE_DISABLED: "DISABLED",
                mt5.SYMBOL_TRADE_MODE_LONGONLY: "LONG_ONLY",
                mt5.SYMBOL_TRADE_MODE_SHORTONLY: "SHORT_ONLY",
                mt5.SYMBOL_TRADE_MODE_CLOSEONLY: "CLOSE_ONLY",
                mt5.SYMBOL_TRADE_MODE_FULL: "FULL",
            }
            
            info_data = {
                "name": symbol_info.name,
                "description": symbol_info.description,
                "path": symbol_info.path,
                "currency_base": symbol_info.currency_base,
                "currency_profit": symbol_info.currency_profit,
                "currency_margin": symbol_info.currency_margin,
                "digits": symbol_info.digits,
                "point": symbol_info.point,
                "trade_contract_size": symbol_info.trade_contract_size,
                "volume_min": symbol_info.volume_min,
                "volume_max": symbol_info.volume_max,
                "volume_step": symbol_info.volume_step,
                "spread": symbol_info.spread,
                "bid": tick.bid if tick else 0.0,
                "ask": tick.ask if tick else 0.0,
                "trade_mode": trade_mode_map.get(symbol_info.trade_mode, "UNKNOWN"),
                "trade_stops_level": symbol_info.trade_stops_level,
                "trade_freeze_level": symbol_info.trade_freeze_level
            }
            
            return MT5Result(success=True, data=info_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_symbol_info: {str(e)}"
            )
    
    def get_ticks(self, symbol: str, count: int = 100, 
                  from_date: Optional[datetime] = None) -> MT5Result:
        """
        Get historical tick data.
        
        Args:
            symbol: Trading symbol
            count: Number of ticks (default 100, max 100000)
            from_date: Start date (if None, gets most recent ticks)
            
        Returns:
            MT5Result with list of ticks
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Ensure symbol is selected
            select_result = self.symbol_select(symbol)
            if not select_result.success:
                return select_result
            
            # Get ticks
            if from_date:
                ticks = mt5.copy_ticks_from(symbol, from_date, count, mt5.COPY_TICKS_ALL)
            else:
                ticks = mt5.copy_ticks_range(symbol, 
                    datetime(2020, 1, 1, tzinfo=self._timezone),
                    datetime.now(tz=self._timezone), 
                    mt5.COPY_TICKS_ALL)[-count:]  # Get last N ticks
            
            if ticks is None or len(ticks) == 0:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.SYMBOL_NOT_FOUND,
                    error_message=f"No tick data available for {symbol}"
                )
            
            # Convert to list
            ticks_list = []
            for tick in ticks:
                tick_info = {
                    "symbol": symbol,
                    "time": datetime.fromtimestamp(tick['time'], tz=self._timezone).isoformat(),
                    "bid": float(tick['bid']),
                    "ask": float(tick['ask']),
                    "last": float(tick['last']),
                    "volume": int(tick['volume']),
                    "flags": int(tick['flags'])
                }
                ticks_list.append(tick_info)
            
            return MT5Result(
                success=True,
                data={"symbol": symbol, "ticks": ticks_list, "count": len(ticks_list)}
            )
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_ticks: {str(e)}"
            )
    
    def place_pending_order(self, symbol: str, side: OrderSide, order_type: str,
                           volume: float, price: float, sl: Optional[float] = None,
                           tp: Optional[float] = None, expiration: Optional[datetime] = None,
                           magic: int = 24001, comment: str = "") -> MT5Result:
        """
        Place a pending order (limit or stop).
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: "LIMIT" or "STOP"
            volume: Lot size
            price: Order price
            sl: Stop loss
            tp: Take profit
            expiration: Order expiration time
            magic: Magic number
            comment: Order comment
            
        Returns:
            MT5Result with order placement data
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Ensure symbol is selected
            select_result = self.symbol_select(symbol)
            if not select_result.success:
                return select_result
            
            # Determine order type constant
            if side == OrderSide.BUY:
                mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type == "LIMIT" else mt5.ORDER_TYPE_BUY_STOP
            else:
                mt5_type = mt5.ORDER_TYPE_SELL_LIMIT if order_type == "LIMIT" else mt5.ORDER_TYPE_SELL_STOP
            
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_type,
                "price": price,
                "magic": magic,
                "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            if expiration is not None:
                request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                request["expiration"] = int(expiration.timestamp())
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_SEND_FAILED,
                    error_message=f"Pending order failed: {error}"
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.ORDER_SEND_FAILED,
                    error_message=f"Pending order failed: {result.comment} (code: {result.retcode})"
                )
            
            order_data = {
                "ticket": result.order,
                "status": "PENDING",
                "price": price,
                "volume": result.volume,
                "time": datetime.now(tz=self._timezone).isoformat(),
                "retcode": result.retcode,
                "comment": result.comment
            }
            
            return MT5Result(success=True, data=order_data)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in place_pending_order: {str(e)}"
            )
    
    def calculate_trade_analytics(self, from_date: datetime, to_date: datetime) -> MT5Result:
        """
        Calculate trading performance analytics.
        
        Args:
            from_date: Start date
            to_date: End date
            
        Returns:
            MT5Result with analytics data
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get deals history
            deals = mt5.history_deals_get(from_date, to_date)
            
            if deals is None or len(deals) == 0:
                # Return empty analytics
                return MT5Result(
                    success=True,
                    data={
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0.0,
                        "total_profit": 0.0,
                        "total_loss": 0.0,
                        "net_profit": 0.0,
                        "profit_factor": 0.0,
                        "average_win": 0.0,
                        "average_loss": 0.0,
                        "largest_win": 0.0,
                        "largest_loss": 0.0,
                        "average_trade": 0.0,
                        "max_consecutive_wins": 0,
                        "max_consecutive_losses": 0,
                        "expectancy": 0.0
                    }
                )
            
            # Filter only exit deals
            exit_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
            
            if len(exit_deals) == 0:
                return MT5Result(success=True, data={
                    "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                    "win_rate": 0.0, "total_profit": 0.0, "total_loss": 0.0,
                    "net_profit": 0.0, "profit_factor": 0.0, "average_win": 0.0,
                    "average_loss": 0.0, "largest_win": 0.0, "largest_loss": 0.0,
                    "average_trade": 0.0, "max_consecutive_wins": 0,
                    "max_consecutive_losses": 0, "expectancy": 0.0
                })
            
            # Calculate metrics
            profits = [d.profit for d in exit_deals]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            
            total_profit = sum(winning_trades) if winning_trades else 0.0
            total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
            net_profit = sum(profits)
            
            profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
            win_rate = len(winning_trades) / len(profits) if profits else 0.0
            
            average_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
            average_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
            
            largest_win = max(winning_trades) if winning_trades else 0.0
            largest_loss = min(losing_trades) if losing_trades else 0.0
            
            average_trade = net_profit / len(profits) if profits else 0.0
            
            # Calculate consecutive wins/losses
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_wins = 0
            current_losses = 0
            
            for profit in profits:
                if profit > 0:
                    current_wins += 1
                    current_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_wins)
                elif profit < 0:
                    current_losses += 1
                    current_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_losses)
            
            # Expectancy
            expectancy = (win_rate * average_win) - ((1 - win_rate) * abs(average_loss))
            
            analytics = {
                "total_trades": len(profits),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": round(win_rate, 4),
                "total_profit": round(total_profit, 2),
                "total_loss": round(total_loss, 2),
                "net_profit": round(net_profit, 2),
                "profit_factor": round(profit_factor, 2),
                "average_win": round(average_win, 2),
                "average_loss": round(average_loss, 2),
                "largest_win": round(largest_win, 2),
                "largest_loss": round(largest_loss, 2),
                "average_trade": round(average_trade, 2),
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
                "expectancy": round(expectancy, 2)
            }
            
            return MT5Result(success=True, data=analytics)
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in calculate_trade_analytics: {str(e)}"
            )
    
    def get_equity_curve(self, from_date: datetime, to_date: datetime) -> MT5Result:
        """
        Calculate equity curve from trade history.
        
        Args:
            from_date: Start date
            to_date: End date
            
        Returns:
            MT5Result with equity curve data
        """
        try:
            if not self._initialized:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.MT5_NOT_INITIALIZED,
                    error_message="MT5 not initialized"
                )
            
            # Get account info for initial balance
            account = mt5.account_info()
            if account is None:
                return MT5Result(
                    success=False,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message="Failed to get account info"
                )
            
            # Get deals history
            deals = mt5.history_deals_get(from_date, to_date)
            
            if deals is None or len(deals) == 0:
                return MT5Result(
                    success=True,
                    data={
                        "points": [],
                        "initial_balance": account.balance,
                        "final_balance": account.balance,
                        "max_drawdown": 0.0,
                        "max_drawdown_percent": 0.0
                    }
                )
            
            # Build equity curve
            points = []
            running_profit = 0.0
            initial_balance = account.balance - sum([d.profit for d in deals])
            
            peak_equity = initial_balance
            max_drawdown = 0.0
            
            for deal in deals:
                if deal.entry == mt5.DEAL_ENTRY_OUT:  # Only count closed trades
                    running_profit += deal.profit
                    current_equity = initial_balance + running_profit
                    
                    # Track drawdown
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    
                    drawdown = peak_equity - current_equity
                    max_drawdown = max(max_drawdown, drawdown)
                    
                    points.append({
                        "time": datetime.fromtimestamp(deal.time, tz=self._timezone).isoformat(),
                        "balance": initial_balance,
                        "equity": current_equity,
                        "profit": running_profit
                    })
            
            final_balance = initial_balance + running_profit
            max_drawdown_percent = (max_drawdown / peak_equity * 100) if peak_equity > 0 else 0.0
            
            return MT5Result(
                success=True,
                data={
                    "points": points,
                    "initial_balance": round(initial_balance, 2),
                    "final_balance": round(final_balance, 2),
                    "max_drawdown": round(max_drawdown, 2),
                    "max_drawdown_percent": round(max_drawdown_percent, 2)
                }
            )
            
        except Exception as e:
            return MT5Result(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Exception in get_equity_curve: {str(e)}"
            )
