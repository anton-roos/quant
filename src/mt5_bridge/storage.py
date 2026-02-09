"""
Logging and storage infrastructure for trade journal and audit trail.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO") -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("mt5_bridge")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_path / "mt5_bridge.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class JSONLLogger:
    """
    Append-only JSONL logger for structured audit trail.
    Each line is a complete JSON object.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize JSONL logger.
        
        Args:
            log_dir: Directory for JSONL log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dated log file
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"audit_{today}.jsonl"
    
    def log_event(
        self,
        action: str,
        request_data: Any,
        response_data: Any,
        result: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Log an event to JSONL file.
        
        Args:
            action: Action type (e.g., "PLACE_ORDER", "CLOSE_POSITION")
            request_data: Request data (dict or Pydantic model)
            response_data: Response data (dict or Pydantic model)
            result: Result status ("SUCCESS" or "ERROR")
            error_code: Error code if result is ERROR
            error_message: Error message if result is ERROR
        """
        # Convert Pydantic models to dict
        if hasattr(request_data, 'model_dump'):
            request_data = request_data.model_dump()
        if hasattr(response_data, 'model_dump'):
            response_data = response_data.model_dump()
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "request": request_data,
            "response": response_data,
            "result": result,
        }
        
        if error_code:
            log_entry["error_code"] = error_code
        if error_message:
            log_entry["error_message"] = error_message
        
        # Append to JSONL file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            # Fail silently, but could log to main logger
            print(f"Failed to write to JSONL log: {e}")
    
    def log_request(
        self,
        endpoint: str,
        method: str,
        request_data: Any,
        response_data: Any,
        status_code: int,
        error: Optional[str] = None
    ) -> None:
        """
        Log an API request/response.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            request_data: Request payload
            response_data: Response payload
            status_code: HTTP status code
            error: Error message if any
        """
        result = "SUCCESS" if 200 <= status_code < 300 else "ERROR"
        action = f"{method} {endpoint}"
        
        self.log_event(
            action=action,
            request_data=request_data or {},
            response_data=response_data or {},
            result=result,
            error_message=error
        )


class TradeJournal:
    """
    SQLite-based trade journal for persistent storage of trade history.
    """
    
    def __init__(self, db_path: str = "./journal.sqlite"):
        """
        Initialize trade journal database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Events table (audit trail)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    request_json TEXT,
                    response_json TEXT,
                    result TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT
                )
            """)
            
            # Trades table (simplified trade history)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    open_time TEXT NOT NULL,
                    sl REAL,
                    tp REAL,
                    close_price REAL,
                    close_time TEXT,
                    profit REAL,
                    status TEXT NOT NULL,
                    magic INTEGER,
                    comment TEXT
                )
            """)
            
            # Daily stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    starting_balance REAL NOT NULL,
                    ending_balance REAL,
                    realized_pnl REAL DEFAULT 0,
                    num_trades INTEGER DEFAULT 0,
                    num_wins INTEGER DEFAULT 0,
                    num_losses INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
    
    def log_event(
        self,
        action: str,
        request_data: Any,
        response_data: Any,
        result: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Log an event to the events table.
        
        Args:
            action: Action type
            request_data: Request data
            response_data: Response data
            result: Result status
            error_code: Error code if applicable
            error_message: Error message if applicable
        """
        # Convert to JSON strings
        request_json = json.dumps(request_data) if request_data else None
        response_json = json.dumps(response_data) if response_data else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (timestamp, action, request_json, response_json, 
                                  result, error_code, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat() + "Z",
                action,
                request_json,
                response_json,
                result,
                error_code,
                error_message
            ))
            conn.commit()
    
    def record_trade_open(
        self,
        ticket: int,
        symbol: str,
        side: str,
        volume: float,
        open_price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        magic: int = 0,
        comment: str = ""
    ) -> None:
        """
        Record a new opened trade.
        
        Args:
            ticket: Position ticket number
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            volume: Lot size
            open_price: Entry price
            sl: Stop loss
            tp: Take profit
            magic: Magic number
            comment: Trade comment
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trades 
                (ticket, symbol, side, volume, open_price, open_time, sl, tp, 
                 status, magic, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """, (
                ticket, symbol, side, volume, open_price,
                datetime.utcnow().isoformat() + "Z",
                sl, tp, magic, comment
            ))
            conn.commit()
    
    def record_trade_close(
        self,
        ticket: int,
        close_price: float,
        profit: float
    ) -> None:
        """
        Update trade record when closed.
        
        Args:
            ticket: Position ticket number
            close_price: Exit price
            profit: Realized profit/loss
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET close_price = ?,
                    close_time = ?,
                    profit = ?,
                    status = 'CLOSED'
                WHERE ticket = ?
            """, (
                close_price,
                datetime.utcnow().isoformat() + "Z",
                profit,
                ticket
            ))
            conn.commit()
            
            # Update daily stats
            self._update_daily_stats(profit)
    
    def _update_daily_stats(self, pnl: float) -> None:
        """
        Update daily statistics with new P&L.
        
        Args:
            pnl: Profit/loss from closed trade
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if record exists for today
            cursor.execute("SELECT date FROM daily_stats WHERE date = ?", (today,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing record
                cursor.execute("""
                    UPDATE daily_stats
                    SET realized_pnl = realized_pnl + ?,
                        num_trades = num_trades + 1,
                        num_wins = num_wins + CASE WHEN ? > 0 THEN 1 ELSE 0 END,
                        num_losses = num_losses + CASE WHEN ? < 0 THEN 1 ELSE 0 END
                    WHERE date = ?
                """, (pnl, pnl, pnl, today))
            else:
                # Create new record (would need starting balance)
                cursor.execute("""
                    INSERT INTO daily_stats (date, starting_balance, realized_pnl, 
                                           num_trades, num_wins, num_losses)
                    VALUES (?, 0, ?, 1, ?, ?)
                """, (
                    today, pnl,
                    1 if pnl > 0 else 0,
                    1 if pnl < 0 else 0
                ))
            
            conn.commit()
    
    def get_daily_pnl(self, date_str: Optional[str] = None) -> float:
        """
        Get realized P&L for a specific date.
        
        Args:
            date_str: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Total realized P&L for the date
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT realized_pnl FROM daily_stats WHERE date = ?",
                (date_str,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0.0
    
    def get_open_trades_count(self) -> int:
        """
        Get count of currently open trades.
        
        Returns:
            Number of open trades in journal
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'")
            result = cursor.fetchone()
            return result[0] if result else 0
