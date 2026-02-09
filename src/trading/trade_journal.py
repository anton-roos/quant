"""
Lightweight trade journal backed by SQLite.

Records every entry/exit with realized P&L and exposes aggregate
statistics (cumulative P&L, Sharpe ratio, win rate, etc.).
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "outputs" / "trade_journal.sqlite"


class TradeJournal:
    """Append-only trade journal using SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info(f"Trade journal opened: {self.db_path}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket          INTEGER,
                symbol          TEXT NOT NULL,
                side            TEXT NOT NULL,
                volume          REAL,
                entry_price     REAL,
                exit_price      REAL,
                sl              REAL,
                tp              REAL,
                entry_time      TEXT,
                exit_time       TEXT,
                realized_pnl    REAL,
                commission      REAL DEFAULT 0,
                swap            REAL DEFAULT 0,
                horizon         TEXT,
                adj_prob        REAL,
                comment         TEXT,
                status          TEXT DEFAULT 'OPEN'
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                equity          REAL NOT NULL,
                balance         REAL NOT NULL,
                open_positions  INTEGER DEFAULT 0
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------
    def record_entry(
        self,
        ticket: int,
        symbol: str,
        side: str,
        volume: float,
        entry_price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        horizon: str = "",
        adj_prob: float = 0.0,
        comment: str = "",
    ) -> int:
        """Record a new trade entry. Returns the journal row id."""
        cur = self._conn.execute(
            """
            INSERT INTO trades
                (ticket, symbol, side, volume, entry_price, sl, tp,
                 entry_time, horizon, adj_prob, comment, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            (
                ticket, symbol, side, volume, entry_price, sl, tp,
                datetime.utcnow().isoformat(), horizon, adj_prob, comment,
            ),
        )
        self._conn.commit()
        logger.debug(f"Journal: recorded entry ticket={ticket} {side} {volume} {symbol}")
        return cur.lastrowid

    def record_exit(
        self,
        ticket: int,
        exit_price: float,
        realized_pnl: float,
        commission: float = 0.0,
        swap: float = 0.0,
    ):
        """Close a trade in the journal with realized P&L."""
        self._conn.execute(
            """
            UPDATE trades
            SET exit_price   = ?,
                exit_time    = ?,
                realized_pnl = ?,
                commission   = ?,
                swap         = ?,
                status       = 'CLOSED'
            WHERE ticket = ? AND status = 'OPEN'
            """,
            (exit_price, datetime.utcnow().isoformat(), realized_pnl, commission, swap, ticket),
        )
        self._conn.commit()
        logger.debug(f"Journal: recorded exit ticket={ticket} pnl={realized_pnl:.2f}")

    def record_equity_snapshot(self, equity: float, balance: float, open_positions: int = 0):
        """Periodic equity snapshot for drawdown / equity curve analysis."""
        self._conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity, balance, open_positions) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), equity, balance, open_positions),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_closed_trades(self, last_n: Optional[int] = None) -> List[Dict]:
        """Return closed trades, most recent first."""
        query = "SELECT * FROM trades WHERE status='CLOSED' ORDER BY exit_time DESC"
        if last_n:
            query += f" LIMIT {int(last_n)}"
        cur = self._conn.execute(query)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_open_trades(self) -> List[Dict]:
        cur = self._conn.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY entry_time DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def compute_stats(self, last_n_days: Optional[int] = None) -> Dict:
        """
        Compute aggregate performance statistics on closed trades.

        Returns dict with: total_trades, wins, losses, win_rate,
        cumulative_pnl, avg_pnl, max_win, max_loss, profit_factor,
        sharpe_ratio (daily, annualized).
        """
        where = "WHERE status='CLOSED'"
        params = []
        if last_n_days:
            where += " AND exit_time >= date('now', ?)"
            params.append(f"-{last_n_days} days")

        rows = self._conn.execute(
            f"SELECT realized_pnl, exit_time FROM trades {where} ORDER BY exit_time",
            params,
        ).fetchall()

        if not rows:
            return {"total_trades": 0, "message": "No closed trades"}

        pnls = [r[0] for r in rows if r[0] is not None]

        if not pnls:
            return {"total_trades": len(rows), "message": "No P&L data"}

        import numpy as np

        pnl_arr = np.array(pnls)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

        # Daily P&L aggregation for Sharpe
        daily_pnl = {}
        for pnl_val, exit_time in rows:
            if pnl_val is None or exit_time is None:
                continue
            day = exit_time[:10]
            daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl_val

        daily_returns = np.array(list(daily_pnl.values())) if daily_pnl else np.array([0.0])
        sharpe = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

        stats = {
            "total_trades": len(pnls),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate": float(len(wins) / len(pnls)) if pnls else 0.0,
            "cumulative_pnl": float(pnl_arr.sum()),
            "avg_pnl": float(pnl_arr.mean()),
            "max_win": float(pnl_arr.max()),
            "max_loss": float(pnl_arr.min()),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "sharpe_ratio": sharpe,
            "trading_days": len(daily_pnl),
        }
        return stats

    def format_stats_report(self, last_n_days: Optional[int] = None) -> str:
        """Return a human-readable performance summary string."""
        s = self.compute_stats(last_n_days)
        if s.get("total_trades", 0) == 0:
            return "No closed trades to report."

        period = f"Last {last_n_days} days" if last_n_days else "All time"
        lines = [
            f"=== Performance Report ({period}) ===",
            f"Trades:        {s['total_trades']} ({s['wins']}W / {s['losses']}L)",
            f"Win rate:      {s['win_rate']:.1%}",
            f"Cumulative PnL: {s['cumulative_pnl']:+.2f}",
            f"Avg PnL/trade: {s['avg_pnl']:+.2f}",
            f"Max win:       {s['max_win']:+.2f}",
            f"Max loss:      {s['max_loss']:+.2f}",
            f"Profit factor: {s['profit_factor']:.2f}",
            f"Sharpe (ann.): {s['sharpe_ratio']:.2f}",
            f"Trading days:  {s['trading_days']}",
        ]
        return "\n".join(lines)

    def close(self):
        self._conn.close()
