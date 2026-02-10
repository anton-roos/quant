import sqlite3

conn = sqlite3.connect('outputs/trade_journal.sqlite')
cursor = conn.cursor()

# First, check what tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Available tables:')
for row in cursor.fetchall():
    print(f'  {row[0]}')
print()

# Get schema for the main table
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' LIMIT 1")
print('Table schema:')
print(cursor.fetchone()[0])
print('\n---\n')

# Overall buy/sell distribution
cursor.execute('SELECT side, COUNT(*) as count FROM trades GROUP BY side')
print('All-time position side counts:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Current open/pending positions
cursor.execute('SELECT side, COUNT(*) FROM trades WHERE status IN ("OPEN", "PENDING") GROUP BY side')
print('\nCurrent open/pending by side:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Recent trades (last 30)
cursor.execute('''
    SELECT side, COUNT(*) 
    FROM trades 
    WHERE entry_time > datetime('now', '-30 days')
    GROUP BY side
''')
print('\nLast 30 days by side:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Check if there are any patterns
cursor.execute('''
    SELECT 
        side,
        COUNT(*) as total,
        SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed,
        SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open,
        AVG(CASE WHEN status = 'CLOSED' AND realized_pnl IS NOT NULL THEN realized_pnl ELSE NULL END) as avg_pnl
    FROM trades
    GROUP BY side
''')
print('\nDetailed statistics:')
for row in cursor.fetchall():
    avg_pnl = row[4] if row[4] is not None else 0
    print(f'  {row[0]}: Total={row[1]}, Closed={row[2]}, Open={row[3]}, Avg PnL=${avg_pnl:.2f}')

conn.close()
