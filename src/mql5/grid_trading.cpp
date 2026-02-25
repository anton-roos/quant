#property strict
#include <Trade/Trade.mqh>

// ===================================================================
//  Grid Trader V4 – Scalping Grid with Trend Filter (M1)
//
//  What killed V1
//  --------------
//  - Traded both directions simultaneously → trending markets destroyed
//    the counter-trend side while tiny TPs closed on the right side.
//  - TP:SL was 1:2 (risk $2 to make $1) → needed >66% win rate.
//  - 16 positions × 0.5% risk = 8% total exposure, then the grid
//    rebuilt and piled on MORE losing trades.
//  - 3-pip min grid on M1 → noise triggered every level instantly.
//
//  V2 Fixes
//  --------
//  1. EMA trend filter – only BUY when price > EMA, only SELL when
//     price < EMA. Never fight the trend.
//  2. TP:SL flipped to 1.5:1 default (risk 1 step, target 1.5 steps).
//  3. Risk reduced to 0.2% per fill, max 6 positions = 1.2% total.
//  4. Min grid step raised to 5 pips to survive M1 noise.
//  5. Rebuild cooldown extended to 5 minutes.
//  6. Per-level cooldown prevents rapid-fire entries on spikes.
//  7. Volatility gate – only trade when ATR is above a minimum
//     threshold (ensures there is actual volatility to capture).
//  8. Lot sizing always reads AccountBalance → compounding.
//
//  V3 Fixes
//  --------
//  1. Trend filter moved to M15 EMA with hysteresis buffer – stops
//     M1 EMA noise from whipsawing trend direction every few bars.
//  2. Counter-trend positions no longer force-closed on flip (TP/SL
//     handle exits). Controlled by CloseOnFlip input.
//  3. Grid rebuilds bypass cooldown on genuine trend change.
//  4. gridDirection tracking ensures grid always matches current trend.
//  5. Min grid reduced to 4 pips, levels/positions increased to 8.
//
//  V4 Fixes
//  --------
//  1. Trailing stop DISABLED – on M1, the 2-pip trail distance gets
//     clipped by noise before TP can fill. Every trade was closing at
//     breakeven ($0.00 × 1035 trades = flat equity).
//  2. TP widened to 2× gridStep, SL widened to 2× gridStep.
//     R:R = 1:1 but TP not sabotaged by trailing.
//  3. Grid tightened: MinGridPips 2.0, ATR mult 1.0, first level at
//     0.5× step for more frequent entries.
//  4. Risk per trade reduced to 0.15% to compensate for wider SL.
//  5. Faster rebuilds (120s) and more entries per bar (3).
// ===================================================================

// ===== INPUTS =====
// --- Grid geometry ---
input int              GridLevels         = 10;           // levels in trend direction only
input int              ATR_Period          = 14;           // ATR period (M1 bars)
input double           GridATR_Mult       = 1.0;          // grid step = ATR * this
input double           MinGridPips        = 2.0;          // floor for grid step (noise filter)
input double           MaxGridPips        = 30.0;         // ceiling for grid step

// --- Trend filter ---
input int              EMA_Trend          = 50;           // single EMA for trend direction
input ENUM_TIMEFRAMES  TrendTF            = PERIOD_M15;   // higher TF for stable trend
input double           TrendBufferPips    = 2.0;          // hysteresis: pips beyond EMA to flip
input bool             CloseOnFlip        = false;        // close counter-trend on trend change

// --- Lot sizing (balance-based compounding) ---
input double           RiskPctPerTrade    = 0.15;         // risk % of balance per grid fill
input double           MinLots            = 0.01;         // broker minimum
input double           MaxLots            = 5.0;          // hard cap

// --- TP / SL per grid order (FIXED: TP > SL now) ---
input double           TP_GridMult        = 2.0;          // TP = grid step * this
input double           SL_GridMult        = 2.0;          // SL = grid step * this

// --- Safety ---
input int              MaxPositions       = 10;           // max open positions (total risk capped)
input double           MaxSpreadPips      = 4.0;          // skip entry above this
input double           DrawdownKillPct    = 8.0;          // close all if DD > this %
input bool             TradeOnFriday      = false;        // allow Friday trading
input int              FridayCutoffHour   = 16;           // hour to stop on Friday

// --- Volatility gate (entries only, grid always builds) ---
input double           MinATR_Pips        = 0.5;          // skip entries if ATR < this

// --- Trailing stop ---
input bool             UseTrailing        = false;        // trail winning positions
input double           TrailStartMult     = 2.0;          // start trailing after this * gridStep profit
input double           TrailDistMult      = 1.0;          // trail distance = this * gridStep

// --- Grid rebuild ---
input int              RebuildCooldownSec = 120;          // 2 min cooldown between rebuilds
input double           RebuildDriftMult   = 1.5;          // rebuild when price drifts > levels * step * this

// --- Per-level cooldown ---
input int              LevelCooldownSec   = 10;           // seconds before same level can re-trigger
input int              MaxEntriesPerBar   = 3;            // max new positions per M1 bar (prevents cascade)

// --- Misc ---
input int              MagicNumber        = 300100;
input ENUM_TIMEFRAMES  TF                 = PERIOD_M1;    // execution timeframe
input string           EAComment          = "GridV4";

// ===== GLOBALS =====
CTrade   trade;
int      hATR;
int      hEMA;

double   gridAnchor    = 0;            // centre price of current grid
double   gridStep      = 0;            // distance between levels
datetime lastRebuild   = 0;            // throttle rebuilds
int      trendDir      = 0;            // +1 = bullish, -1 = bearish, 0 = flat
int      gridDirection  = 0;            // trend dir when grid was last built

double   gridLevels[];                 // trigger prices (trend direction only)
bool     levelFilled[];                // true if this level already triggered
datetime levelLastFill[];              // timestamp of last fill per level

bool     killSwitchActive = false;     // drawdown kill switch
int      entriesThisBar   = 0;         // entries opened in current M1 bar
datetime currentBarTime   = 0;         // track current bar for entry counting

// ===== HELPERS ==========================================================

double PipSize()
{
   if(StringFind(_Symbol, "XAU") >= 0) return 0.10;
   if(_Digits == 3 || _Digits == 5)    return _Point * 10;
   return _Point;
}

double Ask() { return SymbolInfoDouble(_Symbol, SYMBOL_ASK); }
double Bid() { return SymbolInfoDouble(_Symbol, SYMBOL_BID); }

double SpreadPips() { return (Ask() - Bid()) / PipSize(); }

double MidPrice() { return (Ask() + Bid()) / 2.0; }

// Normalise price to tick size
double Norm(double p)
{
   double tick = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick <= 0) tick = _Point;
   return NormalizeDouble(MathRound(p / tick) * tick, _Digits);
}

// Generic indicator read
bool GetBuf(int handle, int shift, double &val)
{
   double buf[1];
   if(CopyBuffer(handle, 0, shift, 1, buf) != 1) return false;
   val = buf[0];
   return true;
}

// Count my own open positions
int CountMyPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)    continue;
      if(PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;
      count++;
   }
   return count;
}

// Floating P/L of my positions
double FloatingPL()
{
   double pl = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)    continue;
      if(PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;
      pl += PositionGetDouble(POSITION_PROFIT)
          + PositionGetDouble(POSITION_SWAP);
   }
   return pl;
}

// Close every position owned by this EA
void CloseAll()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)    continue;
      if(PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;
      trade.PositionClose(ticket);
   }
}

// Close positions that don't match current trend direction
void CloseCounterTrend()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)  != _Symbol)    continue;
      if(PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;

      int type = (int)PositionGetInteger(POSITION_TYPE);
      // If trend is bullish, close any shorts; if bearish, close any longs
      if(trendDir > 0 && type == POSITION_TYPE_SELL)
         trade.PositionClose(ticket);
      else if(trendDir < 0 && type == POSITION_TYPE_BUY)
         trade.PositionClose(ticket);
   }
}

// ===== TREND DETECTION ===================================================
// Uses higher-TF EMA (M15 default) with hysteresis buffer.
// Compares real-time Bid() vs stable EMA → fast detection, stable signal.
// ALWAYS returns +1 or -1 — never blocks trading.

int DetectTrend()
{
   double ema;
   if(!GetBuf(hEMA, 1, ema))
   {
      // EMA not ready – fallback to price action on trend TF
      double c0  = iClose(_Symbol, TrendTF, 1);
      double c20 = iClose(_Symbol, TrendTF, 21);
      int raw = (c0 >= c20) ? +1 : -1;
      if(trendDir != 0) return trendDir;   // keep previous direction
      return raw;
   }

   // Compare real-time price against higher-TF (stable) EMA
   double price  = Bid();
   double buffer = TrendBufferPips * PipSize();

   // Hysteresis: only flip when price clears EMA ± buffer
   if(price > ema + buffer) return +1;
   if(price < ema - buffer) return -1;

   // Inside buffer zone → keep previous direction (no flip on noise)
   if(trendDir != 0) return trendDir;
   return (price >= ema) ? +1 : -1;
}

// ===== LOT SIZING (compounding from balance) =============================

double CalcLots(double slDistPoints)
{
   if(slDistPoints <= 0) slDistPoints = gridStep;

   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * RiskPctPerTrade / 100.0;

   // tick value per lot
   double tickVal  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickVal <= 0 || tickSize <= 0) return MinLots;

   double pointVal = tickVal / tickSize * _Point;   // value per point per lot
   double lots     = riskMoney / (slDistPoints / _Point * pointVal);

   // Normalise to lot step
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lotStep <= 0) lotStep = 0.01;
   lots = MathFloor(lots / lotStep) * lotStep;

   lots = MathMax(lots, MinLots);
   lots = MathMin(lots, MaxLots);

   // Respect broker limits
   double brokerMin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double brokerMax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(brokerMin > 0) lots = MathMax(lots, brokerMin);
   if(brokerMax > 0) lots = MathMin(lots, brokerMax);

   return NormalizeDouble(lots, 2);
}

// ===== GRID CONSTRUCTION =================================================
//  Grid is ONE-DIRECTIONAL based on trend:
//  - Bullish → buy levels BELOW current price (buy dips WITH the trend)
//  - Bearish → sell levels ABOVE current price (sell rallies WITH the trend)

void BuildGrid()
{
   double atr;
   if(!GetBuf(hATR, 1, atr))
   {
      // ATR not ready yet – use a safe default so arrays get allocated
      atr = MinGridPips * PipSize();
   }

   double ps   = PipSize();
   double step = atr * GridATR_Mult;
   step = MathMax(step, MinGridPips * ps);
   step = MathMin(step, MaxGridPips * ps);

   gridStep      = step;
   gridAnchor    = Norm(MidPrice());
   lastRebuild   = TimeCurrent();
   gridDirection  = trendDir;

   ArrayResize(gridLevels,   GridLevels);
   ArrayResize(levelFilled,  GridLevels);
   ArrayResize(levelLastFill, GridLevels);

   for(int i = 0; i < GridLevels; i++)
   {
      // Level 0 at 0.5× step (closer to anchor), rest at full spacing
      double levelDist = (i == 0) ? 0.5 * gridStep : (i + 0.5) * gridStep;

      if(trendDir > 0)
      {
         // Bullish: buy levels below anchor
         gridLevels[i] = Norm(gridAnchor - levelDist);
      }
      else if(trendDir < 0)
      {
         // Bearish: sell levels above anchor
         gridLevels[i] = Norm(gridAnchor + levelDist);
      }
      else
      {
         // No trend: place levels but won't trade (entries are gated)
         gridLevels[i] = Norm(gridAnchor - levelDist);
      }
      levelFilled[i]   = false;
      levelLastFill[i] = 0;
   }

   Print("[Grid] Rebuilt  dir=", (trendDir > 0 ? "LONG" : "SHORT"),
         "  anchor=", gridAnchor,
         "  step=", DoubleToString(gridStep / ps, 1), " pips",
         "  ATR=", DoubleToString(atr / ps, 1), " pips",
         "  levels=", GridLevels,
         "  balance=", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
         "  lot=", DoubleToString(CalcLots(gridStep * SL_GridMult), 2));
}

// ===== SHOULD REBUILD? ===================================================

bool NeedsRebuild()
{
   // First run or arrays never allocated
   if(gridAnchor == 0 || gridStep == 0 || ArraySize(gridLevels) == 0) return true;

   // Trend changed since last grid build → rebuild IMMEDIATELY (bypass cooldown)
   if(trendDir != 0 && trendDir != gridDirection) return true;

   // Everything else respects cooldown
   if((TimeCurrent() - lastRebuild) < RebuildCooldownSec) return false;

   // Price drifted too far from anchor
   double drift    = MathAbs(MidPrice() - gridAnchor);
   double maxDrift = GridLevels * gridStep * RebuildDriftMult;
   if(drift > maxDrift) return true;

   // All levels consumed
   bool allFilled = true;
   for(int i = 0; i < GridLevels; i++)
   {
      if(!levelFilled[i]) { allFilled = false; break; }
   }
   if(allFilled) return true;

   return false;
}

// ===== ENTRY CHECK =======================================================

void CheckGridEntries()
{
   if(killSwitchActive) return;
   if(SpreadPips() > MaxSpreadPips) return;
   if(CountMyPositions() >= MaxPositions) return;
   if(ArraySize(gridLevels) == 0 || gridStep == 0) return;  // grid not built yet

   // Track entries per bar – reset counter on new bar
   datetime barTime = iTime(_Symbol, TF, 0);
   if(barTime != currentBarTime)
   {
      currentBarTime = barTime;
      entriesThisBar = 0;
   }
   if(entriesThisBar >= MaxEntriesPerBar) return;  // already filled enough this bar

   // Friday filter
   MqlDateTime ft;
   TimeToStruct(TimeCurrent(), ft);
   if(!TradeOnFriday && ft.day_of_week == 5 && ft.hour >= FridayCutoffHour)
      return;

   // Volatility gate – only skip entries, never skip grid building
   double atr;
   if(!GetBuf(hATR, 0, atr)) return;
   if(atr / PipSize() < MinATR_Pips) return;

   double bid    = Bid();
   double ask    = Ask();
   double slDist = gridStep * SL_GridMult;
   double tpDist = gridStep * TP_GridMult;
   datetime now  = TimeCurrent();

   for(int i = 0; i < GridLevels; i++)
   {
      if(levelFilled[i]) continue;
      if(CountMyPositions() >= MaxPositions) break;
      if(entriesThisBar >= MaxEntriesPerBar) break;  // one entry per bar max

      // Per-level cooldown
      if(levelLastFill[i] > 0 && (now - levelLastFill[i]) < LevelCooldownSec)
         continue;

      if(trendDir > 0)
      {
         // BULLISH: buy when ask dips to this level
         if(ask <= gridLevels[i])
         {
            double lots = CalcLots(slDist);
            double sl   = Norm(ask - slDist);
            double tp   = Norm(ask + tpDist);

            string comment = EAComment + " B" + IntegerToString(i);
            if(trade.Buy(lots, _Symbol, 0, sl, tp, comment))
            {
               if(trade.ResultRetcode() == TRADE_RETCODE_DONE ||
                  trade.ResultRetcode() == TRADE_RETCODE_PLACED)
               {
                  levelFilled[i]   = true;
                  levelLastFill[i] = now;
                  entriesThisBar++;
                  Print("[Grid] BUY lvl=", i,
                        " @ ", DoubleToString(ask, _Digits),
                        "  lots=", lots,
                        "  SL=", DoubleToString(sl, _Digits),
                        "  TP=", DoubleToString(tp, _Digits),
                        "  bal=", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
               }
            }
         }
      }
      else if(trendDir < 0)
      {
         // BEARISH: sell when bid rises to this level
         if(bid >= gridLevels[i])
         {
            double lots = CalcLots(slDist);
            double sl   = Norm(bid + slDist);
            double tp   = Norm(bid - tpDist);

            string comment = EAComment + " S" + IntegerToString(i);
            if(trade.Sell(lots, _Symbol, 0, sl, tp, comment))
            {
               if(trade.ResultRetcode() == TRADE_RETCODE_DONE ||
                  trade.ResultRetcode() == TRADE_RETCODE_PLACED)
               {
                  levelFilled[i]   = true;
                  levelLastFill[i] = now;
                  entriesThisBar++;
                  Print("[Grid] SELL lvl=", i,
                        " @ ", DoubleToString(bid, _Digits),
                        "  lots=", lots,
                        "  SL=", DoubleToString(sl, _Digits),
                        "  TP=", DoubleToString(tp, _Digits),
                        "  bal=", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
               }
            }
         }
      }
   }
}

// ===== TRAILING STOP =====================================================

void TrailPositions()
{
   if(!UseTrailing) return;

   double trailStart = gridStep * TrailStartMult;
   double trailDist  = gridStep * TrailDistMult;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)    continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      int    type  = (int)PositionGetInteger(POSITION_TYPE);
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double tp    = PositionGetDouble(POSITION_TP);

      if(type == POSITION_TYPE_BUY)
      {
         double profit = Bid() - entry;
         if(profit < trailStart) continue;

         double newSL = Norm(Bid() - trailDist);
         if(newSL > sl + _Point)
            trade.PositionModify(ticket, newSL, tp);
      }
      else if(type == POSITION_TYPE_SELL)
      {
         double profit = entry - Ask();
         if(profit < trailStart) continue;

         double newSL = Norm(Ask() + trailDist);
         if(newSL < sl - _Point)
            trade.PositionModify(ticket, newSL, tp);
      }
   }
}

// ===== DRAWDOWN KILL SWITCH ==============================================

void CheckDrawdown()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance <= 0) return;

   double floatPL = FloatingPL();
   double ddPct   = MathAbs(MathMin(floatPL, 0.0)) / balance * 100.0;

   if(ddPct >= DrawdownKillPct)
   {
      Print("[Grid] DRAWDOWN KILL  DD=", DoubleToString(ddPct, 2),
            "%  closing all ", CountMyPositions(), " positions");
      CloseAll();
      killSwitchActive = true;
   }
}

// ===== VISUAL: draw grid lines on chart ==================================

void DrawGridLines()
{
   // Remove old lines
   for(int i = 0; i < GridLevels; i++)
      ObjectDelete(0, "Grid_" + IntegerToString(i));
   ObjectDelete(0, "GridAnchor");
   ObjectDelete(0, "GridEMA_F");
   ObjectDelete(0, "GridEMA_S");

   if(gridAnchor == 0 || ArraySize(gridLevels) == 0) return;

   // Anchor line
   ObjectCreate(0, "GridAnchor", OBJ_HLINE, 0, 0, gridAnchor);
   ObjectSetInteger(0, "GridAnchor", OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, "GridAnchor", OBJPROP_STYLE, STYLE_DOT);
   ObjectSetInteger(0, "GridAnchor", OBJPROP_WIDTH, 1);

   color activeColor = (trendDir > 0) ? clrLime : clrRed;
   color filledColor = (trendDir > 0) ? clrDarkGreen : clrMaroon;

   for(int i = 0; i < GridLevels; i++)
   {
      string name = "Grid_" + IntegerToString(i);
      ObjectCreate(0, name, OBJ_HLINE, 0, 0, gridLevels[i]);
      ObjectSetInteger(0, name, OBJPROP_COLOR, levelFilled[i] ? filledColor : activeColor);
      ObjectSetInteger(0, name, OBJPROP_STYLE, levelFilled[i] ? STYLE_DOT : STYLE_SOLID);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   }
}

// ===== CHART COMMENT (HUD) ===============================================

void ShowHUD()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double floatPL = FloatingPL();
   double ps      = PipSize();
   int    openPos = CountMyPositions();

   int filled = 0;
   int sz = ArraySize(levelFilled);
   for(int i = 0; i < sz; i++)
      if(levelFilled[i]) filled++;

   double atr;
   double atrPips = 0;
   if(GetBuf(hATR, 0, atr)) atrPips = atr / ps;

   string dir = (trendDir > 0) ? "LONG" : "SHORT";

   string txt = "=== Grid Trader V4 ===";
   txt += "Symbol:    " + _Symbol + "\n";
   txt += "Trend:     " + dir + "\n";
   txt += "Balance:   " + DoubleToString(balance, 2) + "\n";
   txt += "Equity:    " + DoubleToString(equity, 2) + "\n";
   txt += "Float PL:  " + DoubleToString(floatPL, 2) + "\n";
   txt += "DD%:       " + DoubleToString(MathAbs(MathMin(floatPL, 0.0)) / MathMax(balance, 1) * 100, 2) + "%\n";
   txt += "Positions: " + IntegerToString(openPos) + " / " + IntegerToString(MaxPositions) + "\n";
   txt += "Anchor:    " + DoubleToString(gridAnchor, _Digits) + "\n";
   txt += "Step:      " + DoubleToString(gridStep / ps, 1) + " pips\n";
   txt += "ATR:       " + DoubleToString(atrPips, 1) + " pips\n";
   txt += "Spread:    " + DoubleToString(SpreadPips(), 1) + " pips\n";
   txt += "Filled:    " + IntegerToString(filled) + " / " + IntegerToString(GridLevels) + "\n";
   txt += "Kill SW:   " + (killSwitchActive ? "ACTIVE" : "off") + "\n";
   txt += "Next lot:  " + DoubleToString(CalcLots(gridStep * SL_GridMult), 2) + "\n";
   txt += "Risk/fill: " + DoubleToString(RiskPctPerTrade, 2) + "% of " + DoubleToString(balance, 0) + "\n";

   Comment(txt);
}

// ===== CLEANUP CHART OBJECTS ON REMOVAL ==================================

void CleanupChart()
{
   for(int i = 0; i < GridLevels; i++)
      ObjectDelete(0, "Grid_" + IntegerToString(i));
   ObjectDelete(0, "GridAnchor");
   ObjectDelete(0, "GridEMA_F");
   ObjectDelete(0, "GridEMA_S");
   Comment("");
}

// ===== MQL5 EVENT HANDLERS ===============================================

int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(20);

   hATR = iATR(_Symbol, TF, ATR_Period);
   hEMA = iMA(_Symbol, TrendTF, EMA_Trend, 0, MODE_EMA, PRICE_CLOSE);

   if(hATR == INVALID_HANDLE || hEMA == INVALID_HANDLE)
   {
      Print("[Grid] Indicator handle creation FAILED");
      return INIT_FAILED;
   }

   // Pre-allocate arrays so nothing crashes before first BuildGrid
   ArrayResize(gridLevels,   GridLevels);
   ArrayResize(levelFilled,  GridLevels);
   ArrayResize(levelLastFill, GridLevels);
   ArrayInitialize(gridLevels, 0);
   ArrayInitialize(levelLastFill, 0);
   for(int i = 0; i < GridLevels; i++) levelFilled[i] = false;

   // Detect initial trend
   trendDir = DetectTrend();

   // Build initial grid
   BuildGrid();
   DrawGridLines();

   Print("[Grid] Init  symbol=", _Symbol,
         "  magic=", MagicNumber,
         "  trend=", (trendDir > 0 ? "LONG" : "SHORT"),
         "  balance=", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
         "  levels=", GridLevels,
         "  risk%=", RiskPctPerTrade,
         "  TP:SL=", DoubleToString(TP_GridMult, 1), ":", DoubleToString(SL_GridMult, 1));

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(hATR != INVALID_HANDLE) IndicatorRelease(hATR);
   if(hEMA != INVALID_HANDLE) IndicatorRelease(hEMA);
   CleanupChart();
}

// ===== NEW-BAR DETECTION (M1) =============================================
bool IsNewBar()
{
   static datetime lastBar = 0;
   datetime curBar = iTime(_Symbol, TF, 0);
   if(curBar == lastBar) return false;
   lastBar = curBar;
   return true;
}

void OnTick()
{
   // 1. Drawdown protection – every tick
   CheckDrawdown();

   // 2. Update trend on new M1 bar
   if(IsNewBar())
   {
      int prevTrend = trendDir;
      trendDir = DetectTrend();

      // Trend flipped → close counter-trend positions and rebuild
      if(trendDir != 0 && trendDir != prevTrend)
      {
         Print("[Grid] Trend changed  ", (prevTrend > 0 ? "LONG" : "SHORT/FLAT"),
               " → ", (trendDir > 0 ? "LONG" : "SHORT"));
         if(CloseOnFlip) CloseCounterTrend();
      }
   }

   // 3. Rebuild grid if needed
   if(NeedsRebuild())
   {
      if(killSwitchActive)
      {
         if(CountMyPositions() == 0)
            killSwitchActive = false;
         else
            return;
      }
      BuildGrid();
      DrawGridLines();
   }

   // 4. Check grid entries – every tick for M1 responsiveness
   CheckGridEntries();

   // 5. Trail winners
   TrailPositions();

   // 6. HUD (throttle to avoid lag)
   static datetime lastHUD = 0;
   if(TimeCurrent() - lastHUD >= 1)
   {
      ShowHUD();
      lastHUD = TimeCurrent();
   }
}
