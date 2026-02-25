#property strict
#include <Trade/Trade.mqh>

// ===================================================================
//  Holy Grail V3 - "The Minimalist"
//  Direction : 200 EMA filter (longs above, shorts below)
//  Timing    : RSI dip below 30 (long) / spike above 70 (short)
//  Action    : Enter when RSI crosses back through the threshold
//  Management: ATR-based SL + trailing stop
// ===================================================================

// ===== INPUTS =====
input double Lots             = 0.01;
input int    MagicNumber      = 123457;

// --- EMA trend filter ---
input int    EMA_Period       = 200;
input ENUM_TIMEFRAMES TF      = PERIOD_H1;   // timeframe for signals

// --- RSI trigger ---
input int    RSI_Period       = 14;
input double RSI_OversoldLvl  = 30;           // buy when RSI crosses back above
input double RSI_OverboughtLvl= 70;           // sell when RSI crosses back below

// --- ATR-based stop loss & trail ---
input int    ATR_Period       = 14;
input double SL_ATR_Mult      = 2.0;          // SL = ATR * this
input double Trail_ATR_Mult   = 1.5;          // trail distance = ATR * this
input double BE_ATR_Mult      = 1.0;          // move to BE after ATR * this in profit

// --- Risk clamps ---
input double SL_MinPips       = 10;
input double SL_MaxPips       = 100;
input double MaxSpreadPips    = 5.0;

// --- Session filter ---
input bool   SkipFriday       = true;
input int    FridayCutoffHour = 16;

// ===== HANDLES =====
int hEMA, hRSI, hATR;

// ===== STATE =====
CTrade trade;

// ===== INIT =====
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);

   hEMA = iMA  (_Symbol, TF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   hRSI = iRSI (_Symbol, TF, RSI_Period, PRICE_CLOSE);
   hATR = iATR (_Symbol, TF, ATR_Period);

   if(hEMA == INVALID_HANDLE || hRSI == INVALID_HANDLE || hATR == INVALID_HANDLE)
   {
      Print("[HolyGrailV3] Indicator handle creation failed");
      return INIT_FAILED;
   }

   Print("[HolyGrailV3] Init  magic=", MagicNumber,
         "  TF=", EnumToString(TF),
         "  EMA=", EMA_Period,
         "  RSI=", RSI_Period,
         "  ATR_SL=", SL_ATR_Mult, "x");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(hEMA != INVALID_HANDLE) IndicatorRelease(hEMA);
   if(hRSI != INVALID_HANDLE) IndicatorRelease(hRSI);
   if(hATR != INVALID_HANDLE) IndicatorRelease(hATR);
}

// ===== HELPERS =====
double PipSize()
{
   if(StringFind(_Symbol, "XAU") >= 0) return 0.10;
   if(_Digits == 3 || _Digits == 5)    return _Point * 10;
   return _Point;
}

double Ask() { return SymbolInfoDouble(_Symbol, SYMBOL_ASK); }
double Bid() { return SymbolInfoDouble(_Symbol, SYMBOL_BID); }
double SpreadPips() { return (Ask() - Bid()) / PipSize(); }

bool GetInd(int handle, int shift, double &val)
{
   double buf[1];
   if(CopyBuffer(handle, 0, shift, 1, buf) != 1) return false;
   val = buf[0];
   return true;
}

// ===== POSITION HELPERS =====
ulong OwnTicket()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;
      return ticket;
   }
   return 0;
}

bool HasPosition() { return OwnTicket() != 0; }

// ===== NEW-BAR DETECTION =====
bool IsNewBar()
{
   static datetime lastBar = 0;
   datetime curBar = iTime(_Symbol, TF, 0);
   if(curBar == lastBar) return false;
   lastBar = curBar;
   return true;
}

// ===== ENTRY LOGIC =====
void CheckEntry()
{
   if(HasPosition()) return;
   if(SpreadPips() > MaxSpreadPips) return;

   // Friday filter
   MqlDateTime ft;
   TimeToStruct(TimeCurrent(), ft);
   if(SkipFriday && ft.day_of_week == 5 && ft.hour >= FridayCutoffHour)
      return;

   // Read indicators from the CLOSED bar (shift 1 = just-closed, shift 2 = prior)
   double ema1, rsi1, rsi2, atr1;
   if(!GetInd(hEMA, 1, ema1)) return;
   if(!GetInd(hRSI, 1, rsi1)) return;
   if(!GetInd(hRSI, 2, rsi2)) return;
   if(!GetInd(hATR, 1, atr1)) return;

   double close1 = iClose(_Symbol, TF, 1);
   double ps     = PipSize();

   // --- SL from ATR ---
   double slDist = atr1 * SL_ATR_Mult;
   slDist = MathMax(slDist, SL_MinPips * ps);
   slDist = MathMin(slDist, SL_MaxPips * ps);

   // ========== LONG SETUP ==========
   // Price above 200 EMA  +  RSI was < 30 and now crossed back above 30
   if(close1 > ema1 && rsi2 < RSI_OversoldLvl && rsi1 >= RSI_OversoldLvl)
   {
      double sl = Ask() - slDist;
      if(trade.Buy(Lots, _Symbol, 0, sl, 0,
                   "HGv3 long"))
      {
         if(trade.ResultRetcode() == TRADE_RETCODE_DONE)
         {
            ulong ticket = OwnTicket();
            if(ticket > 0)
            {
               double entry = PositionGetDouble(POSITION_PRICE_OPEN);
               double corrSL = entry - slDist;
               trade.PositionModify(ticket, corrSL, 0);
               Print("[HolyGrailV3] BUY @ ", entry,
                     "  SL=", corrSL, " (", slDist / ps, " pips)");
            }
         }
      }
      return;
   }

   // ========== SHORT SETUP ==========
   // Price below 200 EMA  +  RSI was > 70 and now crossed back below 70
   if(close1 < ema1 && rsi2 > RSI_OverboughtLvl && rsi1 <= RSI_OverboughtLvl)
   {
      double sl = Bid() + slDist;
      if(trade.Sell(Lots, _Symbol, 0, sl, 0,
                    "HGv3 short"))
      {
         if(trade.ResultRetcode() == TRADE_RETCODE_DONE)
         {
            ulong ticket = OwnTicket();
            if(ticket > 0)
            {
               double entry = PositionGetDouble(POSITION_PRICE_OPEN);
               double corrSL = entry + slDist;
               trade.PositionModify(ticket, corrSL, 0);
               Print("[HolyGrailV3] SELL @ ", entry,
                     "  SL=", corrSL, " (", slDist / ps, " pips)");
            }
         }
      }
      return;
   }
}

// ===== TRAIL MANAGEMENT (ATR-based) =====
void ManageTrade()
{
   ulong ticket = OwnTicket();
   if(ticket == 0) return;

   int    type  = (int)PositionGetInteger(POSITION_TYPE);
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl    = PositionGetDouble(POSITION_SL);
   double price = (type == POSITION_TYPE_BUY) ? Bid() : Ask();

   double atr1;
   if(!GetInd(hATR, 1, atr1)) return;

   double profit    = (type == POSITION_TYPE_BUY) ? price - entry : entry - price;
   double beTrigger = atr1 * BE_ATR_Mult;
   double trailDist = atr1 * Trail_ATR_Mult;

   if(profit < beTrigger) return;

   double newSL;
   if(type == POSITION_TYPE_BUY)
      newSL = MathMax(entry + PipSize(), price - trailDist);
   else
      newSL = MathMin(entry - PipSize(), price + trailDist);

   bool improve = (type == POSITION_TYPE_BUY  && newSL > sl + _Point)
               || (type == POSITION_TYPE_SELL && newSL < sl - _Point);

   if(improve)
   {
      if(!trade.PositionModify(ticket, newSL, 0))
         Print("[HolyGrailV3] Trail modify failed  ", trade.ResultComment());
   }
}

// ===== MAIN =====
void OnTick()
{
   // Only check entries on new bar to avoid overtrading
   if(IsNewBar())
      CheckEntry();

   // Trail every tick for responsiveness
   ManageTrade();
}
