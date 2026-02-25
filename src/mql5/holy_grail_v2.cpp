#property strict
#include <Trade/Trade.mqh>

// ===== INPUTS =====
input double Lots            = 0.01;

// --- All distances expressed as multiples of the box range ---
// e.g. box = 10 pips, SL_Mult = 1.5 => SL = 15 pips from entry
input double SL_Mult         = 1.5;       // SL distance = box_range * this
input double BE_Mult         = 1.0;       // move SL to BE after price moves 1x box range
input double BE_Lock_Mult    = 0.2;       // lock 0.2x box range above entry at BE
input double Trail_Mult      = 0.8;       // trail distance = 0.8x box range behind price

// --- Fixed-pip safety clamps ---
input double SL_MinPips      = 10;        // minimum SL regardless of box
input double SL_MaxPips      = 80;        // maximum SL regardless of box

// --- Filters ---
input double BodyPctNeeded   = 50;
input int    MagicNumber     = 123456;
input double MaxSpreadPips   = 5.0;
input double MinBoxPips      = 5.0;
input double MaxBoxPips      = 200.0;
input double DailyLossLimit  = -100.0;    // account-currency; 0 = disabled
input bool   SkipFriday      = true;
input int    FridayCutoffHour= 16;
input ENUM_TIMEFRAMES BoxTF  = PERIOD_M1;

// --- Session-hour filter ---
input bool   UseHourFilter   = true;
input string SkipHours       = "19,22";   // comma-separated broker-server hours to skip

// ===== STATE =====
struct SessionBox
{
   double   high;
   double   low;
   bool     locked;
   bool     allowTrade;
   bool     tradedOnce;
   datetime buildStart;
   datetime lockTime;
   datetime sessionEnd;
   string   name;

   double Range() { return high - low; }

   void Reset()
   {
      high = 0;  low = 0;
      locked = false;  allowTrade = true;  tradedOnce = false;
      buildStart = 0;  lockTime = 0;  sessionEnd = 0;
      name = "";
   }
};

SessionBox box;
CTrade     trade;
int        currentSession = 0;
bool       skipHourMap[24];

// ===== INIT / DEINIT =====
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   box.Reset();

   // Parse skip-hours filter
   ArrayInitialize(skipHourMap, false);
   if(UseHourFilter)
   {
      string parts[];
      int cnt = StringSplit(SkipHours, ',', parts);
      for(int i = 0; i < cnt; i++)
      {
         int h = (int)StringToInteger(parts[i]);
         if(h >= 0 && h < 24) skipHourMap[h] = true;
      }
   }

   Print("[HolyGrail] Init  magic=", MagicNumber,
         "  symbol=", _Symbol, "  lots=", Lots,
         "  SL_Mult=", SL_Mult, "  BE_Mult=", BE_Mult,
         "  Trail_Mult=", Trail_Mult);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   Print("[HolyGrail] Removed  reason=", reason);
}

// ===== PIP HELPER (gold-safe) =====
double PipSize()
{
   if(StringFind(_Symbol, "XAU") >= 0) return 0.10;
   if(_Digits == 3 || _Digits == 5)    return _Point * 10;
   return _Point;
}

double Ask() { return SymbolInfoDouble(_Symbol, SYMBOL_ASK); }
double Bid() { return SymbolInfoDouble(_Symbol, SYMBOL_BID); }

double SpreadPips() { return (Ask() - Bid()) / PipSize(); }

// ===== POSITION HELPERS (magic-aware) =====
bool HasOwnPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetTicket(i) == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;
      return true;
   }
   return false;
}

ulong OwnPositionTicket()
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

// ===== DAILY P&L =====
double DailyPL()
{
   double pl = 0;
   datetime dayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));

   HistorySelect(dayStart, TimeCurrent());
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC)  != MagicNumber) continue;
      pl += HistoryDealGetDouble(ticket, DEAL_PROFIT)
          + HistoryDealGetDouble(ticket, DEAL_SWAP)
          + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
   }

   ulong t = OwnPositionTicket();
   if(t > 0)
      pl += PositionGetDouble(POSITION_PROFIT)
          + PositionGetDouble(POSITION_SWAP);

   return pl;
}

// ===== CANDLE HELPERS =====
double BodyPct(int shift)
{
   double o = iOpen (_Symbol, BoxTF, shift);
   double c = iClose(_Symbol, BoxTF, shift);
   double h = iHigh (_Symbol, BoxTF, shift);
   double l = iLow  (_Symbol, BoxTF, shift);
   if(h - l < _Point) return 0;
   return MathAbs(c - o) / (h - l) * 100;
}

// ===== SESSION TIMES =====
void SetSessionTimes(datetime now, int id)
{
   MqlDateTime t;
   TimeToStruct(now, t);
   t.sec = 0;

   switch(id)
   {
      case 1:
         t.hour =  2; t.min = 30; box.buildStart = StructToTime(t);
         t.hour =  3; t.min =  0; box.lockTime   = StructToTime(t);
         t.hour =  9; t.min = 29; box.sessionEnd  = StructToTime(t);
         break;
      case 2:
         t.hour =  9; t.min = 30; box.buildStart = StructToTime(t);
         t.hour = 10; t.min =  0; box.lockTime   = StructToTime(t);
         t.hour = 15; t.min = 59; box.sessionEnd  = StructToTime(t);
         break;
      case 3:
         t.hour = 16; t.min =  0; box.buildStart = StructToTime(t);
         t.hour = 16; t.min = 30; box.lockTime   = StructToTime(t);
         t.hour = 23; t.min = 59; t.sec = 59;
         box.sessionEnd = StructToTime(t);
         break;
   }
}

int DetectSession(datetime now)
{
   MqlDateTime t;
   TimeToStruct(now, t);
   int m = t.hour * 60 + t.min;

   if(m >= 150 && m < 570)  return 1;   // 02:30 - 09:29
   if(m >= 570 && m < 960)  return 2;   // 09:30 - 15:59
   if(m >= 960)             return 3;   // 16:00 - 23:59
   return 0;
}

// ===== BOX =====
void DrawBox()
{
   if(box.high == 0 || box.low == 0) return;

   if(ObjectFind(0, box.name) < 0)
      ObjectCreate(0, box.name, OBJ_RECTANGLE, 0,
                   box.buildStart, box.high, box.sessionEnd, box.low);

   ObjectMove(0, box.name, 0, box.buildStart, box.high);
   ObjectMove(0, box.name, 1, box.sessionEnd,  box.low);
   ObjectSetInteger(0, box.name, OBJPROP_COLOR, clrOrange);
   ObjectSetInteger(0, box.name, OBJPROP_BACK,  true);
}

void BuildBox()
{
   double h = iHigh (_Symbol, BoxTF, 0);
   double l = iLow  (_Symbol, BoxTF, 0);

   if(box.high == 0 || h > box.high) box.high = h;
   if(box.low  == 0 || l < box.low)  box.low  = l;

   DrawBox();
}

bool BoxSizeValid()
{
   double size = box.Range() / PipSize();
   return (size >= MinBoxPips && size <= MaxBoxPips);
}

// ===== SL CALCULATION =====
double CalcSL_Price(bool isBuy, double fromPrice)
{
   double slDist = box.Range() * SL_Mult;
   double ps     = PipSize();

   // Clamp to min/max pips
   slDist = MathMax(slDist, SL_MinPips * ps);
   slDist = MathMin(slDist, SL_MaxPips * ps);

   return isBuy ? fromPrice - slDist : fromPrice + slDist;
}

// ===== ENTRY =====
void TryTrade()
{
   if(!box.locked || !box.allowTrade || box.tradedOnce) return;
   if(HasOwnPosition()) return;
   if(!BoxSizeValid())  return;

   // spread guard
   if(SpreadPips() > MaxSpreadPips) return;

   // daily loss guard
   if(DailyLossLimit < 0 && DailyPL() <= DailyLossLimit) return;

   // Friday guard
   MqlDateTime ft;
   TimeToStruct(TimeCurrent(), ft);
   if(SkipFriday && ft.day_of_week == 5 && ft.hour >= FridayCutoffHour)
      return;

   // Hour filter
   if(UseHourFilter && skipHourMap[ft.hour])
      return;

   if(BodyPct(1) < BodyPctNeeded) return;

   double close = iClose(_Symbol, BoxTF, 1);
   bool buyBreak  = close > box.high;
   bool sellBreak = close < box.low;
   if(!buyBreak && !sellBreak) return;

   bool ok;

   if(buyBreak)
   {
      double sl = CalcSL_Price(true, Ask());
      ok = trade.Buy(Lots, _Symbol, 0, sl, 0,
                     "HG buy s" + IntegerToString(currentSession));
   }
   else
   {
      double sl = CalcSL_Price(false, Bid());
      ok = trade.Sell(Lots, _Symbol, 0, sl, 0,
                      "HG sell s" + IntegerToString(currentSession));
   }

   if(ok && trade.ResultRetcode() == TRADE_RETCODE_DONE)
   {
      box.tradedOnce = true;
      box.allowTrade = false;

      // Adjust SL relative to actual fill price
      ulong ticket = OwnPositionTicket();
      if(ticket > 0)
      {
         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double correctedSL = CalcSL_Price(buyBreak, entry);
         trade.PositionModify(ticket, correctedSL, 0);

         double ps = PipSize();
         Print("[HolyGrail] ", (buyBreak ? "BUY" : "SELL"),
               " @ ", entry,
               "  SL=", correctedSL,
               " (", MathAbs(entry - correctedSL) / ps, " pips)",
               "  box=", box.Range() / ps, " pips",
               "  BE after ", box.Range() * BE_Mult / ps, " pips");
      }
   }
   else
   {
      Print("[HolyGrail] Order FAILED  rc=", trade.ResultRetcode());
   }
}

// ===== BE + TRAIL (all box-relative) =====
void ManageTrade()
{
   ulong ticket = OwnPositionTicket();
   if(ticket == 0) return;

   int    type  = (int)PositionGetInteger(POSITION_TYPE);
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl    = PositionGetDouble(POSITION_SL);
   double price = (type == POSITION_TYPE_BUY) ? Bid() : Ask();

   double profit    = (type == POSITION_TYPE_BUY) ? price - entry : entry - price;
   double boxRange  = box.Range();
   double beTrigger = boxRange * BE_Mult;      // activate trail after 1x box
   double beLock    = boxRange * BE_Lock_Mult;  // lock 0.2x box above entry
   double trailDist = boxRange * Trail_Mult;    // trail 0.8x box behind price

   if(profit < beTrigger) return;

   double newSL;
   if(type == POSITION_TYPE_BUY)
      newSL = MathMax(entry + beLock, price - trailDist);
   else
      newSL = MathMin(entry - beLock, price + trailDist);

   // Only modify if strictly better
   bool improve = (type == POSITION_TYPE_BUY  && newSL > sl + _Point)
               || (type == POSITION_TYPE_SELL && newSL < sl - _Point);

   if(improve)
   {
      if(!trade.PositionModify(ticket, newSL, 0))
         Print("[HolyGrail] SL modify failed  ", trade.ResultComment());
   }
}

// ===== MAIN =====
void OnTick()
{
   datetime now = TimeCurrent();
   int s = DetectSession(now);

   static int lastSession = -1;

   if(s != lastSession)
   {
      lastSession = s;
      if(s != 0)
      {
         box.Reset();
         SetSessionTimes(now, s);
         currentSession = s;
         box.name = "ElektraBox_" + IntegerToString(s)
                    + "_" + TimeToString(now, TIME_DATE);
         Print("[HolyGrail] === Session ", s, " started ===");
      }
   }

   if(s == 0) return;

   if(now >= box.buildStart && now < box.lockTime)
      BuildBox();

   if(now >= box.lockTime && !box.locked)
   {
      if(box.high > 0 && box.low > 0)
      {
         box.locked = true;
         Print("[HolyGrail] Box locked  H=", box.high,
               "  L=", box.low,
               "  size=", box.Range() / PipSize(), " pips");
      }
   }

   if(box.locked)
      TryTrade();

   ManageTrade();
}
