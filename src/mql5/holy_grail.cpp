#property strict
#include <Trade/Trade.mqh>
CTrade trade;

// ===== INPUTS =====
input double Lots = 0.01;
input double FixedSL_Pips = 50;
input double BE_Trigger = 30;
input double BE_Lock = 2;
input double Trail = 10;
input double BodyPctNeeded = 50;

// ===== STATE =====
double boxHigh, boxLow;
bool boxLocked=false;
bool allowTrade=true;
datetime buildStart, lockTime, sessionEnd;
int currentSession=0;
string boxName;

// ===== GOLD SAFE PIP =====
double Pip()
{
   if(StringFind(_Symbol,"XAU")>=0) return 0.10;
   if(_Digits==3 || _Digits==5) return _Point*10;
   return _Point;
}

double AskP(){double a; SymbolInfoDouble(_Symbol,SYMBOL_ASK,a); return a;}
double BidP(){double b; SymbolInfoDouble(_Symbol,SYMBOL_BID,b); return b;}

bool HasTrade(){ return PositionSelect(_Symbol); }

// ===== CANDLE =====
double BodyPct(int s)
{
   double o=iOpen(_Symbol,PERIOD_M1,s);
   double c=iClose(_Symbol,PERIOD_M1,s);
   double h=iHigh(_Symbol,PERIOD_M1,s);
   double l=iLow(_Symbol,PERIOD_M1,s);
   if(h-l==0) return 0;
   return MathAbs(c-o)/(h-l)*100;
}

bool CloseInside(int s)
{
   double c=iClose(_Symbol,PERIOD_M1,s);
   return (c<=boxHigh && c>=boxLow);
}

// ===== SESSION TIME =====
void SessionTimes(datetime now,int id)
{
   MqlDateTime t; TimeToStruct(now,t); t.sec=0;

   if(id==1){t.hour=2;t.min=30;buildStart=StructToTime(t);
             t.hour=3;t.min=0;lockTime=StructToTime(t);
             t.hour=9;t.min=30;sessionEnd=StructToTime(t);}
   if(id==2){t.hour=9;t.min=30;buildStart=StructToTime(t);
             t.hour=10;t.min=0;lockTime=StructToTime(t);
             t.hour=16;t.min=0;sessionEnd=StructToTime(t);}
   if(id==3){t.hour=16;t.min=0;buildStart=StructToTime(t);
             t.hour=16;t.min=30;lockTime=StructToTime(t);
             t.hour=23;t.min=59;t.sec=59;sessionEnd=StructToTime(t);}
}

int GetSession(datetime now)
{
   MqlDateTime t; TimeToStruct(now,t);
   if((t.hour==2 && t.min>=30)||(t.hour>2 && t.hour<9)||(t.hour==9 && t.min<=30)) return 1;
   if((t.hour==9 && t.min>=30)||(t.hour>9 && t.hour<16)) return 2;
   if(t.hour>=16) return 3;
   return 0;
}

// ===== BOX DRAW =====
void DrawBox()
{
   if(boxHigh==0||boxLow==0) return;

   if(ObjectFind(0,boxName)<0)
      ObjectCreate(0,boxName,OBJ_RECTANGLE,0,buildStart,boxHigh,sessionEnd,boxLow);

   ObjectMove(0,boxName,0,buildStart,boxHigh);
   ObjectMove(0,boxName,1,sessionEnd,boxLow);

   ObjectSetInteger(0,boxName,OBJPROP_COLOR,clrOrange);
   ObjectSetInteger(0,boxName,OBJPROP_BACK,true);
}

// ===== BUILD BOX =====
void BuildBox()
{
   double h=iHigh(_Symbol,PERIOD_M1,0);
   double l=iLow(_Symbol,PERIOD_M1,0);

   if(boxHigh==0||h>boxHigh) boxHigh=h;
   if(boxLow==0||l<boxLow) boxLow=l;

   DrawBox();
}

void LockBox(){ if(boxHigh>0&&boxLow>0) boxLocked=true; }

// ===== ENTRY =====
void TryTrade()
{
   if(!boxLocked || !allowTrade || HasTrade()) return;
   if(BodyPct(1)<BodyPctNeeded) return;

   double close=iClose(_Symbol,PERIOD_M1,1);

   bool buyBreak = close>boxHigh;
   bool sellBreak = close<boxLow;

   if(!buyBreak && !sellBreak) return;

   double sl;
   double ps=Pip();

   if(buyBreak)
   {
      sl=BidP()-FixedSL_Pips*ps;
      trade.Buy(Lots,_Symbol,0,sl,0);
   }
   else
   {
      sl=AskP()+FixedSL_Pips*ps;
      trade.Sell(Lots,_Symbol,0,sl,0);
   }

   allowTrade=false;
}

// ===== RESET =====
void ResetCheck()
{
   if(HasTrade()) return;
   if(CloseInside(1) && BodyPct(1)>=BodyPctNeeded)
      allowTrade=true;
}

// ===== BE + TRAIL =====
void ManageTrade()
{
   if(!HasTrade()) return;

   int type=PositionGetInteger(POSITION_TYPE);
   double entry=PositionGetDouble(POSITION_PRICE_OPEN);
   double sl=PositionGetDouble(POSITION_SL);
   double price=(type==POSITION_TYPE_BUY)?BidP():AskP();
   double ps=Pip();

   double profit=(type==POSITION_TYPE_BUY)?(price-entry)/ps:(entry-price)/ps;

   if(profit<BE_Trigger) return;

   double newSL=(type==POSITION_TYPE_BUY)?
      MathMax(entry+BE_Lock*ps,price-Trail*ps):
      MathMin(entry-BE_Lock*ps,price+Trail*ps);

   if(type==POSITION_TYPE_BUY && newSL>sl)
      trade.PositionModify(_Symbol,newSL,0);

   if(type==POSITION_TYPE_SELL && newSL<sl)
      trade.PositionModify(_Symbol,newSL,0);
}

// ===== MAIN =====
void OnTick()
{
   datetime now=TimeCurrent();

   int s=GetSession(now);
   static int last=-1;

   if(s!=last)
   {
      last=s;
      if(s!=0)
      {
         SessionTimes(now,s);
         boxHigh=0; boxLow=0; boxLocked=false; allowTrade=true;
         boxName="ElektraBox_"+IntegerToString(s)+"_"+TimeToString(now,TIME_DATE);
      }
   }

   if(s!=0 && now>=buildStart && now<lockTime) BuildBox();
   if(s!=0 && now>=lockTime) LockBox();

   if(s!=0 && boxLocked) TryTrade();

   ManageTrade();
   ResetCheck();
}