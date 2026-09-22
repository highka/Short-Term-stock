# -*- coding: utf-8 -*-
"""
黑嚕嚕－短線交易雷達 ST V1.16.3
獨立短線研究版：V1.2.2 擴充研究宇宙與AI細產業健診；不沿用原黑嚕嚕 V3.x 策略/分數/帳本。

研究目的
1. 統一指標：MA5 / MA15 / MA30 / MA60 / MA200 + KD(9,3,3)
2. 比較 5m / 15m / 60m
3. 比較當沖、隔日、2日、3日、5日持有
4. 先做研究與回測，不下真單
5. V1.1 新增：KD區間、MA斜率、成交量/20期均量、日內VWAP與多週期研究欄位
6. MACD / RSI / Bollinger / ATR / ADX 暫不加入，避免一次堆疊過多參數

資料：
- V1.0 使用 yfinance 做研究資料
- yfinance intraday 歷史資料有限，因此本版定位為「原型驗證」
- 後續 ST V1.1 再接 Shioaji 保存自己的分鐘K歷史庫
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import urllib.request
import json
import time

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

APP_VERSION = "ST V1.16.3"
APP_NAME = "黑嚕嚕－短線交易雷達"
MA_LIST = [5, 15, 30, 60, 200]
INTERVALS = ["5m", "15m", "60m"]

APP_VERSION = "ST_V1.16.3"
EXPORT_PREFIX = "ST_V1.16.3"

st.set_page_config(page_title=f"{APP_NAME} {APP_VERSION}", page_icon="⚡", layout="wide")


@dataclass
class CostConfig:
    fee_rate: float = 0.001425
    fee_discount: float = 0.28
    tax_rate: float = 0.003
    daytrade_tax_rate: float = 0.0015
    slippage_pct: float = 0.0005

    def roundtrip_cost_pct(self, daytrade: bool) -> float:
        fee = self.fee_rate * self.fee_discount
        tax = self.daytrade_tax_rate if daytrade else self.tax_rate
        return (fee * 2 + tax + self.slippage_pct * 2) * 100


def normalize_symbol(code: str, market: str) -> str:
    code = str(code).strip().upper()
    if "." in code:
        return code
    suffix = ".TW" if market == "上市" else ".TWO"
    return f"{code}{suffix}"


@st.cache_data(ttl=1800, show_spinner=False)
def download_daily_batches(symbols: List[str], period: str="2mo", batch_size: int=25):
    """
    V1.16.3 全市場日K下載：
    - 小批次、threads=False，降低Yahoo大量請求整批失敗。
    - 每批最多3次重試。
    - 批次成功但個股遺漏時，再以5檔小批補抓。
    - 回傳 {symbol: DataFrame}；空資料不冒充成功。
    """
    syms=list(dict.fromkeys(symbols))
    out={}
    if not syms:
        return out

    def _extract(raw, batch):
        got={}
        if raw is None or getattr(raw,"empty",True):
            return got
        for s in batch:
            try:
                if isinstance(raw.columns,pd.MultiIndex):
                    lvl0=raw.columns.get_level_values(0)
                    if s not in lvl0:
                        continue
                    d=raw[s].copy()
                else:
                    if len(batch)!=1:
                        continue
                    d=raw.copy()
                if not {"Close","Volume"}.issubset(d.columns):
                    continue
                d=d.dropna(subset=["Close","Volume"]).copy()
                if len(d):
                    got[s]=d.sort_index()
            except Exception:
                continue
        return got

    def _fetch(batch, attempts=3):
        for attempt in range(attempts):
            try:
                raw=yf.download(
                    tickers=batch,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    prepost=False,
                )
                got=_extract(raw,batch)
                if got:
                    return got
            except Exception:
                pass
            if attempt < attempts-1:
                time.sleep(0.8*(attempt+1))
        return {}

    # 主批次
    for i in range(0,len(syms),batch_size):
        batch=syms[i:i+batch_size]
        got=_fetch(batch,attempts=3)
        out.update(got)

        # 補抓主批次中缺漏者
        missing=[s for s in batch if s not in got]
        for j in range(0,len(missing),5):
            sb=missing[j:j+5]
            out.update(_fetch(sb,attempts=2))

    return out



def build_fullmarket_walkforward_eligibility(lookback_months: int = 6, top_n: int = 100):
    """
    V1.14.0 真正的歷史股票池資格：
    - 官方上市+上櫃完整母池
    - 每個交易日只用該日前已完成日K
    - 核心池：依前20個完成交易日的成交金額中位數，當日全市場排名TOP100
    - 熱門增補：不在核心TOP100、流動性位於當日前20%，且前一完成日
      (量比>=1.5 或 熱門動能條件成立)
    回傳 eligibility_df 與曾經進入資格的股票聯集。
    """
    official, errors = fetch_official_tw_stock_universe()
    if official.empty:
        return pd.DataFrame(), [], errors

    dmap = download_daily_batches(official["股票"].tolist(), period=f"{lookback_months}mo", batch_size=25)
    rows_by_symbol={}
    all_dates=set()
    _wf_diag={
        "官方股票數":len(official),
        "日K下載成功股票數":len(dmap),
        "至少25根日K股票數":0,
        "建立歷史序列股票數":0,
        "歷史交易日數":0,
        "WalkForward資格列數":0,
    }

    for _,meta in official.iterrows():
        s=meta["股票"]
        d=dmap.get(s)
        if d is None or len(d)<25:
            continue
        _wf_diag["至少25根日K股票數"]+=1
        try:
            d=d.copy()
            d.index=pd.to_datetime(d.index,errors="coerce")
            d=d[d.index.notna()]
            d=d[d.index.weekday<5]
            close=pd.to_numeric(d["Close"],errors="coerce")
            vol=pd.to_numeric(d["Volume"],errors="coerce")
            turn=close*vol

            rec=[]
            # signal_date 是「下一個交易日」；此列資料只會在該日開盤前已知
            # i 代表最新完成日；需要其前20日作爆量基準。
            for i in range(20,len(d)):
                latest_date=pd.Timestamp(d.index[i]).date()
                base_vol=vol.iloc[i-20:i]
                base_turn=turn.iloc[max(0,i-19):i+1]  # 核心流動性：最近20個完成交易日
                prev20_vol=float(base_vol.mean()) if len(base_vol) else np.nan
                latest_vol=float(vol.iloc[i])
                latest_turn=float(turn.iloc[i])
                core_turn=float(base_turn.median()) if len(base_turn) else np.nan
                vol_ratio=latest_vol/prev20_vol if prev20_vol>0 else np.nan

                last3=vol.iloc[max(0,i-2):i+1]
                vol3=float(last3.mean())
                vol3_ratio=vol3/prev20_vol if prev20_vol>0 else np.nan

                # 成交金額熱門條件也只使用最新完成日 vs 前20日中位數（不含最新日）
                prev20_turn_excl=turn.iloc[i-20:i]
                turn_base=float(prev20_turn_excl.median()) if len(prev20_turn_excl) else np.nan
                turn_ratio=latest_turn/turn_base if turn_base>0 else np.nan

                rec.append({
                    "基準完成日":latest_date,
                    "核心20日成交金額中位數":core_turn,
                    "前一完成日量比20日":vol_ratio,
                    "前一完成日成交金額比20日":turn_ratio,
                    "近3完成日均量比20日":vol3_ratio,
                })
                all_dates.add(latest_date)
            if rec:
                rows_by_symbol[s]=pd.DataFrame(rec)
        except Exception:
            continue

    _wf_diag["建立歷史序列股票數"]=len(rows_by_symbol)
    _wf_diag["歷史交易日數"]=len(all_dates)

    # 逐日做橫斷面排名，完全不用未來資料。
    elig_rows=[]
    sorted_dates=sorted(all_dates)
    for base_date in sorted_dates:
        day_rows=[]
        for s,df in rows_by_symbol.items():
            q=df[df["基準完成日"]==base_date]
            if q.empty:
                continue
            r=q.iloc[-1]
            day_rows.append({
                "股票":s,
                "基準完成日":base_date,
                "核心20日成交金額中位數":r["核心20日成交金額中位數"],
                "前一完成日量比20日":r["前一完成日量比20日"],
                "前一完成日成交金額比20日":r["前一完成日成交金額比20日"],
                "近3完成日均量比20日":r["近3完成日均量比20日"],
            })
        if not day_rows:
            continue
        day=pd.DataFrame(day_rows)
        day["流動性排名"]=day["核心20日成交金額中位數"].rank(ascending=False,method="min")
        day["成交金額百分位"]=day["核心20日成交金額中位數"].rank(pct=True)*100
        day["核心TOP100"]=day["流動性排名"]<=top_n
        day["爆量異動"]=day["前一完成日量比20日"]>=1.5
        day["熱門動能"]=(day["前一完成日成交金額比20日"]>=1.5)&(day["近3完成日均量比20日"]>=1.2)
        day["熱門增補"]=(
            (~day["核心TOP100"])&
            (day["成交金額百分位"]>=80)&
            (day["爆量異動"]|day["熱門動能"])
        )
        day["WalkForward候選"]=day["核心TOP100"]|day["熱門增補"]
        elig_rows.append(day)

    elig=pd.concat(elig_rows,ignore_index=True) if elig_rows else pd.DataFrame()
    _wf_diag["WalkForward資格列數"]=len(elig)
    if isinstance(elig,pd.DataFrame):
        elig.attrs["wf_diag"]=_wf_diag

    # 只有嚴重不足才列為錯誤；正常診斷另由暖機頁面顯示。
    if len(dmap) < 500:
        errors=list(errors)+[
            f"Yahoo日K覆蓋過低：{len(dmap)}/{len(official)}，可能遇到Yahoo限流或暫時性下載失敗。"
        ]
    union=sorted(elig.loc[elig["WalkForward候選"],"股票"].unique().tolist()) if not elig.empty else []
    return elig, union, errors




def build_market_regime_context(period: str="6mo"):
    """
    V1.15.0 市場環境資料：
    只使用台灣加權指數 ^TWII 的已完成日K，指標沿用既有MA15概念，
    再加上5日報酬作為「近期市場速度」診斷，不直接當交易濾網。
    """
    try:
        d=yf.download("^TWII",period=period,interval="1d",
                      auto_adjust=False,progress=False,threads=False)
        if isinstance(d.columns,pd.MultiIndex):
            # yfinance 單商品有時仍回傳 MultiIndex
            d.columns=[c[0] if isinstance(c,tuple) else c for c in d.columns]
        d=d.dropna(subset=["Close"]).copy()
        d.index=pd.to_datetime(d.index,errors="coerce")
        d=d[d.index.notna()]
        d=d[d.index.weekday<5]
        close=pd.to_numeric(d["Close"],errors="coerce")
        d["市場MA15"]=close.rolling(15).mean()
        d["市場MA15斜率3"]=d["市場MA15"]-d["市場MA15"].shift(3)
        d["市場5日報酬%"]=(close/close.shift(5)-1)*100
        d["市場在MA15之上"]=close>d["市場MA15"]
        d["市場MA15向上"]=d["市場MA15斜率3"]>0

        def ret_bucket(x):
            if pd.isna(x): return "資料不足"
            if x < -2: return "<-2%"
            if x < 0: return "-2~0%"
            if x < 2: return "0~2%"
            return ">=2%"
        d["市場5日報酬區間"]=d["市場5日報酬%"].map(ret_bucket)

        def regime(row):
            if pd.isna(row["市場MA15"]) or pd.isna(row["市場5日報酬%"]):
                return "資料不足"
            if (not row["市場在MA15之上"]) and row["市場5日報酬%"]<0:
                return "弱勢"
            if row["市場在MA15之上"] and row["市場MA15向上"] and row["市場5日報酬%"]>=0:
                return "偏多"
            return "混合"
        d["市場狀態"]=d.apply(regime,axis=1)
        out=d.reset_index().rename(columns={d.index.name or "index":"基準完成日"})
        out["基準完成日"]=pd.to_datetime(out["基準完成日"]).dt.date
        return out[["基準完成日","市場MA15","市場MA15斜率3","市場5日報酬%",
                    "市場在MA15之上","市場MA15向上","市場5日報酬區間","市場狀態"]]
    except Exception:
        return pd.DataFrame()




def _filter_historical_top50(trades: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty or elig is None or elig.empty:
        return pd.DataFrame()
    t=trades.copy()
    sig=_as_taipei_series(t["訊號時間"])
    t["訊號日期"]=sig.dt.date
    t["訊號小時"]=sig.dt.hour
    keep=[]; ranks=[]; base_dates=[]
    for _,r in t.iterrows():
        q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
        if q.empty:
            keep.append(False); ranks.append(np.nan); base_dates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            rank=float(z["流動性排名"])
            keep.append(rank<=50); ranks.append(rank); base_dates.append(z["基準完成日"])
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=base_dates
    return t[pd.Series(keep,index=t.index)].copy()


def _warmup_summary(trades: pd.DataFrame, label: str):
    rows=[]
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame()
    for sample in ["全部","樣本內60%","樣本外40%"]:
        g=trades if sample=="全部" else trades[trades["樣本"]==sample]
        m=aggregate_trade_metrics(g)
        rows.append({
            "版本":label,"樣本":sample,
            "股票數":int(g["股票"].nunique()) if len(g) else 0,
            "交易數":len(g),**m
        })
    summary=pd.DataFrame(rows)

    ts=_as_taipei_series(trades["訊號時間"])
    z=trades.copy()
    edges=pd.date_range(ts.min(),ts.max(),periods=5) if len(z) else []
    blocks=[]
    if len(edges)==5:
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            g=z[z["時間段"]==block]
            m=aggregate_trade_metrics(g)
            blocks.append({
                "版本":label,"時間段":block,
                "起始":str(edges[labels.index(block)]),
                "結束":str(edges[labels.index(block)+1]),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    return summary,pd.DataFrame(blocks)


def validate_top50_warmup_correction(cost: CostConfig):
    """
    V1.16.0 檢查過去第1段失效是否受指標暖機不足影響。
    A：舊方式，抓3mo直接算KD/MA。
    B：抓6mo做指標暖機，只評估最後3mo。
    股票池、策略、成本與持有期完全相同。
    """
    elig, _, errors=build_fullmarket_walkforward_eligibility(lookback_months=6,top_n=100)
    _diag=getattr(elig,"attrs",{}).get("wf_diag",{}) if isinstance(elig,pd.DataFrame) else {}
    if elig is None or elig.empty:
        _rows=[{"檢查":"錯誤","數值":"歷史Walk-Forward資格資料為空"}]
        for k,val in _diag.items():
            _rows.append({"檢查":k,"數值":val})
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors,pd.DataFrame(_rows)
    union=sorted(elig.loc[elig["流動性排名"]<=50,"股票"].dropna().astype(str).unique().tolist())
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors,pd.DataFrame([
            {"檢查":"錯誤","數值":"歷史TOP50股票聯集為空"}
        ])

    _,_,old_trades=run_oos_60m_5d(
        union,cost,"3mo",allow_overlap=False,train_ratio=0.60,evaluation_months=None
    )
    _,_,warm_trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    old_top=_filter_historical_top50(old_trades,elig)
    warm_top=_filter_historical_top50(warm_trades,elig)

    _old_raw_count=0 if old_trades is None else len(old_trades)
    _warm_raw_count=0 if warm_trades is None else len(warm_trades)

    s1,b1=_warmup_summary(old_top,"舊3mo直接計算")
    s2,b2=_warmup_summary(warm_top,"6mo暖機_評估末3mo")
    summary=pd.concat([s1,s2],ignore_index=True)
    blocks=pd.concat([b1,b2],ignore_index=True)

    def _missing_rate(df,col):
        if df is None or df.empty or col not in df.columns:
            return np.nan
        return float(df[col].isna().mean()*100)

    _diag_rows=[{"檢查":k,"數值":val} for k,val in _diag.items()]
    compare=pd.DataFrame(_diag_rows+[
        {"檢查":"舊3mo原始策略交易數","數值":_old_raw_count},
        {"檢查":"6mo暖機原始策略交易數","數值":_warm_raw_count},
        {"檢查":"舊版TOP50逐筆數","數值":len(old_top)},
        {"檢查":"暖機版TOP50逐筆數","數值":len(warm_top)},
        {"檢查":"舊版量比20缺值率%","數值":_missing_rate(old_top,"量比20")},
        {"檢查":"暖機版量比20缺值率%","數值":_missing_rate(warm_top,"量比20")},
        {"檢查":"舊版MA60斜率缺值率%","數值":_missing_rate(old_top,"MA60斜率3")},
        {"檢查":"暖機版MA60斜率缺值率%","數值":_missing_rate(warm_top,"MA60斜率3")},
    ])

    if warm_top is None:
        warm_top=pd.DataFrame()
    if errors is None:
        errors=[]
    elif not isinstance(errors,list):
        errors=list(errors) if isinstance(errors,(tuple,set)) else [str(errors)]

    if not warm_top.empty and "訊號時間" in warm_top.columns:
        _tw=_as_taipei_series(warm_top["訊號時間"])
        warm_top["訊號時間_台北"]=_tw.astype(str)
        warm_top["訊號小時_台北"]=_tw.dt.hour
    else:
        # V1.16.1：不再因空資料KeyError中斷，並把真正問題回報在畫面。
        errors=list(errors)+["6mo暖機版沒有產生有效TOP50交易；請查看資料完整度，不再以KeyError中斷。"]

    return summary,blocks,warm_top,errors,compare


def validate_top50_signal_quality(cost: CostConfig, period: str="3mo"):
    """
    V1.15.1：
    固定真正Walk-Forward TOP50 + 固定60m KD黃金交叉/K<30/5日，
    只診斷既有訊號內部品質，不新增指標、不改買進規則。
    診斷項目：K深度、量比20、MA30/MA60斜率、60m訊號時段。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(elig.loc[elig["流動性排名"]<=50,"股票"].dropna().astype(str).unique().tolist())
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(union,cost,period,allow_overlap=False,train_ratio=0.60)
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=trades.copy()
    sig=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce").dt.tz_convert("Asia/Taipei")
    t["訊號日期"]=sig.dt.date
    t["訊號小時"]=sig.dt.hour

    # 歷史TOP50資格
    keep=[]; ranks=[]; pool_dates=[]
    for _,r in t.iterrows():
        q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
        if q.empty:
            keep.append(False); ranks.append(np.nan); pool_dates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            rank=float(z["流動性排名"])
            keep.append(rank<=50); ranks.append(rank); pool_dates.append(z["基準完成日"])
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=pool_dates
    t=t[pd.Series(keep,index=t.index)].copy()

    # 預先固定分桶，避免看完績效後再微調邊界。
    t["K深度"]=pd.cut(pd.to_numeric(t["訊號K"],errors="coerce"),
                       bins=[-np.inf,10,20,30],labels=["K<10","K10-20","K20-30"]).astype(str)
    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    t["量比區間"]=pd.cut(vr,bins=[-np.inf,0.8,1.0,1.5,np.inf],
                         labels=["<0.8","0.8-1.0","1.0-1.5",">=1.5"]).astype(str)
    ma30=pd.to_numeric(t.get("MA30斜率3"),errors="coerce")
    ma60=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["MA30方向"]=np.where(ma30>0,"向上","未向上")
    t["MA60方向"]=np.where(ma60>0,"向上","未向上")
    t["60m時段"]=t["訊號小時"].map(lambda h:f"{int(h):02d}:00" if pd.notna(h) else "資料不足")

    specs=[
        ("K深度",["K<10","K10-20","K20-30"]),
        ("量比區間",["<0.8","0.8-1.0","1.0-1.5",">=1.5"]),
        ("MA30方向",["向上","未向上"]),
        ("MA60方向",["向上","未向上"]),
        ("60m時段",["09:00","10:00","11:00","12:00","13:00"]),
    ]

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for col,vals in specs:
            for val in vals:
                g=xs[xs[col]==val]
                m=aggregate_trade_metrics(g)
                rows.append({
                    "樣本":sample,"診斷分類":col,"分組":val,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })
    summary=pd.DataFrame(rows)

    # 四段時間只追三個最核心既有結構：K深度、量比、MA60方向。
    blocks=[]
    ts=pd.to_datetime(t["訊號時間"],errors="coerce")
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for col,vals in [
                ("K深度",["K<10","K10-20","K20-30"]),
                ("量比區間",["<0.8","0.8-1.0","1.0-1.5",">=1.5"]),
                ("MA60方向",["向上","未向上"]),
            ]:
                for val in vals:
                    g=q[q[col]==val]
                    m=aggregate_trade_metrics(g)
                    blocks.append({
                        "時間段":block,
                        "起始":str(edges[labels.index(block)]),
                        "結束":str(edges[labels.index(block)+1]),
                        "診斷分類":col,"分組":val,
                        "股票數":int(g["股票"].nunique()) if len(g) else 0,
                        "交易數":len(g),**m
                    })
    return summary,pd.DataFrame(blocks),t,errors


def validate_market_regime_top50(cost: CostConfig, period: str="3mo"):
    """
    V1.15.0：
    固定真正Walk-Forward TOP50 + 固定60m KD黃金交叉/K<30/5日，
    只診斷「訊號前一完成日」市場環境，不改交易規則。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(elig.loc[elig["流動性排名"]<=50,"股票"].dropna().astype(str).unique().tolist())
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(union,cost,period,allow_overlap=False,train_ratio=0.60)
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=trades.copy()
    sig=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce").dt.tz_convert("Asia/Taipei")
    t["訊號日期"]=sig.dt.date

    # 先確認該筆交易在訊號日前的歷史資格確實為TOP50。
    keep=[]; ranks=[]; pool_dates=[]
    for _,r in t.iterrows():
        q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
        if q.empty:
            keep.append(False); ranks.append(np.nan); pool_dates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            rank=float(z["流動性排名"])
            keep.append(rank<=50); ranks.append(rank); pool_dates.append(z["基準完成日"])
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=pool_dates
    t=t[pd.Series(keep,index=t.index)].copy()

    market=build_market_regime_context(period="6mo")
    if market.empty:
        return pd.DataFrame(),pd.DataFrame(),t,errors+["^TWII市場環境資料下載失敗"]

    # 嚴格使用訊號日前最近一個已完成日K。
    mrows=[]
    for _,r in t.iterrows():
        q=market[market["基準完成日"]<r["訊號日期"]]
        if q.empty:
            mrows.append({})
        else:
            mrows.append(q.sort_values("基準完成日").iloc[-1].to_dict())
    mdf=pd.DataFrame(mrows,index=t.index)
    for c in mdf.columns:
        if c!="基準完成日":
            t[c]=mdf[c]
    t["市場基準日"]=mdf.get("基準完成日")

    rows=[]
    specs=[]
    for state in ["偏多","混合","弱勢"]:
        specs.append(("市場狀態",state,t["市場狀態"]==state))
    for bucket in ["<-2%","-2~0%","0~2%",">=2%"]:
        specs.append(("5日報酬",bucket,t["市場5日報酬區間"]==bucket))
    specs += [
        ("MA15位置","站上MA15",t["市場在MA15之上"]==True),
        ("MA15位置","跌破MA15",t["市場在MA15之上"]==False),
        ("MA15方向","MA15向上",t["市場MA15向上"]==True),
        ("MA15方向","MA15未向上",t["市場MA15向上"]==False),
    ]

    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for typ,name,mask in specs:
            g=xs[mask.reindex(xs.index,fill_value=False)]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"環境分類":typ,"環境":name,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    summary=pd.DataFrame(rows)

    # 四段時間 × 市場狀態，檢查是否能解釋先前第1段失效。
    blocks=[]
    ts=pd.to_datetime(t["訊號時間"],errors="coerce")
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for state in ["偏多","混合","弱勢"]:
                g=q[q["市場狀態"]==state]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "市場狀態":state,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })
    return summary,pd.DataFrame(blocks),t,errors


def validate_core_pool_sizes(cost: CostConfig, period: str="3mo"):
    """
    V1.14.1：真正Walk-Forward核心池規模健診。
    不再研究熱門增補；固定比較每日歷史流動性 TOP50 / TOP100 / TOP150 / TOP200。
    所有資格只使用訊號日前已完成日K，不以報酬反推門檻。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    max_n=200
    union=sorted(
        elig.loc[elig["流動性排名"]<=max_n,"股票"].dropna().astype(str).unique().tolist()
    )
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(union,cost,period,allow_overlap=False,train_ratio=0.60)
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=trades.copy()
    sig=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce").dt.tz_convert("Asia/Taipei")
    t["訊號日期"]=sig.dt.date

    ranks=[]; base_dates=[]
    for _,r in t.iterrows():
        s=r["股票"]; sd=r["訊號日期"]
        q=elig[(elig["股票"]==s)&(elig["基準完成日"]<sd)]
        if q.empty:
            ranks.append(np.nan); base_dates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            ranks.append(float(z["流動性排名"]))
            base_dates.append(z["基準完成日"])
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=base_dates
    t=t[t["WF流動性排名"].notna()].copy()

    sizes=[50,100,150,200]
    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for n in sizes:
            g=xs[xs["WF流動性排名"]<=n].copy()
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"核心池規模":f"TOP{n}",
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    summary=pd.DataFrame(rows)

    # 四段時間穩定度
    blocks=[]
    ts=pd.to_datetime(t["訊號時間"],errors="coerce")
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for n in sizes:
                g=q[q["WF流動性排名"]<=n].copy()
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "核心池規模":f"TOP{n}",
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })

    # 每日各規模有效股票數；正常情況應固定50/100/150/200，資料不足時可察覺。
    daily_rows=[]
    for d,q in elig.groupby("基準完成日"):
        daily_rows.append({
            "基準完成日":d,
            "TOP50":int((q["流動性排名"]<=50).sum()),
            "TOP100":int((q["流動性排名"]<=100).sum()),
            "TOP150":int((q["流動性排名"]<=150).sum()),
            "TOP200":int((q["流動性排名"]<=200).sum()),
        })
    daily=pd.DataFrame(daily_rows).sort_values("基準完成日")
    return summary,pd.DataFrame(blocks),t,daily,errors


def validate_fullmarket_walkforward(cost: CostConfig, period: str="3mo"):
    """
    V1.14.0：固定60m KD黃金交叉 + K<30 + 5日，
    交易是否納入由『訊號當日之前』的歷史Walk-Forward股票池決定。
    """
    elig, union, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig.empty or not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),elig,errors

    _,_,trades=run_oos_60m_5d(union,cost,period,allow_overlap=False,train_ratio=0.60)
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),elig,errors

    t=trades.copy()
    sig=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce").dt.tz_convert("Asia/Taipei")
    t["訊號日期"]=sig.dt.date

    # 對每筆訊號，使用訊號日前最近一個「基準完成日」的資格。
    core_flags=[]; hot_flags=[]; ranks=[]; base_dates=[]
    for _,r in t.iterrows():
        s=r["股票"]; sd=r["訊號日期"]
        q=elig[(elig["股票"]==s)&(elig["基準完成日"]<sd)]
        if q.empty:
            core_flags.append(False); hot_flags.append(False); ranks.append(np.nan); base_dates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            core_flags.append(bool(z["核心TOP100"]))
            hot_flags.append(bool(z["熱門增補"]))
            ranks.append(float(z["流動性排名"]))
            base_dates.append(z["基準完成日"])

    t["WF核心TOP100"]=core_flags
    t["WF熱門增補"]=hot_flags
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=base_dates
    t["WF納入"]=t["WF核心TOP100"]|t["WF熱門增補"]
    t=t[t["WF納入"]].copy()
    t["WF組別"]=np.where(t["WF核心TOP100"],"核心TOP100","熱門增補")

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for group in ["核心TOP100","熱門增補","全部候選"]:
            g=xs if group=="全部候選" else xs[xs["WF組別"]==group]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"組別":group,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    summary=pd.DataFrame(rows)

    # 四段時間穩定度
    blocks=[]
    ts=pd.to_datetime(t["訊號時間"],errors="coerce")
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for group in ["核心TOP100","熱門增補","全部候選"]:
                g=q if group=="全部候選" else q[q["WF組別"]==group]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "組別":group,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })

    # 每日股票池規模，檢查TOP100與熱門增補是否穩定。
    daily=elig.groupby("基準完成日").agg(
        核心TOP100=("核心TOP100","sum"),
        熱門增補=("熱門增補","sum"),
        候選總數=("WalkForward候選","sum")
    ).reset_index()
    return summary,pd.DataFrame(blocks),t,daily,errors


def validate_fullmarket_candidate_groups(cost: CostConfig, period: str="3mo"):
    """
    V1.13.2 初步策略A/B：
    1) 先用當前全市場資料建立核心TOP100 + 前20%流動性中的熱門增補；
    2) 對兩組分別跑固定核心策略 60m KD黃金交叉 + K<30 + 5日；
    3) 僅做結構驗證。由於股票組別是用『目前』資料定義，仍存在 current-selection bias，
       不可視為最終walk-forward驗證。
    """
    pool, pool_summary, errors = build_full_market_pool_diagnostics(top_n=100)
    if pool is None or pool.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pool_summary, errors

    core = pool.loc[pool["核心TOP100"]=="是","股票"].dropna().astype(str).tolist()
    hot = pool.loc[pool["熱門增補候選"]=="是","股票"].dropna().astype(str).tolist()
    all_syms = list(dict.fromkeys(core + hot))
    group_map={s:"核心TOP100" for s in core}
    for s in hot:
        group_map[s]="熱門增補"

    summary, _, trades = run_oos_60m_5d(all_syms, cost, period, allow_overlap=False, train_ratio=0.60)
    if trades is None or trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pool, pool_summary, errors

    t=trades.copy()
    t["股票池組別"]=t["股票"].map(group_map).fillna("其他")
    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for group in ["核心TOP100","熱門增補","全部候選"]:
            g=xs if group=="全部候選" else xs[xs["股票池組別"]==group]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"組別":group,"股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    ab=pd.DataFrame(rows)

    # 四段時間穩定度
    ts=pd.to_datetime(t["訊號時間"],errors="coerce")
    tb=[]
    ok=ts.notna()
    z=t.loc[ok].copy()
    ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for group in ["核心TOP100","熱門增補","全部候選"]:
                g=q if group=="全部候選" else q[q["股票池組別"]==group]
                m=aggregate_trade_metrics(g)
                tb.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "組別":group,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })
    return ab,pd.DataFrame(tb),pool,pool_summary,errors


def summarize_liquidity_thresholds(full_df: pd.DataFrame):
    """
    V1.13.1：用絕對成交金額與相對百分位兩種方式觀察候選池大小。
    目的不是直接選出『最佳』門檻，而是避免把極低流動股票只因爆量納入。
    """
    if full_df is None or full_df.empty:
        return pd.DataFrame()
    x=full_df.copy()
    rows=[]
    specs=[
        ("成交金額≥5,000萬", x["20日成交金額中位數"]>=50_000_000),
        ("成交金額≥1億", x["20日成交金額中位數"]>=100_000_000),
        ("成交金額≥2億", x["20日成交金額中位數"]>=200_000_000),
        ("成交金額≥5億", x["20日成交金額中位數"]>=500_000_000),
        ("全市場流動性前20%", x["成交金額百分位"]>=80),
        ("全市場流動性前15%", x["成交金額百分位"]>=85),
        ("全市場流動性前10%", x["成交金額百分位"]>=90),
    ]
    for name,eligible in specs:
        add=(x["核心TOP100"]=="否") & eligible & (
            (x["爆量異動"]=="是") | (x["熱門動能"]=="是")
        )
        rows.append({
            "門檻":name,
            "符合基本流動股票":int(eligible.sum()),
            "TOP100外熱門增補":int(add.sum()),
            "研究候選池總數":int(100+add.sum()),
            "最低20日成交金額中位數":float(x.loc[eligible,"20日成交金額中位數"].min()) if eligible.any() else np.nan,
            "上市增補":int((add & (x["市場"]=="上市")).sum()),
            "上櫃增補":int((add & (x["市場"]=="上櫃")).sum()),
        })
    return pd.DataFrame(rows)


def build_full_market_pool_diagnostics(top_n: int=100):
    """
    V1.13.0 全市場股票池研究：
    官方上市+上櫃普通公司 → Yahoo近期日K → 流動性/爆量/熱門標籤。
    爆量只作資訊欄位，不作報酬預測分數。
    """
    official,errors=fetch_official_tw_stock_universe()
    if official.empty:
        return pd.DataFrame(),pd.DataFrame(),errors

    dmap=download_daily_batches(official["股票"].tolist(),period="2mo",batch_size=80)
    rows=[]
    now_tw=pd.Timestamp.now(tz="Asia/Taipei")
    for _,meta in official.iterrows():
        s=meta["股票"]
        d=dmap.get(s)
        if d is None or len(d)<21:
            continue
        try:
            d=d.copy()
            d.index=pd.to_datetime(d.index,errors="coerce")
            d=d[d.index.notna()]
            d=d[d.index.weekday<5]
            if not len(d):
                continue
            if pd.Timestamp(d.index[-1]).date()==now_tw.date() and now_tw.time()<pd.Timestamp("13:30").time():
                d=d.iloc[:-1]
            if len(d)<21:
                continue
            close=pd.to_numeric(d["Close"],errors="coerce")
            vol=pd.to_numeric(d["Volume"],errors="coerce")
            high=pd.to_numeric(d["High"],errors="coerce")
            low=pd.to_numeric(d["Low"],errors="coerce")
            turn=close*vol
            amp=(high-low)/close.replace(0,np.nan)*100
            pv=vol.iloc[-21:-1]; pt=turn.iloc[-21:-1]; pa=amp.iloc[-21:-1]
            vol20=float(pv.mean()); turn20=float(pt.median()); amp20=float(pa.median())
            today_vol=float(vol.iloc[-1]); today_turn=float(turn.iloc[-1])
            vratio=today_vol/vol20 if vol20>0 else np.nan
            tratio=today_turn/turn20 if turn20>0 else np.nan
            vol3=float(vol.iloc[-3:].mean())
            v3ratio=vol3/vol20 if vol20>0 else np.nan
            amp3=float(amp.iloc[-3:].mean())
            aratio=amp3/amp20 if amp20>0 else np.nan
            rows.append({
                "股票":s,"代號":meta["代號"],"公司":meta["公司"],"市場":meta["市場"],
                "官方產業別":meta["官方產業別"],"最後交易日":str(pd.Timestamp(d.index[-1]).date()),
                "20日成交金額中位數":turn20,"20日成交量均值":vol20,
                "今日量比20日":vratio,"3日均量比20日":v3ratio,
                "今日成交金額比20日":tratio,"3日振幅比20日":aratio,
            })
        except Exception:
            continue

    out=pd.DataFrame(rows)
    if out.empty:
        return out,pd.DataFrame(),errors

    out["成交金額百分位"]=out["20日成交金額中位數"].rank(pct=True)*100
    out["流動性排名"]=out["20日成交金額中位數"].rank(ascending=False,method="min")
    out["核心TOP100"]=np.where(out["流動性排名"]<=top_n,"是","否")
    out["爆量異動"]=np.where(out["今日量比20日"]>=1.5,"是","否")
    out["熱門動能"]=np.where(
        (out["今日成交金額比20日"]>=1.5)&(out["3日均量比20日"]>=1.2),"是","否")

    # V1.13.1：正式研究候選先收斂到「全市場流動性前20%」。
    # 原V1.13.0使用>=40百分位，實際可低到日成交金額約數百萬，過於寬鬆。
    out["寬鬆熱門增補_V1130"]=np.where(
        (out["核心TOP100"]=="否")&(out["成交金額百分位"]>=40)&(
            (out["爆量異動"]=="是")|(out["熱門動能"]=="是")),"是","否")
    hot_add=(out["核心TOP100"]=="否")&(out["成交金額百分位"]>=80)&(
        (out["爆量異動"]=="是")|(out["熱門動能"]=="是"))
    out["熱門增補候選"]=np.where(hot_add,"是","否")
    out["研究候選池"]=np.where((out["核心TOP100"]=="是")|hot_add,"是","否")
    out["爆量分層"]=pd.cut(out["今日量比20日"],
        bins=[-np.inf,1.0,1.5,2.0,3.0,np.inf],
        labels=["<1倍","1-1.5倍","1.5-2倍","2-3倍",">3倍"]).astype(str)

    common=pd.to_datetime(out["最後交易日"],errors="coerce").dt.date.mode()
    common_date=str(common.iloc[0]) if len(common) else ""
    summary=pd.DataFrame([
        {"指標":"官方上市+上櫃公司數","數值":len(official)},
        {"指標":"Yahoo有效日K數","數值":len(out)},
        {"指標":"核心TOP100","數值":int((out["核心TOP100"]=="是").sum())},
        {"指標":"TOP100外爆量","數值":int(((out["核心TOP100"]=="否")&(out["爆量異動"]=="是")).sum())},
        {"指標":"TOP100外熱門動能","數值":int(((out["核心TOP100"]=="否")&(out["熱門動能"]=="是")).sum())},
        {"指標":"熱門增補候選","數值":int((out["熱門增補候選"]=="是").sum())},
        {"指標":"研究候選池總數","數值":int((out["研究候選池"]=="是").sum())},
    ])
    out["共同資料基準日"]=common_date
    return out.sort_values(["研究候選池","核心TOP100","今日量比20日","20日成交金額中位數"],
                           ascending=[False,False,False,False]).reset_index(drop=True),summary,errors


def build_pool_20_diagnostics(universe: List[str], top_n: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    V1.12.0 股票池2.0研究：
    核心流動性、短期爆量、熱門動能分開標記，不把熱門度直接混入既有可交易分。
    只做研究標籤，不改正式60m雷達股票池。
    """
    tickers=list(dict.fromkeys(universe))
    raw=yf.download(tickers=tickers, period="2mo", interval="1d",
                    group_by="ticker", auto_adjust=False, progress=False,
                    threads=True)
    rows=[]
    for s in tickers:
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                if s not in raw.columns.get_level_values(0):
                    continue
                df=raw[s].copy()
            else:
                df=raw.copy()
            df=df.dropna(subset=["Close","Volume"])
            # V1.12.1：只使用「已完成的台股日K」。
            # Yahoo 偶爾會在週末/非交易時段附上一根日期為今天、但成交量不完整的日K，
            # 會讓爆量比失真；週末列一律排除，交易日13:30前也排除當日日K。
            _idx=pd.to_datetime(df.index,errors="coerce")
            _weekday=pd.Series(_idx.weekday,index=df.index)
            df=df.loc[_weekday.values < 5].copy()
            _now_tw=pd.Timestamp.now(tz="Asia/Taipei")
            if not df.empty:
                _last_date=pd.Timestamp(df.index[-1]).date()
                if _last_date == _now_tw.date() and _now_tw.time() < pd.Timestamp("13:30").time():
                    df=df.iloc[:-1]
            if len(df)<21:
                continue
            close=pd.to_numeric(df["Close"],errors="coerce")
            vol=pd.to_numeric(df["Volume"],errors="coerce")
            high=pd.to_numeric(df["High"],errors="coerce")
            low=pd.to_numeric(df["Low"],errors="coerce")
            turnover=close*vol
            amp=(high-low)/close.replace(0,np.nan)*100

            prev20_vol=vol.iloc[-21:-1]
            prev20_turn=turnover.iloc[-21:-1]
            prev20_amp=amp.iloc[-21:-1]
            today_vol=float(vol.iloc[-1])
            today_turn=float(turnover.iloc[-1])
            vol20=float(prev20_vol.mean()) if len(prev20_vol) else np.nan
            turn20=float(prev20_turn.median()) if len(prev20_turn) else np.nan
            amp20=float(prev20_amp.median()) if len(prev20_amp) else np.nan
            vol_ratio=today_vol/vol20 if vol20 and vol20>0 else np.nan
            turn_ratio=today_turn/turn20 if turn20 and turn20>0 else np.nan
            vol3=float(vol.iloc[-3:].mean())
            vol3_ratio=vol3/vol20 if vol20 and vol20>0 else np.nan
            amp3=float(amp.iloc[-3:].mean())
            amp_ratio=amp3/amp20 if amp20 and amp20>0 else np.nan

            rows.append({
                "股票":s,"研究主題":research_theme(s),
                "最後交易日":str(pd.Timestamp(df.index[-1]).date()),
                "20日成交金額中位數":turn20,
                "20日成交量均值":vol20,
                "20日振幅中位數%":amp20,
                "今日成交量":today_vol,
                "今日量比20日":vol_ratio,
                "3日均量比20日":vol3_ratio,
                "今日成交金額比20日":turn_ratio,
                "3日振幅比20日":amp_ratio,
            })
        except Exception:
            continue

    out=pd.DataFrame(rows)
    if out.empty:
        return out,pd.DataFrame()

    # 核心流動池：只按既有流動性概念排序，不把爆量/熱門當預測分數。
    out["流動性百分位"]=out["20日成交金額中位數"].rank(pct=True)*100
    out["核心流動池"]=np.where(out["流動性百分位"]>=40,"是","否")

    # 爆量只分層，門檻先作研究桶，不宣稱哪個最好。
    out["爆量層級"]=pd.cut(
        out["今日量比20日"],
        bins=[-np.inf,1.0,1.5,2.0,3.0,np.inf],
        labels=["<1倍","1-1.5倍","1.5-2倍","2-3倍",">3倍"]
    ).astype(str)

    # 熱門動能：成交金額與量能同時高於自己的20日基準；僅觀察標籤。
    hot=(out["今日成交金額比20日"]>=1.5) & (out["3日均量比20日"]>=1.2)
    out["熱門動能觀察"]=np.where(hot,"是","否")
    out["爆量異動觀察"]=np.where(out["今日量比20日"]>=1.5,"是","否")

    # 保留既有TOP比較組，避免直接替換正式池。
    out["既有TOP比較組"]=np.where(
        out["20日成交金額中位數"].rank(ascending=False,method="min")<=top_n,"是","否"
    )

    _dates=pd.to_datetime(out["最後交易日"],errors="coerce").dt.date
    _common_date=str(pd.Series(_dates).mode().iloc[0]) if len(_dates.dropna()) else ""
    out["是否共同最新交易日"]=np.where(out["最後交易日"]==_common_date,"是","否")
    summary=pd.DataFrame([
        {"指標":"有效股票數","數值":len(out)},
        {"指標":"核心流動池","數值":int((out["核心流動池"]=="是").sum())},
        {"指標":"爆量異動觀察","數值":int((out["爆量異動觀察"]=="是").sum())},
        {"指標":"熱門動能觀察","數值":int((out["熱門動能觀察"]=="是").sum())},
        {"指標":f"既有TOP{top_n}比較組","數值":int((out["既有TOP比較組"]=="是").sum())},
    ])
    return out.sort_values(["爆量異動觀察","今日量比20日","今日成交金額比20日"],
                           ascending=[True,False,False]).reset_index(drop=True),summary


def diagnose_stock_pool(universe: List[str], ranked_pool: pd.DataFrame, top_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    V1.11.1 股票池健診：
    透明化目前人工母池 -> 近期可交易性排名 -> TOP N 的形成。
    這版不改核心策略，也不把排名當成報酬預測。
    """
    rows=[]
    rank_df=ranked_pool.copy() if ranked_pool is not None else pd.DataFrame()
    if not rank_df.empty and "股票" in rank_df.columns:
        rank_df=rank_df.reset_index(drop=True)
        rank_df["目前排名"]=np.arange(1,len(rank_df)+1)
        rmap=rank_df.set_index("股票").to_dict("index")
    else:
        rmap={}
    for s in universe:
        m=rmap.get(s,{})
        rows.append({
            "股票":s,
            "研究主題":research_theme(s),
            "是否有近期排名資料":"是" if s in rmap else "否",
            "目前排名":m.get("目前排名",np.nan),
            "短線可交易分":m.get("短線可交易分",np.nan),
            "近期成交金額":m.get("20日成交金額中位數",np.nan),
            "近期成交量":m.get("20日成交量中位數",np.nan),
            "近期振幅":m.get("20日振幅中位數%",np.nan),
            "目前TOP池":"是" if (pd.notna(m.get("目前排名",np.nan)) and m.get("目前排名")<=top_n) else "否",
        })
    detail=pd.DataFrame(rows)
    if detail.empty:
        return detail,pd.DataFrame()
    summary=(detail.groupby(["研究主題","目前TOP池"],dropna=False)
             .agg(股票數=("股票","count"),
                  平均可交易分=("短線可交易分","mean"))
             .reset_index())
    total=pd.DataFrame([{
        "研究主題":"【全部】","目前TOP池":"母池",
        "股票數":len(detail),"平均可交易分":pd.to_numeric(detail["短線可交易分"],errors="coerce").mean()
    }])
    summary=pd.concat([total,summary],ignore_index=True)
    return detail,summary


def time_block_stability(trades: pd.DataFrame, blocks: int = 4) -> pd.DataFrame:
    """把固定模型的逐筆交易按時間切成連續區塊，檢查edge是否只集中在單一時段。"""
    if trades is None or trades.empty:
        return pd.DataFrame()
    t=trades.copy()
    t["_dt"]=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce")
    t=t.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    if len(t)<blocks:
        return pd.DataFrame()
    # 用時間分位數建立連續區塊
    qs=np.linspace(0,1,blocks+1)
    bounds=t["_dt"].quantile(qs).tolist()
    rows=[]
    for i in range(blocks):
        lo,hi=bounds[i],bounds[i+1]
        g=t[(t["_dt"]>=lo) & (t["_dt"]<=hi if i==blocks-1 else t["_dt"]<hi)]
        m=aggregate_trade_metrics(g)
        by_stock=g.groupby("股票")["淨報酬%"].mean() if not g.empty else pd.Series(dtype=float)
        rows.append({
            "區段":f"時間區段{i+1}/{blocks}","開始":lo,"結束":hi,"交易數":len(g),
            "股票數":int(g["股票"].nunique()) if not g.empty else 0,
            "正期望股票比例":float((by_stock>0).mean()*100) if len(by_stock) else np.nan,
            "期望值中位數":float(by_stock.median()) if len(by_stock) else np.nan,
            **m
        })
    return pd.DataFrame(rows)


def build_multitimeframe_5m_signal(
    d5: pd.DataFrame,
    d15: pd.DataFrame,
    d60: pd.DataFrame,
    entry_rule: str,
) -> pd.DataFrame:
    """
    V1.3.0 多週期研究：
    60m 環境 = K<30
    15m 確認 = KD黃金交叉
    5m 觸發 = KD黃金交叉 / MA5上穿MA15 / 站上VWAP後KD黃金交叉

    高週期欄位以「K棒完成時間」向後平移後再對齊5m，
    避免在高週期K棒尚未收完時偷看該根資料。
    """
    if d5 is None or d15 is None or d60 is None or d5.empty or d15.empty or d60.empty:
        return pd.DataFrame()

    x5 = d5.copy().sort_index()
    x15 = d15.copy().sort_index()
    x60 = d60.copy().sort_index()

    # Yahoo 分K索引通常代表K棒起始時間；轉成可使用時間。
    a15 = pd.DataFrame(index=x15.index + pd.Timedelta(minutes=15))
    a15["CONFIRM_15M"] = x15["KD_GOLD"].fillna(False).to_numpy()
    # 15m黃金交叉確認後，給後續45分鐘作為5m進場窗口。
    a15["CONFIRM_15M_RECENT"] = a15["CONFIRM_15M"].rolling(3, min_periods=1).max().astype(bool)

    a60 = pd.DataFrame(index=x60.index + pd.Timedelta(minutes=60))
    a60["ENV_60M_K"] = x60["K"].to_numpy()
    a60["ENV_60M_LOW"] = (a60["ENV_60M_K"] < 30).fillna(False)

    base = x5.reset_index()
    time_col = base.columns[0]
    base = base.rename(columns={time_col: "_time"})

    z15 = a15.reset_index()
    z15 = z15.rename(columns={z15.columns[0]: "_time"})
    z60 = a60.reset_index()
    z60 = z60.rename(columns={z60.columns[0]: "_time"})

    # V1.3.2：Yahoo 不同 interval 可能回傳 datetime64[ns] / datetime64[us]，
    # merge_asof 要求完全相同 dtype。統一轉為 UTC datetime64[ns] 後再合併。
    def _normalize_merge_time(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ts = pd.to_datetime(out["_time"], utc=True, errors="coerce")
        out["_time"] = ts.astype("datetime64[ns, UTC]")
        out = out.dropna(subset=["_time"]).sort_values("_time").reset_index(drop=True)
        return out

    base = _normalize_merge_time(base)
    z15 = _normalize_merge_time(z15)
    z60 = _normalize_merge_time(z60)

    base = pd.merge_asof(base, z15, on="_time", direction="backward")
    base = pd.merge_asof(base, z60, on="_time", direction="backward")
    base = base.set_index("_time")
    base.index.name = x5.index.name

    env = base["ENV_60M_LOW"].fillna(False)
    confirm = base["CONFIRM_15M_RECENT"].fillna(False)

    confirm_start = confirm & ~confirm.shift(1, fill_value=False)

    if entry_rule == "15m確認後首根5m":
        # 對照組：不再等待5m第二次KD交叉，直接用15m確認後第一根可交易5m。
        trigger = confirm_start
    elif entry_rule == "5m KD黃金交叉":
        trigger = base["KD_GOLD"].fillna(False)
    elif entry_rule == "5m MA5上穿MA15":
        trigger = base["MA5_XUP_MA15"].fillna(False)
    else:
        trigger = base["KD_GOLD"].fillna(False) & base["PRICE_GT_VWAP"].fillna(False)

    base["MTF_SIGNAL"] = env & confirm & trigger
    base["MTF_60M_K"] = base["ENV_60M_K"]
    base["MTF_15M_CONFIRM"] = confirm
    base["MTF_5M_TRIGGER"] = trigger
    return base


def run_multitimeframe_batch(
    symbols: List[str],
    entry_rules: List[str],
    modes: List[str],
    cost: CostConfig,
    period: str,
    allow_overlap: bool = False,
):
    """60m環境 → 15m確認 → 5m觸發；用5m價格執行當沖/隔日/2日回測。"""
    all_rows, all_trades = [], []
    p = st.progress(0, text="下載 5m / 15m / 60m 多週期資料…")

    raw5 = download_intraday_batch(symbols, "5m", period)
    p.progress(0.10, text="5m下載完成")
    raw15 = download_intraday_batch(symbols, "15m", period)
    p.progress(0.20, text="15m下載完成")
    raw60 = download_intraday_batch(symbols, "60m", period)
    p.progress(0.30, text="60m下載完成")

    diagnostics = []
    for iv, raw in [("5m", raw5), ("15m", raw15), ("60m", raw60)]:
        ok = sum(1 for s in symbols if s in raw and not raw[s].empty)
        sample_dtype = ""
        for s in symbols:
            if s in raw and not raw[s].empty:
                sample_dtype = str(raw[s].index.dtype)
                break
        diagnostics.append({
            "週期": iv,
            "要求股票數": len(symbols),
            "成功下載": ok,
            "失敗/空資料": len(symbols)-ok,
            "原始時間型別": sample_dtype,
        })

    for n, symbol in enumerate(symbols, 1):
        if symbol not in raw5 or symbol not in raw15 or symbol not in raw60:
            continue
        if raw5[symbol].empty or raw15[symbol].empty or raw60[symbol].empty:
            continue

        d5 = add_indicators(raw5[symbol])
        d15 = add_indicators(raw15[symbol])
        d60 = add_indicators(raw60[symbol])

        for er in entry_rules:
            mtf = build_multitimeframe_5m_signal(d5, d15, d60, er)
            if mtf.empty:
                continue
            for mode in modes:
                t = backtest(mtf, "5m", "多週期條件", mode, cost)
                raw_count = len(t)
                if not allow_overlap:
                    t = enforce_non_overlapping(t)
                m = metrics(t)
                m["原始訊號數"] = raw_count
                m["重疊排除數"] = raw_count - len(t)
                if not t.empty and "進場時間" in t.columns:
                    et = pd.to_datetime(t["進場時間"], errors="coerce")
                    mins = et.dt.hour * 60 + et.dt.minute
                    m["09:00-10:29交易數"] = int((mins < 630).sum())
                    m["10:30-11:59交易數"] = int(((mins >= 630) & (mins < 720)).sum())
                    m["12:00-12:59交易數"] = int(((mins >= 720) & (mins < 780)).sum())
                    m["13:00後交易數"] = int((mins >= 780).sum())
                rule_name = f"60m K<30 → 15m KD黃金交叉 → {er}"
                all_rows.append({
                    "股票": symbol,
                    "研究主題": research_theme(symbol),
                    "週期": "60m→15m→5m",
                    "規則": rule_name,
                    "持有": mode,
                    **m,
                })
                if not t.empty:
                    tt = t.copy()
                    tt.insert(0, "股票", symbol)
                    tt["多週期規則"] = rule_name
                    tt["回測版本"] = APP_VERSION
                    tt["持倉模式"] = "允許重疊" if allow_overlap else "禁止重疊"
                    all_trades.append(tt)

        p.progress(0.30 + 0.70*n/max(1, len(symbols)), text=f"多週期計算 {symbol}｜{n}/{len(symbols)}")

    p.empty()
    detail = pd.DataFrame(all_rows)
    if not detail.empty:
        detail["回測版本"] = APP_VERSION
        detail["持倉模式"] = "允許重疊" if allow_overlap else "禁止重疊"
        if "PF" not in detail.columns and "Profit Factor" in detail.columns:
            detail["PF"] = detail["Profit Factor"]
    cross = cross_stock_summary(detail[detail["交易數"] > 0].copy()) if not detail.empty else pd.DataFrame()
    if not cross.empty:
        cross.insert(0, "回測版本", APP_VERSION)
        cross.insert(1, "持倉模式", "允許重疊" if allow_overlap else "禁止重疊")
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return detail, cross, trades, pd.DataFrame(diagnostics)


def bars_per_day(interval: str) -> int:
    # 09:00~13:30 約 270 分鐘
    return {"5m": 54, "15m": 18, "60m": 5}.get(interval, 1)


def _session_date(ts):
    try:
        return pd.Timestamp(ts).date()
    except Exception:
        return None


def find_exit_index(d: pd.DataFrame, entry_i: int, mode: str, interval: str) -> int | None:
    if entry_i >= len(d) - 1:
        return None
    entry_date = _session_date(d.index[entry_i])

    if mode == "當沖":
        # 同一交易日最後一根K棒收盤出場
        j = entry_i
        while j + 1 < len(d) and _session_date(d.index[j + 1]) == entry_date:
            j += 1
        return j if j > entry_i else None

    day_map = {"隔日": 1, "2日": 2, "3日": 3, "4日": 4, "5日": 5, "6日": 6, "7日": 7}
    target_days = day_map.get(mode, 1)
    seen = []
    for j in range(entry_i + 1, len(d)):
        dt = _session_date(d.index[j])
        if dt != entry_date and dt not in seen:
            seen.append(dt)
        if len(seen) >= target_days:
            target = seen[target_days - 1]
            k = j
            while k + 1 < len(d) and _session_date(d.index[k + 1]) == target:
                k += 1
            return k
    return None


def backtest(
    d: pd.DataFrame,
    interval: str,
    rule: str,
    mode: str,
    cost: CostConfig,
    min_gap_bars: int = 1,
) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()

    sig = signal_mask(d, rule)
    rows = []
    last_entry = -10**9

    for i in np.flatnonzero(sig.to_numpy()):
        # 訊號必須用已收完的K棒；下一根開盤進場，避免偷看。
        entry_i = int(i) + 1
        if entry_i >= len(d) or entry_i - last_entry < min_gap_bars:
            continue

        # 當沖訊號若下一根已跨日，不建立交易。
        if mode == "當沖" and _session_date(d.index[entry_i]) != _session_date(d.index[i]):
            continue

        exit_i = find_exit_index(d, entry_i, mode, interval)
        if exit_i is None or exit_i <= entry_i:
            continue

        entry = float(d["Open"].iloc[entry_i])
        exitp = float(d["Close"].iloc[exit_i])
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exitp):
            continue

        gross = (exitp / entry - 1) * 100
        cost_pct = cost.roundtrip_cost_pct(daytrade=(mode == "當沖"))
        net = gross - cost_pct

        path = d.iloc[entry_i:exit_i + 1]
        mfe = (float(path["High"].max()) / entry - 1) * 100
        mae = (float(path["Low"].min()) / entry - 1) * 100

        rows.append({
            "訊號時間": d.index[i],
            "進場時間": d.index[entry_i],
            "出場時間": d.index[exit_i],
            "週期": interval,
            "規則": rule,
            "持有": mode,
            "進場價": entry,
            "出場價": exitp,
            "毛報酬%": gross,
            "成本%": cost_pct,
            "淨報酬%": net,
            "MFE%": mfe,
            "MAE%": mae,
            "訊號K": float(d["K"].iloc[i]) if pd.notna(d["K"].iloc[i]) else np.nan,
            "訊號D": float(d["D"].iloc[i]) if pd.notna(d["D"].iloc[i]) else np.nan,
            "K區間": str(d["K_ZONE"].iloc[i]) if pd.notna(d["K_ZONE"].iloc[i]) else "",
            "訊號收盤": float(d["Close"].iloc[i]),
            "訊號VWAP": float(d["VWAP"].iloc[i]) if "VWAP" in d and pd.notna(d["VWAP"].iloc[i]) else np.nan,
            "量比20": float(d["VOL_RATIO20"].iloc[i]) if "VOL_RATIO20" in d and pd.notna(d["VOL_RATIO20"].iloc[i]) else np.nan,
            "MA30斜率3": float(d["MA30_SLOPE3"].iloc[i]) if "MA30_SLOPE3" in d and pd.notna(d["MA30_SLOPE3"].iloc[i]) else np.nan,
            "MA60斜率3": float(d["MA60_SLOPE3"].iloc[i]) if "MA60_SLOPE3" in d and pd.notna(d["MA60_SLOPE3"].iloc[i]) else np.nan,
        })
        last_entry = entry_i

    return pd.DataFrame(rows)


def enforce_non_overlapping(trades: pd.DataFrame) -> pd.DataFrame:
    """同一股票/策略採單一持倉：前一筆尚未出場時，忽略後續新訊號。"""
    if trades is None or trades.empty:
        return trades
    entry_col = next((c for c in ["進場時間","進場日期","EntryTime"] if c in trades.columns), None)
    exit_col = next((c for c in ["出場時間","出場日期","ExitTime"] if c in trades.columns), None)
    if not entry_col or not exit_col:
        return trades
    t = trades.copy()
    t[entry_col] = pd.to_datetime(t[entry_col])
    t[exit_col] = pd.to_datetime(t[exit_col])
    t = t.sort_values(entry_col)
    keep, last_exit = [], None
    for idx, row in t.iterrows():
        if last_exit is None or row[entry_col] > last_exit:
            keep.append(idx)
            last_exit = row[exit_col]
    return t.loc[keep].reset_index(drop=True)

def metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {
            "交易數": 0, "勝率%": np.nan, "平均淨報酬%": np.nan,
            "平均獲利%": np.nan, "平均虧損%": np.nan,
            "盈虧比": np.nan, "ProfitFactor": np.nan,
            "期望值%": np.nan, "最大回撤%": np.nan, "累積報酬%": np.nan,
        }

    r = pd.to_numeric(trades["淨報酬%"], errors="coerce").dropna()
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = (r > 0).mean() * 100
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    payoff = avg_win / abs(avg_loss) if len(wins) and len(losses) and avg_loss != 0 else np.nan
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    expectancy = r.mean()

    equity = (1 + r / 100).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100
    mdd = dd.min() if len(dd) else np.nan
    total = (equity.iloc[-1] - 1) * 100 if len(equity) else np.nan

    return {
        "交易數": int(len(r)),
        "勝率%": win_rate,
        "平均淨報酬%": r.mean(),
        "平均獲利%": avg_win,
        "平均虧損%": avg_loss,
        "盈虧比": payoff,
        "ProfitFactor": pf,
        "期望值%": expectancy,
        "最大回撤%": mdd,
        "累積報酬%": total,
    }



def classify_stock_state(data_map: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """
    V1.2 初版狀態分類：只使用既有 MA / KD / 量價特徵，
    不把股票永久貼標籤，而是描述「目前資料尾端」的狀態。
    """
    d = data_map.get("60m", pd.DataFrame())
    if d.empty:
        d = data_map.get("15m", pd.DataFrame())
    if d.empty:
        d = data_map.get("5m", pd.DataFrame())
    if d.empty:
        return {"市場狀態": "資料不足", "狀態分數": 0, "說明": "無可用分K"}

    z = d.dropna(subset=["Close"]).copy()
    if len(z) < 30:
        return {"市場狀態": "資料不足", "狀態分數": 0, "說明": "K棒數不足"}

    last = z.iloc[-1]
    trend_score = 0
    reversal_score = 0
    volatile_score = 0

    if pd.notna(last.get("MA30_SLOPE3")) and last["MA30_SLOPE3"] > 0:
        trend_score += 1
    if pd.notna(last.get("MA60_SLOPE3")) and last["MA60_SLOPE3"] > 0:
        trend_score += 1
    if bool(last.get("PRICE_GT_MA200", False)):
        trend_score += 1
    if bool(last.get("MA_BULL_5_15", False)) and bool(last.get("MA_BULL_15_30", False)):
        trend_score += 1
    if pd.notna(last.get("VWAP")) and last["Close"] > last["VWAP"]:
        trend_score += 1

    k = last.get("K", np.nan)
    if pd.notna(k) and k < 30:
        reversal_score += 2
    if bool(last.get("KD_GOLD", False)):
        reversal_score += 1
    if pd.notna(last.get("MA30_SLOPE3")) and last["MA30_SLOPE3"] <= 0:
        reversal_score += 1

    # 以近20根 true range / close 的中位數作相對波動描述；不加入 ATR 交易規則。
    tr = pd.concat([
        (z["High"] - z["Low"]).abs(),
        (z["High"] - z["Close"].shift(1)).abs(),
        (z["Low"] - z["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    rel_tr = (tr / z["Close"].replace(0, np.nan) * 100).tail(20).median()
    if pd.notna(rel_tr) and rel_tr >= 2.0:
        volatile_score = 2
    elif pd.notna(rel_tr) and rel_tr >= 1.2:
        volatile_score = 1

    ma_spread = np.nan
    if pd.notna(last.get("MA5")) and pd.notna(last.get("MA60")) and last["Close"] != 0:
        ma_spread = abs(last["MA5"] - last["MA60"]) / last["Close"] * 100

    if volatile_score >= 2:
        state = "高波動"
        score = volatile_score
    elif trend_score >= 4:
        state = "趨勢"
        score = trend_score
    elif reversal_score >= 3:
        state = "低檔轉折"
        score = reversal_score
    elif pd.notna(ma_spread) and ma_spread < 1.0:
        state = "震盪/均線糾結"
        score = 1
    else:
        state = "混合"
        score = max(trend_score, reversal_score, volatile_score)

    return {
        "市場狀態": state,
        "狀態分數": score,
        "趨勢分": trend_score,
        "轉折分": reversal_score,
        "相對波動%": round(float(rel_tr), 3) if pd.notna(rel_tr) else np.nan,
        "MA5-MA60距離%": round(float(ma_spread), 3) if pd.notna(ma_spread) else np.nan,
    }




# V1.2.2 候選母池：不是「固定熱門排名」；真正入選名單會用近期日線重新排序。
# 刻意涵蓋大型權值、電子次產業、金融、傳產與高交易活躍族群。
SHORT_TERM_UNIVERSE = [
    # 半導體 / IC設計 / 封測
    "2330.TW","2303.TW","2454.TW","3034.TW","2379.TW","3443.TW","3661.TW","3711.TW","6239.TW","2449.TW",
    "5347.TWO","6770.TW","6531.TW","5269.TW",
    # AI伺服器 / ODM / 板卡
    "2317.TW","2382.TW","3231.TW","6669.TW","2356.TW","4938.TW","2357.TW","2376.TW","2377.TW","2395.TW",
    # PCB / 載板 / CCL
    "2368.TW","3037.TW","8046.TW","3189.TW","4958.TW","2383.TW","6213.TW","6274.TW","5439.TWO","6191.TWO",
    # 玻纖布 / PCB材料
    "1815.TW","1802.TW","4763.TW",
    # 光通訊 / CPO / 網通
    "2345.TW","3081.TWO","3163.TWO","3363.TWO","4979.TWO","6442.TWO","6530.TW","3596.TW","6285.TW","5388.TWO","3704.TW","4904.TW",
    # DRAM / NAND / 儲存
    "2408.TW","2344.TW","8299.TWO","3260.TWO","3006.TW","5351.TWO","5289.TW","4967.TW",
    # 散熱 / 機殼 / 電源
    "3017.TW","3324.TWO","3653.TW","8210.TW","2308.TW","6412.TW","6409.TW",
    # 被動元件 / 面板
    "2327.TW","2492.TW","3026.TW","2375.TW","2409.TW","3481.TW",
    # 金融對照
    "2881.TW","2882.TW","2891.TW","2886.TW","2884.TW","2885.TW","2887.TW","2892.TW","5871.TW","5880.TW",
    # 航運 / 鋼鐵 / 塑化 / 營建 / 生技對照
    "2603.TW","2609.TW","2615.TW","2618.TW","2606.TW","2610.TW","2002.TW","2014.TW","2027.TW",
    "1301.TW","1303.TW","1326.TW","6505.TW","2542.TW","5522.TW","2501.TW","6446.TW","4743.TW","1795.TW",
    # 其他大型 / 活躍股
    "2207.TW","2301.TW","2353.TW","2354.TW","2385.TW","2404.TW","2441.TW","2451.TW","2474.TW",
    "3008.TW","3019.TW","3234.TW","3706.TW","5876.TW","6781.TW","6805.TW","8454.TW","1477.TW","1590.TW","2105.TW","2201.TW","2634.TW","9904.TW","9910.TW","9914.TW"
]

THEME_MAP = {
    "PCB/載板/CCL": {"2368","3037","8046","3189","4958","2383","6213","6274","5439","6191"},
    "玻纖布/材料": {"1815","1802","4763"},
    "光通訊/CPO": {"3081","3163","3363","4979","6442","6530"},
    "DRAM/NAND/記憶體": {"2408","2344","8299","3260","3006","5351","5289","4967"},
    "AI伺服器/ODM": {"2317","2382","3231","6669","2356","4938"},
    "半導體/IC設計": {"2330","2303","2454","3034","2379","3443","3661","6531","5269"},
    "散熱/電源": {"3017","3324","3653","8210","2308","6412","6409"},
    "網通": {"2345","3596","6285","5388","3704","4904"},
    "金融對照": {"2881","2882","2891","2886","2884","2885","2887","2892","5871","5880"},
    "航運對照": {"2603","2609","2615","2618","2606","2610"},
}
def research_theme(symbol: str) -> str:
    code = str(symbol).split(".")[0]
    for theme, codes in THEME_MAP.items():
        if code in codes:
            return theme
    return "其他/對照"

@st.cache_data(ttl=1800, show_spinner=False)
def rank_short_term_pool(symbols: List[str], top_n: int = 30, lookback: str = "1mo") -> pd.DataFrame:
    """一次批次下載母池日線，避免逐檔請求造成 Streamlit Cloud 中斷。"""
    if not symbols:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers=" ".join(symbols),
            period=lookback,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for symbol in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(0):
                    continue
                d = raw[symbol].copy()
            else:
                d = raw.copy()
            needed = {"Close","High","Low","Volume"}
            if not needed.issubset(d.columns):
                continue
            x = d.dropna(subset=["Close","High","Low","Volume"]).tail(20)
            if len(x) < 8:
                continue
            turn = (x["Close"] * x["Volume"]).replace([np.inf,-np.inf], np.nan)
            amp = ((x["High"]-x["Low"]) / x["Close"].replace(0,np.nan) * 100).replace([np.inf,-np.inf],np.nan)
            rows.append({
                "股票": symbol,
                "研究主題": research_theme(symbol),
                "有效日數": len(x),
                "20日成交金額中位數": float(turn.median()),
                "20日成交量中位數": float(x["Volume"].median()),
                "20日振幅中位數%": float(amp.median()),
            })
        except Exception:
            continue

    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r["成交金額分位"] = r["20日成交金額中位數"].rank(pct=True)
    r["成交量分位"] = r["20日成交量中位數"].rank(pct=True)
    r["振幅分位"] = r["20日振幅中位數%"].rank(pct=True)
    r["短線可交易分"] = (
        r["成交金額分位"] * 50 +
        r["成交量分位"] * 20 +
        r["振幅分位"] * 30
    )
    return r.sort_values("短線可交易分", ascending=False).head(int(top_n)).reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def download_intraday_batch(symbols: List[str], interval: str, period: str) -> Dict[str, pd.DataFrame]:
    """
    V1.16.1：
    Yahoo 60m 在「較長期間 + 較多股票」一次下載時，可能整批失敗。
    改為分批下載，單批失敗時再縮小重試，避免整個暖機版變成空資料。
    """
    symbols=list(dict.fromkeys(symbols))
    out={s:pd.DataFrame() for s in symbols}
    if not symbols:
        return out

    def _fetch(batch):
        try:
            return yf.download(
                tickers=" ".join(batch),
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
                prepost=False,
                group_by="ticker",
            )
        except Exception:
            return pd.DataFrame()

    # 60m + 長期間時用較小批次，減少Yahoo整批失敗。
    batch_size=30 if interval=="60m" and period in ["6mo","1y","2y","5y","10y","max"] else 60

    for i in range(0,len(symbols),batch_size):
        batch=symbols[i:i+batch_size]
        raw=_fetch(batch)

        # 若整批失敗，再拆成10檔重試。
        sub_batches=[batch]
        if raw is None or raw.empty:
            sub_batches=[batch[j:j+10] for j in range(0,len(batch),10)]
        else:
            sub_batches=None

        raws=[]
        if sub_batches is None:
            raws=[(batch,raw)]
        else:
            for sb in sub_batches:
                rr=_fetch(sb)
                raws.append((sb,rr))

        for sb,rr in raws:
            if rr is None or rr.empty:
                continue
            for symbol in sb:
                try:
                    if isinstance(rr.columns,pd.MultiIndex):
                        if symbol not in rr.columns.get_level_values(0):
                            continue
                        d=rr[symbol].copy()
                    else:
                        if len(sb)!=1:
                            continue
                        d=rr.copy()

                    need=["Open","High","Low","Close","Volume"]
                    if not all(c in d.columns for c in need):
                        continue
                    d=d[need].copy()
                    for c in need:
                        d[c]=pd.to_numeric(d[c],errors="coerce")
                    d=d.dropna(subset=["Open","High","Low","Close"])
                    if d.empty:
                        continue

                    idx=pd.to_datetime(d.index)
                    if getattr(idx,"tz",None) is not None:
                        try:
                            idx=idx.tz_convert("Asia/Taipei").tz_localize(None)
                        except Exception:
                            idx=idx.tz_localize(None)
                    d.index=idx
                    out[symbol]=d.sort_index()
                except Exception:
                    continue
    return out


SECTOR_POOLS = {
    "半導體/晶圓": ["2330", "2303", "5347", "6770"],
    "IC設計": ["2454", "3034", "2379", "3443"],
    "AI伺服器/ODM": ["2382", "3231", "6669", "2356"],
    "電腦品牌/板卡": ["2357", "2376", "2377", "2395"],
    "PCB/載板": ["3037", "8046", "3189", "4958"],
    "散熱/機殼": ["3017", "3324", "3653", "8210"],
    "網通": ["2345", "3596", "6285", "5388"],
    "記憶體": ["2408", "2344", "8299", "3260"],
    "面板": ["2409", "3481"],
    "被動元件": ["2327", "2492", "3026"],
    "電源/能源管理": ["2308", "6412", "6409"],
    "封測": ["3711", "6239", "2449"],
    "金融": ["2881", "2882", "2891", "2886"],
    "航運": ["2603", "2609", "2615", "2618"],
    "鋼鐵": ["2002", "2014", "2027"],
    "塑化": ["1301", "1303", "1326"],
    "生技": ["6446", "4743", "1795"],
    "營建": ["2542", "5522", "2501"],
}

def sector_symbols(selected_sectors: List[str], per_sector: int, market: str) -> List[str]:
    out = []
    for sec in selected_sectors:
        for code in SECTOR_POOLS.get(sec, [])[:per_sector]:
            s = normalize_symbol(code, market)
            if s not in out:
                out.append(s)
    return out


def parse_batch_codes(text: str, market: str) -> List[str]:
    raw = text.replace("，", ",").replace("、", ",").replace("\n", ",").replace(" ", ",")
    codes = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        s = normalize_symbol(x, market)
        if s not in codes:
            codes.append(s)
    return codes



def code_from_symbol(symbol: str) -> str:
    return str(symbol).split(".")[0]

def sector_of_symbol(symbol: str) -> str:
    code = code_from_symbol(symbol)
    for sec, codes in SECTOR_POOLS.items():
        if code in codes:
            return sec
    return "其他/手動"

def sector_strategy_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    x = detail[detail["交易數"] > 0].copy()
    if x.empty:
        return pd.DataFrame()
    x["族群"] = x["股票"].map(sector_of_symbol)

    def positive_rate(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return (s > 0).mean() * 100 if len(s) else np.nan

    return x.groupby(["族群", "週期", "規則", "持有"], dropna=False).agg(
        股票數=("股票", "nunique"),
        總交易數=("交易數", "sum"),
        正期望股票比例=("期望值%", positive_rate),
        期望值中位數=("期望值%", "median"),
        PF中位數=("ProfitFactor", "median"),
        平均勝率=("勝率%", "mean"),
    ).reset_index()



def cross_stock_summary(batch_summary: pd.DataFrame) -> pd.DataFrame:
    if batch_summary.empty:
        return pd.DataFrame()

    def positive_rate(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return (x > 0).mean() * 100 if len(x) else np.nan

    g = batch_summary.groupby(["週期", "規則", "持有"], dropna=False).agg(
        股票數=("股票", "nunique"),
        總交易數=("交易數", "sum"),
        正期望股票比例=("期望值%", positive_rate),
        平均期望值=("期望值%", "mean"),
        期望值中位數=("期望值%", "median"),
        平均PF=("ProfitFactor", "mean"),
        PF中位數=("ProfitFactor", "median"),
        平均勝率=("勝率%", "mean"),
        平均最大回撤=("最大回撤%", "mean"),
    ).reset_index()

    # 穩定度不是「最佳策略評分」，只是描述跨股票一致性。
    g["樣本覆蓋率"] = g["股票數"] / max(1, batch_summary["股票"].nunique()) * 100
    # 研究標記，不是投資評級：避免只看最高期望值。
    g["研究穩健標記"] = np.select(
        [
            (g["股票數"] >= 10) & (g["總交易數"] >= 100) &
            (g["正期望股票比例"] >= 70) & (g["期望值中位數"] > 0) & (g["PF中位數"] > 1.2),
            (g["股票數"] >= 5) & (g["總交易數"] >= 50) &
            (g["正期望股票比例"] >= 60) & (g["期望值中位數"] > 0) & (g["PF中位數"] > 1.0),
        ],
        ["跨股一致性較高", "值得續測"],
        default="證據不足"
    )
    def evidence_gap(row):
        gaps = []
        if row["股票數"] < 10: gaps.append("股票數<10")
        if row["總交易數"] < 100: gaps.append("交易數<100")
        if row["正期望股票比例"] < 70: gaps.append("正期望比例<70%")
        if pd.isna(row["期望值中位數"]) or row["期望值中位數"] <= 0: gaps.append("期望值中位數<=0")
        if pd.isna(row["PF中位數"]) or row["PF中位數"] <= 1.2: gaps.append("PF中位數<=1.2")
        return "；".join(gaps) if gaps else "主要研究門檻通過"
    g["證據缺口"] = g.apply(evidence_gap, axis=1)
    return g


def theme_strategy_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty or "研究主題" not in detail.columns:
        return pd.DataFrame()
    d = detail[detail["交易數"] > 0].copy()
    if d.empty:
        return pd.DataFrame()
    rows = []
    for key, x in d.groupby(["研究主題","週期","規則","持有"], dropna=False):
        pf_col = "PF" if "PF" in x.columns else ("Profit Factor" if "Profit Factor" in x.columns else None)
        pf = x[pf_col].replace([np.inf,-np.inf],np.nan) if pf_col else pd.Series(dtype=float)
        rows.append({
            "研究主題":key[0],"週期":key[1],"規則":key[2],"持有":key[3],
            "股票數":int(x["股票"].nunique()),"總交易數":int(x["交易數"].sum()),
            "正期望股票比例":float((x["期望值%"]>0).mean()*100),
            "期望值中位數":float(x["期望值%"].median()),
            "PF中位數":float(pf.median()) if not pf.empty else np.nan,"平均勝率":float(x["勝率%"].mean()),
            "平均最大回撤":float(x["最大回撤%"].mean())
        })
    return pd.DataFrame(rows).sort_values(
        ["正期望股票比例","期望值中位數","PF中位數","總交易數"],
        ascending=[False,False,False,False]).reset_index(drop=True)

def run_batch_matrix(symbols: List[str], intervals: List[str], rules: List[str], modes: List[str],
                     cost: CostConfig, period: str, allow_overlap: bool = False):
    all_rows, states = [], []
    total = max(1, len(symbols))
    p = st.progress(0, text="批次下載分K資料…")

    # 每個 interval 只呼叫 Yahoo 一次
    interval_raw = {}
    for i, interval in enumerate(intervals, 1):
        interval_raw[interval] = download_intraday_batch(symbols, interval, period)
        p.progress(min(0.25, 0.25*i/max(1,len(intervals))),
                   text=f"完成 {interval} 批次下載｜{i}/{len(intervals)}")

    success_count = {iv: 0 for iv in intervals}
    for idx, symbol in enumerate(symbols, 1):
        local_data = {}
        for interval in intervals:
            raw = interval_raw.get(interval, {}).get(symbol, pd.DataFrame())
            if not raw.empty:
                success_count[interval] += 1
                local_data[interval] = add_indicators(raw)
            else:
                local_data[interval] = pd.DataFrame()

        state = classify_stock_state(local_data)
        states.append({"股票": symbol, **state})

        for interval in intervals:
            d = local_data[interval]
            for rule in rules:
                for mode in modes:
                    t = backtest(d, interval, rule, mode, cost)
                    raw_signal_count = len(t)
                    if not allow_overlap:
                        t = enforce_non_overlapping(t)
                    m = metrics(t)
                    m["原始訊號數"] = raw_signal_count
                    m["重疊排除數"] = raw_signal_count - len(t)
                    all_rows.append({
                        "股票": symbol, "研究主題": research_theme(symbol),
                        "市場狀態": state["市場狀態"],
                        "週期": interval, "規則": rule, "持有": mode, **m
                    })
        p.progress(0.25 + 0.75*idx/total, text=f"策略計算 {symbol}｜{idx}/{total}")

    p.empty()
    detail = pd.DataFrame(all_rows)
    if not detail.empty:
        detail["回測版本"] = APP_VERSION
        detail["持倉模式"] = "允許重疊" if allow_overlap else "禁止重疊"
    if not detail.empty and "PF" not in detail.columns and "Profit Factor" in detail.columns:
        detail["PF"] = detail["Profit Factor"]
    states_df = pd.DataFrame(states)
    cross = cross_stock_summary(detail[detail["交易數"] > 0].copy()) if not detail.empty else pd.DataFrame()
    if not cross.empty:
        cross.insert(0, "回測版本", APP_VERSION)
        cross.insert(1, "持倉模式", "允許重疊" if allow_overlap else "禁止重疊")
    sector_summary = sector_strategy_summary(detail)
    theme_summary = theme_strategy_summary(detail)
    diagnostics = pd.DataFrame([
        {"週期": iv, "要求股票數": len(symbols), "成功下載": success_count[iv],
         "失敗/空資料": len(symbols)-success_count[iv]}
        for iv in intervals
    ])
    return detail, cross, states_df, sector_summary, theme_summary, diagnostics


def run_matrix(symbol: str, intervals: List[str], rules: List[str], modes: List[str],
               cost: CostConfig, period: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    summary = []
    data_map = {}
    trade_map = {}

    total = max(1, len(intervals) * len(rules) * len(modes))
    done = 0
    bar = st.progress(0, text="建立多週期回測矩陣…")

    for interval in intervals:
        raw = download_intraday(symbol, interval, period)
        d = add_indicators(raw) if not raw.empty else pd.DataFrame()
        data_map[interval] = d

        for rule in rules:
            for mode in modes:
                key = f"{interval}|{rule}|{mode}"
                t = backtest(d, interval, rule, mode, cost)
                trade_map[key] = t
                m = metrics(t)
                summary.append({"週期": interval, "規則": rule, "持有": mode, **m})
                done += 1
                bar.progress(done / total, text=f"回測 {interval}｜{rule}｜{mode}")

    bar.empty()
    return pd.DataFrame(summary), data_map, trade_map


def fmt_summary(s: pd.DataFrame) -> pd.DataFrame:
    z = s.copy()
    for c in ["勝率%", "平均淨報酬%", "平均獲利%", "平均虧損%", "盈虧比",
              "ProfitFactor", "期望值%", "最大回撤%", "累積報酬%"]:
        if c in z:
            z[c] = pd.to_numeric(z[c], errors="coerce").round(3)
    return z


def plot_chart(d: pd.DataFrame, interval: str):
    if d.empty:
        st.warning("沒有可畫的資料。")
        return
    show = d.tail(300)
    if not PLOTLY_OK:
        st.line_chart(show[["Close", "MA5", "MA15", "MA30", "MA60", "MA200"]])
        st.line_chart(show[["K", "D"]])
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(
        x=show.index, open=show.Open, high=show.High, low=show.Low, close=show.Close, name=interval
    ), row=1, col=1)
    for n in MA_LIST:
        fig.add_trace(go.Scatter(x=show.index, y=show[f"MA{n}"], mode="lines", name=f"MA{n}"), row=1, col=1)
    if "VWAP" in show.columns:
        fig.add_trace(go.Scatter(x=show.index, y=show["VWAP"], mode="lines", name="VWAP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["K"], mode="lines", name="K"), row=2, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["D"], mode="lines", name="D"), row=2, col=1)
    fig.add_hline(y=80, row=2, col=1)
    fig.add_hline(y=20, row=2, col=1)
    fig.update_layout(height=760, xaxis_rangeslider_visible=False, hovermode="x unified", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})


st.title(f"⚡ {APP_NAME}")
st.caption(f"{APP_VERSION}｜今日60m短線雷達｜核心：KD黃金交叉 + K<30")
st.info("主畫面只保留每天會看的雷達；研究參數與成本設定收進進階區，減少操作干擾。")

with st.sidebar:
    st.header("⚡ 今日雷達")
    st.caption("標準研究設定已固定。日常使用只要按更新。")

    simple_mode = st.radio("操作模式", ["今日雷達", "進階研究"], index=0)
    research_mode = "60m五日OOS驗證"
    top_n = st.slider("掃描股票數", 20, 100, 100, step=10)
    period = st.selectbox("資料長度", ["60d", "1mo"], index=0)

    selected_intervals = ["60m"]
    selected_rules = ["KD黃金交叉 + K<30"]
    selected_modes = ["5日"]
    overlap_mode = "禁止重疊（較接近實際單一持倉）"
    allow_overlap = False
    code, market = "2330", "上市"
    symbol = normalize_symbol(code, market)
    pool_mode = "動態短線TOP池"
    batch_text = "2330,2357,3711,2317,2454,2382"
    selected_sectors = ["半導體/晶圓", "IC設計", "AI伺服器/ODM"]
    per_sector = 3
    mtf_entry_rules, mtf_modes = [], []

    with st.expander("⚙️ 進階研究設定", expanded=(simple_mode=="進階研究")):
        if simple_mode == "進階研究":
            research_mode = st.radio("研究模式",
                ["TOP50暖機修正驗證","TOP50訊號品質健診","市場環境健診_TOP50","核心池規模WalkForward","全市場WalkForward驗證","全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診","單一股票","跨股票批次","多週期當沖/隔日驗證","60m五日OOS驗證"], index=0)
            if research_mode == "TOP50暖機修正驗證":
                st.caption("比較舊3mo直接計算 vs 6mo指標暖機後只評估最後3mo；同時修正訊號時間誤當UTC的問題。")
            elif research_mode == "TOP50訊號品質健診":
                st.caption("固定真正Walk-Forward TOP50與核心策略，只診斷K深度、量比20、MA30/60方向與60m訊號時段。")
            elif research_mode == "市場環境健診_TOP50":
                st.caption("固定真正Walk-Forward TOP50與核心策略，只診斷訊號前一完成日的TAIEX MA15與5日市場狀態。")
            elif research_mode == "核心池規模WalkForward":
                st.caption("真正歷史Walk-Forward比較每日流動性TOP50 / TOP100 / TOP150 / TOP200；不使用熱門增補。")
            elif research_mode == "全市場WalkForward驗證":
                st.caption("真正歷史Walk-Forward：每個交易日只用當時已完成資料建立TOP100與熱門增補，再跑固定60m策略。")
            elif research_mode == "全市場候選策略驗證":
                st.caption("本模式會執行固定策略回測：60m KD黃金交叉＋K<30＋持有5日，並比較核心TOP100與熱門增補。")
            elif research_mode in ["全市場股票池研究","股票池2.0研究","股票池2.0歷史驗證","股票池健診"]:
                st.caption("本模式只分析股票池／研究資料。")
            else:
                code = st.text_input("股票代號", value="2330")
                market = st.radio("市場", ["上市","上櫃"], horizontal=True)
                symbol = normalize_symbol(code, market)
                selected_intervals = st.multiselect("K棒週期", INTERVALS, default=["60m"])
            all_rules = [
                "MA5上穿MA15","MA15上穿MA30","KD黃金交叉","MA5>15 + KD黃金交叉",
                "MA5>15>30 + KD黃金交叉","MA5>15>30>60 + KD黃金交叉","完整多頭排列",
                "完整多頭排列 + KD黃金交叉","站上MA200 + KD黃金交叉","KD黃金交叉 + K<30",
                "KD黃金交叉 + K30-50","KD黃金交叉 + K50-80","KD黃金交叉 + K>80",
                "KD黃金交叉 + MA30向上","KD黃金交叉 + MA60向上","KD黃金交叉 + 量比>1.2",
                "KD黃金交叉 + 量比>1.5","KD黃金交叉 + 站上VWAP","MA5>15 + KD + 站上VWAP"
            ]
            if research_mode not in ["TOP50暖機修正驗證","TOP50訊號品質健診","市場環境健診_TOP50","核心池規模WalkForward","全市場WalkForward驗證","全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診"]:
                selected_rules = st.multiselect("進場規則", all_rules, default=["KD黃金交叉 + K<30"])
                selected_modes = st.multiselect("持有方式",
                    ["當沖","隔日","2日","3日","4日","5日","6日","7日"], default=["5日"])
                overlap_mode = st.radio("持倉期間新訊號",
                    ["禁止重疊（較接近實際單一持倉）","允許重疊（訊號事件研究）"])
                allow_overlap = overlap_mode.startswith("允許")
            if research_mode == "跨股票批次":
                pool_mode = st.radio("批次股票池", ["手動輸入","族群代表池","動態短線TOP池"], index=2)
                batch_text = st.text_area("批次股票代號", value=batch_text)
                selected_sectors = st.multiselect("選擇族群", list(SECTOR_POOLS.keys()), default=selected_sectors)
                per_sector = st.slider("每族群代表檔數",1,4,3)
            if research_mode == "多週期當沖/隔日驗證":
                mtf_entry_rules = st.multiselect("5分鐘進場觸發",
                    ["15m確認後首根5m","5m KD黃金交叉","5m MA5上穿MA15","5m KD黃金交叉+站上VWAP"],
                    default=["15m確認後首根5m","5m KD黃金交叉"])
                mtf_modes = st.multiselect("短線出場方式", ["當沖","隔日","2日"], default=["當沖","隔日","2日"])

    with st.expander("💰 成本設定"):
        fee_discount = st.number_input("手續費折數", min_value=0.1, max_value=1.0, value=0.28, step=0.01)
        slip_bp = st.number_input("單邊滑價（bp）", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
    cost = CostConfig(fee_discount=fee_discount, slippage_pct=slip_bp / 10000)

    if simple_mode=="今日雷達":
        _btn_label="🔄 更新今日雷達"
    else:
        _btn_label={
            "TOP50暖機修正驗證":"🧰 執行TOP50暖機修正驗證",
            "TOP50訊號品質健診":"🔬 執行TOP50訊號品質健診",
            "市場環境健診_TOP50":"🌤️ 執行TOP50市場環境健診",
            "核心池規模WalkForward":"📏 驗證核心池TOP50/100/150/200",
            "全市場WalkForward驗證":"🧭 執行全市場Walk-Forward",
            "全市場股票池研究":"🌐 建立全市場研究股票池",
            "全市場候選策略驗證":"🧪 驗證核心TOP100 vs 熱門增補",
            "股票池2.0研究":"🧭 開始股票池2.0即時診斷",
            "股票池2.0歷史驗證":"🧪 開始股票池2.0歷史驗證",
            "股票池健診":"🧭 開始股票池健診",
        }.get(research_mode,"🚀 開始策略健診")
    run = st.button(_btn_label, type="primary", use_container_width=True)


if simple_mode == "今日雷達":
    st.markdown("""<style>div[data-baseweb="tab-list"]{display:none!important;}</style>""", unsafe_allow_html=True)

if simple_mode=="進階研究" and research_mode=="TOP50暖機修正驗證" and not run:
    st.warning("V1.16.2 已不再崩潰，但畫面顯示「歷史Walk-Forward資格資料為空」，表示真正問題在全市場6mo日K下載階段。V1.16.3 將日K改成25檔小批、關閉threads、失敗重試並對缺漏股票5檔補抓，同時把官方股票數、日K成功數、有效歷史序列數與資格列數直接顯示在資料完整度。")

if run and simple_mode=="進階研究" and research_mode=="TOP50暖機修正驗證":
    st.subheader("🧰 TOP50暖機修正驗證")
    with st.spinner("同時跑舊3mo版本與6mo暖機版本，比較最後3個月結果…"):
        _wu_sum,_wu_blocks,_wu_trades,_wu_errs,_wu_check=validate_top50_warmup_correction(cost)
    st.session_state["st_v1163_warmup"]={
        "summary":_wu_sum,"blocks":_wu_blocks,"trades":_wu_trades,
        "check":_wu_check,"errors":_wu_errs
    }

_wu=st.session_state.get("st_v1163_warmup")
if simple_mode=="進階研究" and research_mode=="TOP50暖機修正驗證" and _wu:
    _us=_wu.get("summary",pd.DataFrame()); _ub=_wu.get("blocks",pd.DataFrame())
    _ut=_wu.get("trades",pd.DataFrame()); _uc=_wu.get("check",pd.DataFrame()); _ue=_wu.get("errors",[])
    st.subheader("🧰 TOP50暖機修正結果")
    if _ue:
        st.warning("資料來源異常："+"；".join(_ue))
    st.info("先確認暖機修正是否改變第1段與整體結論；若有明顯差異，後續所有研究統一改用『長資料暖機＋固定評估窗』。")
    if not _uc.empty:
        st.markdown("#### 資料完整度")
        st.dataframe(_uc.round(3),use_container_width=True,hide_index=True)
    if not _us.empty:
        st.markdown("#### 舊版 vs 暖機版｜全部 / 樣本內 / 樣本外")
        st.dataframe(_us.round(3),use_container_width=True,hide_index=True)
    if not _ub.empty:
        st.markdown("#### 舊版 vs 暖機版｜四段時間")
        st.dataframe(_ub.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看暖機版逐筆交易"):
        st.dataframe(_ut.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【TOP50暖機修正摘要】",_us.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50暖機修正摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50暖機修正四段穩定度】",_ub.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50暖機修正四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50暖機修正逐筆交易】",_ut.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50暖機修正逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50暖機資料完整度】",_uc.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50暖機資料完整度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="TOP50訊號品質健診" and not run:
    st.info("V1.15.0 顯示MA15/市場狀態無法穩定解釋第1段失效，且部分關係在樣本內外反轉。這一版不加市場Gate，改固定TOP50檢查訊號本身品質。")

if run and simple_mode=="進階研究" and research_mode=="TOP50訊號品質健診":
    st.subheader("🔬 TOP50訊號品質健診")
    with st.spinner("重建真正Walk-Forward TOP50交易，檢查K深度、量比20、MA30/60方向與60m時段…"):
        _sq_sum,_sq_blocks,_sq_trades,_sq_errs=validate_top50_signal_quality(cost,period="3mo")
    st.session_state["st_v1151_signal_quality"]={
        "summary":_sq_sum,"blocks":_sq_blocks,"trades":_sq_trades,"errors":_sq_errs
    }

_sq=st.session_state.get("st_v1151_signal_quality")
if simple_mode=="進階研究" and research_mode=="TOP50訊號品質健診" and _sq:
    _qs=_sq.get("summary",pd.DataFrame()); _qb=_sq.get("blocks",pd.DataFrame())
    _qt=_sq.get("trades",pd.DataFrame()); _qe=_sq.get("errors",[])
    st.subheader("🔬 TOP50訊號品質健診結果")
    if _qe:
        st.warning("資料來源異常："+"；".join(_qe))
    st.info("這一版仍然只做診斷；不會因單一分桶績效最好就直接改策略。")
    if not _qs.empty:
        st.markdown("#### 訊號品質分組｜全部 / 樣本內 / 樣本外")
        st.dataframe(_qs.round(3),use_container_width=True,hide_index=True)
    if not _qb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_qb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆交易品質標籤"):
        st.dataframe(_qt.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【TOP50訊號品質健診摘要】",_qs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50訊號品質健診摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50訊號品質四段穩定度】",_qb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50訊號品質四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50訊號品質逐筆交易】",_qt.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50訊號品質逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="市場環境健診_TOP50" and not run:
    st.info("V1.14.1 顯示TOP50在全部、樣本外與第2~4段皆優於更大的核心池，但第1段所有規模都失效。這一版固定TOP50，不再調股票池，只檢查市場環境。")

if run and simple_mode=="進階研究" and research_mode=="市場環境健診_TOP50":
    st.subheader("🌤️ TOP50市場環境健診")
    with st.spinner("重建TOP50 Walk-Forward交易，並對齊每筆訊號前一完成日的TAIEX市場狀態…"):
        _mr_sum,_mr_blocks,_mr_trades,_mr_errs=validate_market_regime_top50(cost,period="3mo")
    st.session_state["st_v1150_market_regime"]={
        "summary":_mr_sum,"blocks":_mr_blocks,"trades":_mr_trades,"errors":_mr_errs
    }

_mr=st.session_state.get("st_v1150_market_regime")
if simple_mode=="進階研究" and research_mode=="市場環境健診_TOP50" and _mr:
    _ms=_mr.get("summary",pd.DataFrame()); _mb=_mr.get("blocks",pd.DataFrame())
    _mt=_mr.get("trades",pd.DataFrame()); _me=_mr.get("errors",[])
    st.subheader("🌤️ TOP50市場環境健診結果")
    if _me:
        st.warning("資料來源異常："+"；".join(_me))
    st.info("這一版只做診斷，不會因結果直接把MA15或5日報酬寫成濾網。必須先看樣本外與四段是否一致。")
    if not _ms.empty:
        st.markdown("#### 市場環境分組｜全部 / 樣本內 / 樣本外")
        st.dataframe(_ms.round(3),use_container_width=True,hide_index=True)
    if not _mb.empty:
        st.markdown("#### 四段時間 × 市場狀態")
        st.dataframe(_mb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆交易市場標籤"):
        st.dataframe(_mt.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【TOP50市場環境健診摘要】",_ms.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50市場環境健診摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50市場環境四段穩定度】",_mb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50市場環境四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50市場環境逐筆交易】",_mt.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50市場環境逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="核心池規模WalkForward" and not run:
    st.info("V1.14.0 已確認熱門增補在真正Walk-Forward下拖累結果。這一版不調策略，只檢查核心流動池選50、100、150、200檔時是否穩定。")

if run and simple_mode=="進階研究" and research_mode=="核心池規模WalkForward":
    st.subheader("📏 核心池規模Walk-Forward")
    with st.spinner("逐日建立全市場歷史流動性排名，下載TOP200曾入選股票的60m資料並比較四種核心池規模…"):
        _cs_sum,_cs_blocks,_cs_trades,_cs_daily,_cs_errs=validate_core_pool_sizes(cost,period="3mo")
    st.session_state["st_v1141_core_sizes"]={
        "summary":_cs_sum,"blocks":_cs_blocks,"trades":_cs_trades,
        "daily":_cs_daily,"errors":_cs_errs
    }

_cs=st.session_state.get("st_v1141_core_sizes")
if simple_mode=="進階研究" and research_mode=="核心池規模WalkForward" and _cs:
    _ss=_cs.get("summary",pd.DataFrame()); _sb=_cs.get("blocks",pd.DataFrame())
    _st=_cs.get("trades",pd.DataFrame()); _sd=_cs.get("daily",pd.DataFrame()); _se=_cs.get("errors",[])
    st.subheader("📏 核心池規模驗證結果")
    if _se:
        st.warning("部分官方來源讀取異常："+"；".join(_se))
    st.info("TOP50/100/150/200 是事先固定的結構測試，不會因本次報酬結果去微調成TOP83、TOP117等事後最佳化數字。")
    if not _ss.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ss.round(3),use_container_width=True,hide_index=True)
    if not _sb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_sb.round(3),use_container_width=True,hide_index=True)
    if not _sd.empty:
        st.markdown("#### 每日核心池完整度")
        st.dataframe(_sd.tail(30),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【核心池規模WalkForward摘要】",_ss.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_核心池規模WalkForward摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【核心池規模四段穩定度】",_sb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_核心池規模四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【核心池規模逐筆交易】",_st.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_核心池規模逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【核心池規模每日完整度】",_sd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_核心池規模每日完整度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="全市場WalkForward驗證" and not run:
    st.info("這是目前最重要的驗證：股票池資格會隨歷史日期變動，不再用今天的TOP100回測過去。第一次執行時間會較長。")

if run and simple_mode=="進階研究" and research_mode=="全市場WalkForward驗證":
    st.subheader("🧭 全市場Walk-Forward驗證")
    with st.spinner("逐日重建全市場歷史股票池，接著下載曾入選股票的60m資料並跑固定策略…"):
        _wf_sum,_wf_blocks,_wf_trades,_wf_daily,_wf_errs=validate_fullmarket_walkforward(cost,period="3mo")
    st.session_state["st_v1140_wf"]={
        "summary":_wf_sum,"blocks":_wf_blocks,"trades":_wf_trades,
        "daily":_wf_daily,"errors":_wf_errs
    }

_wf=st.session_state.get("st_v1140_wf")
if simple_mode=="進階研究" and research_mode=="全市場WalkForward驗證" and _wf:
    _ws=_wf.get("summary",pd.DataFrame()); _wb=_wf.get("blocks",pd.DataFrame())
    _wt=_wf.get("trades",pd.DataFrame()); _wd=_wf.get("daily",pd.DataFrame()); _we=_wf.get("errors",[])
    st.subheader("🧭 全市場Walk-Forward結果")
    if _we:
        st.warning("部分官方來源讀取異常："+"；".join(_we))
    st.success("這份結果已移除『用今天股票池回測過去』的主要選股偏誤；仍保留Yahoo資料覆蓋與短期間樣本限制。")
    if not _ws.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ws.round(3),use_container_width=True,hide_index=True)
    if not _wb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_wb.round(3),use_container_width=True,hide_index=True)
    if not _wd.empty:
        st.markdown("#### 歷史每日股票池規模")
        st.dataframe(_wd.tail(30),use_container_width=True,hide_index=True)
    with st.expander("查看Walk-Forward逐筆交易"):
        st.dataframe(_wt.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【全市場WalkForward摘要】",_ws.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場WalkForward摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【全市場WalkForward四段穩定度】",_wb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場WalkForward四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【全市場WalkForward逐筆交易】",_wt.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場WalkForward逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【全市場WalkForward每日股票池】",_wd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場WalkForward每日股票池.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="全市場候選策略驗證" and not run:
    st.info("目前已選擇【全市場候選策略驗證】。請按「🧪 驗證核心TOP100 vs 熱門增補」。完成後應出現三個以【全市場候選策略】開頭的下載檔。")

if run and simple_mode=="進階研究" and research_mode=="全市場候選策略驗證":
    st.subheader("🧪 全市場候選策略驗證")
    st.caption("固定60m KD黃金交叉＋K<30＋持有5日；比較核心TOP100與熱門增補。此版為目前股票池定義的初步A/B，尚非最終Walk-Forward。")
    with st.spinner("建立全市場候選池並下載60m資料進行固定策略驗證…"):
        _ab,_blocks,_pool,_pool_sum,_errs=validate_fullmarket_candidate_groups(cost,period="3mo")
    st.session_state["st_v1133_fullmarket_ab"]={
        "ab":_ab,"blocks":_blocks,"pool":_pool,"pool_summary":_pool_sum,"errors":_errs
    }

_fab=st.session_state.get("st_v1133_fullmarket_ab")
if simple_mode=="進階研究" and research_mode=="全市場候選策略驗證" and _fab:
    _ab=_fab.get("ab",pd.DataFrame()); _bl=_fab.get("blocks",pd.DataFrame())
    _pp=_fab.get("pool",pd.DataFrame()); _ps=_fab.get("pool_summary",pd.DataFrame()); _er=_fab.get("errors",[])
    st.subheader("🧪 全市場候選策略驗證結果")
    st.success("如果你看到這個區塊，代表A/B策略驗證已完成。請下載下方三個『全市場候選策略』檔案。")
    if _er:
        st.warning("部分官方來源讀取異常："+"；".join(_er))
    st.warning("重要：本版股票組別由『目前』全市場資料定義，因此仍有 current-selection bias。結果只用來判斷熱門增補是否值得進入下一階段Walk-Forward，不作最終策略結論。")
    if not _ab.empty:
        st.markdown("#### 核心TOP100 vs 熱門增補｜全部 / 樣本內 / 樣本外")
        st.dataframe(_ab.round(3),use_container_width=True,hide_index=True)
    if not _bl.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_bl.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【全市場候選策略A_B摘要】",_ab.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場候選策略A_B摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【全市場候選策略四段穩定度】",_bl.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場候選策略四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【全市場候選股票池】",_pp.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場候選股票池.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("結果已保存；三份檔案可連續下載，不需重跑。")

if run and simple_mode=="進階研究" and research_mode=="全市場股票池研究":
    st.subheader("🌐 全市場股票池研究")
    st.caption("官方上市+上櫃公司名單 → Yahoo近期日K → 核心TOP100 + 熱門增補候選。爆量只作資訊，不作買進分數。")
    with st.spinner("下載官方上市/上櫃名單並分批建立全市場股票池，第一次可能需要較久…"):
        _fm_detail,_fm_summary,_fm_errors=build_full_market_pool_diagnostics(top_n=100)
        _fm_thresholds=summarize_liquidity_thresholds(_fm_detail)
    st.session_state["st_v1131_fullmarket"]={
        "detail":_fm_detail,"summary":_fm_summary,"errors":_fm_errors,
        "thresholds":_fm_thresholds
    }

_fm=st.session_state.get("st_v1131_fullmarket")
if simple_mode=="進階研究" and research_mode=="全市場股票池研究" and _fm:
    _fd=_fm.get("detail",pd.DataFrame()); _fs=_fm.get("summary",pd.DataFrame()); _fe=_fm.get("errors",[])
    _ft=_fm.get("thresholds",pd.DataFrame())
    st.subheader("🌐 全市場股票池研究結果")
    if _fe:
        st.warning("部分官方來源讀取異常："+"；".join(_fe))
    if not _fs.empty:
        cols=st.columns(min(4,len(_fs)))
        for i,(_,r) in enumerate(_fs.iterrows()):
            cols[i%len(cols)].metric(str(r["指標"]),int(r["數值"]))
    if not _fd.empty:
        _common=_fd["共同資料基準日"].mode()
        if len(_common):
            st.caption(f"共同資料基準日：{_common.iloc[0]}")
        st.info("V1.13.1 將正式研究增補收斂到『全市場流動性前20%』；爆量/熱門仍只作候選，不直接改買進策略。")
        if not _ft.empty:
            st.markdown("#### 流動性門檻敏感度")
            st.dataframe(_ft.round(0),use_container_width=True,hide_index=True)
        show=[c for c in ["股票","公司","市場","核心TOP100","熱門增補候選","研究候選池",
                          "流動性排名","成交金額百分位","爆量分層","今日量比20日",
                          "今日成交金額比20日","3日均量比20日","熱門動能"] if c in _fd.columns]
        st.dataframe(_fd[show].round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【全市場股票池明細】",_fd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場股票池明細.csv",mime="text/csv",
                       use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【全市場股票池摘要】",_fs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場股票池摘要.csv",mime="text/csv",
                       use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【全市場流動性門檻比較】",_ft.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_全市場流動性門檻比較.csv",mime="text/csv",
                       use_container_width=True,on_click="ignore")
    st.success("結果已保存；三份檔案可連續下載，不需重跑。")

if run and simple_mode=="進階研究" and research_mode=="股票池2.0歷史驗證":
    st.subheader("🧪 股票池2.0歷史驗證")
    st.caption("固定60m KD黃金交叉＋K<30＋持有5日；量能條件只使用訊號前一個已完成日K，避免偷看。")
    with st.spinner("建立核心策略交易並對齊歷史量能…"):
        _p20_hist,_p20_trades,_p20_sample,_p20_blocks=validate_pool20_historical(SHORT_TERM_UNIVERSE,cost,period="3mo")
    st.session_state["st_v1124_pool20_hist"]={
        "summary":_p20_hist,"trades":_p20_trades,
        "sample":_p20_sample,"blocks":_p20_blocks
    }

_ph=st.session_state.get("st_v1124_pool20_hist")
if simple_mode=="進階研究" and research_mode=="股票池2.0歷史驗證" and _ph:
    _hs=_ph.get("summary",pd.DataFrame()); _ht=_ph.get("trades",pd.DataFrame())
    _hsa=_ph.get("sample",pd.DataFrame()); _hbl=_ph.get("blocks",pd.DataFrame())
    st.subheader("🧪 股票池2.0歷史驗證結果")
    st.success("歷史驗證已完成。下方應出現兩個歷史驗證下載檔，不是『逐檔診斷／摘要』。")
    st.warning("這一輪只檢驗『前一完成日』爆量/熱門是否對核心策略有資訊價值；尚未把條件寫進正式雷達。")
    if not _hs.empty:
        st.dataframe(_hs.round(3),use_container_width=True,hide_index=True)
    if not _hsa.empty:
        st.markdown("#### 樣本內 / 樣本外穩定度")
        st.dataframe(_hsa.round(3),use_container_width=True,hide_index=True)
    if not _hbl.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_hbl.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆交易與量能標籤"):
        st.dataframe(_ht.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【股票池2.0歷史驗證摘要】",_hs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池2.0歷史驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池2.0歷史逐筆交易】",_ht.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池2.0歷史逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池2.0樣本內外穩定度】",_hsa.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池2.0樣本內外穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池2.0四段時間穩定度】",_hbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池2.0四段時間穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("結果已保存；四份檔案可連續下載，不需重跑。")

if run and simple_mode=="進階研究" and research_mode=="股票池2.0研究":
    st.subheader("🧭 股票池2.0研究")
    st.caption("核心流動、爆量異動、熱門動能分開觀察；本版不改正式今日雷達。")
    with st.spinner("計算近期流動性、爆量與熱門動能…"):
        _p20_detail,_p20_summary=build_pool_20_diagnostics(SHORT_TERM_UNIVERSE,top_n=top_n)
    st.session_state["st_v1120_pool20"]={"detail":_p20_detail,"summary":_p20_summary}

_p20=st.session_state.get("st_v1120_pool20")
if simple_mode=="進階研究" and research_mode=="股票池2.0研究" and _p20:
    _d=_p20.get("detail",pd.DataFrame()); _s=_p20.get("summary",pd.DataFrame())
    if not _s.empty:
        cols=st.columns(len(_s))
        for c,(_,r) in zip(cols,_s.iterrows()):
            c.metric(str(r["指標"]),int(r["數值"]))
    if not _d.empty:
        _common=pd.to_datetime(_d["最後交易日"],errors="coerce").dt.date.mode()
        if len(_common):
            st.caption(f"資料基準交易日：{_common.iloc[0]}｜V1.12.1 已排除週末與未收盤日K。")
    st.info("1.5倍、2倍、3倍目前只是研究分桶，不直接當買進條件；下一輪會拿來和60m核心策略做歷史驗證。")
    showcols=[c for c in ["股票","研究主題","核心流動池","爆量層級","熱門動能觀察","今日量比20日","3日均量比20日","今日成交金額比20日","3日振幅比20日","既有TOP比較組","是否共同最新交易日"] if c in _d.columns]
    st.dataframe(_d[showcols].round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【股票池2.0逐檔診斷】",_d.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池2.0逐檔診斷.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池2.0摘要】",_s.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池2.0摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("結果已保存在工作階段；下載不會要求重新計算。")

if run and simple_mode=="進階研究" and research_mode=="股票池健診":
    st.subheader("🧭 股票池健診")
    st.caption("分析人工母池 → 全母池近期資料 → 可交易性排名 → TOP100截斷；本次不跑策略回測。")
    with st.spinner("分析股票池…"):
        # V1.11.4 必須先排名完整母池，不能先 head(100)，否則第101名以後會被誤判成「無資料」。
        _ranked_all = rank_short_term_pool(SHORT_TERM_UNIVERSE, top_n=len(SHORT_TERM_UNIVERSE))
        _detail, _summary = diagnose_stock_pool(SHORT_TERM_UNIVERSE, _ranked_all, top_n)
    st.session_state["st_v1120_pool_diag"]={"detail":_detail,"summary":_summary}

_pool_state=st.session_state.get("st_v1120_pool_diag")
if simple_mode=="進階研究" and research_mode=="股票池健診" and _pool_state:
    _detail=_pool_state.get("detail",pd.DataFrame())
    _summary=_pool_state.get("summary",pd.DataFrame())
    st.subheader("🧭 股票池健診結果")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("人工母池",len(_detail))
    c2.metric("有近期資料",int((_detail["是否有近期排名資料"]=="是").sum()) if not _detail.empty else 0)
    c3.metric(f"TOP{top_n}",int((_detail["目前TOP池"]=="是").sum()) if not _detail.empty else 0)
    c4.metric("TOP外",int(((_detail["是否有近期排名資料"]=="是") & (_detail["目前TOP池"]=="否")).sum()) if not _detail.empty else 0)
    st.warning("短線可交易分只代表流動性/活躍度，不代表未來報酬。TOP100只是比較組。")
    if not _summary.empty:
        st.markdown("#### 族群分布")
        st.dataframe(_summary.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看股票池逐檔明細"):
        st.dataframe(_detail.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【股票池健診明細】",_detail.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池健診明細.csv",mime="text/csv",
                       use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池族群分布】",_summary.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池族群分布.csv",mime="text/csv",
                       use_container_width=True,on_click="ignore")
    st.success("結果已保存在本次工作階段。下載檔案不會清掉健診結果，不需要重新計算。")


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📊 單股總表", "🌐 跨股穩定度", "🔥 動態短線池", "🏭 族群比較", "🧬 狀態分類", "📈 K線/KD", "🔬 KD分區", "📦 量價/斜率", "🧾 交易明細"])

if run:
    if research_mode not in ["股票池2.0研究","股票池2.0歷史驗證","股票池健診","多週期當沖/隔日驗證","60m五日OOS驗證"] and (not selected_intervals or not selected_rules or not selected_modes):
        st.error("請至少選擇一個K棒週期、進場規則與持有方式。")
        st.stop()

    if research_mode in ["TOP50暖機修正驗證","TOP50訊號品質健診","市場環境健診_TOP50","核心池規模WalkForward","全市場WalkForward驗證","全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診"]:
        # 股票池研究已在上方獨立完成。：股票池健診已在上方獨立完成。
        # 初始化舊版共用變數，避免後續 session 儲存引用未定義的 summary。
        summary, data_map, trade_map = pd.DataFrame(), {}, {}
    elif research_mode == "60m五日OOS驗證":
        # V1.4.1：OOS 執行移到按下「開始策略健診」之後；
        # 此時交易成本 cost 已完成建立，避免 V1.4.0 的 NameError。
        with st.spinner("建立固定股票池並進行60m五日時間OOS…"):
            ranked_pool = rank_short_term_pool(SHORT_TERM_UNIVERSE, top_n=top_n)
        symbols = ranked_pool["股票"].tolist() if not ranked_pool.empty else []
        if not symbols:
            st.error("股票池建立失敗：沒有可用股票。")
            st.stop()
        try:
            with st.spinner(f"60m五日OOS驗證：{len(symbols)}檔…"):
                oos_detail, oos_summary, oos_trades = run_oos_60m_5d(
                    symbols, cost, period, allow_overlap=False, train_ratio=0.60
                )
        except Exception as e:
            st.error(f"OOS驗證中斷：{type(e).__name__}: {e}")
            st.stop()
        block_summary = time_block_stability(oos_trades, blocks=4)
        state_diag = state_at_entry_diagnostics(oos_trades)
        filter_robust = state_filter_robustness(oos_trades, blocks=4)
        market_diag = market_regime_diagnostics(oos_trades, period)
        market_blocks = market_regime_by_timeblock(oos_trades, blocks=4)
        wf_summary, wf_detail, wf_status = walkforward_tradability_validation(oos_trades, daily_lookback=20)
        wf_time = walkforward_layer_time_validation(wf_detail, blocks=4)
        radar_candidates = build_research_radar_candidates(oos_trades, wf_detail)
        radar_for_validation = radar_candidates.copy()
        if not radar_for_validation.empty:
            radar_for_validation["雷達研究分數"] = radar_for_validation["訊號新鮮度分"]
            _hk = pd.to_numeric(radar_for_validation["K深度分"],errors="coerce").notna()
            radar_for_validation.loc[_hk,"雷達研究分數"] = (
                radar_for_validation.loc[_hk,"訊號新鮮度分"]*0.70 +
                radar_for_validation.loc[_hk,"K深度分"]*0.30
            )
        radar_validation = validate_radar_ranking(radar_for_validation, oos_trades)
        daily_radar = build_daily_radar_status(oos_trades, wf_detail, observe_days=5)
        latest_signal = pd.to_datetime(oos_trades["訊號時間"],utc=True,errors="coerce").max() if not oos_trades.empty else pd.NaT
        pool_detail, pool_summary = diagnose_stock_pool(SHORT_TERM_UNIVERSE, ranked_pool, top_n)
        with st.spinner("掃描最新60m行情，建立今日雷達…"):
            live_radar, live_diag = scan_latest_60m_radar(symbols, ranked_pool, period=period, observe_days=5)
        st.session_state["st_v1120_oos"] = {
            "detail": oos_detail, "summary": oos_summary, "trades": oos_trades,
            "blocks": block_summary, "state_diag": state_diag,
            "filter_robust": filter_robust, "market_diag": market_diag,
            "market_blocks": market_blocks,
            "wf_summary": wf_summary, "wf_detail": wf_detail, "wf_status": wf_status, "wf_time": wf_time,
            "radar_candidates": radar_candidates, "radar_validation": radar_validation, "daily_radar": daily_radar, "latest_signal": latest_signal, "live_radar": live_radar, "live_diag": live_diag, "pool_detail": pool_detail, "pool_summary": pool_summary,
            "pool": ranked_pool, "symbols": symbols
        }
        summary, data_map, trade_map = pd.DataFrame(), {}, {}
        symbol = symbols[0]
    elif research_mode == "多週期當沖/隔日驗證":
        # 多週期版沿用動態短線池；預設50檔即可，避免再擴大樣本造成不必要負載。
        with st.spinner("建立多週期驗證股票池…"):
            ranked_pool = rank_short_term_pool(SHORT_TERM_UNIVERSE, top_n=top_n)
        symbols = ranked_pool["股票"].tolist() if not ranked_pool.empty else []
        if not symbols:
            st.error("股票池建立失敗：沒有可用股票。")
            st.stop()
        if not mtf_entry_rules or not mtf_modes:
            st.error("請至少選擇一個5分鐘進場觸發與一個短線出場方式。")
            st.stop()
        try:
            with st.spinner(f"多週期當沖/隔日驗證：{len(symbols)}檔…"):
                mtf_detail, mtf_cross, mtf_trades, mtf_diag = run_multitimeframe_batch(
                    symbols, mtf_entry_rules, mtf_modes, cost, period, allow_overlap
                )
        except Exception as e:
            st.error(f"多週期健診中斷：{type(e).__name__}: {e}")
            st.stop()
        st.session_state["st_v130_mtf"] = {
            "detail": mtf_detail, "cross": mtf_cross, "trades": mtf_trades,
            "diagnostics": mtf_diag, "symbols": symbols, "ranked_pool": ranked_pool,
        }
        # 不進入舊批次流程
        summary, data_map, trade_map = pd.DataFrame(), {}, {}
        symbol = symbols[0]
    elif research_mode == "跨股票批次":
        if pool_mode == "手動輸入":
            symbols = parse_batch_codes(batch_text, market)
            ranked_pool = pd.DataFrame()
        elif pool_mode == "族群代表池":
            symbols = sector_symbols(selected_sectors, per_sector, market)
            ranked_pool = pd.DataFrame()
        else:
            with st.spinner("建立動態短線股票池：讀取近期日線流動性與振幅…"):
                ranked_pool = rank_short_term_pool(SHORT_TERM_UNIVERSE, top_n=top_n)
            symbols = ranked_pool["股票"].tolist() if not ranked_pool.empty else []
        if not symbols:
            st.error("股票池建立失敗：沒有可用股票。請先降低TOP數或稍後重試。")
            if not ranked_pool.empty:
                st.dataframe(ranked_pool, use_container_width=True, hide_index=True)
            st.stop()

        # 股票池先存入 session，避免長時間回測失敗後畫面完全空白。
        st.session_state["st_v1231_pool_preview"] = ranked_pool.copy()
        st.success(f"股票池建立完成：{len(symbols)} 檔。開始分K健診。")
        with st.expander("查看本次實際健診名單", expanded=False):
            st.write("、".join(symbols))

        try:
            with st.spinner(f"跨股票策略健診：{len(symbols)} 檔…"):
                batch_detail, cross, states_df, sector_summary, theme_summary, diagnostics = run_batch_matrix(
                    symbols, selected_intervals, selected_rules, selected_modes, cost, period, allow_overlap
                )
        except Exception as e:
            st.error(f"分K健診中斷：{type(e).__name__}: {e}")
            st.warning("股票池已保留。若仍中斷，請把紅色錯誤訊息截圖給我。")
            st.stop()
        st.session_state["st_v120_batch"] = {
            "detail": batch_detail, "cross": cross, "states": states_df, "sector_summary": sector_summary, "theme_summary": theme_summary, "diagnostics": diagnostics, "symbols": symbols, "ranked_pool": ranked_pool, "allow_overlap": allow_overlap
        }
        # 批次完成後不再重跑第一檔，避免再次下載造成中斷。
        symbol = symbols[0]
        summary, data_map, trade_map = pd.DataFrame(), {}, {}
    elif research_mode == "單一股票":
        with st.spinner(f"下載 {symbol} 分K並回測…"):
            summary, data_map, trade_map = run_matrix(
                symbol, selected_intervals, selected_rules, selected_modes, cost, period
            )

    st.session_state["st_v120"] = {
        "symbol": symbol,
        "summary": summary,
        "data_map": data_map,
        "trade_map": trade_map,
    }

oos_state = st.session_state.get("st_v1120_oos")
if research_mode == "60m五日OOS驗證" and oos_state:
    st.markdown("## 🧪 60m＋K<30＋5日｜時間穩定度驗證")
    st.caption("規則完全固定；所有股票共用同一個全市場時間切點，前60%時間區段為樣本內、後40%為樣本外。股票池仍由近期流動性建立，因此仍屬固定股票池時間OOS。")
    osum=oos_state.get("summary",pd.DataFrame())
    odet=oos_state.get("detail",pd.DataFrame())
    otr=oos_state.get("trades",pd.DataFrame())
    oblocks=oos_state.get("blocks",pd.DataFrame())
    ostate=oos_state.get("state_diag",pd.DataFrame())
    ofilter=oos_state.get("filter_robust",pd.DataFrame())
    omarket=oos_state.get("market_diag",pd.DataFrame())
    omarket_blocks=oos_state.get("market_blocks",pd.DataFrame())
    owf=oos_state.get("wf_summary",pd.DataFrame())
    owfd=oos_state.get("wf_detail",pd.DataFrame())
    owf_status=oos_state.get("wf_status","尚未執行Walk-Forward診斷。")
    owf_time=oos_state.get("wf_time",pd.DataFrame())
    oradar=oos_state.get("radar_candidates",pd.DataFrame())
    oradar_val=oos_state.get("radar_validation",pd.DataFrame())
    odaily=oos_state.get("daily_radar",pd.DataFrame())
    olatest=oos_state.get("latest_signal",pd.NaT)
    olive=oos_state.get("live_radar",pd.DataFrame())
    olive_diag=oos_state.get("live_diag",pd.DataFrame())
    opool=oos_state.get("pool_detail",pd.DataFrame())
    opool_sum=oos_state.get("pool_summary",pd.DataFrame())

    st.markdown("## 📡 今日60m短線雷達")
    st.caption("直接掃描最新60m行情。以行情最後交易日判斷新訊號；週末仍保留週五新訊號。只使用已完成60m K棒。")
    if not olive_diag.empty:
        _ok=int((olive_diag["狀態"]=="掃描完成").sum())
        _last=pd.to_datetime(olive_diag["行情最後K棒"],errors="coerce").max()
        c1,c2,c3=st.columns(3)
        c1.metric("成功掃描",f"{_ok}/{len(olive_diag)}")
        c2.metric("新訊號",int((olive.get("目前狀態",pd.Series(dtype=str))=="🟢 新訊號").sum()) if not olive.empty else 0)
        c3.metric("觀察中",int((olive.get("目前狀態",pd.Series(dtype=str))=="🟡 觀察中").sum()) if not olive.empty else 0)
        if pd.notna(_last):
            st.info(f"本批行情最後60m K棒：{_last:%Y-%m-%d %H:%M}（台灣時間欄位）。")
    if olive.empty:
        st.info("本次最新60m掃描沒有找到KD黃金交叉＋K<30的近期訊號。這不代表程式失敗；請同時查看掃描診斷。")
    else:
        _active=olive[olive["目前狀態"].isin(["🟢 新訊號","🟡 觀察中"])]
        _show=_active if not _active.empty else olive.head(20)
        _cols=[c for c in ["股票","研究主題","目前狀態","訊號時間","訊號K","目前K","目前D","預計觀察至","目前短線可交易分"] if c in _show.columns]
        st.dataframe(_show[_cols],use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【今日60m短線雷達】",olive.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_今日60m短線雷達.csv",mime="text/csv",use_container_width=True)
    if simple_mode=="進階研究" and not opool.empty:
        st.markdown("### 🧭 股票池健診")
        _mother=len(opool)
        _ranked=int((opool["是否有近期排名資料"]=="是").sum())
        _top=int((opool["目前TOP池"]=="是").sum())
        p1,p2,p3=st.columns(3)
        p1.metric("人工母池",_mother)
        p2.metric("有近期排名資料",_ranked)
        p3.metric("目前TOP池",_top)
        st.caption("這裡只揭露股票池怎麼形成；短線可交易分是流動性/活躍度背景，不代表未來報酬。")
        if not opool_sum.empty:
            st.dataframe(opool_sum.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【股票池健診明細】",opool.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_股票池健診明細.csv",mime="text/csv",use_container_width=True)
        if not opool_sum.empty:
            st.download_button("⬇️ 下載【股票池族群分布】",opool_sum.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{APP_VERSION}_股票池族群分布.csv",mime="text/csv",use_container_width=True)

    if not olive_diag.empty and simple_mode=="進階研究":
        with st.expander("🔧 掃描診斷"):
            st.dataframe(olive_diag,use_container_width=True,hide_index=True)
            st.download_button("⬇️ 下載【今日60m掃描診斷】",olive_diag.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{APP_VERSION}_今日60m掃描診斷.csv",mime="text/csv",use_container_width=True)

    if simple_mode=="進階研究":
        st.markdown("---")
        st.caption("歷史OOS、Walk-Forward、狀態診斷等研究資料仍保留在程式內；主畫面不再重複鋪滿舊版研究表格。")
        if not osum.empty:
            with st.expander("🧪 OOS研究摘要"):
                st.dataframe(osum.round(3),use_container_width=True,hide_index=True)
                st.download_button("⬇️ 下載【OOS驗證總表】",osum.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{APP_VERSION}_OOS驗證總表.csv",mime="text/csv")
        if not owfd.empty:
            with st.expander("📦 研究資料下載"):
                st.download_button("⬇️ 下載【WalkForward逐筆明細】",owfd.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{APP_VERSION}_WalkForward逐筆明細.csv",mime="text/csv")
                st.download_button("⬇️ 下載【OOS逐筆交易明細】",otr.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{APP_VERSION}_OOS逐筆交易明細.csv",mime="text/csv")


mtf_state = st.session_state.get("st_v130_mtf")
if research_mode == "多週期當沖/隔日驗證" and mtf_state:
    st.markdown("## ⚡ 多週期當沖／隔日驗證結果")
    mcross = mtf_state.get("cross", pd.DataFrame())
    mdetail = mtf_state.get("detail", pd.DataFrame())
    mtrades = mtf_state.get("trades", pd.DataFrame())
    mdiag = mtf_state.get("diagnostics", pd.DataFrame())

    if not mdiag.empty:
        st.dataframe(mdiag, use_container_width=True, hide_index=True)
    if mcross.empty:
        st.warning("目前沒有產生多週期跨股結果。請先確認5m/15m/60m三個週期的成功下載數，再把畫面截圖給我。")
    if not mcross.empty:
        st.markdown("### 跨股票穩定度")
        st.dataframe(mcross.round(3), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 下載【多週期跨股票穩定度】",
            mcross.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{APP_VERSION}_多週期跨股票穩定度.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if not mdetail.empty:
        st.download_button(
            "⬇️ 下載【多週期批次策略明細】",
            mdetail.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{APP_VERSION}_多週期批次策略明細.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if not mtrades.empty:
        st.download_button(
            "⬇️ 下載【多週期逐筆交易明細】",
            mtrades.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{APP_VERSION}_多週期逐筆交易明細.csv",
            mime="text/csv",
            use_container_width=True,
        )
    rp = mtf_state.get("ranked_pool", pd.DataFrame())
    if not rp.empty:
        st.download_button(
            "⬇️ 下載【多週期驗證股票池】",
            rp.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{APP_VERSION}_多週期驗證股票池.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.info("下一輪請提供：①【多週期跨股票穩定度】②【多週期批次策略明細】③【多週期逐筆交易明細】。本輪會比較「15m確認立即進場」與「再等5m KD」及當沖/隔日/2日。")

state = st.session_state.get("st_v120")
batch_state = st.session_state.get("st_v120_batch")
if state:
    symbol = state["symbol"]
    summary = state["summary"]
    data_map = state["data_map"]
    trade_map = state["trade_map"]

    with tab1:
        if research_mode == "跨股票批次":
            st.subheader("📊 跨股票批次健診")
            _diag = batch_state.get("diagnostics", pd.DataFrame()) if batch_state else pd.DataFrame()
            _cross = batch_state.get("cross", pd.DataFrame()) if batch_state else pd.DataFrame()
            c1, c2, c3 = st.columns(3)
            c1.metric("實際健診股票", len(batch_state.get("symbols", [])) if batch_state else 0)
            c2.metric("跨股策略組合", len(_cross))
            c3.metric("成功週期", int((_diag["成功下載"] > 0).sum()) if not _diag.empty and "成功下載" in _diag.columns else 0)
            st.info("批次模式不重複下載第一檔做單股基準；請查看「跨股票穩定度／族群比較／動態短線池」。")
            valid = pd.DataFrame()
        else:
            st.subheader(f"{symbol}｜多週期基準比較")
            valid = (
                summary[summary["交易數"] > 0].copy()
                if isinstance(summary, pd.DataFrame) and (not summary.empty) and ("交易數" in summary.columns)
                else pd.DataFrame()
            )
        if research_mode != "跨股票批次" and valid.empty:
            st.warning("目前條件沒有產生足夠交易。可換股票、延長資料或放寬進場規則。")
        elif research_mode != "跨股票批次":
            # 不宣告單一「最佳策略」；提供客觀排序欄位供研究。
            sort_col = st.selectbox(
                "總表排序依據",
                ["ProfitFactor", "期望值%", "平均淨報酬%", "勝率%", "最大回撤%", "交易數"],
                index=0,
            )
            asc = sort_col == "最大回撤%"
            shown = valid.sort_values(sort_col, ascending=asc, na_position="last")
            st.dataframe(fmt_summary(shown), use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("已驗證組合", f"{len(valid):,}")
            c2.metric("總交易樣本", f"{int(valid['交易數'].sum()):,}")
            c3.metric("使用指標", "MA + KD")

            csv = shown.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ 下載驗證總表 CSV",
                csv,
                file_name=f"{symbol}_ST_V1.0_matrix.csv",
                mime="text/csv",
            )

            st.markdown("#### 週期彙總")
            agg = valid.groupby("週期").agg(
                組合數=("規則", "count"),
                交易數=("交易數", "sum"),
                平均勝率=("勝率%", "mean"),
                平均期望值=("期望值%", "mean"),
                平均PF=("ProfitFactor", "mean"),
            ).reset_index()
            st.dataframe(agg.round(3), use_container_width=True, hide_index=True)

            st.markdown("#### 持有方式彙總")
            holdagg = valid.groupby("持有").agg(
                組合數=("規則", "count"),
                交易數=("交易數", "sum"),
                平均勝率=("勝率%", "mean"),
                平均期望值=("期望值%", "mean"),
                平均PF=("ProfitFactor", "mean"),
            ).reset_index()
            st.dataframe(holdagg.round(3), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("跨股票策略穩定度")
        if not batch_state or batch_state["cross"].empty:
            st.info("切換左側「跨股票批次」並執行後，這裡會比較同一策略在不同股票上的一致性。")
        else:
            cross = batch_state["cross"].copy()
            min_stocks = st.slider(
                "至少涵蓋股票數",
                1,
                max(1, len(batch_state["symbols"])),
                min(3, len(batch_state["symbols"])),
            )
            cx = cross[cross["股票數"] >= min_stocks].copy()
            sort2 = st.selectbox(
                "跨股排序欄位",
                ["正期望股票比例", "期望值中位數", "PF中位數", "總交易數", "樣本覆蓋率"],
                key="cross_sort",
            )
            cx = cx.sort_values(sort2, ascending=False, na_position="last")
            st.dataframe(cx.round(3), use_container_width=True, hide_index=True)
            st.caption("重點看『正期望股票比例＋期望值中位數＋股票數/交易數』，避免被單一股票或少量交易拉高。")
            st.download_button(
                "⬇️ 下載【跨股票穩定度】",
                cx.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{EXPORT_PREFIX}_cross_stock_stability_nonoverlap.csv",
                mime="text/csv",
            )
            detail_export = batch_state.get("detail", pd.DataFrame())
            if not detail_export.empty:
                st.download_button(
                    "⬇️ 下載【批次策略明細】",
                    data=detail_export.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{EXPORT_PREFIX}_batch_strategy_detail_nonoverlap.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with tab3:
        st.subheader("動態短線股票池")
        preview_pool = st.session_state.get("st_v1231_pool_preview", pd.DataFrame())
        if (not batch_state or batch_state.get("ranked_pool", pd.DataFrame()).empty) and preview_pool.empty:
            st.info("左側選擇「跨股票批次 → 動態短線TOP池」後執行，即會顯示近期可交易性排名與實際健診名單。")
        else:
            rp = (batch_state["ranked_pool"].copy()
                  if batch_state and not batch_state.get("ranked_pool", pd.DataFrame()).empty
                  else preview_pool.copy())
            show = rp.copy()
            show["20日成交金額中位數(億)"] = show["20日成交金額中位數"] / 1e8
            cols = ["股票","短線可交易分","20日成交金額中位數(億)","20日成交量中位數","20日振幅中位數%","有效日數"]
            st.dataframe(show[cols].round(2), use_container_width=True, hide_index=True)
            st.caption("此排名只衡量近期流動性與波動是否適合短線研究，不代表預測報酬或推薦買賣。")
            if "研究主題" in rp.columns:
                st.markdown("#### 入選股票主題分布")
                tc = rp.groupby("研究主題")["股票"].count().sort_values(ascending=False).rename("入選檔數").reset_index()
                st.dataframe(tc, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ 下載【動態短線股票池】",
                show[cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{EXPORT_PREFIX}_dynamic_short_term_pool.csv",
                mime="text/csv",
            )

    with tab4:
        st.subheader("族群 × 策略健診")
        if not batch_state or batch_state.get("sector_summary", pd.DataFrame()).empty:
            st.info("使用「跨股票批次 → 族群代表池」執行後，這裡會比較不同族群的策略表現。")
        else:
            ss = batch_state["sector_summary"].copy()
            sec_filter = st.multiselect(
                "顯示族群",
                sorted(ss["族群"].dropna().unique().tolist()),
                default=sorted(ss["族群"].dropna().unique().tolist()),
                key="sector_filter",
            )
            sx = ss[ss["族群"].isin(sec_filter)].copy()
            sec_sort = st.selectbox(
                "族群表排序",
                ["正期望股票比例", "期望值中位數", "PF中位數", "總交易數"],
                key="sector_sort",
            )
            sx = sx.sort_values(["族群", sec_sort], ascending=[True, False], na_position="last")
            st.dataframe(sx.round(3), use_container_width=True, hide_index=True)
            st.caption("族群結果用來找『策略在哪些產業環境較穩定』，不把單一族群的最高數字直接視為最終策略。")
            st.download_button(
                "⬇️ 下載【族群策略健診】",
                sx.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{EXPORT_PREFIX}_sector_strategy_nonoverlap.csv",
                mime="text/csv",
            )

    with tab5:
        st.subheader("股票目前狀態分類")
        if not batch_state or batch_state["states"].empty:
            local_state = classify_stock_state(data_map)
            st.dataframe(pd.DataFrame([{"股票": symbol, **local_state}]), use_container_width=True, hide_index=True)
        else:
            st.dataframe(batch_state["states"], use_container_width=True, hide_index=True)
            st.caption("這是研究用『當前狀態』，不是把股票永久分類；同一股票日後可能從趨勢轉為震盪或高波動。")

    with tab6:
        iv = st.selectbox("圖表週期", list(data_map.keys()), key="chart_iv")
        d = data_map.get(iv, pd.DataFrame())
        if d.empty:
            st.error(f"{iv} 沒有取得資料。")
        else:
            last = d.iloc[-1]
            mcols = st.columns(7)
            mcols[0].metric("Close", f"{last['Close']:.2f}")
            for j, n in enumerate(MA_LIST, 1):
                val = last.get(f"MA{n}", np.nan)
                mcols[j].metric(f"MA{n}", "-" if pd.isna(val) else f"{val:.2f}")
            mcols[6].metric("KD", "-" if pd.isna(last.K) else f"{last.K:.1f}/{last.D:.1f}")
            plot_chart(d, iv)

    with tab7:
        st.subheader("KD 所在區間是否真的影響結果？")
        all_t = [t for t in trade_map.values() if t is not None and not t.empty]
        if not all_t:
            st.warning("沒有交易樣本。")
        else:
            tt = pd.concat(all_t, ignore_index=True)
            kd = tt.groupby(["週期", "K區間"]).agg(
                交易數=("淨報酬%", "count"),
                勝率=("淨報酬%", lambda x: (x > 0).mean() * 100),
                平均淨報酬=("淨報酬%", "mean"),
                平均MFE=("MFE%", "mean"),
                平均MAE=("MAE%", "mean"),
            ).reset_index()
            st.dataframe(kd.round(3), use_container_width=True, hide_index=True)
            st.caption(
                "這張表用來檢查 K<20、20~50、50~80、>80 哪個區間的訊號實際較有延續性；"
                "先觀察，不先假設超買一定要賣、超賣一定要買。"
            )

    with tab8:
        st.subheader("量價 / 均線斜率研究")
        all_t2 = [t for t in trade_map.values() if t is not None and not t.empty]
        if not all_t2:
            st.warning("沒有交易樣本。")
        else:
            ft = pd.concat(all_t2, ignore_index=True)
            ft["量比分組"] = pd.cut(
                ft["量比20"],
                bins=[-np.inf, 0.8, 1.2, 1.5, 2.0, np.inf],
                labels=["<0.8", "0.8-1.2", "1.2-1.5", "1.5-2.0", ">2.0"],
            )
            voltab = ft.groupby(["週期", "量比分組"], observed=True).agg(
                交易數=("淨報酬%", "count"),
                勝率=("淨報酬%", lambda x: (x > 0).mean() * 100),
                平均淨報酬=("淨報酬%", "mean"),
                平均MFE=("MFE%", "mean"),
                平均MAE=("MAE%", "mean"),
            ).reset_index()
            st.markdown("#### 成交量 / 20期均量")
            st.dataframe(voltab.round(3), use_container_width=True, hide_index=True)

            ft["MA30方向"] = np.where(ft["MA30斜率3"] > 0, "向上", "非向上")
            slopetab = ft.groupby(["週期", "MA30方向"]).agg(
                交易數=("淨報酬%", "count"),
                勝率=("淨報酬%", lambda x: (x > 0).mean() * 100),
                平均淨報酬=("淨報酬%", "mean"),
            ).reset_index()
            st.markdown("#### MA30 3根斜率")
            st.dataframe(slopetab.round(3), use_container_width=True, hide_index=True)

    with tab9:
        keys = [k for k, t in trade_map.items() if t is not None and not t.empty]
        if not keys:
            st.warning("沒有交易明細。")
        else:
            key = st.selectbox("選擇策略組合", keys)
            t = trade_map[key]
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ 下載此組交易明細",
                t.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{symbol}_{key.replace('|','_')}.csv",
                mime="text/csv",
            )
else:
    with tab1:
        st.write("左側選擇研究模式與條件後，按「開始策略健診」。")
    with tab2:
        st.write("跨股票批次完成後顯示策略跨股穩定度。")
    with tab3:
        st.write("動態短線TOP池完成後顯示本次入選股票與可交易性。")
    with tab4:
        st.write("族群代表池完成後顯示族群 × 策略比較。")
    with tab5:
        st.write("完成驗證後顯示股票目前狀態分類。")
    with tab6:
        st.write("完成第一次驗證後顯示 MA5/15/30/60/200、VWAP 與 KD。")
    with tab7:
        st.write("完成第一次驗證後分析 KD 區間。")
    with tab8:
        st.write("完成第一次驗證後分析量比與均線斜率。")
    with tab9:
        st.write("完成第一次驗證後顯示逐筆交易。")

st.divider()
st.caption(
    "ST V1.16.3 僅供策略研究與程式驗證，不送出證券委託。"
    "下一階段將根據實際回測結果，再判斷是否增加 VWAP、成交量/量比、MACD、ATR 或其他參數。"
)
