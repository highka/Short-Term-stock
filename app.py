# -*- coding: utf-8 -*-
"""
黑嚕嚕－短線交易雷達 ST V1.16.33
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
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# V1.16.6：Streamlit Cloud 資源保護。
# 在 numpy/pandas 載入前限制底層執行緒，避免記憶體/Thread耗盡。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import urllib.request
import json

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

APP_VERSION = "ST V1.16.33"
APP_NAME = "黑嚕嚕－短線交易雷達"
MA_LIST = [5, 15, 30, 60, 200]
INTERVALS = ["5m", "15m", "60m"]

APP_VERSION = "ST_V1.16.33"
EXPORT_PREFIX = "ST_V1.16.33"

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


@st.cache_data(ttl=900, show_spinner=False)
def download_intraday(symbol: str, interval: str, period: str = "60d") -> pd.DataFrame:
    try:
        d = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
        )
    except Exception:
        return pd.DataFrame()

    if d is None or d.empty:
        return pd.DataFrame()

    if isinstance(d.columns, pd.MultiIndex):
        # yfinance 單檔有時仍回 MultiIndex
        if symbol in d.columns.get_level_values(-1):
            try:
                d = d.xs(symbol, axis=1, level=-1)
            except Exception:
                pass
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [str(c[0]) for c in d.columns]

    rename = {c: str(c).title() for c in d.columns}
    d = d.rename(columns=rename)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in d.columns for c in need):
        return pd.DataFrame()

    d = d[need].copy()
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d[~d.index.duplicated(keep="last")].sort_index()

    # 僅保留台股一般交易時段。Yahoo 時區若可用，轉台北。
    try:
        if d.index.tz is not None:
            d.index = d.index.tz_convert("Asia/Taipei")
    except Exception:
        pass
    try:
        d = d.between_time("09:00", "13:30")
    except Exception:
        pass
    return d


def add_indicators(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    for n in MA_LIST:
        x[f"MA{n}"] = x["Close"].rolling(n, min_periods=n).mean()

    # 台灣常用 KD：RSV 9，K/D 平滑 1/3；以 50 為初始值。
    low9 = x["Low"].rolling(9, min_periods=9).min()
    high9 = x["High"].rolling(9, min_periods=9).max()
    den = (high9 - low9).replace(0, np.nan)
    x["RSV"] = ((x["Close"] - low9) / den * 100).clip(0, 100)

    k_vals, d_vals = [], []
    k_prev = 50.0
    d_prev = 50.0
    for rsv in x["RSV"]:
        if pd.isna(rsv):
            k_vals.append(np.nan)
            d_vals.append(np.nan)
            continue
        k_prev = (2/3) * k_prev + (1/3) * float(rsv)
        d_prev = (2/3) * d_prev + (1/3) * k_prev
        k_vals.append(k_prev)
        d_vals.append(d_prev)
    x["K"] = k_vals
    x["D"] = d_vals

    x["MA_BULL_5_15"] = x["MA5"] > x["MA15"]
    x["MA_BULL_15_30"] = x["MA15"] > x["MA30"]
    x["MA_BULL_30_60"] = x["MA30"] > x["MA60"]
    x["FULL_BULL"] = (
        (x["MA5"] > x["MA15"]) &
        (x["MA15"] > x["MA30"]) &
        (x["MA30"] > x["MA60"]) &
        (x["MA60"] > x["MA200"])
    )
    x["PRICE_GT_MA200"] = x["Close"] > x["MA200"]

    x["MA5_XUP_MA15"] = (x["MA5"] > x["MA15"]) & (x["MA5"].shift(1) <= x["MA15"].shift(1))
    x["MA15_XUP_MA30"] = (x["MA15"] > x["MA30"]) & (x["MA15"].shift(1) <= x["MA30"].shift(1))
    x["KD_GOLD"] = (x["K"] > x["D"]) & (x["K"].shift(1) <= x["D"].shift(1))
    x["KD_DEAD"] = (x["K"] < x["D"]) & (x["K"].shift(1) >= x["D"].shift(1))

    x["K_ZONE"] = pd.cut(
        x["K"],
        bins=[-np.inf, 20, 30, 50, 80, np.inf],
        labels=["K<20", "K20-30", "K30-50", "K50-80", "K>80"],
    )

    # V1.1：均線斜率，以目前 MA - 3 根前 MA 表示方向。
    for n in MA_LIST:
        x[f"MA{n}_SLOPE3"] = x[f"MA{n}"] - x[f"MA{n}"].shift(3)

    # V1.1：量能比 = 當根成交量 / 前20期平均量（shift 1 避免把當根放入基準）。
    x["VOL_MA20_PREV"] = x["Volume"].shift(1).rolling(20, min_periods=20).mean()
    x["VOL_RATIO20"] = x["Volume"] / x["VOL_MA20_PREV"].replace(0, np.nan)

    # V1.1：日內 VWAP，每個交易日重新累積。
    tp = (x["High"] + x["Low"] + x["Close"]) / 3.0
    dates = pd.Index([pd.Timestamp(i).date() for i in x.index])
    pv = tp * x["Volume"].fillna(0)
    cum_pv = pv.groupby(dates).cumsum()
    cum_v = x["Volume"].fillna(0).groupby(dates).cumsum().replace(0, np.nan)
    x["VWAP"] = cum_pv / cum_v
    x["PRICE_GT_VWAP"] = x["Close"] > x["VWAP"]

    return x


def signal_mask(d: pd.DataFrame, rule: str) -> pd.Series:
    false = pd.Series(False, index=d.index)
    rules = {
        "MA5上穿MA15": d.get("MA5_XUP_MA15", false),
        "MA15上穿MA30": d.get("MA15_XUP_MA30", false),
        "KD黃金交叉": d.get("KD_GOLD", false),
        "MA5>15 + KD黃金交叉": d.get("MA_BULL_5_15", false) & d.get("KD_GOLD", false),
        "MA5>15>30 + KD黃金交叉": (
            d.get("MA_BULL_5_15", false) &
            d.get("MA_BULL_15_30", false) &
            d.get("KD_GOLD", false)
        ),
        "MA5>15>30>60 + KD黃金交叉": (
            d.get("MA_BULL_5_15", false) &
            d.get("MA_BULL_15_30", false) &
            d.get("MA_BULL_30_60", false) &
            d.get("KD_GOLD", false)
        ),
        "完整多頭排列": d.get("FULL_BULL", false) & (~d.get("FULL_BULL", false).shift(1).fillna(False)),
        "完整多頭排列 + KD黃金交叉": d.get("FULL_BULL", false) & d.get("KD_GOLD", false),
        "站上MA200 + KD黃金交叉": d.get("PRICE_GT_MA200", false) & d.get("KD_GOLD", false),
        "KD黃金交叉 + K<30": d.get("KD_GOLD", false) & (d.get("K", pd.Series(np.nan, index=d.index)) < 30),
        "KD黃金交叉 + K30-50": d.get("KD_GOLD", false) & (d.get("K", pd.Series(np.nan, index=d.index)) >= 30) & (d.get("K", pd.Series(np.nan, index=d.index)) < 50),
        "KD黃金交叉 + K50-80": d.get("KD_GOLD", false) & (d.get("K", pd.Series(np.nan, index=d.index)) >= 50) & (d.get("K", pd.Series(np.nan, index=d.index)) < 80),
        "KD黃金交叉 + K>80": d.get("KD_GOLD", false) & (d.get("K", pd.Series(np.nan, index=d.index)) >= 80),
        "KD黃金交叉 + MA30向上": d.get("KD_GOLD", false) & (d.get("MA30_SLOPE3", pd.Series(np.nan, index=d.index)) > 0),
        "KD黃金交叉 + MA60向上": d.get("KD_GOLD", false) & (d.get("MA60_SLOPE3", pd.Series(np.nan, index=d.index)) > 0),
        "KD黃金交叉 + 量比>1.2": d.get("KD_GOLD", false) & (d.get("VOL_RATIO20", pd.Series(np.nan, index=d.index)) > 1.2),
        "KD黃金交叉 + 量比>1.5": d.get("KD_GOLD", false) & (d.get("VOL_RATIO20", pd.Series(np.nan, index=d.index)) > 1.5),
        "KD黃金交叉 + 站上VWAP": d.get("KD_GOLD", false) & d.get("PRICE_GT_VWAP", false),
        "MA5>15 + KD + 站上VWAP": d.get("MA_BULL_5_15", false) & d.get("KD_GOLD", false) & d.get("PRICE_GT_VWAP", false),
        "多週期條件": d.get("MTF_SIGNAL", false),
    }
    return rules.get(rule, false).fillna(False)




def _profit_factor_series(df: pd.DataFrame) -> pd.Series:
    for c in ["PF", "ProfitFactor", "Profit Factor"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(dtype=float)


def aggregate_trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty or "淨報酬%" not in trades.columns:
        return {"整體交易勝率":np.nan,"整體平均淨報酬":np.nan,"整體PF":np.nan}
    r=pd.to_numeric(trades["淨報酬%"],errors="coerce").dropna()
    if r.empty:
        return {"整體交易勝率":np.nan,"整體平均淨報酬":np.nan,"整體PF":np.nan}
    gp=r[r>0].sum()
    gl=-r[r<0].sum()
    pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
    return {"整體交易勝率":float((r>0).mean()*100),"整體平均淨報酬":float(r.mean()),"整體PF":float(pf) if np.isfinite(pf) else pf}




def summarize_pool20_stability(trades: pd.DataFrame):
    """V1.12.4：爆量分層的樣本內外與四段時間穩定度。"""
    if trades is None or trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x=trades.copy()
    buckets=["全部基準","<1倍","1-1.5倍","1.5-2倍","2-3倍",">3倍","爆量>=1.5倍","熱門動能"]

    def mask_for(name):
        if name=="全部基準":
            return pd.Series(True,index=x.index)
        if name in ["<1倍","1-1.5倍","1.5-2倍","2-3倍",">3倍"]:
            return x["爆量分層"]==name
        if name=="爆量>=1.5倍":
            return x["前日爆量>=1.5"]=="是"
        return x["前日熱門動能"]=="是"

    # IS / OOS
    sample_rows=[]
    for sample in [z for z in ["樣本內60%","樣本外40%"] if z in set(x["樣本"].dropna())]:
        xs=x[x["樣本"]==sample]
        for b in buckets:
            g=xs[mask_for(b).reindex(xs.index,fill_value=False)]
            m=aggregate_trade_metrics(g)
            sample_rows.append({"樣本":sample,"分組":b,"交易數":len(g),
                                "涵蓋股票數":int(g["股票"].nunique()) if len(g) else 0,**m})
    sample_df=pd.DataFrame(sample_rows)

    # 四段等時間區間，不以交易數等分，避免大量交易期被強迫平均。
    ts=pd.to_datetime(x["訊號時間"],errors="coerce")
    ok=ts.notna()
    xb=x.loc[ok].copy()
    ts=ts.loc[ok]
    block_rows=[]
    if len(xb):
        t0,t1=ts.min(),ts.max()
        edges=pd.date_range(t0,t1,periods=5)
        # 最後一段包含右界。
        labels=["第1段","第2段","第3段","第4段"]
        xb["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            z=xb[xb["時間段"]==block]
            for b in buckets:
                if b=="全部基準":
                    g=z
                elif b in ["<1倍","1-1.5倍","1.5-2倍","2-3倍",">3倍"]:
                    g=z[z["爆量分層"]==b]
                elif b=="爆量>=1.5倍":
                    g=z[z["前日爆量>=1.5"]=="是"]
                else:
                    g=z[z["前日熱門動能"]=="是"]
                m=aggregate_trade_metrics(g)
                block_rows.append({"時間段":block,
                                   "起始":str(edges[labels.index(block)]),
                                   "結束":str(edges[labels.index(block)+1]),
                                   "分組":b,"交易數":len(g),
                                   "涵蓋股票數":int(g["股票"].nunique()) if len(g) else 0,**m})
    return sample_df,pd.DataFrame(block_rows)


def validate_pool20_historical(symbols: List[str], cost: CostConfig, period: str = "3mo"):
    """
    V1.12.2：固定核心策略 60m KD黃金交叉+K<30+5日，
    以「訊號當下已知的前一個完成日K」建立爆量/熱門標籤，避免偷看當日收盤量。
    """
    _, _, trades = run_oos_60m_5d(symbols, cost, period, allow_overlap=False, train_ratio=0.60)
    if trades is None or trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    tickers=list(dict.fromkeys(symbols))
    daily=yf.download(tickers=tickers, period="6mo", interval="1d",
                      group_by="ticker", auto_adjust=False, progress=False, threads=False)
    contexts={}
    for s in tickers:
        try:
            if isinstance(daily.columns,pd.MultiIndex):
                if s not in daily.columns.get_level_values(0):
                    continue
                d=daily[s].copy()
            else:
                d=daily.copy()
            d=d.dropna(subset=["Close","Volume"])
            if d.empty:
                continue
            d.index=pd.to_datetime(d.index,errors="coerce")
            d=d[d.index.notna()].copy()
            d=d[d.index.weekday < 5]
            close=pd.to_numeric(d["Close"],errors="coerce")
            vol=pd.to_numeric(d["Volume"],errors="coerce")
            turn=close*vol
            rows=[]
            for i in range(20,len(d)):
                prev20=vol.iloc[i-20:i]
                prev20_turn=turn.iloc[i-20:i]
                base_vol=float(prev20.mean()) if len(prev20) else np.nan
                base_turn=float(prev20_turn.median()) if len(prev20_turn) else np.nan
                day_vol=float(vol.iloc[i])
                day_turn=float(turn.iloc[i])
                vol_ratio=day_vol/base_vol if base_vol>0 else np.nan
                turn_ratio=day_turn/base_turn if base_turn>0 else np.nan
                # 近3個「已完成」交易日（含該日）
                vol3=float(vol.iloc[max(0,i-2):i+1].mean())
                vol3_ratio=vol3/base_vol if base_vol>0 else np.nan
                rows.append({
                    "context_date":d.index[i].date(),
                    "前一完成日量比20日":vol_ratio,
                    "前一完成日成交金額比20日":turn_ratio,
                    "近3完成日均量比20日":vol3_ratio,
                })
            contexts[s]=pd.DataFrame(rows)
        except Exception:
            continue

    x=trades.copy()
    sig=pd.to_datetime(x["訊號時間"],utc=True,errors="coerce").dt.tz_convert("Asia/Taipei")
    x["訊號日期"]=sig.dt.date
    vals=[]
    for _,r in x.iterrows():
        s=r["股票"]; sd=r["訊號日期"]
        c=contexts.get(s,pd.DataFrame())
        if c.empty or pd.isna(sd):
            vals.append((np.nan,np.nan,np.nan,None))
            continue
        # 嚴格使用訊號日期以前的完成日K，杜絕使用訊號當日收盤量。
        q=c[c["context_date"] < sd]
        if q.empty:
            vals.append((np.nan,np.nan,np.nan,None))
        else:
            z=q.iloc[-1]
            vals.append((z["前一完成日量比20日"],z["前一完成日成交金額比20日"],
                         z["近3完成日均量比20日"],z["context_date"]))
    vv=pd.DataFrame(vals,columns=["前一完成日量比20日","前一完成日成交金額比20日","近3完成日均量比20日","量能基準日"],index=x.index)
    x=pd.concat([x,vv],axis=1)

    x["爆量分層"]=pd.cut(x["前一完成日量比20日"],
        bins=[-np.inf,1.0,1.5,2.0,3.0,np.inf],
        labels=["<1倍","1-1.5倍","1.5-2倍","2-3倍",">3倍"]).astype(str)
    x["前日爆量>=1.5"]=np.where(x["前一完成日量比20日"]>=1.5,"是","否")
    x["前日熱門動能"]=np.where(
        (x["前一完成日成交金額比20日"]>=1.5)&(x["近3完成日均量比20日"]>=1.2),"是","否")

    groups=[]
    specs=[
        ("全部基準",pd.Series(True,index=x.index)),
        ("<1倍",x["爆量分層"]=="<1倍"),
        ("1-1.5倍",x["爆量分層"]=="1-1.5倍"),
        ("1.5-2倍",x["爆量分層"]=="1.5-2倍"),
        ("2-3倍",x["爆量分層"]=="2-3倍"),
        (">3倍",x["爆量分層"]==">3倍"),
        ("爆量>=1.5倍",x["前日爆量>=1.5"]=="是"),
        ("熱門動能",x["前日熱門動能"]=="是"),
    ]
    for name,mask in specs:
        g=x[mask].copy()
        m=aggregate_trade_metrics(g)
        groups.append({"分組":name,"交易數":len(g),
                       "涵蓋股票數":int(g["股票"].nunique()) if len(g) else 0,**m})
    _sample_stability,_block_stability=summarize_pool20_stability(x)
    return pd.DataFrame(groups),x,_sample_stability,_block_stability



def _as_taipei_series(s):
    """將交易時間統一視為台北時間；naive時間不再誤當UTC。"""
    x=pd.to_datetime(s,errors="coerce")
    try:
        if x.dt.tz is None:
            return x.dt.tz_localize("Asia/Taipei")
        return x.dt.tz_convert("Asia/Taipei")
    except Exception:
        return x


def run_oos_60m_5d(symbols: List[str], cost: CostConfig, period: str, allow_overlap: bool = False,
                     train_ratio: float = 0.60, evaluation_months: Optional[int] = None):
    """
    V1.4.2 固定模型驗證：
    60m KD黃金交叉 + K<30，持有5日。
    先收集所有股票交易，再以「全體訊號時間」建立同一個時間切點：
    前60%時間區段 = 樣本內；後40%時間區段 = 樣本外。
    這比每檔依交易筆數各自切割更接近真正的時間OOS。
    股票池仍由近期流動性建立，因此仍不是完整 walk-forward 股票池OOS。
    """
    raw = download_intraday_batch(symbols, "60m", period)
    raw_trades = []
    p = st.progress(0, text="60m固定模型：建立全體交易…")
    for n, symbol in enumerate(symbols, 1):
        if symbol in raw and not raw[symbol].empty:
            d = add_indicators(raw[symbol])
            t = backtest(d, "60m", "KD黃金交叉 + K<30", "5日", cost)
            if not allow_overlap:
                t = enforce_non_overlapping(t)
            if not t.empty:
                x=t.copy()
                x.insert(0,"股票",symbol)
                x["研究主題"]=research_theme(symbol)
                x["_signal_dt"]=_as_taipei_series(x["訊號時間"])
                raw_trades.append(x)
        p.progress(n/max(1,len(symbols)), text=f"建立交易 {symbol}｜{n}/{len(symbols)}")
    p.empty()

    if not raw_trades:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_t=pd.concat(raw_trades,ignore_index=True).dropna(subset=["_signal_dt"]).sort_values("_signal_dt")
    # V1.16.0：可用更長資料做指標暖機，但只評估最後N個月。
    if evaluation_months is not None and not all_t.empty:
        _eval_end=all_t["_signal_dt"].max()
        _eval_start=_eval_end-pd.DateOffset(months=int(evaluation_months))
        all_t=all_t[all_t["_signal_dt"]>=_eval_start].copy()
    unique_times=pd.Series(all_t["_signal_dt"].drop_duplicates().sort_values().to_list())
    if len(unique_times)<2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cut_idx=max(1,min(len(unique_times)-1,int(len(unique_times)*train_ratio)))
    cutoff=unique_times.iloc[cut_idx]
    all_t["樣本"]=np.where(all_t["_signal_dt"] < cutoff, "樣本內60%", "樣本外40%")
    all_t["切割時間"]=cutoff
    all_t["回測版本"]=APP_VERSION

    rows=[]
    for symbol in symbols:
        stx=all_t[all_t["股票"]==symbol]
        for sample_name in ["樣本內60%","樣本外40%"]:
            part=stx[stx["樣本"]==sample_name].drop(columns=["_signal_dt"],errors="ignore")
            m=metrics(part)
            rows.append({
                "股票":symbol,"研究主題":research_theme(symbol),"樣本":sample_name,
                "切割時間":cutoff,"週期":"60m","規則":"KD黃金交叉 + K<30","持有":"5日",**m
            })

    detail=pd.DataFrame(rows)
    trades=all_t.drop(columns=["_signal_dt"],errors="ignore").reset_index(drop=True)
    summaries=[]
    for sample_name,g in detail.groupby("樣本"):
        valid=g[g["交易數"]>0].copy()
        pf=_profit_factor_series(valid)
        tm=aggregate_trade_metrics(trades[trades["樣本"]==sample_name])
        summaries.append({
            "回測版本":APP_VERSION,"樣本":sample_name,"共同切割時間":cutoff,
            "股票數":int(valid["股票"].nunique()),"總交易數":int(valid["交易數"].sum()),
            "正期望股票比例":float((valid["期望值%"]>0).mean()*100) if len(valid) else np.nan,
            "平均期望值":float(valid["期望值%"].mean()) if len(valid) else np.nan,
            "期望值中位數":float(valid["期望值%"].median()) if len(valid) else np.nan,
            "PF中位數":float(pf.median()) if not pf.empty else np.nan,
            **tm,
            "平均勝率":float(valid["勝率%"].mean()) if len(valid) else np.nan,
            "平均最大回撤":float(valid["最大回撤%"].mean()) if len(valid) else np.nan,
        })
    return detail,pd.DataFrame(summaries),trades


def state_at_entry_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    """描述訊號發生當下的狀態；只做診斷，不自動把最佳分組變成新規則。"""
    if trades is None or trades.empty:
        return pd.DataFrame()
    t=trades.copy()
    groups=[]

    def add_group(factor, label, mask):
        g=t.loc[mask].copy()
        if g.empty:
            return
        r=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna()
        if r.empty:
            return
        gp=r[r>0].sum(); gl=-r[r<0].sum()
        pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
        groups.append({
            "狀態因子":factor,"分組":label,"交易數":len(r),
            "股票數":int(g["股票"].nunique()),
            "勝率%":float((r>0).mean()*100),"平均淨報酬%":float(r.mean()),
            "中位淨報酬%":float(r.median()),"整體PF":pf
        })

    if "K區間" in t:
        add_group("K深度","K<20",t["K區間"].eq("K<20"))
        add_group("K深度","K20-30",t["K區間"].eq("K20-30"))
    if {"訊號收盤","訊號VWAP"}.issubset(t.columns):
        add_group("VWAP位置","收盤>VWAP",t["訊號收盤"]>t["訊號VWAP"])
        add_group("VWAP位置","收盤<=VWAP",t["訊號收盤"]<=t["訊號VWAP"])
    for c,label in [("MA30斜率3","MA30"),("MA60斜率3","MA60")]:
        if c in t:
            x=pd.to_numeric(t[c],errors="coerce")
            add_group(f"{label}方向",f"{label}上彎",x>0)
            add_group(f"{label}方向",f"{label}下彎/平",x<=0)
    if "量比20" in t:
        x=pd.to_numeric(t["量比20"],errors="coerce")
        add_group("量比","量比<1",x<1)
        add_group("量比","量比>=1",x>=1)
    if "研究主題" in t:
        for theme,gidx in t.groupby("研究主題").groups.items():
            add_group("研究主題",str(theme),t.index.isin(gidx))
    return pd.DataFrame(groups)


def state_filter_robustness(trades: pd.DataFrame, blocks: int = 4) -> pd.DataFrame:
    """
    V1.5.2：把V1.5.1觀察到的狀態只當「候選假說」。
    比較基準與少量預先固定的候選濾網，要求跨連續時間區段仍成立；
    不自動挑最佳參數、不回寫成正式交易規則。
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    t=trades.copy()
    t["_dt"]=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce")
    t=t.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    if len(t)<blocks:
        return pd.DataFrame()

    q=np.linspace(0,1,blocks+1)
    bounds=t["_dt"].quantile(q).tolist()

    def num(col):
        return pd.to_numeric(t[col],errors="coerce") if col in t.columns else pd.Series(np.nan,index=t.index)

    k=t["K區間"] if "K區間" in t.columns else pd.Series("",index=t.index)
    vol=num("量比20")
    ma60=num("MA60斜率3")

    candidates={
        "基準｜K<30": pd.Series(True,index=t.index),
        "候選A｜K<20": k.eq("K<20"),
        "候選B｜量比>=1": vol.ge(1),
        "候選C｜MA60下彎/平": ma60.le(0),
        "候選D｜MA60下彎/平＋量比>=1": ma60.le(0) & vol.ge(1),
    }

    rows=[]
    for name,mask in candidates.items():
        for i in range(blocks):
            lo,hi=bounds[i],bounds[i+1]
            tm=(t["_dt"]>=lo) & ((t["_dt"]<=hi) if i==blocks-1 else (t["_dt"]<hi))
            g=t.loc[mask & tm].copy()
            m=aggregate_trade_metrics(g)
            r=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna() if not g.empty else pd.Series(dtype=float)
            rows.append({
                "候選狀態":name,"區段":f"時間區段{i+1}/{blocks}",
                "開始":lo,"結束":hi,"交易數":len(r),
                "股票數":int(g["股票"].nunique()) if not g.empty else 0,
                "涵蓋率%":float(len(r)/max(1,int(tm.sum()))*100),
                "勝率%":float((r>0).mean()*100) if len(r) else np.nan,
                "平均淨報酬%":float(r.mean()) if len(r) else np.nan,
                "中位淨報酬%":float(r.median()) if len(r) else np.nan,
                "整體PF":m["整體PF"],
            })
    return pd.DataFrame(rows)


def market_regime_diagnostics(trades: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    V1.6.0：用台股加權指數 ^TWII 的日線，描述每筆60m訊號當時的大盤環境。
    僅做診斷，不把結果直接變成交易濾網。
    使用前一個已完成交易日的日線，避免把訊號之後才知道的資料帶入。
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    try:
        idx=yf.download("^TWII", period="6mo", interval="1d", auto_adjust=False,
                        progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if idx is None or idx.empty:
        return pd.DataFrame()
    if isinstance(idx.columns,pd.MultiIndex):
        # yfinance 單一ticker在不同版本可能仍回傳MultiIndex
        try:
            idx=idx.xs("^TWII",axis=1,level=-1)
        except Exception:
            idx.columns=[c[0] if isinstance(c,tuple) else c for c in idx.columns]
    idx=idx.rename(columns={c:str(c).title() for c in idx.columns})
    if "Close" not in idx.columns:
        return pd.DataFrame()
    idx=idx.copy()
    idx.index=pd.to_datetime(idx.index,utc=True,errors="coerce")
    idx=idx[~idx.index.isna()].sort_index()
    idx["MKT_MA5"]=idx["Close"].rolling(5).mean()
    idx["MKT_MA15"]=idx["Close"].rolling(15).mean()
    idx["MKT_RET5"]=idx["Close"].pct_change(5)*100
    idx["MKT_RET15"]=idx["Close"].pct_change(15)*100
    idx["MKT_MA15_SLOPE"]=idx["MKT_MA15"].diff(3)
    # shift(1)：訊號當天只使用前一個完成日
    m=idx[["Close","MKT_MA5","MKT_MA15","MKT_RET5","MKT_RET15","MKT_MA15_SLOPE"]].shift(1).dropna().reset_index()
    m=m.rename(columns={m.columns[0]:"_mkt_dt","Close":"大盤收盤"})
    m["_mkt_dt"]=pd.to_datetime(m["_mkt_dt"],utc=True,errors="coerce")

    t=trades.copy()
    t["_dt"]=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce")
    t=t.dropna(subset=["_dt"]).sort_values("_dt")
    x=pd.merge_asof(t,m.sort_values("_mkt_dt"),left_on="_dt",right_on="_mkt_dt",direction="backward")
    x["大盤站上MA15"]=x["大盤收盤"]>=x["MKT_MA15"]
    x["大盤MA15上彎"]=x["MKT_MA15_SLOPE"]>0
    x["大盤5日報酬正"]=x["MKT_RET5"]>0

    rows=[]
    for factor,col in [("大盤位置","大盤站上MA15"),("大盤趨勢","大盤MA15上彎"),("大盤短動能","大盤5日報酬正")]:
        for val,label in [(True,"是"),(False,"否")]:
            g=x[x[col].eq(val)]
            if g.empty: continue
            r=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna()
            gp=r[r>0].sum(); gl=-r[r<0].sum()
            pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
            rows.append({"市場因子":factor,"狀態":label,"交易數":len(r),
                         "股票數":int(g["股票"].nunique()),"勝率%":float((r>0).mean()*100),
                         "平均淨報酬%":float(r.mean()),"中位淨報酬%":float(r.median()),"整體PF":pf})
    return pd.DataFrame(rows)


def market_regime_by_timeblock(trades: pd.DataFrame, blocks: int = 4) -> pd.DataFrame:
    """
    把 ^TWII 前一完成交易日的市場狀態，與四段時間直接交叉。
    目的：確認第1段失效是否真的由某一大盤regime主導，而非只看全期間分組。
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    try:
        idx=yf.download("^TWII", period="6mo", interval="1d", auto_adjust=False,
                        progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if idx is None or idx.empty:
        return pd.DataFrame()
    if isinstance(idx.columns,pd.MultiIndex):
        try:
            idx=idx.xs("^TWII",axis=1,level=-1)
        except Exception:
            idx.columns=[c[0] if isinstance(c,tuple) else c for c in idx.columns]
    idx=idx.rename(columns={c:str(c).title() for c in idx.columns})
    if "Close" not in idx.columns:
        return pd.DataFrame()
    idx.index=pd.to_datetime(idx.index,utc=True,errors="coerce")
    idx=idx[~idx.index.isna()].sort_index()
    idx["MKT_MA15"]=idx["Close"].rolling(15).mean()
    idx["MKT_RET5"]=idx["Close"].pct_change(5)*100
    idx["MKT_MA15_SLOPE"]=idx["MKT_MA15"].diff(3)
    m=idx[["Close","MKT_MA15","MKT_RET5","MKT_MA15_SLOPE"]].shift(1).dropna().reset_index()
    m=m.rename(columns={m.columns[0]:"_mkt_dt","Close":"大盤收盤"})
    m["_mkt_dt"]=pd.to_datetime(m["_mkt_dt"],utc=True,errors="coerce")

    t=trades.copy()
    t["_dt"]=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce")
    t=t.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    x=pd.merge_asof(t,m.sort_values("_mkt_dt"),left_on="_dt",right_on="_mkt_dt",direction="backward")
    x["大盤MA15上彎"]=x["MKT_MA15_SLOPE"]>0
    x["大盤5日報酬正"]=x["MKT_RET5"]>0

    bounds=x["_dt"].quantile(np.linspace(0,1,blocks+1)).tolist()
    rows=[]
    for i in range(blocks):
        lo,hi=bounds[i],bounds[i+1]
        tm=(x["_dt"]>=lo) & ((x["_dt"]<=hi) if i==blocks-1 else (x["_dt"]<hi))
        for factor,col in [("大盤MA15上彎","大盤MA15上彎"),("大盤5日報酬正","大盤5日報酬正")]:
            for val,label in [(True,"是"),(False,"否")]:
                g=x[tm & x[col].eq(val)]
                r=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna()
                if r.empty: continue
                gp=r[r>0].sum(); gl=-r[r<0].sum()
                pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
                rows.append({
                    "區段":f"時間區段{i+1}/{blocks}","開始":lo,"結束":hi,
                    "市場因子":factor,"狀態":label,"交易數":len(r),
                    "占該段交易%":float(len(r)/max(1,int(tm.sum()))*100),
                    "勝率%":float((r>0).mean()*100),"平均淨報酬%":float(r.mean()),
                    "中位淨報酬%":float(r.median()),"整體PF":pf
                })
    return pd.DataFrame(rows)


def walkforward_tradability_validation(trades: pd.DataFrame, daily_lookback: int = 20) -> pd.DataFrame:
    """
    V1.7.0：Walk-Forward股票池偏誤診斷。
    對每筆既有60m交易，使用該股票「訊號日前一交易日以前」的日線資料，
    計算過去20日成交金額/成交量/振幅的相對排名。
    不使用訊號日之後資料，也不使用目前最新流動性來決定歷史排名。
    注意：這仍是在目前候選universe內做歷史可交易性重建，不等於完整歷史上市櫃成分重建。
    """
    if trades is None or trades.empty or "股票" not in trades.columns:
        return pd.DataFrame(), pd.DataFrame(), "沒有可用的OOS逐筆交易或缺少股票欄位。"

    t=trades.copy()
    t["_dt"]=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce")
    t=t.dropna(subset=["_dt"]).copy()
    symbols=sorted(t["股票"].dropna().astype(str).unique().tolist())
    if not symbols:
        return pd.DataFrame(), pd.DataFrame(), "逐筆交易中沒有股票代號。"

    def wf_yf_symbol(s: str) -> str:
        s=str(s).strip()
        if s.endswith(".TW") or s.endswith(".TWO"):
            return s
        # 現有universe多數已在SHORT_TERM_UNIVERSE；優先沿用其yfinance代碼。
        for item in SHORT_TERM_UNIVERSE:
            code=str(item).split(".")[0]
            if code==s:
                return str(item)
        # fallback：台股四位數先以上市 .TW 嘗試
        return f"{s}.TW"

    # 下載足夠長的日線，供每個訊號點向前看20個完成交易日。
    try:
        raw=yf.download(
            tickers=" ".join([wf_yf_symbol(s) for s in symbols]),
            period="6mo", interval="1d", auto_adjust=False,
            progress=False, threads=False, group_by="ticker"
        )
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Yahoo日線下載失敗：{type(e).__name__}: {e}"
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame(), "Yahoo日線下載結果為空。"

    rows=[]
    for symbol in symbols:
        yf_sym=wf_yf_symbol(symbol)
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                if yf_sym in raw.columns.get_level_values(0):
                    d=raw[yf_sym].copy()
                elif yf_sym in raw.columns.get_level_values(-1):
                    d=raw.xs(yf_sym,axis=1,level=-1).copy()
                else:
                    continue
            else:
                d=raw.copy()
            d.columns=[str(c).title() for c in d.columns]
            if not {"Close","High","Low","Volume"}.issubset(d.columns):
                continue
            d=d.dropna(subset=["Close"]).copy()
            d.index=pd.to_datetime(d.index,utc=True,errors="coerce")
            d=d[~d.index.isna()].sort_index()
            d["成交金額代理"]=pd.to_numeric(d["Close"],errors="coerce")*pd.to_numeric(d["Volume"],errors="coerce")
            d["振幅%"]=(pd.to_numeric(d["High"],errors="coerce")-pd.to_numeric(d["Low"],errors="coerce"))/pd.to_numeric(d["Close"],errors="coerce").replace(0,np.nan)*100
            d["WF成交金額"]=d["成交金額代理"].rolling(daily_lookback,min_periods=10).median().shift(1)
            d["WF成交量"]=pd.to_numeric(d["Volume"],errors="coerce").rolling(daily_lookback,min_periods=10).median().shift(1)
            d["WF振幅"]=d["振幅%"].rolling(daily_lookback,min_periods=10).median().shift(1)
            dd=d[["WF成交金額","WF成交量","WF振幅"]].dropna().reset_index()
            dd=dd.rename(columns={dd.columns[0]:"_daily_dt"})
            dd["_daily_dt"]=pd.to_datetime(dd["_daily_dt"],utc=True,errors="coerce")
            stx=t[t["股票"].astype(str)==symbol].sort_values("_dt").copy()
            if stx.empty or dd.empty: continue
            z=pd.merge_asof(stx,dd.sort_values("_daily_dt"),left_on="_dt",right_on="_daily_dt",direction="backward")
            rows.append(z)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), "沒有任何股票成功完成歷史日線與訊號時間配對。"

    x=pd.concat(rows,ignore_index=True)
    x=x.dropna(subset=["WF成交金額","WF成交量","WF振幅"]).copy()
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(), "完成配對後，20日歷史可交易性欄位全部為空；可能是日線暖機資料不足。"

    # 每個交易日橫向排名：只用當時可取得的歷史20日資訊。
    x["_signal_day"]=x["_dt"].dt.floor("D")
    for c,outc in [("WF成交金額","成交金額百分位"),("WF成交量","成交量百分位"),("WF振幅","振幅百分位")]:
        x[outc]=x.groupby("_signal_day")[c].rank(pct=True,method="average")*100
    x["WF可交易分數"]=x["成交金額百分位"]*0.50+x["成交量百分位"]*0.20+x["振幅百分位"]*0.30

    # 固定分層，不事後最佳化切點。
    x["WF分層"]=pd.cut(
        x["WF可交易分數"],[-np.inf,40,60,80,np.inf],
        labels=["低於40","40-60","60-80","80以上"],right=False
    )

    summary=[]
    for layer,g in x.groupby("WF分層",observed=True):
        r=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna()
        if r.empty: continue
        gp=r[r>0].sum(); gl=-r[r<0].sum()
        pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
        summary.append({
            "歷史可交易性分層":str(layer),"交易數":len(r),
            "股票數":int(g["股票"].nunique()),
            "勝率%":float((r>0).mean()*100),
            "平均淨報酬%":float(r.mean()),"中位淨報酬%":float(r.median()),
            "整體PF":pf
        })

    detail_cols=[c for c in [
        "股票","研究主題","訊號時間","進場時間","出場時間","淨報酬%",
        "WF成交金額","WF成交量","WF振幅","成交金額百分位","成交量百分位",
        "振幅百分位","WF可交易分數","WF分層"
    ] if c in x.columns]
    return pd.DataFrame(summary),x[detail_cols].sort_values("訊號時間").reset_index(drop=True), f"成功：{x['股票'].nunique()}檔、{len(x)}筆交易完成Walk-Forward歷史可交易性配對。"


def walkforward_layer_time_validation(wf_detail: pd.DataFrame, blocks: int = 4) -> pd.DataFrame:
    """固定WF分層 × 連續四段時間；避免只看全期間平均後誤判股票池規則。"""
    if wf_detail is None or wf_detail.empty:
        return pd.DataFrame()
    x=wf_detail.copy()
    x["_dt"]=pd.to_datetime(x["訊號時間"],utc=True,errors="coerce")
    x=x.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    if len(x)<blocks:
        return pd.DataFrame()
    bounds=x["_dt"].quantile(np.linspace(0,1,blocks+1)).tolist()
    rows=[]
    order=["低於40","40-60","60-80","80以上"]
    for i in range(blocks):
        lo,hi=bounds[i],bounds[i+1]
        tm=(x["_dt"]>=lo) & ((x["_dt"]<=hi) if i==blocks-1 else (x["_dt"]<hi))
        for layer in order:
            g=x[tm & x["WF分層"].astype(str).eq(layer)]
            r=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna()
            if r.empty: continue
            gp=r[r>0].sum(); gl=-r[r<0].sum()
            pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
            rows.append({
                "區段":f"時間區段{i+1}/{blocks}","開始":lo,"結束":hi,
                "歷史可交易性分層":layer,"交易數":len(r),
                "勝率%":float((r>0).mean()*100),
                "平均淨報酬%":float(r.mean()),"中位淨報酬%":float(r.median()),
                "整體PF":pf
            })
    return pd.DataFrame(rows)


def build_research_radar_candidates(trades: pd.DataFrame, wf_detail: pd.DataFrame) -> pd.DataFrame:
    """
    V1.8.0 研究雷達候選：
    - 核心訊號固定：60m KD黃金交叉 + K<30
    - 不把WF可交易分數當預測報酬排名；只標示歷史可交易性層級
    - 80以上不直接排除，因V1.7.2顯示其弱勢並非所有時間段都成立
    - 排序優先使用訊號新鮮度與K深度；此表是研究候選，不是投資建議/自動下單
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    t=trades.copy()
    t["_dt"]=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce")
    t=t.dropna(subset=["_dt"]).copy()

    # 合併當時的WF可交易性資訊
    if wf_detail is not None and not wf_detail.empty:
        w=wf_detail.copy()
        w["_dt"]=pd.to_datetime(w["訊號時間"],utc=True,errors="coerce")
        keep=[c for c in ["股票","_dt","WF可交易分數","WF分層","WF成交金額","WF成交量","WF振幅"] if c in w.columns]
        w=w[keep].drop_duplicates(["股票","_dt"])
        t=t.merge(w,on=["股票","_dt"],how="left",suffixes=("","_wf"))

    # K值欄位若存在，K越低只作研究排序輔助；不改核心訊號門檻K<30。
    kval=None
    for c in ["訊號K","K"]:
        if c in t.columns:
            kval=pd.to_numeric(t[c],errors="coerce")
            break
    if kval is None:
        kval=pd.Series(np.nan,index=t.index)

    latest=t["_dt"].max()
    t["距最新訊號小時"]=(latest-t["_dt"]).dt.total_seconds()/3600
    t["K深度分"]=np.where(kval.notna(),(30-kval).clip(lower=0,upper=30)/30*100,np.nan)
    t["訊號新鮮度分"]=(100-(t["距最新訊號小時"]/24*8)).clip(lower=0,upper=100)

    # 不用WF分數預測報酬；只給可交易性標籤。研究排序=新鮮度70% + K深度30%(有K時)
    # V1.8.1 不再把「新鮮度70%＋K深度30%」當成有效預測排名。
    # 保留兩個原始維度供使用者判讀；排序只依訊號時間由新到舊。
    t["核心規則"]="60m KD黃金交叉 + K<30｜研究持有5日"
    t["用途"]="研究雷達候選（非投資建議）"

    cols=[c for c in [
        "股票","研究主題","訊號時間","核心規則","WF分層","WF可交易分數",
        "訊號新鮮度分","K深度分","用途"
    ] if c in t.columns]
    return t[cols].sort_values(["訊號時間","股票"],ascending=[False,True]).reset_index(drop=True)


def validate_radar_ranking(radar: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    V1.8.1：驗證雷達研究分數是否真的具有排序資訊。
    以既有逐筆交易實現報酬回填，固定切成五分位；只做驗證，不自動調權重。
    """
    if radar is None or radar.empty or trades is None or trades.empty:
        return pd.DataFrame()
    r=radar.copy()
    t=trades.copy()
    r["_dt"]=pd.to_datetime(r["訊號時間"],utc=True,errors="coerce")
    t["_dt"]=pd.to_datetime(t["訊號時間"],utc=True,errors="coerce")
    keep=[c for c in ["股票","_dt","淨報酬%"] if c in t.columns]
    x=r.merge(t[keep].drop_duplicates(["股票","_dt"]),on=["股票","_dt"],how="left")
    x=x.dropna(subset=["雷達研究分數","淨報酬%"]).copy()
    if len(x)<20:
        return pd.DataFrame()
    # rank(method=first) only resolves duplicate score edges; quintile definitions are fixed.
    x["_rank"]=pd.to_numeric(x["雷達研究分數"],errors="coerce").rank(method="first")
    x["雷達分數五分位"]=pd.qcut(x["_rank"],5,labels=["Q1低","Q2","Q3","Q4","Q5高"])
    rows=[]
    for q,g in x.groupby("雷達分數五分位",observed=True):
        ret=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna()
        gp=ret[ret>0].sum(); gl=-ret[ret<0].sum()
        pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
        rows.append({
            "雷達分數五分位":str(q),"交易數":len(ret),
            "勝率%":float((ret>0).mean()*100),
            "平均淨報酬%":float(ret.mean()),"中位淨報酬%":float(ret.median()),
            "整體PF":pf
        })
    return pd.DataFrame(rows)


def build_daily_radar_status(trades: pd.DataFrame, wf_detail: pd.DataFrame, observe_days: int = 5) -> pd.DataFrame:
    """
    V1.9.0 每日雷達狀態表。
    固定研究主線：60m KD黃金交叉 + K<30，5個交易日觀察。
    不建立預測性總分；只呈現訊號、時間、KD與可交易性背景。
    """
    if trades is None or trades.empty:
        return pd.DataFrame()

    x = trades.copy()
    x["_dt"] = pd.to_datetime(x["訊號時間"], utc=True, errors="coerce")
    x = x.dropna(subset=["_dt"]).copy()
    if x.empty:
        return pd.DataFrame()

    # 合併 WF 背景
    if wf_detail is not None and not wf_detail.empty:
        w = wf_detail.copy()
        w["_dt"] = pd.to_datetime(w["訊號時間"], utc=True, errors="coerce")
        keep = [c for c in ["股票","_dt","WF可交易分數","WF分層"] if c in w.columns]
        if "股票" in keep and "_dt" in keep:
            w = w[keep].drop_duplicates(["股票","_dt"])
            x = x.merge(w, on=["股票","_dt"], how="left", suffixes=("","_wf"))

    # 每檔只留最新一個有效歷史訊號，避免雷達同股重複
    x = x.sort_values("_dt").groupby("股票", as_index=False).tail(1).copy()
    latest_signal_time = x["_dt"].max()
    now_tw = pd.Timestamp.now(tz="Asia/Taipei")
    now_utc = now_tw.tz_convert("UTC")

    # 以工作日估算5日研究觀察窗；後續接正式交易日曆/Shioaji時再處理國定假日。
    sig_date = x["_dt"].dt.tz_convert("Asia/Taipei").dt.date
    x["預計觀察至"] = [pd.Timestamp(np.busday_offset(d, observe_days, roll="forward")).date() for d in sig_date]
    current_date = now_tw.date()

    # V1.9.1：狀態必須相對「現在」而不是相對「最後一筆訊號」。
    # 新訊號暫定24小時內；觀察中則依5工作日觀察窗。
    age_hours = (now_utc - x["_dt"]).dt.total_seconds() / 3600
    x["距現在小時"] = age_hours.round(1)
    x["目前狀態"] = np.where(
        (age_hours >= 0) & (age_hours <= 24), "🟢 新訊號",
        np.where(pd.to_datetime(x["預計觀察至"]) >= pd.Timestamp(current_date), "🟡 觀察中", "⚪ 已逾期")
    )

    kval = None
    for c in ["訊號K","K"]:
        if c in x.columns:
            kval = pd.to_numeric(x[c], errors="coerce")
            break
    if kval is not None:
        x["60m K值"] = kval.round(2)

    x["KD狀態"] = "黃金交叉＋K<30"
    x["訊號時間"] = x["_dt"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
    x["核心規則"] = "60m KD黃金交叉 + K<30｜5日研究觀察"

    cols = [c for c in [
        "股票","研究主題","60m K值","KD狀態","訊號時間","距現在小時",
        "預計觀察至","WF分層","WF可交易分數","目前狀態","核心規則"
    ] if c in x.columns]

    status_order = {"🟢 新訊號":0, "🟡 觀察中":1, "⚪ 已逾期":2}
    x["_status_order"] = x["目前狀態"].map(status_order).fillna(9)
    x = x.sort_values(["_status_order","_dt"], ascending=[True,False])
    return x[cols].reset_index(drop=True)


def scan_latest_60m_radar(symbols: List[str], ranked_pool: pd.DataFrame, period: str = "3mo",
                            observe_days: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    V1.10.0 真正的最新60m雷達：
    直接掃描最新60m行情，不從歷史回測交易表反推「今天」。
    規則固定：KD黃金交叉 + K<30。
    回傳：(有效/近期雷達, 掃描診斷)
    """
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()

    raw_map = download_intraday_batch(symbols, "60m", period)
    now_tw = pd.Timestamp.now(tz="Asia/Taipei")
    rows, diag = [], []

    pool_map = {}
    if ranked_pool is not None and not ranked_pool.empty and "股票" in ranked_pool.columns:
        pool_map = ranked_pool.set_index("股票").to_dict("index")

    for symbol in symbols:
        d = raw_map.get(symbol, pd.DataFrame())
        if d is None or d.empty:
            diag.append({"股票":symbol,"狀態":"無60m資料","行情最後K棒":pd.NaT,"有效K棒數":0})
            continue
        x = add_indicators(d)
        if x.empty:
            diag.append({"股票":symbol,"狀態":"指標資料不足","行情最後K棒":pd.NaT,"有效K棒數":0})
            continue

        # 保守處理：只使用已開始至少60分鐘的bar，避免盤中未完成K棒觸發假訊號。
        idx = pd.DatetimeIndex(x.index)
        if idx.tz is None:
            idx_tw = idx.tz_localize("Asia/Taipei")
        else:
            idx_tw = idx.tz_convert("Asia/Taipei")
        complete = (idx_tw + pd.Timedelta(minutes=60)) <= now_tw
        xc = x.loc[complete].copy()
        if xc.empty:
            diag.append({"股票":symbol,"狀態":"沒有已完成60m K棒","行情最後K棒":pd.NaT,"有效K棒數":0})
            continue

        last_bar = pd.Timestamp(xc.index[-1])
        last_bar_tw = last_bar.tz_localize("Asia/Taipei") if last_bar.tzinfo is None else last_bar.tz_convert("Asia/Taipei")
        mask = xc["KD_GOLD"].fillna(False) & (pd.to_numeric(xc["K"],errors="coerce") < 30)
        sig = xc.loc[mask]
        diag.append({
            "股票":symbol,"狀態":"掃描完成","行情最後K棒":last_bar_tw.strftime("%Y-%m-%d %H:%M"),
            "有效K棒數":len(xc),"目前K":float(xc["K"].iloc[-1]) if pd.notna(xc["K"].iloc[-1]) else np.nan,
            "目前D":float(xc["D"].iloc[-1]) if pd.notna(xc["D"].iloc[-1]) else np.nan
        })
        if sig.empty:
            continue

        srow=sig.iloc[-1]
        stime=pd.Timestamp(sig.index[-1])
        stime_tw=stime.tz_localize("Asia/Taipei") if stime.tzinfo is None else stime.tz_convert("Asia/Taipei")
        sdate=stime_tw.date()
        observe_to=pd.Timestamp(np.busday_offset(sdate, observe_days, roll="forward")).date()
        age_h=(now_tw-stime_tw).total_seconds()/3600
        # 最新交易日判斷稍後依全體行情最後K棒日期統一修正；
        # 這裡先依5工作日窗判斷觀察中/逾期，避免週末用24小時誤殺週五新訊號。
        if pd.Timestamp(observe_to) >= pd.Timestamp(now_tw.date()):
            status="🟡 觀察中"
        else:
            status="⚪ 已逾期"

        pm=pool_map.get(symbol,{})
        _vr=float(srow["VOL_RATIO20"]) if "VOL_RATIO20" in srow.index and pd.notna(srow["VOL_RATIO20"]) else np.nan
        _ma60=float(srow["MA60_SLOPE3"]) if "MA60_SLOPE3" in srow.index and pd.notna(srow["MA60_SLOPE3"]) else np.nan
        _g1=pd.notna(_vr) and _vr>=1.5
        _g2=pd.notna(_ma60) and _ma60<=0
        _grade="S級" if (_g1 and _g2) else ("A級" if (_g1 or _g2) else "B級")
        _rank=pd.to_numeric(pd.Series([pm.get("流動性排名",np.nan)]),errors="coerce").iloc[0]
        if pd.notna(_rank) and _rank<=50:
            _layer="核心_TOP1-50"
        elif pd.notna(_rank) and _rank<=100:
            _layer="觀察_TOP51-100"
        elif pd.notna(_rank) and _rank<=150:
            _layer="擴充_TOP101-150"
        else:
            _layer=pm.get("股票池層級","")

        rows.append({
            "股票":symbol,
            "公司":pm.get("公司",""),
            "市場":pm.get("市場",""),
            "研究主題":research_theme(symbol),
            "流動性排名":_rank,
            "股票池層級":_layer,
            "訊號等級":_grade,
            "量比20":round(_vr,3) if pd.notna(_vr) else np.nan,
            "MA60斜率3":round(_ma60,3) if pd.notna(_ma60) else np.nan,
            "目前狀態":status,
            "訊號時間":stime_tw.strftime("%Y-%m-%d %H:%M"),
            "訊號K":round(float(srow["K"]),2) if pd.notna(srow["K"]) else np.nan,
            "訊號D":round(float(srow["D"]),2) if pd.notna(srow["D"]) else np.nan,
            "目前K":round(float(xc["K"].iloc[-1]),2) if pd.notna(xc["K"].iloc[-1]) else np.nan,
            "目前D":round(float(xc["D"].iloc[-1]),2) if pd.notna(xc["D"].iloc[-1]) else np.nan,
            "行情最後K棒":last_bar_tw.strftime("%Y-%m-%d %H:%M"),
            "距訊號小時":round(age_h,1),
            "預計觀察至":observe_to,
            "目前短線可交易分":pm.get("短線可交易分",np.nan),
            "核心規則":"60m KD黃金交叉 + K<30｜5日研究觀察"
        })

    radar=pd.DataFrame(rows)
    diagnostics=pd.DataFrame(diag)
    if radar.empty:
        return radar, diagnostics

    # V1.10.1：以本批100檔共同的「行情最後交易日」定義新訊號。
    # 例如週日執行時，週五訊號仍應是新訊號，而不是因超過24小時被降成觀察中。
    last_market_dates=pd.to_datetime(diagnostics["行情最後K棒"],errors="coerce").dropna()
    if not last_market_dates.empty:
        latest_market_date=last_market_dates.max().date()
        sig_dates=pd.to_datetime(radar["訊號時間"],errors="coerce").dt.date
        radar.loc[sig_dates==latest_market_date,"目前狀態"]="🟢 新訊號"

    order={"🟢 新訊號":0,"🟡 觀察中":1,"⚪ 已逾期":2}
    radar["_o"]=radar["目前狀態"].map(order).fillna(9)
    _grade_order={"S級":0,"A級":1,"B級":2}
    radar["_g"]=radar.get("訊號等級",pd.Series(index=radar.index,dtype=str)).map(_grade_order).fillna(9)
    radar=radar.sort_values(["_o","_g","流動性排名","訊號時間"],
                            ascending=[True,True,True,False]).drop(columns=["_o","_g"]).reset_index(drop=True)
    return radar, diagnostics



def fetch_official_tw_stock_universe():
    """
    V1.16.32：
    官方上市/上櫃公司基本資料，加入3次重試與漸進等待。
    只保留4位數字公司代號；上市加.TW、上櫃加.TWO。
    官方來源多次失敗時仍明確回報，不靜默冒充全市場。
    """
    endpoints=[
        ("上市","https://openapi.twse.com.tw/v1/opendata/t187ap03_L",".TW"),
        ("上櫃","https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",".TWO"),
    ]
    rows=[]
    errors=[]

    for market,url,suffix in endpoints:
        data=None
        last_error=None

        for attempt in range(1,4):
            try:
                req=urllib.request.Request(
                    url,
                    headers={
                        "User-Agent":"Mozilla/5.0",
                        "Accept":"application/json,text/plain,*/*",
                        "Connection":"close",
                    }
                )
                with urllib.request.urlopen(req,timeout=25) as resp:
                    raw=resp.read().decode("utf-8")
                data=json.loads(raw)
                if isinstance(data,list) and len(data)>0:
                    break
                last_error=RuntimeError("官方API回傳空資料")
            except Exception as e:
                last_error=e

            if attempt<3:
                time.sleep(1.2*attempt)

        if data is None or not isinstance(data,list) or len(data)==0:
            errors.append(f"{market}: 重試3次仍失敗｜{last_error}")
            continue

        for item in data:
            code=str(item.get("公司代號",item.get("SecuritiesCompanyCode",""))).strip()
            name=str(item.get("公司簡稱",item.get("CompanyName",""))).strip()
            industry=str(item.get("產業別",item.get("SecuritiesIndustryCode",""))).strip()
            if len(code)==4 and code.isdigit():
                rows.append({
                    "股票":code+suffix,
                    "代號":code,
                    "公司":name,
                    "市場":market,
                    "官方產業別":industry,
                    "官方來源":url
                })

    df=pd.DataFrame(rows).drop_duplicates("股票") if rows else pd.DataFrame()
    return df,errors


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






@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)
def build_current_formal_radar_pool(max_rank: int = 150):
    """
    V1.16.16 正式今日雷達股票池：
    - 母池：官方上市+上櫃完整公司名單。
    - 只用「今天以前」已完成日K，避免盤中日K不完整造成排名前視。
    - 以最近20個完成交易日的成交金額中位數做全市場流動性排名。
    - 正式掃描到TOP150：
        TOP1-50 核心
        TOP51-100 觀察
        TOP101-150 擴充（只有S級訊號才進正式雷達）
    """
    official, errors=fetch_official_tw_stock_universe()
    if official is None or official.empty:
        return pd.DataFrame(),pd.DataFrame(),errors

    dmap=download_daily_batches(official["股票"].tolist(),period="2mo",batch_size=25)
    today_tw=pd.Timestamp.now(tz="Asia/Taipei").date()
    rows=[]
    for _,meta in official.iterrows():
        s=meta["股票"]
        d=dmap.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy()
            x.index=pd.to_datetime(x.index,errors="coerce")
            x=x[x.index.notna()]
            # 永遠排除今天這根日K，只用前一完成交易日以前資料。
            x=x[pd.Index([pd.Timestamp(i).date() for i in x.index]) < today_tw]
            x=x.dropna(subset=["Close","Volume"]).tail(20)
            if len(x)<20:
                continue
            close=pd.to_numeric(x["Close"],errors="coerce")
            vol=pd.to_numeric(x["Volume"],errors="coerce")
            turn=(close*vol).replace([np.inf,-np.inf],np.nan)
            med=float(turn.median()) if turn.notna().any() else np.nan
            if not np.isfinite(med):
                continue
            rows.append({
                "股票":s,
                "公司":meta.get("公司",""),
                "市場":meta.get("市場",""),
                "官方產業別":meta.get("官方產業別",""),
                "20日成交金額中位數":med,
                "排名基準完成日":pd.Timestamp(x.index[-1]).date(),
            })
        except Exception:
            continue

    pool=pd.DataFrame(rows)
    if pool.empty:
        diag=pd.DataFrame([{
            "官方股票數":len(official),"日K成功股票數":len(dmap),
            "可排名股票數":0,"正式掃描股票數":0
        }])
        return pool,diag,errors

    pool=pool.sort_values(["20日成交金額中位數","股票"],ascending=[False,True]).reset_index(drop=True)
    pool["流動性排名"]=np.arange(1,len(pool)+1)
    pool["股票池層級"]=np.select(
        [pool["流動性排名"]<=50,pool["流動性排名"]<=100,pool["流動性排名"]<=150],
        ["核心_TOP1-50","觀察_TOP51-100","擴充_TOP101-150"],
        default="池外"
    )
    selected=pool[pool["流動性排名"]<=int(max_rank)].copy()

    diag=pd.DataFrame([{
        "官方股票數":len(official),
        "日K成功股票數":len(dmap),
        "可排名股票數":len(pool),
        "正式掃描股票數":len(selected),
        "核心_TOP1-50":int((selected["流動性排名"]<=50).sum()),
        "觀察_TOP51-100":int(((selected["流動性排名"]>50)&(selected["流動性排名"]<=100)).sum()),
        "擴充_TOP101-150":int(((selected["流動性排名"]>100)&(selected["流動性排名"]<=150)).sum()),
        "最新排名基準日":str(selected["排名基準完成日"].max()) if len(selected) else "",
    }])
    return selected.reset_index(drop=True),diag,errors



@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)
def build_current_market_breadth(top200_pool: pd.DataFrame):
    """
    V1.16.22 今日市場環境：
    使用今天以前已完成日K，針對目前流動性TOP200計算市場廣度。
    僅作風險提示，不直接過濾今日雷達訊號。
    """
    if top200_pool is None or top200_pool.empty:
        return pd.DataFrame(), ["TOP200股票池為空，無法計算市場廣度。"]

    symbols=top200_pool.loc[top200_pool["流動性排名"]<=200,"股票"].astype(str).tolist()
    if not symbols:
        return pd.DataFrame(), ["TOP200股票清單為空。"]

    dmap=download_daily_batches(symbols,period="1y",batch_size=25)
    today_tw=pd.Timestamp.now(tz="Asia/Taipei").date()
    rec=[]
    latest_dates=[]

    for s in symbols:
        d=dmap.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy()
            x.index=pd.to_datetime(x.index,errors="coerce")
            x=x[x.index.notna()].sort_index()
            x=x[pd.Index([pd.Timestamp(i).date() for i in x.index]) < today_tw]
            x=x.dropna(subset=["Close"]).copy()
            if len(x)<60:
                continue

            c=pd.to_numeric(x["Close"],errors="coerce")
            ma15=c.rolling(15,min_periods=15).mean()
            ma30=c.rolling(30,min_periods=30).mean()
            ma60=c.rolling(60,min_periods=60).mean()
            ret5=(c/c.shift(5)-1)*100

            rec.append({
                "股票":s,
                "Close":float(c.iloc[-1]) if pd.notna(c.iloc[-1]) else np.nan,
                "MA15":float(ma15.iloc[-1]) if pd.notna(ma15.iloc[-1]) else np.nan,
                "MA30":float(ma30.iloc[-1]) if pd.notna(ma30.iloc[-1]) else np.nan,
                "MA60":float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else np.nan,
                "RET5%":float(ret5.iloc[-1]) if pd.notna(ret5.iloc[-1]) else np.nan,
            })
            latest_dates.append(pd.Timestamp(x.index[-1]).date())
        except Exception:
            continue

    z=pd.DataFrame(rec)
    if z.empty:
        return pd.DataFrame(), ["TOP200市場廣度日K資料不足。"]

    v15=z["Close"].notna()&z["MA15"].notna()
    v30=z["Close"].notna()&z["MA30"].notna()
    v60=z["Close"].notna()&z["MA60"].notna()
    v5=z["RET5%"].notna()

    ma15_pct=float((z.loc[v15,"Close"]>z.loc[v15,"MA15"]).mean()*100) if v15.any() else np.nan
    ma30_pct=float((z.loc[v30,"Close"]>z.loc[v30,"MA30"]).mean()*100) if v30.any() else np.nan
    ma60_pct=float((z.loc[v60,"Close"]>z.loc[v60,"MA60"]).mean()*100) if v60.any() else np.nan
    up5_pct=float((z.loc[v5,"RET5%"]>0).mean()*100) if v5.any() else np.nan
    med5=float(z.loc[v5,"RET5%"].median()) if v5.any() else np.nan

    if pd.isna(ma60_pct):
        risk="⚪ 資料不足"
    elif ma60_pct>=65:
        risk="🔴 高檔風險"
    else:
        risk="🟢 正常"

    out=pd.DataFrame([{
        "市場狀態":risk,
        "基準完成日":str(max(latest_dates)) if latest_dates else "",
        "TOP200有效股票數":len(z),
        "站上MA15比例%":ma15_pct,
        "站上MA30比例%":ma30_pct,
        "站上MA60比例%":ma60_pct,
        "5日上漲家數比例%":up5_pct,
        "5日報酬中位數%":med5,
        "風險規則":"MA60廣度>=65%僅顯示高檔風險警示，不過濾訊號"
    }])
    return out, []


def build_formal_daily_radar(cost: CostConfig):
    """
    V1.16.22 正式今日雷達：
    - 全市場先排名到TOP200，用於市場廣度風險提示。
    - 60m正式掃描仍只掃TOP150：
        TOP1-100：保留全部核心KD訊號
        TOP101-150：只保留S級
    - MA60市場廣度>=65%目前只顯示風險警示，不作硬Gate。
    """
    pool200,pool_diag,errors=build_current_formal_radar_pool(max_rank=200)
    if pool200 is None or pool200.empty:
        return pd.DataFrame(),pd.DataFrame(),pool_diag,pd.DataFrame(),pd.DataFrame(),errors

    env,env_errors=build_current_market_breadth(pool200)
    errors=list(errors or [])+list(env_errors or [])

    pool=pool200[pool200["流動性排名"]<=150].copy().reset_index(drop=True)
    if not pool_diag.empty:
        pool_diag=pool_diag.copy()
        pool_diag["正式掃描股票數"]=len(pool)

    symbols=pool["股票"].tolist()
    radar,scan_diag=scan_latest_60m_radar(symbols,pool,period="60d",observe_days=5)
    if radar is None or radar.empty:
        return pd.DataFrame(),scan_diag,pool_diag,pool,env,errors

    rank=pd.to_numeric(radar["流動性排名"],errors="coerce")
    formal=(rank<=100)|((rank>100)&(rank<=150)&(radar["訊號等級"]=="S級"))
    radar["正式雷達納入"]=np.where(formal,"是","否")
    radar=radar[formal].copy()

    active=radar["目前狀態"].isin(["🟢 新訊號","🟡 觀察中"])
    radar=radar[active].copy()

    if env is not None and not env.empty:
        radar["市場狀態"]=env.iloc[0]["市場狀態"]
        radar["市場MA60廣度%"]=env.iloc[0]["站上MA60比例%"]
    else:
        radar["市場狀態"]="⚪ 資料不足"
        radar["市場MA60廣度%"]=np.nan

    layer_order={"核心_TOP1-50":0,"觀察_TOP51-100":1,"擴充_TOP101-150":2}
    grade_order={"S級":0,"A級":1,"B級":2}
    status_order={"🟢 新訊號":0,"🟡 觀察中":1}
    radar["_s"]=radar["目前狀態"].map(status_order).fillna(9)
    radar["_g"]=radar["訊號等級"].map(grade_order).fillna(9)
    radar["_l"]=radar["股票池層級"].map(layer_order).fillna(9)
    radar=radar.sort_values(["_s","_g","_l","流動性排名","訊號時間"],
                            ascending=[True,True,True,True,False]).drop(columns=["_s","_g","_l"])
    return radar.reset_index(drop=True),scan_diag,pool_diag,pool,env,errors


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








@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)
def build_top200_market_breadth_context():
    """
    V1.16.18：建立歷史TOP200的日K橫斷面市場廣度。
    只下載歷史TOP200曾出現過的股票聯集；每個基準日只用當日及以前資料。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(), elig, errors

    union=sorted(
        elig.loc[elig["流動性排名"]<=200,"股票"].dropna().astype(str).unique().tolist()
    )
    if not union:
        return pd.DataFrame(), elig, errors

    # V1.16.21：市場廣度指標需要MA60暖機。
    # 先前6mo在評估區最前端仍會有少量MA60缺值，因此改抓1y；
    # 正式交易評估區仍維持6mo暖機/最後3mo，不改策略樣本。
    dmap=download_daily_batches(union,period="1y",batch_size=25)
    per_symbol={}
    for s in union:
        d=dmap.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy()
            x.index=pd.to_datetime(x.index,errors="coerce")
            x=x[x.index.notna()].sort_index().dropna(subset=["Close"]).copy()
            c=pd.to_numeric(x["Close"],errors="coerce")
            x["MA15"]=c.rolling(15,min_periods=15).mean()
            x["MA30"]=c.rolling(30,min_periods=30).mean()
            x["MA60"]=c.rolling(60,min_periods=60).mean()
            x["RET5%"]=(c/c.shift(5)-1)*100
            x["日期"]=[pd.Timestamp(i).date() for i in x.index]
            per_symbol[s]=x[["日期","Close","MA15","MA30","MA60","RET5%"]].set_index("日期")
        except Exception:
            continue

    rows=[]
    for base_date,q in elig[elig["流動性排名"]<=200].groupby("基準完成日"):
        rec=[]
        for s in q["股票"].astype(str):
            x=per_symbol.get(s)
            if x is None or base_date not in x.index:
                continue
            r=x.loc[base_date]
            if isinstance(r,pd.DataFrame):
                r=r.iloc[-1]
            rec.append({
                "Close":pd.to_numeric(pd.Series([r.get("Close")]),errors="coerce").iloc[0],
                "MA15":pd.to_numeric(pd.Series([r.get("MA15")]),errors="coerce").iloc[0],
                "MA30":pd.to_numeric(pd.Series([r.get("MA30")]),errors="coerce").iloc[0],
                "MA60":pd.to_numeric(pd.Series([r.get("MA60")]),errors="coerce").iloc[0],
                "RET5%":pd.to_numeric(pd.Series([r.get("RET5%")]),errors="coerce").iloc[0],
            })
        if not rec:
            continue
        z=pd.DataFrame(rec)
        v15=z["Close"].notna()&z["MA15"].notna()
        v30=z["Close"].notna()&z["MA30"].notna()
        v60=z["Close"].notna()&z["MA60"].notna()
        v5=z["RET5%"].notna()
        rows.append({
            "基準完成日":base_date,
            "TOP200有效股票數":len(z),
            "站上MA15比例%":float((z.loc[v15,"Close"]>z.loc[v15,"MA15"]).mean()*100) if v15.any() else np.nan,
            "站上MA30比例%":float((z.loc[v30,"Close"]>z.loc[v30,"MA30"]).mean()*100) if v30.any() else np.nan,
            "站上MA60比例%":float((z.loc[v60,"Close"]>z.loc[v60,"MA60"]).mean()*100) if v60.any() else np.nan,
            "5日上漲家數比例%":float((z.loc[v5,"RET5%"]>0).mean()*100) if v5.any() else np.nan,
            "5日報酬中位數%":float(z.loc[v5,"RET5%"].median()) if v5.any() else np.nan,
        })
    return pd.DataFrame(rows), elig, errors















def validate_long_horizon_concentration(cost: CostConfig):
    """
    V1.16.33 長期報酬分布 / 集中度健診：
    沿用V1.16.32完全相同的1y資料、最後9mo評估、正式架構C與5日持有。
    不新增任何策略條件，只檢查長期正期望是否被少數月份 / 少數股票 / 極端大賺單支撐。

    輸出：
      1) 月度穩定度
      2) 股票貢獻集中度
      3) 移除Top5 / Top10 / Top20正貢獻股票後，核心績效是否仍為正
      4) Mean / Median / 5% trimmed mean / P10 / P90
    """
    _,_,t,errors=validate_long_horizon_robustness(cost)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    x=t.copy()
    x["淨報酬%"]=pd.to_numeric(x["淨報酬%"],errors="coerce")
    x=x[x["淨報酬%"].notna()].copy()
    x["訊號月份"]=pd.to_datetime(x["訊號時間_台北"],errors="coerce").dt.strftime("%Y-%m")

    # ---------------- 月度 ----------------
    month_rows=[]
    for month,g in x.groupby("訊號月份",sort=True):
        arr=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna().sort_values()
        n=len(arr)
        trim_n=int(np.floor(n*0.05))
        trimmed=arr.iloc[trim_n:n-trim_n] if n>2*trim_n and trim_n>0 else arr
        m=aggregate_trade_metrics(g)
        month_rows.append({
            "月份":month,
            "交易數":len(g),
            "股票數":int(g["股票"].nunique()),
            "平均淨報酬%":float(arr.mean()) if n else np.nan,
            "中位數淨報酬%":float(arr.median()) if n else np.nan,
            "5%TrimmedMean%":float(trimmed.mean()) if len(trimmed) else np.nan,
            "P10%":float(arr.quantile(0.10)) if n else np.nan,
            "P90%":float(arr.quantile(0.90)) if n else np.nan,
            "勝率%":m.get("整體交易勝率",np.nan),
            "PF":m.get("整體PF",np.nan),
        })
    monthly=pd.DataFrame(month_rows)

    # ---------------- 股票貢獻 ----------------
    stock_rows=[]
    for stock,g in x.groupby("股票"):
        arr=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna()
        stock_rows.append({
            "股票":stock,
            "交易數":len(g),
            "總淨報酬貢獻":float(arr.sum()),
            "平均淨報酬%":float(arr.mean()) if len(arr) else np.nan,
            "中位數淨報酬%":float(arr.median()) if len(arr) else np.nan,
            "勝率%":float((arr>0).mean()*100) if len(arr) else np.nan,
        })
    stocks=pd.DataFrame(stock_rows).sort_values(
        ["總淨報酬貢獻","交易數"],ascending=[False,False]
    ).reset_index(drop=True)

    total_sum=float(x["淨報酬%"].sum())
    stocks["累積總淨報酬貢獻"]=stocks["總淨報酬貢獻"].cumsum()
    stocks["累積貢獻占總淨報酬%"]=(
        stocks["累積總淨報酬貢獻"]/total_sum*100 if total_sum!=0 else np.nan
    )

    # ---------------- 影響力移除 ----------------
    positive_rank=stocks[stocks["總淨報酬貢獻"]>0]["股票"].tolist()
    influence_rows=[]

    def _dist_metrics(g):
        arr=pd.to_numeric(g["淨報酬%"],errors="coerce").dropna().sort_values()
        n=len(arr)
        trim_n=int(np.floor(n*0.05))
        trimmed=arr.iloc[trim_n:n-trim_n] if n>2*trim_n and trim_n>0 else arr
        m=aggregate_trade_metrics(g)
        return {
            "交易數":len(g),
            "股票數":int(g["股票"].nunique()) if len(g) else 0,
            "平均淨報酬%":float(arr.mean()) if n else np.nan,
            "中位數淨報酬%":float(arr.median()) if n else np.nan,
            "5%TrimmedMean%":float(trimmed.mean()) if len(trimmed) else np.nan,
            "勝率%":m.get("整體交易勝率",np.nan),
            "PF":m.get("整體PF",np.nan),
        }

    influence_rows.append({"情境":"基準_全部",**_dist_metrics(x)})
    for k in [5,10,20]:
        remove=set(positive_rank[:k])
        g=x[~x["股票"].isin(remove)].copy()
        influence_rows.append({
            "情境":f"移除Top{k}正貢獻股票",
            **_dist_metrics(g)
        })
    influence=pd.DataFrame(influence_rows)

    # ---------------- 全體分布摘要 ----------------
    arr=x["淨報酬%"].dropna().sort_values()
    n=len(arr)
    trim_n=int(np.floor(n*0.05))
    trimmed=arr.iloc[trim_n:n-trim_n] if n>2*trim_n and trim_n>0 else arr

    stock_avg=stocks["平均淨報酬%"].dropna()
    distribution=pd.DataFrame([{
        "交易數":len(x),
        "股票數":int(x["股票"].nunique()),
        "平均淨報酬%":float(arr.mean()) if n else np.nan,
        "中位數淨報酬%":float(arr.median()) if n else np.nan,
        "5%TrimmedMean%":float(trimmed.mean()) if len(trimmed) else np.nan,
        "P10%":float(arr.quantile(0.10)) if n else np.nan,
        "P90%":float(arr.quantile(0.90)) if n else np.nan,
        "股票平均報酬為正占比%":float((stock_avg>0).mean()*100) if len(stock_avg) else np.nan,
        "Top5股票貢獻占總淨報酬%":float(stocks.head(5)["總淨報酬貢獻"].sum()/total_sum*100) if total_sum!=0 else np.nan,
        "Top10股票貢獻占總淨報酬%":float(stocks.head(10)["總淨報酬貢獻"].sum()/total_sum*100) if total_sum!=0 else np.nan,
        "Top20股票貢獻占總淨報酬%":float(stocks.head(20)["總淨報酬貢獻"].sum()/total_sum*100) if total_sum!=0 else np.nan,
        "正平均月份數":int((monthly["平均淨報酬%"]>0).sum()) if not monthly.empty else 0,
        "總月份數":len(monthly),
    }])

    return monthly,stocks,influence,distribution,errors
def validate_long_horizon_robustness(cost: CostConfig):
    """
    V1.16.32 長期穩健度驗證：
    停止繼續微調出場參數，回頭驗證正式核心策略本身在更長時間是否成立。

    固定：
      - 官方全市場母池
      - 歷史Point-in-Time流動性排名
      - 正式架構C：TOP1-100全部 + TOP101-150僅S級
      - 60m KD黃金交叉 + K<30
      - non-overlap
      - 持有5日
      - 交易成本不變

    資料：
      - 日K資格：12mo
      - 60m資料：1y
      - 前3mo作暖機，正式評估最後9mo
      - 同一時間切割：前60% / 後40%
      - 再切6個等長時間區塊看穩定度

    注意：
      這版不比較新Gate、不比較新停損，只回答「核心策略長期是否穩健」。
    """
    elig,_,errors=build_fullmarket_walkforward_eligibility(
        lookback_months=12, top_n=100
    )
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(
        elig.loc[elig["流動性排名"]<=200,"股票"]
        .dropna().astype(str).unique().tolist()
    )
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors+["長期TOP200聯集為空"]

    _,_,trades=run_oos_60m_5d(
        union,cost,"1y",
        allow_overlap=False,
        train_ratio=0.60,
        evaluation_months=9
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors+["1y 60m交易資料為空"]

    t=trades.copy()
    sig=_as_taipei_series(t["訊號時間"])
    t["訊號時間_台北"]=sig
    t["訊號日期"]=sig.dt.date

    # Point-in-time 流動性排名：只能使用訊號日前已完成日K。
    ranks=[]
    base_dates=[]
    for _,r in t.iterrows():
        q=elig[
            (elig["股票"]==r["股票"]) &
            (elig["基準完成日"]<r["訊號日期"])
        ]
        if q.empty:
            ranks.append(np.nan)
            base_dates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            ranks.append(float(z["流動性排名"]))
            base_dates.append(z["基準完成日"])

    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=base_dates
    t=t[t["WF流動性排名"].notna()].copy()

    # 正式S級定義不變。
    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60s=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["S級"]=(vr>=1.5)&(ma60s<=0)

    rnk=pd.to_numeric(t["WF流動性排名"],errors="coerce")
    formal=(rnk<=100)|((rnk>100)&(rnk<=150)&t["S級"])
    t=t[formal].copy().sort_values("訊號時間_台北").reset_index(drop=True)

    if t.empty:
        return pd.DataFrame(),pd.DataFrame(),t,errors+["正式架構C長期交易為空"]

    # 全部 / 樣本內 / 樣本外
    summary_rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        g=t if sample=="全部" else t[t["樣本"]==sample]
        m=aggregate_trade_metrics(g)
        summary_rows.append({
            "樣本":sample,
            "交易數":len(g),
            "股票數":int(g["股票"].nunique()) if len(g) else 0,
            "起始訊號":str(g["訊號時間_台北"].min()) if len(g) else "",
            "結束訊號":str(g["訊號時間_台北"].max()) if len(g) else "",
            **m
        })
    summary=pd.DataFrame(summary_rows)

    # 6個等長時間區塊：避免只看單一OOS平均。
    ts=t["訊號時間_台北"]
    edges=pd.date_range(ts.min(),ts.max(),periods=7)
    labels=[f"區塊{i}" for i in range(1,7)]
    t["長期時間區塊"]=pd.cut(
        ts,bins=edges,labels=labels,include_lowest=True,right=True
    )

    block_rows=[]
    for i,block in enumerate(labels):
        g=t[t["長期時間區塊"]==block]
        m=aggregate_trade_metrics(g)
        block_rows.append({
            "時間區塊":block,
            "起始":str(edges[i]),
            "結束":str(edges[i+1]),
            "交易數":len(g),
            "股票數":int(g["股票"].nunique()) if len(g) else 0,
            **m
        })
    blocks=pd.DataFrame(block_rows)

    return summary,blocks,t,errors
def validate_delayed_timestop_candidates(cost: CostConfig):
    """
    V1.16.31 延遲Time-Stop A/B：
    沿用V1.16.30完全相同進場母體，只在第2/3日收盤後判斷是否提早結束。

    比較：
      A_5日基準
      B_第2日<=-3%出場
      C_第2日<=-5%出場
      D_第3日<=-3%出場
      E_第3日<=-5%出場
      F_第2日<=-3%否則第3日<=-3%
      G_第2日<=-5%否則第3日<=-5%

    注意：
      - 所有條件都只用「已完成交易日收盤」。
      - 不使用盤中Low，因此避開固定停損過度洗出的問題。
      - 不新增訊號、不改股票池、不改S/A/B、不改5日基準母體。
    """
    _,_,detail,errors=validate_early_path_diagnostics(cost)
    if detail is None or detail.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),detail,errors

    x=detail.copy()

    policies=[
        "A_5日基準",
        "B_第2日<=-3%出場",
        "C_第2日<=-5%出場",
        "D_第3日<=-3%出場",
        "E_第3日<=-5%出場",
        "F_第2日<=-3%否則第3日<=-3%",
        "G_第2日<=-5%否則第3日<=-5%",
    ]

    rows=[]
    for _,r in x.iterrows():
        d2=float(r["第2日淨報酬%"])
        d3=float(r["第3日淨報酬%"])
        d5=float(r["第5日淨報酬%"])

        decisions={
            "A_5日基準":(d5,"第5日"),
            "B_第2日<=-3%出場":(d2,"第2日") if d2<=-3 else (d5,"第5日"),
            "C_第2日<=-5%出場":(d2,"第2日") if d2<=-5 else (d5,"第5日"),
            "D_第3日<=-3%出場":(d3,"第3日") if d3<=-3 else (d5,"第5日"),
            "E_第3日<=-5%出場":(d3,"第3日") if d3<=-5 else (d5,"第5日"),
            "F_第2日<=-3%否則第3日<=-3%":(
                (d2,"第2日") if d2<=-3 else ((d3,"第3日") if d3<=-3 else (d5,"第5日"))
            ),
            "G_第2日<=-5%否則第3日<=-5%":(
                (d2,"第2日") if d2<=-5 else ((d3,"第3日") if d3<=-5 else (d5,"第5日"))
            ),
        }

        for name,(ret,exit_day) in decisions.items():
            z=r.to_dict()
            z["TimeStop方案"]=name
            z["比較淨報酬%"]=ret
            z["比較出場日"]=exit_day
            z["提早出場"]="否" if exit_day=="第5日" else "是"
            rows.append(z)

    out=pd.DataFrame(rows)

    def _risk(g):
        rr=pd.to_numeric(g["比較淨報酬%"],errors="coerce").dropna()
        if rr.empty:
            return {"報酬P5%":np.nan,"報酬P10%":np.nan,"CVaR10%":np.nan,
                    "最差單筆%":np.nan,"提早出場率%":np.nan}
        p10=float(rr.quantile(0.10))
        return {
            "報酬P5%":float(rr.quantile(0.05)),
            "報酬P10%":p10,
            "CVaR10%":float(rr[rr<=p10].mean()),
            "最差單筆%":float(rr.min()),
            "提早出場率%":float((g["提早出場"]=="是").mean()*100),
        }

    summary_rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=out if sample=="全部" else out[out["樣本"]==sample]
        for name in policies:
            g=xs[xs["TimeStop方案"]==name].copy()
            tmp=g.copy()
            tmp["淨報酬%"]=pd.to_numeric(g["比較淨報酬%"],errors="coerce")
            m=aggregate_trade_metrics(tmp)
            summary_rows.append({
                "樣本":sample,"TimeStop方案":name,
                "交易數":len(g),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m,**_risk(g)
            })
    summary=pd.DataFrame(summary_rows)

    block_rows=[]
    for block in ["第1段","第2段","第3段","第4段"]:
        xb=out[out["時間段"]==block]
        for name in policies:
            g=xb[xb["TimeStop方案"]==name].copy()
            tmp=g.copy()
            tmp["淨報酬%"]=pd.to_numeric(g["比較淨報酬%"],errors="coerce")
            m=aggregate_trade_metrics(tmp)
            block_rows.append({
                "時間段":block,"TimeStop方案":name,
                "交易數":len(g),
                **m,**_risk(g)
            })
    blocks=pd.DataFrame(block_rows)

    # Paired差異
    key_cols=["股票","訊號時間_台北","進場時間"]
    ret=out.pivot_table(index=key_cols,columns="TimeStop方案",values="比較淨報酬%",aggfunc="first")
    paired=[]
    base_col="A_5日基準"
    for name in policies[1:]:
        delta=pd.to_numeric(ret[name],errors="coerce")-pd.to_numeric(ret[base_col],errors="coerce")
        paired.append({
            "TimeStop方案":name,
            "平均Δvs5日%":float(delta.mean()),
            "中位數Δvs5日%":float(delta.median()),
            "改善率%":float((delta>0).mean()*100),
            "惡化率%":float((delta<0).mean()*100),
            "不變率%":float((delta==0).mean()*100),
            "ΔP10%":float(delta.quantile(0.10)),
            "ΔP90%":float(delta.quantile(0.90)),
            "單筆改善>=5%占比":float((delta>=5).mean()*100),
            "單筆惡化<=-5%占比":float((delta<=-5).mean()*100),
        })
    paired_df=pd.DataFrame(paired)

    return summary,blocks,paired_df,out,errors
def validate_early_path_diagnostics(cost: CostConfig):
    """
    V1.16.30 早期路徑健診：
    固定正式架構C與5日基準，不新增停損規則。
    目的：找出「最後會輸」的交易，在第1/2/3日收盤時是否已有可辨識特徵。

    觀察：
      - 第1 / 2 / 3日收盤相對進場報酬
      - 最終5日報酬
      - 固定分桶：<=-5、-5~-3、-3~0、0~3、>3
      - 各分桶後續5日平均、PF、勝率、最後翻正率
      - 特別比較第1段 vs 後三段

    這版只診斷，不直接新增 time-stop。
    """
    _,_,_,base,errors=validate_breadth_transition(cost)
    if base is None or base.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    symbols=sorted(base["股票"].dropna().astype(str).unique().tolist())
    raw=download_intraday_batch(symbols,"60m","6mo")

    data_map={}
    idx_map={}
    for s in symbols:
        d=raw.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy().sort_index()
            tw=_as_taipei_series(pd.Series(x.index))
            data_map[s]=x
            idx_map[s]={pd.Timestamp(t):i for i,t in enumerate(tw) if pd.notna(t)}
        except Exception:
            continue

    rows=[]
    for _,r in base.iterrows():
        s=str(r["股票"])
        d=data_map.get(s)
        mp=idx_map.get(s)
        if d is None or mp is None:
            continue

        et=_as_taipei_series(pd.Series([r["進場時間"]])).iloc[0]
        if pd.isna(et):
            continue
        entry_i=mp.get(pd.Timestamp(et))
        if entry_i is None or entry_i>=len(d):
            continue

        entry=float(d["Open"].iloc[entry_i])
        if not np.isfinite(entry) or entry<=0:
            continue

        z=r.to_dict()
        ok=True
        for day in [1,2,3,5]:
            ei=find_exit_index(d,entry_i,f"{day}日","60m")
            if ei is None or ei<=entry_i or ei>=len(d):
                ok=False
                break
            px=float(d["Close"].iloc[ei])
            if not np.isfinite(px):
                ok=False
                break
            gross=(px/entry-1)*100
            net=gross-cost.roundtrip_cost_pct(daytrade=False)
            z[f"第{day}日淨報酬%"]=net
        if not ok:
            continue

        z["最終5日結果"]="獲利" if z["第5日淨報酬%"]>0 else "虧損"
        rows.append(z)

    detail=pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(),pd.DataFrame(),detail,errors

    bins=[-np.inf,-5,-3,0,3,np.inf]
    names=["<=-5%","-5~-3%","-3~0%","0~3%",">3%"]

    bucket_rows=[]
    for day in [1,2,3]:
        col=f"第{day}日淨報酬%"
        bucket=pd.cut(pd.to_numeric(detail[col],errors="coerce"),
                      bins=bins,labels=names,include_lowest=True,right=True)
        for name in names:
            g=detail[bucket==name].copy()
            if g.empty:
                bucket_rows.append({
                    "觀察日":f"第{day}日","早期報酬分桶":name,
                    "交易數":0,"占比%":np.nan,"最終翻正率%":np.nan,
                    "最終5日平均%":np.nan,"最終5日勝率%":np.nan,"最終5日PF":np.nan
                })
                continue
            tmp=g.copy()
            tmp["淨報酬%"]=pd.to_numeric(g["第5日淨報酬%"],errors="coerce")
            m=aggregate_trade_metrics(tmp)
            bucket_rows.append({
                "觀察日":f"第{day}日",
                "早期報酬分桶":name,
                "交易數":len(g),
                "占比%":float(len(g)/len(detail)*100),
                "最終翻正率%":float((pd.to_numeric(g["第5日淨報酬%"],errors="coerce")>0).mean()*100),
                "最終5日平均%":float(pd.to_numeric(g["第5日淨報酬%"],errors="coerce").mean()),
                "最終5日勝率%":m.get("整體交易勝率",np.nan),
                "最終5日PF":m.get("整體PF",np.nan),
            })
    buckets=pd.DataFrame(bucket_rows)

    # 第1段 vs 後三段：看早期負報酬是否在失效段更具預測性。
    compare_rows=[]
    groups=[
        ("第1段",detail[detail["時間段"]=="第1段"]),
        ("第2~4段",detail[detail["時間段"].isin(["第2段","第3段","第4段"])]),
        ("全部",detail),
    ]
    for gname,g0 in groups:
        for day in [1,2,3]:
            x=pd.to_numeric(g0[f"第{day}日淨報酬%"],errors="coerce")
            for threshold in [-3,-5]:
                g=g0[x<=threshold].copy()
                if g.empty:
                    compare_rows.append({
                        "區段":gname,"觀察日":f"第{day}日","條件":f"<={threshold}%",
                        "交易數":0,"占區段比例%":np.nan,"最終翻正率%":np.nan,
                        "最終5日平均%":np.nan,"最終5日PF":np.nan
                    })
                    continue
                tmp=g.copy()
                tmp["淨報酬%"]=pd.to_numeric(g["第5日淨報酬%"],errors="coerce")
                m=aggregate_trade_metrics(tmp)
                compare_rows.append({
                    "區段":gname,
                    "觀察日":f"第{day}日",
                    "條件":f"<={threshold}%",
                    "交易數":len(g),
                    "占區段比例%":float(len(g)/len(g0)*100) if len(g0) else np.nan,
                    "最終翻正率%":float((pd.to_numeric(g["第5日淨報酬%"],errors="coerce")>0).mean()*100),
                    "最終5日平均%":float(pd.to_numeric(g["第5日淨報酬%"],errors="coerce").mean()),
                    "最終5日PF":m.get("整體PF",np.nan),
                })
    compare=pd.DataFrame(compare_rows)

    return buckets,compare,detail,errors
def validate_stoploss_candidates(cost: CostConfig):
    """
    V1.16.29 固定停損A/B：
    固定正式架構C、固定同一批進場、固定最多持有5日，
    只比較固定虧損保護是否能改善「一路偏弱」交易的左尾。

    比較：
      A_5日基準
      B_停損-5%
      C_停損-7.5%
      D_停損-10%

    執行假設：
      - 進場在訊號後下一根60m K的 Open。
      - 停損從進場後立即有效；同一進場K若 Low <= 停損價，可在停損價出場。
      - 後續K若 Open已低於停損價，以實際Open出場（gap conservative）。
      - 否則 Low<=停損價則用停損價出場。
      - 若未停損，維持第5個交易日最後一根Close出場。
      - 不加獲利保護，不改股票池、不改進場訊號。
    """
    _,_,_,base,errors=validate_breadth_transition(cost)
    if base is None or base.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    symbols=sorted(base["股票"].dropna().astype(str).unique().tolist())
    raw=download_intraday_batch(symbols,"60m","6mo")

    data_map={}
    idx_map={}
    tw_index_map={}
    for s in symbols:
        d=raw.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy().sort_index()
            tw=_as_taipei_series(pd.Series(x.index))
            data_map[s]=x
            idx_map[s]={pd.Timestamp(t):i for i,t in enumerate(tw) if pd.notna(t)}
            tw_index_map[s]=list(tw)
        except Exception:
            continue

    policies=[
        ("A_5日基準",None),
        ("B_停損-5%",-0.05),
        ("C_停損-7.5%",-0.075),
        ("D_停損-10%",-0.10),
    ]
    rows=[]

    for _,r in base.iterrows():
        s=str(r["股票"])
        d=data_map.get(s)
        mp=idx_map.get(s)
        twidx=tw_index_map.get(s)
        if d is None or mp is None or twidx is None:
            continue

        et=_as_taipei_series(pd.Series([r["進場時間"]])).iloc[0]
        if pd.isna(et):
            continue
        entry_i=mp.get(pd.Timestamp(et))
        if entry_i is None or entry_i>=len(d):
            continue

        entry=float(d["Open"].iloc[entry_i])
        if not np.isfinite(entry) or entry<=0:
            continue

        exit5=find_exit_index(d,entry_i,"5日","60m")
        if exit5 is None:
            continue

        for name,stop_pct in policies:
            exit_i=None
            exit_price=None
            exit_reason=""

            if stop_pct is None:
                exit_i=exit5
                exit_price=float(d["Close"].iloc[exit_i])
                exit_reason="5日到期"
            else:
                stop_price=entry*(1.0+float(stop_pct))

                for j in range(entry_i,exit5+1):
                    op=float(d["Open"].iloc[j])
                    lo=float(d["Low"].iloc[j])

                    # entry bar: already entered at its Open, so only Low can trigger.
                    if j==entry_i:
                        if np.isfinite(lo) and lo<=stop_price:
                            exit_i=j
                            exit_price=stop_price
                            exit_reason=f"停損_{stop_pct*100:.1f}%"
                            break
                        continue

                    # later bars: gap below stop uses actual opening price.
                    if np.isfinite(op) and op<=stop_price:
                        exit_i=j
                        exit_price=op
                        exit_reason=f"gap停損_{stop_pct*100:.1f}%"
                        break
                    if np.isfinite(lo) and lo<=stop_price:
                        exit_i=j
                        exit_price=stop_price
                        exit_reason=f"停損_{stop_pct*100:.1f}%"
                        break

                if exit_i is None:
                    exit_i=exit5
                    exit_price=float(d["Close"].iloc[exit_i])
                    exit_reason="5日到期"

            if exit_i is None or not np.isfinite(exit_price):
                continue

            gross=(exit_price/entry-1)*100
            cost_pct=cost.roundtrip_cost_pct(daytrade=False)
            net=gross-cost_pct
            path=d.iloc[entry_i:exit_i+1]

            z=r.to_dict()
            z["停損方案"]=name
            z["停損門檻%"]=np.nan if stop_pct is None else stop_pct*100
            z["比較出場時間"]=twidx[exit_i] if exit_i<len(twidx) else d.index[exit_i]
            z["比較出場價"]=exit_price
            z["比較淨報酬%"]=net
            z["比較MFE%"]=(float(path["High"].max())/entry-1)*100
            z["比較MAE%"]=(float(path["Low"].min())/entry-1)*100
            z["比較出場原因"]=exit_reason
            z["持有60m棒數"]=int(exit_i-entry_i+1)
            rows.append(z)

    detail=pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),detail,errors

    # 全部方案共同母體。
    key_cols=["股票","訊號時間_台北","進場時間"]
    counts=detail.groupby(key_cols)["停損方案"].nunique().reset_index(name="_n")
    common=counts[counts["_n"]==len(policies)][key_cols]
    detail=detail.merge(common,on=key_cols,how="inner")

    def _risk_metrics(g):
        x=pd.to_numeric(g["比較淨報酬%"],errors="coerce").dropna()
        if x.empty:
            return {
                "平均淨報酬%":np.nan,"報酬P5%":np.nan,"報酬P10%":np.nan,
                "CVaR10%":np.nan,"最差單筆%":np.nan,"虧損中位數%":np.nan,
                "平均持有60m棒數":np.nan,"停損出場率%":np.nan
            }
        p10=float(x.quantile(0.10))
        worst10=x[x<=p10]
        losses=x[x<0]
        stopped=g["比較出場原因"].astype(str).str.contains("停損")
        return {
            "平均淨報酬%":float(x.mean()),
            "報酬P5%":float(x.quantile(0.05)),
            "報酬P10%":p10,
            "CVaR10%":float(worst10.mean()) if len(worst10) else np.nan,
            "最差單筆%":float(x.min()),
            "虧損中位數%":float(losses.median()) if len(losses) else np.nan,
            "平均持有60m棒數":float(pd.to_numeric(g["持有60m棒數"],errors="coerce").mean()),
            "停損出場率%":float(stopped.mean()*100),
        }

    names=[p[0] for p in policies]
    summary_rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=detail if sample=="全部" else detail[detail["樣本"]==sample]
        for name in names:
            g=xs[xs["停損方案"]==name].copy()
            tmp=g.copy()
            tmp["淨報酬%"]=pd.to_numeric(g["比較淨報酬%"],errors="coerce")
            m=aggregate_trade_metrics(tmp)
            rm=_risk_metrics(g)
            summary_rows.append({
                "樣本":sample,"停損方案":name,
                "交易數":len(g),"股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m,**{k:v for k,v in rm.items() if k not in ["平均淨報酬%"]}
            })
    summary=pd.DataFrame(summary_rows)

    block_rows=[]
    for block in ["第1段","第2段","第3段","第4段"]:
        xb=detail[detail["時間段"]==block]
        for name in names:
            g=xb[xb["停損方案"]==name].copy()
            tmp=g.copy()
            tmp["淨報酬%"]=pd.to_numeric(g["比較淨報酬%"],errors="coerce")
            m=aggregate_trade_metrics(tmp)
            rm=_risk_metrics(g)
            block_rows.append({
                "時間段":block,"停損方案":name,
                "交易數":len(g),"股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m,**{k:v for k,v in rm.items() if k not in ["平均淨報酬%"]}
            })
    blocks=pd.DataFrame(block_rows)

    # paired delta：直接比較每筆停損方案 vs 5日基準。
    ret=detail.pivot_table(index=key_cols,columns="停損方案",values="比較淨報酬%",aggfunc="first")
    paired=[]
    base_col="A_5日基準"
    for name in names[1:]:
        delta=pd.to_numeric(ret[name],errors="coerce")-pd.to_numeric(ret[base_col],errors="coerce")
        paired.append({
            "停損方案":name,
            "平均Δvs5日%":float(delta.mean()),
            "中位數Δvs5日%":float(delta.median()),
            "改善率%":float((delta>0).mean()*100),
            "惡化率%":float((delta<0).mean()*100),
            "不變率%":float((delta==0).mean()*100),
            "ΔP10%":float(delta.quantile(0.10)),
            "ΔP90%":float(delta.quantile(0.90)),
            "單筆改善>=5%占比":float((delta>=5).mean()*100),
            "單筆惡化<=-5%占比":float((delta<=-5).mean()*100),
        })
    paired_df=pd.DataFrame(paired)

    return summary,blocks,paired_df,detail,errors
def validate_profit_protection_risk_benefit(cost: CostConfig):
    """
    V1.16.28 獲利保護風險效益健診：
    沿用V1.16.27同一批進場與9組敏感度結果，
    不再找最高PF，而是檢查每筆交易相對5日基準的改善/犧牲分布。
    """
    summary,blocks,detail,errors=validate_profit_protection_sensitivity(cost)
    if detail is None or detail.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    key_cols=["股票","訊號時間_台北","進場時間"]
    keep_cols=key_cols+["樣本","時間段","參數組合","比較淨報酬%","比較出場原因","保護曾啟動"]
    d=detail[keep_cols].copy()
    ret=d.pivot_table(index=key_cols,columns="參數組合",values="比較淨報酬%",aggfunc="first")
    meta=d.drop_duplicates(key_cols)[key_cols+["樣本","時間段"]].set_index(key_cols)

    base_col="A_5日基準"
    if base_col not in ret.columns:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors+["找不到5日基準欄位"]

    candidates=[c for c in ret.columns if c!=base_col]

    def _tail_metrics(x):
        x=pd.to_numeric(x,errors="coerce").dropna()
        if x.empty:
            return {"報酬P5%":np.nan,"報酬P10%":np.nan,"CVaR10%":np.nan,
                    "最差單筆%":np.nan,"虧損中位數%":np.nan}
        p10=float(x.quantile(0.10))
        worst10=x[x<=p10]
        losses=x[x<0]
        return {
            "報酬P5%":float(x.quantile(0.05)),
            "報酬P10%":p10,
            "CVaR10%":float(worst10.mean()) if len(worst10) else np.nan,
            "最差單筆%":float(x.min()),
            "虧損中位數%":float(losses.median()) if len(losses) else np.nan,
        }

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        idx=ret.index if sample=="全部" else meta[meta["樣本"]==sample].index
        rsub=ret.loc[ret.index.intersection(idx)]
        for cand in [base_col]+candidates:
            x=pd.to_numeric(rsub[cand],errors="coerce")
            delta=pd.Series(0.0,index=x.index) if cand==base_col else x-pd.to_numeric(rsub[base_col],errors="coerce")
            rows.append({
                "樣本":sample,
                "參數組合":cand,
                "交易數":int(x.notna().sum()),
                "平均淨報酬%":float(x.mean()) if x.notna().any() else np.nan,
                "平均Δvs5日%":float(delta.mean()) if delta.notna().any() else np.nan,
                "中位數Δvs5日%":float(delta.median()) if delta.notna().any() else np.nan,
                "改善率%":float((delta>0).mean()*100) if cand!=base_col else 0.0,
                "惡化率%":float((delta<0).mean()*100) if cand!=base_col else 0.0,
                "不變率%":float((delta==0).mean()*100),
                "ΔP10%":float(delta.quantile(0.10)) if delta.notna().any() else np.nan,
                "ΔP90%":float(delta.quantile(0.90)) if delta.notna().any() else np.nan,
                "單筆改善>=5%占比":float((delta>=5).mean()*100) if cand!=base_col else 0.0,
                "單筆惡化<=-5%占比":float((delta<=-5).mean()*100) if cand!=base_col else 0.0,
                **_tail_metrics(x)
            })
    risk_summary=pd.DataFrame(rows)

    block_rows=[]
    for block in ["第1段","第2段","第3段","第4段"]:
        idx=meta[meta["時間段"]==block].index
        rsub=ret.loc[ret.index.intersection(idx)]
        for cand in candidates:
            x=pd.to_numeric(rsub[cand],errors="coerce")
            b=pd.to_numeric(rsub[base_col],errors="coerce")
            delta=x-b
            block_rows.append({
                "時間段":block,
                "參數組合":cand,
                "交易數":int(delta.notna().sum()),
                "平均Δvs5日%":float(delta.mean()) if delta.notna().any() else np.nan,
                "中位數Δvs5日%":float(delta.median()) if delta.notna().any() else np.nan,
                "改善率%":float((delta>0).mean()*100) if delta.notna().any() else np.nan,
                "惡化率%":float((delta<0).mean()*100) if delta.notna().any() else np.nan,
                "ΔP10%":float(delta.quantile(0.10)) if delta.notna().any() else np.nan,
                "ΔP90%":float(delta.quantile(0.90)) if delta.notna().any() else np.nan,
                "單筆改善>=5%占比":float((delta>=5).mean()*100) if delta.notna().any() else np.nan,
                "單筆惡化<=-5%占比":float((delta<=-5).mean()*100) if delta.notna().any() else np.nan,
            })
    block_pair=pd.DataFrame(block_rows)

    focus=["T4_P2","T5_P1","T5_P2","T5_P3","T6_P2"]
    compare_rows=[]
    for cand in focus:
        if cand not in ret.columns:
            continue
        delta=pd.to_numeric(ret[cand],errors="coerce")-pd.to_numeric(ret[base_col],errors="coerce")
        compare_rows.append({
            "參數組合":cand,
            "全體平均Δ%":float(delta.mean()),
            "全體改善率%":float((delta>0).mean()*100),
            "全體惡化率%":float((delta<0).mean()*100),
            "全體不變率%":float((delta==0).mean()*100),
            "全體ΔP10%":float(delta.quantile(0.10)),
            "全體ΔP90%":float(delta.quantile(0.90)),
            "全體改善>=5%占比":float((delta>=5).mean()*100),
            "全體惡化<=-5%占比":float((delta<=-5).mean()*100),
        })
    focus_df=pd.DataFrame(compare_rows)
    return risk_summary,block_pair,focus_df,errors
def validate_profit_protection_sensitivity(cost: CostConfig):
    """
    V1.16.27 獲利保護敏感度驗證：
    固定正式架構C、固定同一批進場，只測附近參數是否穩健，
    不以單一最高PF選點。

    觸發門檻：+4 / +5 / +6
    保護位置：+1 / +2 / +3
    另保留5日基準作對照。

    執行假設同V1.16.26：
      - 某根已完成60m K的 High 首次 >= 觸發價，視為啟動。
      - 保護從下一根60m K才生效。
      - 若下一根開盤已跌破保護價，用開盤價出場。
      - 否則若 Low <= 保護價，用保護價出場。
      - 未觸發則第5日最後收盤出場。
    """
    _,_,_,base,errors=validate_breadth_transition(cost)
    if base is None or base.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    symbols=sorted(base["股票"].dropna().astype(str).unique().tolist())
    raw=download_intraday_batch(symbols,"60m","6mo")

    data_map={}
    idx_map={}
    tw_index_map={}
    for s in symbols:
        d=raw.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy().sort_index()
            tw=_as_taipei_series(pd.Series(x.index))
            data_map[s]=x
            idx_map[s]={pd.Timestamp(t):i for i,t in enumerate(tw) if pd.notna(t)}
            tw_index_map[s]=list(tw)
        except Exception:
            continue

    configs=[("A_5日基準",None,None)]
    for trigger in [0.04,0.05,0.06]:
        for protect in [0.01,0.02,0.03]:
            configs.append((f"T{int(trigger*100)}_P{int(protect*100)}",trigger,protect))

    rows=[]
    for _,r in base.iterrows():
        s=str(r["股票"])
        d=data_map.get(s)
        mp=idx_map.get(s)
        twidx=tw_index_map.get(s)
        if d is None or mp is None or twidx is None:
            continue

        et=_as_taipei_series(pd.Series([r["進場時間"]])).iloc[0]
        if pd.isna(et):
            continue
        entry_i=mp.get(pd.Timestamp(et))
        if entry_i is None or entry_i>=len(d):
            continue

        entry=float(d["Open"].iloc[entry_i])
        if not np.isfinite(entry) or entry<=0:
            continue

        exit5=find_exit_index(d,entry_i,"5日","60m")
        if exit5 is None:
            continue

        for name,trigger,protect in configs:
            activated=False
            activation_i=None
            exit_i=None
            exit_price=None
            exit_reason=""

            if name=="A_5日基準":
                exit_i=exit5
                exit_price=float(d["Close"].iloc[exit_i])
                exit_reason="5日到期"
            else:
                trigger_price=entry*(1.0+float(trigger))
                stop_price=entry*(1.0+float(protect))

                for j in range(entry_i,exit5+1):
                    if activated and j>activation_i:
                        op=float(d["Open"].iloc[j])
                        lo=float(d["Low"].iloc[j])
                        if np.isfinite(op) and op<=stop_price:
                            exit_i=j
                            exit_price=op
                            exit_reason=f"gap保護_T{int(trigger*100)}_P{int(protect*100)}"
                            break
                        if np.isfinite(lo) and lo<=stop_price:
                            exit_i=j
                            exit_price=stop_price
                            exit_reason=f"保護_T{int(trigger*100)}_P{int(protect*100)}"
                            break

                    hi=float(d["High"].iloc[j])
                    if (not activated) and np.isfinite(hi) and hi>=trigger_price:
                        activated=True
                        activation_i=j

                if exit_i is None:
                    exit_i=exit5
                    exit_price=float(d["Close"].iloc[exit_i])
                    exit_reason="5日到期"

            if exit_i is None or not np.isfinite(exit_price):
                continue

            gross=(exit_price/entry-1)*100
            cost_pct=cost.roundtrip_cost_pct(daytrade=False)
            net=gross-cost_pct
            path=d.iloc[entry_i:exit_i+1]

            z=r.to_dict()
            z["參數組合"]=name
            z["觸發門檻%"]=np.nan if trigger is None else trigger*100
            z["保護位置%"]=np.nan if protect is None else protect*100
            z["比較出場時間"]=twidx[exit_i] if exit_i<len(twidx) else d.index[exit_i]
            z["比較出場價"]=exit_price
            z["比較淨報酬%"]=net
            z["比較MFE%"]=(float(path["High"].max())/entry-1)*100
            z["比較MAE%"]=(float(path["Low"].min())/entry-1)*100
            z["保護曾啟動"]="是" if activated else "否"
            z["比較出場原因"]=exit_reason
            z["持有60m棒數"]=int(exit_i-entry_i+1)
            rows.append(z)

    detail=pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(),pd.DataFrame(),detail,errors

    # 僅保留10個方案都存在的共同母體。
    key_cols=["股票","訊號時間_台北","進場時間"]
    counts=detail.groupby(key_cols)["參數組合"].nunique().reset_index(name="_n")
    common=counts[counts["_n"]==len(configs)][key_cols]
    detail=detail.merge(common,on=key_cols,how="inner")

    def _metrics(g,sample,name):
        tmp=g.copy()
        tmp["淨報酬%"]=pd.to_numeric(tmp["比較淨報酬%"],errors="coerce")
        m=aggregate_trade_metrics(tmp)
        activated=(tmp["保護曾啟動"]=="是") if "保護曾啟動" in tmp.columns else pd.Series(False,index=tmp.index)
        protective=tmp["比較出場原因"].astype(str).str.contains("保護_") if "比較出場原因" in tmp.columns else pd.Series(False,index=tmp.index)
        return {
            "樣本":sample,"參數組合":name,
            "交易數":len(tmp),
            "股票數":int(tmp["股票"].nunique()) if len(tmp) else 0,
            **m,
            "MFE中位數%":float(pd.to_numeric(tmp["比較MFE%"],errors="coerce").median()) if len(tmp) else np.nan,
            "MAE中位數%":float(pd.to_numeric(tmp["比較MAE%"],errors="coerce").median()) if len(tmp) else np.nan,
            "平均持有60m棒數":float(pd.to_numeric(tmp["持有60m棒數"],errors="coerce").mean()) if len(tmp) else np.nan,
            "保護啟動率%":float(activated.mean()*100) if len(tmp) else np.nan,
            "保護出場率%":float(protective.mean()*100) if len(tmp) else np.nan,
        }

    names=[x[0] for x in configs]
    summary_rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=detail if sample=="全部" else detail[detail["樣本"]==sample]
        for name in names:
            summary_rows.append(_metrics(xs[xs["參數組合"]==name],sample,name))
    summary=pd.DataFrame(summary_rows)

    block_rows=[]
    for block in ["第1段","第2段","第3段","第4段"]:
        xb=detail[detail["時間段"]==block]
        for name in names:
            row=_metrics(xb[xb["參數組合"]==name],block,name)
            row["時間段"]=block
            row.pop("樣本",None)
            block_rows.append(row)
    blocks=pd.DataFrame(block_rows)

    return summary,blocks,detail,errors
def validate_profit_protection_candidates(cost: CostConfig):
    """
    V1.16.26 獲利保護A/B：
    固定正式架構C、固定同一批進場，不增加新訊號。

    比較：
      A_5日基準
      B_4日固定出場
      C_5日_曾達+5後保本
      D_5日_曾達+5後保+2
      E_5日_曾達+5後保+3

    執行假設（避免60m OHLC同K先後順序偏誤）：
      - 某根已完成60m K的 High 首次 >= 進場價*1.05，視為「+5%曾達」。
      - 保護停損從「下一根60m K」才開始生效。
      - 若下一根開盤已低於保護價，用開盤價出場（gap conservative）。
      - 否則若該K Low <= 保護價，使用保護價出場。
      - 若始終未觸發保護，仍於第5個交易日最後一根收盤出場。
    """
    _,_,_,base,errors=validate_breadth_transition(cost)
    if base is None or base.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    symbols=sorted(base["股票"].dropna().astype(str).unique().tolist())
    raw=download_intraday_batch(symbols,"60m","6mo")

    data_map={}
    idx_map={}
    tw_index_map={}
    for s in symbols:
        d=raw.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy().sort_index()
            tw=_as_taipei_series(pd.Series(x.index))
            data_map[s]=x
            idx_map[s]={pd.Timestamp(t):i for i,t in enumerate(tw) if pd.notna(t)}
            tw_index_map[s]=list(tw)
        except Exception:
            continue

    policies=[
        ("A_5日基準",None),
        ("B_4日固定出場","4d"),
        ("C_5日_曾達+5後保本",0.00),
        ("D_5日_曾達+5後保+2",0.02),
        ("E_5日_曾達+5後保+3",0.03),
    ]
    rows=[]

    for _,r in base.iterrows():
        s=str(r["股票"])
        d=data_map.get(s)
        mp=idx_map.get(s)
        twidx=tw_index_map.get(s)
        if d is None or mp is None or twidx is None:
            continue

        et=_as_taipei_series(pd.Series([r["進場時間"]])).iloc[0]
        if pd.isna(et):
            continue
        entry_i=mp.get(pd.Timestamp(et))
        if entry_i is None or entry_i>=len(d):
            continue

        entry=float(d["Open"].iloc[entry_i])
        if not np.isfinite(entry) or entry<=0:
            continue

        exit5=find_exit_index(d,entry_i,"5日","60m")
        exit4=find_exit_index(d,entry_i,"4日","60m")
        if exit5 is None or exit4 is None:
            continue

        for name,protect in policies:
            exit_i=None
            exit_price=None
            exit_reason=""
            activated=False
            activation_i=None

            if name=="A_5日基準":
                exit_i=exit5
                exit_price=float(d["Close"].iloc[exit_i])
                exit_reason="5日到期"
            elif name=="B_4日固定出場":
                exit_i=exit4
                exit_price=float(d["Close"].iloc[exit_i])
                exit_reason="4日到期"
            else:
                stop_price=entry*(1.0+float(protect))
                # 使用完成K的High判斷曾達+5%，保護從下一根才生效。
                for j in range(entry_i,exit5+1):
                    if activated and j>activation_i:
                        op=float(d["Open"].iloc[j])
                        lo=float(d["Low"].iloc[j])
                        if np.isfinite(op) and op<=stop_price:
                            exit_i=j
                            exit_price=op
                            exit_reason=f"保護觸發_gap_{int(protect*100)}%"
                            break
                        if np.isfinite(lo) and lo<=stop_price:
                            exit_i=j
                            exit_price=stop_price
                            exit_reason=f"保護觸發_{int(protect*100)}%"
                            break

                    hi=float(d["High"].iloc[j])
                    if (not activated) and np.isfinite(hi) and hi>=entry*1.05:
                        activated=True
                        activation_i=j

                if exit_i is None:
                    exit_i=exit5
                    exit_price=float(d["Close"].iloc[exit_i])
                    exit_reason="5日到期"

            if exit_i is None or not np.isfinite(exit_price):
                continue

            gross=(exit_price/entry-1)*100
            cost_pct=cost.roundtrip_cost_pct(daytrade=False)
            net=gross-cost_pct
            path=d.iloc[entry_i:exit_i+1]
            mfe=(float(path["High"].max())/entry-1)*100
            mae=(float(path["Low"].min())/entry-1)*100

            z=r.to_dict()
            z["出場政策"]=name
            z["比較出場時間"]=twidx[exit_i] if exit_i<len(twidx) else d.index[exit_i]
            z["比較出場價"]=exit_price
            z["比較淨報酬%"]=net
            z["比較MFE%"]=mfe
            z["比較MAE%"]=mae
            z["保護曾啟動"]="是" if activated else "否"
            z["比較出場原因"]=exit_reason
            z["持有60m棒數"]=int(exit_i-entry_i+1)
            rows.append(z)

    detail=pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(),pd.DataFrame(),detail,errors

    # 僅保留所有政策都有結果的共同進場母體。
    key_cols=["股票","訊號時間_台北","進場時間"]
    counts=detail.groupby(key_cols)["出場政策"].nunique().reset_index(name="_n")
    common=counts[counts["_n"]==len(policies)][key_cols]
    detail=detail.merge(common,on=key_cols,how="inner")

    def _summary_row(g,sample,name):
        tmp=g.copy()
        tmp["淨報酬%"]=pd.to_numeric(tmp["比較淨報酬%"],errors="coerce")
        m=aggregate_trade_metrics(tmp)
        activated=(tmp["保護曾啟動"]=="是") if "保護曾啟動" in tmp.columns else pd.Series(False,index=tmp.index)
        protective=tmp["比較出場原因"].astype(str).str.startswith("保護觸發") if "比較出場原因" in tmp.columns else pd.Series(False,index=tmp.index)
        return {
            "樣本":sample,"出場政策":name,
            "交易數":len(tmp),
            "股票數":int(tmp["股票"].nunique()) if len(tmp) else 0,
            **m,
            "MFE中位數%":float(pd.to_numeric(tmp["比較MFE%"],errors="coerce").median()) if len(tmp) else np.nan,
            "MAE中位數%":float(pd.to_numeric(tmp["比較MAE%"],errors="coerce").median()) if len(tmp) else np.nan,
            "平均持有60m棒數":float(pd.to_numeric(tmp["持有60m棒數"],errors="coerce").mean()) if len(tmp) else np.nan,
            "保護啟動率%":float(activated.mean()*100) if len(tmp) else np.nan,
            "保護實際出場率%":float(protective.mean()*100) if len(tmp) else np.nan,
        }

    summary_rows=[]
    policy_names=[p[0] for p in policies]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=detail if sample=="全部" else detail[detail["樣本"]==sample]
        for name in policy_names:
            g=xs[xs["出場政策"]==name]
            summary_rows.append(_summary_row(g,sample,name))
    summary=pd.DataFrame(summary_rows)

    block_rows=[]
    for block in ["第1段","第2段","第3段","第4段"]:
        xb=detail[detail["時間段"]==block]
        for name in policy_names:
            g=xb[xb["出場政策"]==name]
            row=_summary_row(g,block,name)
            row["時間段"]=block
            row.pop("樣本",None)
            block_rows.append(row)
    blocks=pd.DataFrame(block_rows)

    return summary,blocks,detail,errors
def validate_holding_period_candidates(cost: CostConfig):
    """
    V1.16.25 持有天數A/B：
    沿用V1.16.24已確認的正式架構C交易母體，固定「同一批進場」，
    只改出場持有天數，避免短持有因提早空倉而多出新訊號造成比較偏誤。

    比較：
      2日 / 3日 / 4日 / 5日基準

    重要：
      - 進場時間、股票、訊號完全相同。
      - 只重新計算不同持有天數的出場價、淨報酬、MFE、MAE。
      - 不加入停利停損，不改股票池，不改S/A/B。
    """
    _,_,_,base,errors=validate_breadth_transition(cost)
    if base is None or base.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    # 只需要實際出現在正式架構C交易母體中的股票，降低下載量。
    symbols=sorted(base["股票"].dropna().astype(str).unique().tolist())
    raw=download_intraday_batch(symbols,"60m","6mo")

    # 建立每檔60m資料與台北時間索引對照。
    data_map={}
    idx_map={}
    for s in symbols:
        d=raw.get(s)
        if d is None or d.empty:
            continue
        try:
            x=d.copy().sort_index()
            tw=_as_taipei_series(pd.Series(x.index))
            data_map[s]=x
            idx_map[s]={pd.Timestamp(t):i for i,t in enumerate(tw) if pd.notna(t)}
        except Exception:
            continue

    modes=["2日","3日","4日","5日"]
    repriced=[]

    for _,r in base.iterrows():
        s=str(r["股票"])
        d=data_map.get(s)
        mp=idx_map.get(s)
        if d is None or mp is None:
            continue

        et=_as_taipei_series(pd.Series([r["進場時間"]])).iloc[0]
        if pd.isna(et):
            continue
        entry_i=mp.get(pd.Timestamp(et))
        if entry_i is None or entry_i>=len(d):
            continue

        entry=float(d["Open"].iloc[entry_i])
        if not np.isfinite(entry) or entry<=0:
            continue

        for mode in modes:
            exit_i=find_exit_index(d,entry_i,mode,"60m")
            if exit_i is None or exit_i<=entry_i:
                continue
            exitp=float(d["Close"].iloc[exit_i])
            if not np.isfinite(exitp):
                continue

            gross=(exitp/entry-1)*100
            cost_pct=cost.roundtrip_cost_pct(daytrade=False)
            net=gross-cost_pct
            path=d.iloc[entry_i:exit_i+1]
            mfe=(float(path["High"].max())/entry-1)*100
            mae=(float(path["Low"].min())/entry-1)*100

            z=r.to_dict()
            z["比較持有"]=mode
            z["比較出場時間"]=d.index[exit_i]
            z["比較出場價"]=exitp
            z["比較毛報酬%"]=gross
            z["比較成本%"]=cost_pct
            z["比較淨報酬%"]=net
            z["比較MFE%"]=mfe
            z["比較MAE%"]=mae
            repriced.append(z)

    detail=pd.DataFrame(repriced)
    if detail.empty:
        return pd.DataFrame(),pd.DataFrame(),detail,errors

    # 只有四種模式都能重算的共同交易才納入比較，確保完全同母體。
    key_cols=["股票","訊號時間_台北","進場時間"]
    counts=detail.groupby(key_cols)["比較持有"].nunique().reset_index(name="_mode_n")
    common=counts[counts["_mode_n"]==len(modes)][key_cols]
    detail=detail.merge(common,on=key_cols,how="inner")

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=detail if sample=="全部" else detail[detail["樣本"]==sample]
        for mode in modes:
            g=xs[xs["比較持有"]==mode].copy()
            # aggregate_trade_metrics 預期欄名為淨報酬%
            gm=g.rename(columns={"比較淨報酬%":"淨報酬%_AB"})
            tmp=g.copy()
            tmp["淨報酬%"]=pd.to_numeric(g["比較淨報酬%"],errors="coerce")
            m=aggregate_trade_metrics(tmp)
            rows.append({
                "樣本":sample,"持有方案":mode,
                "交易數":len(g),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m,
                "MFE中位數%":float(pd.to_numeric(g["比較MFE%"],errors="coerce").median()) if len(g) else np.nan,
                "MAE中位數%":float(pd.to_numeric(g["比較MAE%"],errors="coerce").median()) if len(g) else np.nan,
            })
    summary=pd.DataFrame(rows)

    # 四段時間穩定度，仍沿用原本訊號時間四段，不因出場模式重切。
    block_rows=[]
    for block in ["第1段","第2段","第3段","第4段"]:
        xb=detail[detail["時間段"]==block]
        for mode in modes:
            g=xb[xb["比較持有"]==mode].copy()
            tmp=g.copy()
            tmp["淨報酬%"]=pd.to_numeric(g["比較淨報酬%"],errors="coerce")
            m=aggregate_trade_metrics(tmp)
            block_rows.append({
                "時間段":block,"持有方案":mode,
                "交易數":len(g),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m,
                "MFE中位數%":float(pd.to_numeric(g["比較MFE%"],errors="coerce").median()) if len(g) else np.nan,
                "MAE中位數%":float(pd.to_numeric(g["比較MAE%"],errors="coerce").median()) if len(g) else np.nan,
            })
    blocks=pd.DataFrame(block_rows)

    return summary,blocks,detail,errors
def validate_holding_path_diagnostics(cost: CostConfig):
    """
    V1.16.24 持有路徑健診：
    固定正式架構C與5日持有，不改策略，只用逐筆 MFE / MAE 檢查：
      - 第1段是否其實有先反彈、之後再回吐
      - 問題較像進場品質，還是固定持有5日過久
    這版只診斷，不直接加入停利/停損。
    """
    # 沿用目前最完整的正式架構C point-in-time逐筆資料。
    _,_,_,t,errors=validate_breadth_transition(cost)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    x=t.copy()
    net=pd.to_numeric(x.get("淨報酬%"),errors="coerce")
    mfe=pd.to_numeric(x.get("MFE%"),errors="coerce")
    mae=pd.to_numeric(x.get("MAE%"),errors="coerce")

    # 固定、事先定義的描述性門檻，不依本輪績效最佳化。
    x["曾達+3%"]=mfe>=3
    x["曾達+5%"]=mfe>=5
    x["曾達+10%"]=mfe>=10
    x["曾跌-3%"]=mae<=-3
    x["曾跌-5%"]=mae<=-5
    x["曾跌-10%"]=mae<=-10
    x["+3後最終虧損"]=(mfe>=3)&(net<0)
    x["+5後最終虧損"]=(mfe>=5)&(net<0)
    x["+10後最終虧損"]=(mfe>=10)&(net<0)
    x["-5後最終翻正"]=(mae<=-5)&(net>0)

    # MFE利用率：最終淨報酬 / 最大有利幅度，只在MFE>0時觀察。
    x["MFE利用率%"]=np.where(mfe>0,net/mfe*100,np.nan)

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=x if sample=="全部" else x[x["樣本"]==sample]
        for block in ["第1段","第2段","第3段","第4段"]:
            g=xs[xs["時間段"]==block]
            if g.empty:
                rows.append({
                    "樣本":sample,"時間段":block,"交易數":0,
                    "平均淨報酬%":np.nan,"MFE中位數%":np.nan,"MAE中位數%":np.nan,
                    "曾達+3%占比":np.nan,"曾達+5%占比":np.nan,"曾達+10%占比":np.nan,
                    "曾跌-5%占比":np.nan,"+3後最終虧損占比":np.nan,
                    "+5後最終虧損占比":np.nan,"-5後最終翻正占比":np.nan,
                    "MFE利用率中位數%":np.nan
                })
                continue
            rows.append({
                "樣本":sample,
                "時間段":block,
                "交易數":len(g),
                "平均淨報酬%":float(pd.to_numeric(g["淨報酬%"],errors="coerce").mean()),
                "MFE中位數%":float(pd.to_numeric(g["MFE%"],errors="coerce").median()),
                "MAE中位數%":float(pd.to_numeric(g["MAE%"],errors="coerce").median()),
                "曾達+3%占比":float(g["曾達+3%"].mean()*100),
                "曾達+5%占比":float(g["曾達+5%"].mean()*100),
                "曾達+10%占比":float(g["曾達+10%"].mean()*100),
                "曾跌-5%占比":float(g["曾跌-5%"].mean()*100),
                "+3後最終虧損占比":float(g["+3後最終虧損"].mean()*100),
                "+5後最終虧損占比":float(g["+5後最終虧損"].mean()*100),
                "-5後最終翻正占比":float(g["-5後最終翻正"].mean()*100),
                "MFE利用率中位數%":float(pd.to_numeric(g["MFE利用率%"],errors="coerce").median())
            })
    summary=pd.DataFrame(rows)

    # 路徑類型：互斥分組，避免重疊。
    def _path(r):
        n=pd.to_numeric(pd.Series([r.get("淨報酬%")]),errors="coerce").iloc[0]
        f=pd.to_numeric(pd.Series([r.get("MFE%")]),errors="coerce").iloc[0]
        a=pd.to_numeric(pd.Series([r.get("MAE%")]),errors="coerce").iloc[0]
        if pd.isna(n) or pd.isna(f) or pd.isna(a):
            return "資料不足"
        if f>=5 and n<0:
            return "先漲後吐_曾+5最終虧"
        if a<=-5 and n>0:
            return "先跌後拉_-5後翻正"
        if n>0:
            return "正常獲利"
        return "一路偏弱/未達+5"
    x["持有路徑類型"]=x.apply(_path,axis=1)

    path_rows=[]
    path_order=["先漲後吐_曾+5最終虧","先跌後拉_-5後翻正","正常獲利","一路偏弱/未達+5","資料不足"]
    for block in ["第1段","第2段","第3段","第4段"]:
        xb=x[x["時間段"]==block]
        for path in path_order:
            g=xb[xb["持有路徑類型"]==path]
            m=aggregate_trade_metrics(g)
            path_rows.append({
                "時間段":block,"持有路徑類型":path,
                "交易數":len(g),
                "占比%":float(len(g)/len(xb)*100) if len(xb) else np.nan,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m
            })
    paths=pd.DataFrame(path_rows)

    return summary,paths,x,errors
def validate_breadth_transition(cost: CostConfig):
    """
    V1.16.23 市場廣度轉折健診：
    不再只看「MA60廣度高不高」，改看廣度是正在擴散還是收斂。

    固定正式架構C：
      TOP1-100全部 + TOP101-150僅S
      60m KD黃金交叉 + K<30 + 持有5日
      6mo暖機 / 最後3mo評估

    市場轉折狀態：
      高檔擴散：MA60廣度>=65 且 MA15廣度5日變化>=0
      高檔收斂：MA60廣度>=65 且 MA15廣度5日變化<0
      非高檔擴散：MA60廣度<65 且 MA15廣度5日變化>=0
      非高檔收斂：MA60廣度<65 且 MA15廣度5日變化<0

    另外固定檢查 MA15/30/60 廣度 5日變化分桶。
    這版只診斷，不直接改今日雷達Gate。
    """
    breadth, elig, errors = build_top200_market_breadth_context()
    if breadth is None or breadth.empty or elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    b=breadth.sort_values("基準完成日").copy()
    for col in ["站上MA15比例%","站上MA30比例%","站上MA60比例%","5日上漲家數比例%"]:
        x=pd.to_numeric(b[col],errors="coerce")
        b[col+"_3日變化"]=x.diff(3)
        b[col+"_5日變化"]=x.diff(5)

    union=sorted(
        elig.loc[elig["流動性排名"]<=200,"股票"].dropna().astype(str).unique().tolist()
    )
    _,_,trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=trades.copy()
    sig=_as_taipei_series(t["訊號時間"])
    t["訊號時間_台北"]=sig
    t["訊號日期"]=sig.dt.date

    # 歷史流動性排名：只用訊號日前完成日。
    ranks=[]
    for _,r in t.iterrows():
        q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
        ranks.append(float(q.sort_values("基準完成日").iloc[-1]["流動性排名"]) if not q.empty else np.nan)
    t["WF流動性排名"]=ranks
    t=t[t["WF流動性排名"].notna() & (t["WF流動性排名"]<=200)].copy()

    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60s=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["S級"]=(vr>=1.5)&(ma60s<=0)
    rnk=pd.to_numeric(t["WF流動性排名"],errors="coerce")
    t=t[(rnk<=100)|((rnk>100)&(rnk<=150)&t["S級"])].copy()

    # 合併訊號日前一個完成日的市場廣度與變化值。
    merge_cols=[
        "基準完成日","站上MA15比例%","站上MA30比例%","站上MA60比例%",
        "5日上漲家數比例%","5日報酬中位數%",
        "站上MA15比例%_3日變化","站上MA15比例%_5日變化",
        "站上MA30比例%_3日變化","站上MA30比例%_5日變化",
        "站上MA60比例%_3日變化","站上MA60比例%_5日變化",
        "5日上漲家數比例%_3日變化","5日上漲家數比例%_5日變化",
    ]
    mrows=[]
    for _,r in t.iterrows():
        q=b[b["基準完成日"]<r["訊號日期"]]
        mrows.append(q.iloc[-1][merge_cols].to_dict() if not q.empty else {})
    mx=pd.DataFrame(mrows,index=t.index)
    for c in merge_cols:
        if c in mx.columns:
            t["市場_"+c]=mx[c]

    ma60_level=pd.to_numeric(t.get("市場_站上MA60比例%"),errors="coerce")
    ma15_d5=pd.to_numeric(t.get("市場_站上MA15比例%_5日變化"),errors="coerce")

    def _state(level,delta):
        if pd.isna(level) or pd.isna(delta):
            return "資料不足"
        if level>=65 and delta>=0:
            return "高檔擴散"
        if level>=65 and delta<0:
            return "高檔收斂"
        if level<65 and delta>=0:
            return "非高檔擴散"
        return "非高檔收斂"

    t["市場廣度轉折狀態"]=[_state(a,d) for a,d in zip(ma60_level,ma15_d5)]

    # 固定四段，與前面研究一致。
    ts=t["訊號時間_台北"]
    edges=pd.date_range(ts.min(),ts.max(),periods=5)
    labels=["第1段","第2段","第3段","第4段"]
    t["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)

    states=["高檔擴散","高檔收斂","非高檔擴散","非高檔收斂","資料不足"]
    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for state in states:
            g=xs[xs["市場廣度轉折狀態"]==state]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"市場廣度轉折狀態":state,
                "交易數":len(g),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "占比%":float(len(g)/len(xs)*100) if len(xs) else np.nan,
                **m
            })
    summary=pd.DataFrame(rows)

    # 固定變化分桶，不依績效調切點。
    bucket_rows=[]
    specs=[
        ("市場_站上MA15比例%_5日變化",[-np.inf,-15,-5,5,15,np.inf],["<=-15","-15~-5","-5~5","5~15",">15"]),
        ("市場_站上MA30比例%_5日變化",[-np.inf,-15,-5,5,15,np.inf],["<=-15","-15~-5","-5~5","5~15",">15"]),
        ("市場_站上MA60比例%_5日變化",[-np.inf,-10,-3,3,10,np.inf],["<=-10","-10~-3","-3~3","3~10",">10"]),
        ("市場_5日上漲家數比例%_5日變化",[-np.inf,-20,-5,5,20,np.inf],["<=-20","-20~-5","-5~5","5~20",">20"]),
    ]
    for col,bins,names in specs:
        x=pd.to_numeric(t.get(col),errors="coerce")
        bucket=pd.cut(x,bins=bins,labels=names,include_lowest=True,right=True)
        for name in names:
            g=t[bucket==name]
            m=aggregate_trade_metrics(g)
            bucket_rows.append({
                "轉折因子":col,"分組":name,
                "交易數":len(g),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m
            })
    buckets=pd.DataFrame(bucket_rows)

    # 四段 × 轉折狀態
    blocks=[]
    for block in labels:
        xb=t[t["時間段"]==block]
        for state in states:
            g=xb[xb["市場廣度轉折狀態"]==state]
            m=aggregate_trade_metrics(g)
            blocks.append({
                "時間段":block,"市場廣度轉折狀態":state,
                "交易數":len(g),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m
            })
    block_df=pd.DataFrame(blocks)

    return summary,buckets,block_df,t,errors
def validate_environment_signal_interaction(cost: CostConfig):
    """
    V1.16.20 市場環境 × 訊號等級交互驗證：
    沿用V1.16.18/19的point-in-time正式架構C逐筆交易。

    市場狀態：
      高檔風險：TOP200站上MA60比例 >=65%
      正常環境：TOP200站上MA60比例 <65%

    訊號等級：
      S級：量比20>=1.5 且 MA60斜率3<=0
      A級：兩者只符合一項
      B級：兩者皆不符合

    目的：
      判斷高檔風險環境是否應「全部封鎖」，
      還是只需要降低A/B級權重、保留S級。
    """
    _,_,t,errors=validate_failure_environment(cost)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    market_ma60=pd.to_numeric(t.get("市場_站上MA60比例%"),errors="coerce")
    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60s=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")

    t["市場風險狀態"]=np.select(
        [market_ma60.isna(), market_ma60>=65, market_ma60<65],
        ["資料不足","高檔風險_MA60廣度>=65","正常_MA60廣度<65"],
        default="資料不足"
    )
    t["訊號等級"]=np.where(
        (vr>=1.5)&(ma60s<=0),"S級",
        np.where((vr>=1.5)|(ma60s<=0),"A級","B級")
    )

    envs=["高檔風險_MA60廣度>=65","正常_MA60廣度<65","資料不足"]
    grades=["S級","A級","B級"]

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for env in envs:
            xe=xs[xs["市場風險狀態"]==env]
            for grade in grades:
                g=xe[xe["訊號等級"]==grade]
                m=aggregate_trade_metrics(g)
                rows.append({
                    "樣本":sample,
                    "市場風險狀態":env,
                    "訊號等級":grade,
                    "交易數":len(g),
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "環境內占比%":float(len(g)/len(xe)*100) if len(xe) else np.nan,
                    **m
                })
    summary=pd.DataFrame(rows)

    # 四段時間：檢查同一交互關係是否只發生在第1段。
    blocks=[]
    for block in ["第1段","第2段","第3段","第4段"]:
        xb=t[t["時間段"]==block]
        for env in envs:
            xe=xb[xb["市場風險狀態"]==env]
            for grade in grades:
                g=xe[xe["訊號等級"]==grade]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "市場風險狀態":env,
                    "訊號等級":grade,
                    "交易數":len(g),
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "環境內占比%":float(len(g)/len(xe)*100) if len(xe) else np.nan,
                    **m
                })
    block_df=pd.DataFrame(blocks)

    # 模擬三種可落地政策，但此版仍只研究、不改正式雷達：
    # P0 全收；P1 高檔風險全部排除；P2 高檔風險只保留S級。
    valid_env=t["市場風險狀態"]!="資料不足"
    policies=[
        ("P0_全部保留",valid_env),
        ("P1_高檔風險全部排除",valid_env & (t["市場風險狀態"]!="高檔風險_MA60廣度>=65")),
        ("P2_高檔風險只保留S級",
         valid_env & ((t["市場風險狀態"]!="高檔風險_MA60廣度>=65")|(t["訊號等級"]=="S級"))),
    ]
    policy_rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        valid_xs=xs[xs["市場風險狀態"]!="資料不足"]
        for name,mask in policies:
            g=xs[mask.reindex(xs.index,fill_value=False)]
            m=aggregate_trade_metrics(g)
            policy_rows.append({
                "樣本":sample,
                "政策":name,
                "交易數":len(g),
                "有效環境樣本數":len(valid_xs),
                "市場廣度缺值筆數":int((xs["市場風險狀態"]=="資料不足").sum()),
                "保留率%":float(len(g)/len(valid_xs)*100) if len(valid_xs) else np.nan,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m
            })
    policies_df=pd.DataFrame(policy_rows)

    return summary,block_df,policies_df,t,errors
def validate_environment_gate_candidates(cost: CostConfig):
    """
    V1.16.19 環境Gate候選A/B：
    直接沿用 V1.16.18 的 point-in-time 正式架構C逐筆資料，
    不重定義股票池、不改進出場，只比較幾個事先鎖定的環境排除條件。

    A 基準：不排除
    B 排除長期高檔：市場站上MA60比例 >=65%
    C 排除高檔轉弱：MA60>=65% 且 5日上漲家數比例 35~50%
    D 排除高檔轉弱2：MA60>=65% 且 5日報酬中位數 -2~0%
    E 排除中度訊號擁擠：同時60m訊號數 6~10

    注意：
    - 這些是研究候選，不會在此版直接寫進今日正式雷達。
    - C/D 是根據「長期廣度仍高、短期已轉弱」的同一市場狀態做兩種觀測定義。
    """
    _,_,t,errors=validate_failure_environment(cost)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    ma60=pd.to_numeric(t.get("市場_站上MA60比例%"),errors="coerce")
    up5=pd.to_numeric(t.get("市場_5日上漲家數比例%"),errors="coerce")
    med5=pd.to_numeric(t.get("市場_5日報酬中位數%"),errors="coerce")
    crowd=pd.to_numeric(t.get("同時60m訊號數"),errors="coerce")

    trap_c=(ma60>=65)&(up5>=35)&(up5<50)
    trap_d=(ma60>=65)&(med5>=-2)&(med5<0)
    crowd_mid=(crowd>=6)&(crowd<=10)

    gates=[
        ("A_基準",pd.Series(True,index=t.index)),
        ("B_排除MA60廣度>=65",~(ma60>=65)),
        ("C_排除高檔轉弱_MA60>=65且5日上漲35-50",~trap_c),
        ("D_排除高檔轉弱_MA60>=65且5日報酬-2~0",~trap_d),
        ("E_排除同時60m訊號6-10",~crowd_mid),
    ]

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for name,mask in gates:
            g=xs[mask.reindex(xs.index,fill_value=False)]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,
                "環境Gate":name,
                "交易數":len(g),
                "保留率%":float(len(g)/len(xs)*100) if len(xs) else np.nan,
                "排除交易數":int(len(xs)-len(g)),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m
            })
    summary=pd.DataFrame(rows)

    blocks=[]
    labels=["第1段","第2段","第3段","第4段"]
    for block in labels:
        q=t[t["時間段"]==block]
        for name,mask in gates:
            g=q[mask.reindex(q.index,fill_value=False)]
            m=aggregate_trade_metrics(g)
            blocks.append({
                "時間段":block,
                "環境Gate":name,
                "交易數":len(g),
                "保留率%":float(len(g)/len(q)*100) if len(q) else np.nan,
                "排除交易數":int(len(q)-len(g)),
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                **m
            })
    block_df=pd.DataFrame(blocks)

    # 逐筆標示每個候選是否會被排除，方便後續檢查。
    detail=t.copy()
    detail["GateB_排除"]=ma60>=65
    detail["GateC_排除"]=trap_c
    detail["GateD_排除"]=trap_d
    detail["GateE_排除"]=crowd_mid

    return summary,block_df,detail,errors
def validate_failure_environment(cost: CostConfig):
    """
    V1.16.18 第一段失效環境健診：
    固定正式架構C，不新增Gate，只比較第1段與後3段的市場廣度/訊號擁擠差異。
    """
    breadth, elig, errors = build_top200_market_breadth_context()
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(
        elig.loc[elig["流動性排名"]<=200,"股票"].dropna().astype(str).unique().tolist()
    )
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=trades.copy()
    sig=_as_taipei_series(t["訊號時間"])
    t["訊號時間_台北"]=sig
    t["訊號日期"]=sig.dt.date

    # 歷史流動性排名只取訊號日前一個完成日。
    ranks=[]; basedates=[]
    for _,r in t.iterrows():
        q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
        if q.empty:
            ranks.append(np.nan); basedates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            ranks.append(float(z["流動性排名"]))
            basedates.append(z["基準完成日"])
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=basedates
    t=t[t["WF流動性排名"].notna() & (t["WF流動性排名"]<=200)].copy()

    # 正式架構C：TOP1-100全部 + TOP101-150僅S。
    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60s=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["S級"]=(vr>=1.5)&(ma60s<=0)
    rnk=pd.to_numeric(t["WF流動性排名"],errors="coerce")
    formal=(rnk<=100)|((rnk>100)&(rnk<=150)&t["S級"])
    t=t[formal].copy()

    # 市場廣度嚴格取訊號日前一個已完成日K。
    if breadth is not None and not breadth.empty:
        b=breadth.sort_values("基準完成日").copy()
        b_rows=[]
        for _,r in t.iterrows():
            q=b[b["基準完成日"]<r["訊號日期"]]
            b_rows.append(q.iloc[-1].to_dict() if not q.empty else {})
        bx=pd.DataFrame(b_rows,index=t.index)
        for c in ["基準完成日","TOP200有效股票數","站上MA15比例%","站上MA30比例%",
                  "站上MA60比例%","5日上漲家數比例%","5日報酬中位數%"]:
            if c in bx.columns:
                t["市場_"+c]=bx[c]

    # 同一個完成60m bar的訊號擁擠度，只看當下同時訊號。
    t["同時60m訊號數"]=t.groupby("訊號時間_台北")["股票"].transform("count")

    ts=t["訊號時間_台北"]
    edges=pd.date_range(ts.min(),ts.max(),periods=5)
    labels=["第1段","第2段","第3段","第4段"]
    t["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)

    env_cols=[
        "市場_站上MA15比例%","市場_站上MA30比例%","市場_站上MA60比例%",
        "市場_5日上漲家數比例%","市場_5日報酬中位數%","同時60m訊號數"
    ]
    block_rows=[]
    for block in labels:
        g=t[t["時間段"]==block]
        m=aggregate_trade_metrics(g)
        row={
            "時間段":block,
            "起始":str(edges[labels.index(block)]),
            "結束":str(edges[labels.index(block)+1]),
            "股票數":int(g["股票"].nunique()) if len(g) else 0,
            "交易數":len(g),**m
        }
        for c in env_cols:
            x=pd.to_numeric(g[c],errors="coerce") if c in g.columns else pd.Series(dtype=float)
            row[c+"_中位數"]=float(x.median()) if x.notna().any() else np.nan
            row[c+"_平均"]=float(x.mean()) if x.notna().any() else np.nan
        block_rows.append(row)
    blocks=pd.DataFrame(block_rows)

    # 固定分桶，避免根據結果臨時尋找最佳門檻。
    specs=[
        ("市場_站上MA15比例%",[-np.inf,35,50,65,np.inf],["<35%","35-50%","50-65%",">=65%"]),
        ("市場_站上MA30比例%",[-np.inf,35,50,65,np.inf],["<35%","35-50%","50-65%",">=65%"]),
        ("市場_站上MA60比例%",[-np.inf,35,50,65,np.inf],["<35%","35-50%","50-65%",">=65%"]),
        ("市場_5日上漲家數比例%",[-np.inf,35,50,65,np.inf],["<35%","35-50%","50-65%",">=65%"]),
        ("市場_5日報酬中位數%",[-np.inf,-2,0,2,np.inf],["<-2%","-2~0%","0~2%",">=2%"]),
        ("同時60m訊號數",[-np.inf,2,5,10,np.inf],["1-2","3-5","6-10",">10"]),
    ]
    rows=[]
    for col,bins,names in specs:
        x=pd.to_numeric(t[col],errors="coerce") if col in t.columns else pd.Series(np.nan,index=t.index)
        bucket=pd.cut(x,bins=bins,labels=names,include_lowest=True,right=True)
        for name in names:
            g=t[bucket==name]
            m=aggregate_trade_metrics(g)
            rows.append({
                "環境因子":col,"分組":name,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    return blocks,pd.DataFrame(rows),t,errors
def validate_radar_architectures(cost: CostConfig):
    """
    V1.16.15 雷達架構驗證：
    固定真正Walk-Forward + 6mo暖機/末3mo評估 + 核心策略，
    比較不同「廣度 vs 品質」雷達架構。

    A 核心50：TOP1-50全部訊號
    B 核心+觀察100：TOP1-100全部訊號
    C 100 + 擴充S：TOP1-100全部 + TOP101-150僅S級
    D 100 + 外圍S：TOP1-100全部 + TOP101-200僅S級
    E 全TOP150：TOP1-150全部訊號（廣度對照）
    F 全TOP200：TOP1-200全部訊號（最大廣度對照）

    S級定義沿用既有驗證：量比20>=1.5 且 MA60斜率3<=0。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(
        elig.loc[elig["流動性排名"]<=200,"股票"].dropna().astype(str).unique().tolist()
    )
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=trades.copy()
    t["訊號日期"]=_as_taipei_series(t["訊號時間"]).dt.date

    ranks=[]; basedates=[]
    for _,r in t.iterrows():
        q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
        if q.empty:
            ranks.append(np.nan); basedates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            ranks.append(float(z["流動性排名"]))
            basedates.append(z["基準完成日"])
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=basedates
    t=t[t["WF流動性排名"].notna() & (t["WF流動性排名"]<=200)].copy()

    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["S級"]=(vr>=1.5)&(ma60<=0)
    t["訊號時間_台北"]=_as_taipei_series(t["訊號時間"]).astype(str)

    def mask_arch(name,df):
        r=df["WF流動性排名"]
        s=df["S級"]
        if name=="A_核心50":
            return r<=50
        if name=="B_核心+觀察100":
            return r<=100
        if name=="C_TOP100+101-150僅S":
            return (r<=100)|((r>100)&(r<=150)&s)
        if name=="D_TOP100+101-200僅S":
            return (r<=100)|((r>100)&(r<=200)&s)
        if name=="E_全TOP150":
            return r<=150
        return r<=200

    archs=[
        "A_核心50",
        "B_核心+觀察100",
        "C_TOP100+101-150僅S",
        "D_TOP100+101-200僅S",
        "E_全TOP150",
        "F_全TOP200",
    ]

    rows=[]
    total_all=len(t)
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for name in archs:
            mask=mask_arch(name,xs)
            g=xs[mask]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"雷達架構":name,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),
                "相對TOP200訊號涵蓋率%":float(len(g)/len(xs)*100) if len(xs) else np.nan,
                "S級交易數":int(g["S級"].sum()) if len(g) else 0,
                **m
            })
    summary=pd.DataFrame(rows)

    # 四段時間穩定度
    blocks=[]
    ts=_as_taipei_series(t["訊號時間"])
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for name in archs:
                g=q[mask_arch(name,q)]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "雷達架構":name,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),
                    "相對該段TOP200訊號涵蓋率%":float(len(g)/len(q)*100) if len(q) else np.nan,
                    **m
                })

    # 各架構新增訊號來源，方便理解廣度增加從哪裡來
    detail=t.copy()
    detail["A_核心50"]=mask_arch("A_核心50",detail)
    detail["B_核心+觀察100"]=mask_arch("B_核心+觀察100",detail)
    detail["C_TOP100+101-150僅S"]=mask_arch("C_TOP100+101-150僅S",detail)
    detail["D_TOP100+101-200僅S"]=mask_arch("D_TOP100+101-200僅S",detail)
    detail["E_全TOP150"]=mask_arch("E_全TOP150",detail)
    detail["F_全TOP200"]=mask_arch("F_全TOP200",detail)

    return summary,pd.DataFrame(blocks),detail,errors
def validate_pool_band_layers(cost: CostConfig):
    """
    V1.16.14 股票池分層驗證：
    固定真正Walk-Forward + 6mo暖機/末3mo評估 + 核心策略，
    將歷史流動性排名拆成互斥四層：
      核心層 TOP1-50
      觀察層 TOP51-100
      擴充層 TOP101-150
      外圍層 TOP151-200
    目的：在不犧牲全面性的前提下，判斷哪些層值得進正式雷達。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(
        elig.loc[elig["流動性排名"]<=200,"股票"].dropna().astype(str).unique().tolist()
    )
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=trades.copy()
    t["訊號日期"]=_as_taipei_series(t["訊號時間"]).dt.date

    ranks=[]; basedates=[]
    for _,r in t.iterrows():
        q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
        if q.empty:
            ranks.append(np.nan); basedates.append(None)
        else:
            z=q.sort_values("基準完成日").iloc[-1]
            ranks.append(float(z["流動性排名"]))
            basedates.append(z["基準完成日"])
    t["WF流動性排名"]=ranks
    t["WF股票池基準日"]=basedates
    t=t[t["WF流動性排名"].notna() & (t["WF流動性排名"]<=200)].copy()

    def _band(rank):
        if pd.isna(rank): return "資料不足"
        if rank<=50: return "核心層_TOP1-50"
        if rank<=100: return "觀察層_TOP51-100"
        if rank<=150: return "擴充層_TOP101-150"
        return "外圍層_TOP151-200"
    t["股票池層級"]=t["WF流動性排名"].map(_band)

    # 同步標示目前已驗證過的訊號等級，方便後續決定是否保留次核心層。
    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["量比Gate"]=vr>=1.5
    t["MA60Gate"]=ma60<=0
    t["訊號等級"]=np.where(
        t["量比Gate"] & t["MA60Gate"],"S級",
        np.where(t["量比Gate"] | t["MA60Gate"],"A級","B級")
    )
    t["訊號時間_台北"]=_as_taipei_series(t["訊號時間"]).astype(str)

    bands=["核心層_TOP1-50","觀察層_TOP51-100","擴充層_TOP101-150","外圍層_TOP151-200"]
    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for band in bands:
            g=xs[xs["股票池層級"]==band]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"股票池層級":band,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),
                "交易占TOP200比例%":float(len(g)/len(xs)*100) if len(xs) else np.nan,
                **m
            })
    summary=pd.DataFrame(rows)

    # 每一層的 S/A/B 結構
    grade_rows=[]
    for band in bands:
        q=t[t["股票池層級"]==band]
        for grade in ["S級","A級","B級"]:
            g=q[q["訊號等級"]==grade]
            m=aggregate_trade_metrics(g)
            grade_rows.append({
                "股票池層級":band,"訊號等級":grade,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),
                "該層訊號占比%":float(len(g)/len(q)*100) if len(q) else np.nan,
                **m
            })
    grades=pd.DataFrame(grade_rows)

    # 四段時間穩定度
    blocks=[]
    ts=_as_taipei_series(t["訊號時間"])
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for band in bands:
                g=q[q["股票池層級"]==band]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "股票池層級":band,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })

    return summary,pd.DataFrame(blocks),grades,t,errors
def diagnose_stock_pool_coverage(cost: CostConfig):
    """V1.16.13：檢查上市/上櫃母池、Yahoo覆蓋、動態TOP50/100/150/200與訊號涵蓋率。"""
    official, official_errors = fetch_official_tw_stock_universe()
    elig, _, wf_errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    errors=list(official_errors or [])+list(wf_errors or [])
    rows=[]

    official_count=len(official) if isinstance(official,pd.DataFrame) else 0
    listed=int((official["市場"]=="上市").sum()) if official_count and "市場" in official.columns else 0
    otc=int((official["市場"]=="上櫃").sum()) if official_count and "市場" in official.columns else 0
    diag=getattr(elig,"attrs",{}).get("wf_diag",{}) if isinstance(elig,pd.DataFrame) else {}
    daily_ok=int(diag.get("日K下載成功股票數",0))
    seq_ok=int(diag.get("建立歷史序列股票數",0))
    rows += [
        {"類別":"母池覆蓋","指標":"官方上市+上櫃公司數","數值":official_count},
        {"類別":"母池覆蓋","指標":"上市公司數","數值":listed},
        {"類別":"母池覆蓋","指標":"上櫃公司數","數值":otc},
        {"類別":"資料覆蓋","指標":"Yahoo 6mo日K成功股票數","數值":daily_ok},
        {"類別":"資料覆蓋","指標":"可建立歷史序列股票數","數值":seq_ok},
        {"類別":"資料覆蓋","指標":"Yahoo日K覆蓋率%","數值":(daily_ok/official_count*100) if official_count else np.nan},
    ]
    if elig is None or elig.empty:
        return pd.DataFrame(rows),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    sizes=[50,100,150,200]
    union_rows=[]
    for n in sizes:
        q=elig[elig["流動性排名"]<=n]
        perday=q.groupby("基準完成日")["股票"].nunique()
        union_rows.append({
            "核心池規模":f"TOP{n}",
            "歷史曾入選股票數":int(q["股票"].nunique()),
            "平均每日股票數":float(perday.mean()) if len(perday) else np.nan,
            "最少每日股票數":int(perday.min()) if len(perday) else 0,
            "最多每日股票數":int(perday.max()) if len(perday) else 0,
            "歷史交易日數":int(perday.size),
        })

    market_rows=[]
    latest_date=max(elig["基準完成日"])
    latest=elig[elig["基準完成日"]==latest_date].copy()
    market_map=dict(zip(official["股票"],official["市場"])) if official_count and {"股票","市場"}.issubset(official.columns) else {}
    for n in sizes:
        q=latest[latest["流動性排名"]<=n].copy()
        q["市場"]=q["股票"].map(market_map).fillna("未知")
        total=len(q)
        for market,count in q["市場"].value_counts().items():
            market_rows.append({
                "基準完成日":latest_date,"核心池規模":f"TOP{n}","市場":market,
                "股票數":int(count),"占比%":float(count/total*100) if total else np.nan
            })

    capture_rows=[]
    top200_union=sorted(elig.loc[elig["流動性排名"]<=200,"股票"].astype(str).unique().tolist())
    if top200_union:
        _,_,trades=run_oos_60m_5d(top200_union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3)
        if trades is not None and not trades.empty:
            t=trades.copy()
            t["訊號日期"]=_as_taipei_series(t["訊號時間"]).dt.date
            ranks=[]
            for _,r in t.iterrows():
                q=elig[(elig["股票"]==r["股票"])&(elig["基準完成日"]<r["訊號日期"])]
                ranks.append(float(q.sort_values("基準完成日").iloc[-1]["流動性排名"]) if not q.empty else np.nan)
            t["WF流動性排名"]=ranks
            t=t[t["WF流動性排名"].notna() & (t["WF流動性排名"]<=200)]
            total=len(t)
            for n in sizes:
                g=t[t["WF流動性排名"]<=n]
                capture_rows.append({
                    "核心池規模":f"TOP{n}","交易數":len(g),
                    "TOP200交易涵蓋率%":float(len(g)/total*100) if total else np.nan,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "平均淨報酬%":float(pd.to_numeric(g["淨報酬%"],errors="coerce").mean()) if len(g) else np.nan,
                })
            rows.append({"類別":"訊號涵蓋","指標":"TOP200範圍核心策略交易總數","數值":total})

    return pd.DataFrame(rows),pd.DataFrame(union_rows),pd.DataFrame(market_rows),pd.DataFrame(capture_rows),errors


def validate_top50_signal_grades(cost: CostConfig):
    """
    V1.16.12 訊號等級驗證：
    固定真正Walk-Forward TOP50 + 6mo暖機/末3mo評估 + 核心策略。
    不硬砍訊號，改成三層品質標籤：
      S級：量比20>=1.5 且 MA60未向上
      A級：兩者只符合一項
      B級：兩者皆不符合
    驗證重點：等級是否在全部/樣本外/四段時間呈現合理品質差異。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(elig.loc[elig["流動性排名"]<=50,"股票"].dropna().astype(str).unique().tolist())
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=_filter_historical_top50(trades,elig)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["量比Gate"]=vr>=1.5
    t["MA60Gate"]=ma60<=0

    def _grade(r):
        a=bool(r["量比Gate"]); b=bool(r["MA60Gate"])
        if a and b:
            return "S級_量比+MA60"
        if a or b:
            return "A級_符合一項"
        return "B級_基準"
    t["訊號等級"]=t.apply(_grade,axis=1)
    t["訊號時間_台北"]=_as_taipei_series(t["訊號時間"]).astype(str)
    t["指標資料期間"]="6mo"
    t["評估期間"]="最後3mo"

    grades=["S級_量比+MA60","A級_符合一項","B級_基準"]
    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for grade in grades:
            g=xs[xs["訊號等級"]==grade]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"訊號等級":grade,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),
                "交易占比%":float(len(g)/len(xs)*100) if len(xs) else np.nan,
                **m
            })
    summary=pd.DataFrame(rows)

    blocks=[]
    ts=_as_taipei_series(t["訊號時間"])
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for grade in grades:
                g=q[q["訊號等級"]==grade]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "訊號等級":grade,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),
                    "交易占比%":float(len(g)/len(q)*100) if len(q) else np.nan,
                    **m
                })
    return summary,pd.DataFrame(blocks),t,errors
def validate_top50_gate_decomposition(cost: CostConfig):
    """
    V1.16.11 Gate拆解驗證：
    固定真正Walk-Forward TOP50 + 6mo暖機/末3mo評估 + 核心策略。
    將兩個候選Gate拆成互斥四組，避免B/C/D重疊造成誤判：
      1) 都不符合
      2) 只有量比>=1.5
      3) 只有MA60未向上
      4) 兩者同時符合
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(elig.loc[elig["流動性排名"]<=50,"股票"].dropna().astype(str).unique().tolist())
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=_filter_historical_top50(trades,elig)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["量比Gate"]=vr>=1.5
    t["MA60Gate"]=ma60<=0

    def _grp(r):
        a=bool(r["量比Gate"]); b=bool(r["MA60Gate"])
        if a and b: return "D_量比>=1.5且MA60未向上"
        if a and not b: return "B_only_只有量比>=1.5"
        if (not a) and b: return "C_only_只有MA60未向上"
        return "N_兩者皆否"
    t["互斥Gate組別"]=t.apply(_grp,axis=1)
    t["訊號時間_台北"]=_as_taipei_series(t["訊號時間"]).astype(str)
    t["指標資料期間"]="6mo"
    t["評估期間"]="最後3mo"

    groups=[
        "N_兩者皆否",
        "B_only_只有量比>=1.5",
        "C_only_只有MA60未向上",
        "D_量比>=1.5且MA60未向上",
    ]

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for gname in groups:
            g=xs[xs["互斥Gate組別"]==gname]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"互斥Gate組別":gname,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    summary=pd.DataFrame(rows)

    blocks=[]
    ts=_as_taipei_series(t["訊號時間"])
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            for gname in groups:
                g=q[q["互斥Gate組別"]==gname]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "互斥Gate組別":gname,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })
    return summary,pd.DataFrame(blocks),t,errors
def validate_top50_candidate_gates(cost: CostConfig):
    """
    V1.16.9 候選Gate A/B：
    固定真正Walk-Forward TOP50 + 6mo暖機/末3mo評估 +
    60m KD黃金交叉/K<30/5日。
    只比較事先鎖定的三個候選：
      A 基準：不加Gate
      B 量比20>=1.5
      C MA60斜率3<=0（MA60未向上）
      D B+C 組合
    不加入時段或其他事後最佳化條件。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(elig.loc[elig["流動性排名"]<=50,"股票"].dropna().astype(str).unique().tolist())
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(
        union,cost,"6mo",allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=_filter_historical_top50(trades,elig)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    ma60=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["Gate_量比>=1.5"]=vr>=1.5
    t["Gate_MA60未向上"]=ma60<=0
    t["Gate_組合"]=(t["Gate_量比>=1.5"] & t["Gate_MA60未向上"])
    t["訊號時間_台北"]=_as_taipei_series(t["訊號時間"]).astype(str)
    t["指標資料期間"]="6mo"
    t["評估期間"]="最後3mo"

    gates=[
        ("A_基準",pd.Series(True,index=t.index)),
        ("B_量比>=1.5",t["Gate_量比>=1.5"]),
        ("C_MA60未向上",t["Gate_MA60未向上"]),
        ("D_量比>=1.5且MA60未向上",t["Gate_組合"]),
    ]

    rows=[]
    for sample in ["全部","樣本內60%","樣本外40%"]:
        xs=t if sample=="全部" else t[t["樣本"]==sample]
        for name,mask in gates:
            g=xs[mask.reindex(xs.index,fill_value=False)]
            m=aggregate_trade_metrics(g)
            rows.append({
                "樣本":sample,"Gate":name,
                "股票數":int(g["股票"].nunique()) if len(g) else 0,
                "交易數":len(g),**m
            })
    summary=pd.DataFrame(rows)

    blocks=[]
    ts=_as_taipei_series(t["訊號時間"])
    ok=ts.notna()
    z=t.loc[ok].copy(); ts=ts.loc[ok]
    if len(z):
        edges=pd.date_range(ts.min(),ts.max(),periods=5)
        labels=["第1段","第2段","第3段","第4段"]
        z["時間段"]=pd.cut(ts,bins=edges,labels=labels,include_lowest=True,right=True)
        for block in labels:
            q=z[z["時間段"]==block]
            local_gates=[
                ("A_基準",pd.Series(True,index=q.index)),
                ("B_量比>=1.5",q["Gate_量比>=1.5"]),
                ("C_MA60未向上",q["Gate_MA60未向上"]),
                ("D_量比>=1.5且MA60未向上",q["Gate_組合"]),
            ]
            for name,mask in local_gates:
                g=q[mask.reindex(q.index,fill_value=False)]
                m=aggregate_trade_metrics(g)
                blocks.append({
                    "時間段":block,
                    "起始":str(edges[labels.index(block)]),
                    "結束":str(edges[labels.index(block)+1]),
                    "Gate":name,
                    "股票數":int(g["股票"].nunique()) if len(g) else 0,
                    "交易數":len(g),**m
                })
    return summary,pd.DataFrame(blocks),t,errors
def validate_top50_signal_quality(cost: CostConfig, period: str="6mo"):
    """
    V1.16.7：
    固定真正Walk-Forward TOP50 + 60m KD黃金交叉/K<30/5日，
    使用6mo資料暖機、只評估最後3mo。
    訊號時間統一為台北時間。
    只診斷既有訊號品質：K深度、量比20、MA30/MA60方向、60m時段。
    """
    elig, _, errors = build_fullmarket_walkforward_eligibility(lookback_months=6, top_n=100)
    if elig is None or elig.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    union=sorted(elig.loc[elig["流動性排名"]<=50,"股票"].dropna().astype(str).unique().tolist())
    if not union:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    _,_,trades=run_oos_60m_5d(
        union,cost,period,allow_overlap=False,train_ratio=0.60,evaluation_months=3
    )
    if trades is None or trades.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    t=_filter_historical_top50(trades,elig)
    if t is None or t.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),errors

    # V1.16.8：將實際資料窗寫進輸出，避免UI覆蓋期間後不易察覺。
    t["指標資料期間"]="6mo"
    t["評估期間"]="最後3mo"

    _tw=_as_taipei_series(t["訊號時間"])
    t["訊號時間_台北"]=_tw.astype(str)
    t["訊號小時_台北"]=_tw.dt.hour

    t["K深度"]=pd.cut(
        pd.to_numeric(t["訊號K"],errors="coerce"),
        bins=[-np.inf,10,20,30],
        labels=["K<10","K10-20","K20-30"]
    ).astype(str)

    vr=pd.to_numeric(t.get("量比20"),errors="coerce")
    t["量比區間"]=pd.cut(
        vr,bins=[-np.inf,0.8,1.0,1.5,np.inf],
        labels=["<0.8","0.8-1.0","1.0-1.5",">=1.5"]
    ).astype(str)

    ma30=pd.to_numeric(t.get("MA30斜率3"),errors="coerce")
    ma60=pd.to_numeric(t.get("MA60斜率3"),errors="coerce")
    t["MA30方向"]=np.where(ma30>0,"向上","未向上")
    t["MA60方向"]=np.where(ma60>0,"向上","未向上")
    t["60m時段"]=t["訊號小時_台北"].map(
        lambda h:f"{int(h):02d}:00" if pd.notna(h) else "資料不足"
    )

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

    # 暖機資料品質監控：三項缺值率會直接出現在摘要底部。
    _quality_rows=[]
    for _col,_label in [
        ("量比20","量比20缺值率%"),
        ("MA30斜率3","MA30斜率缺值率%"),
        ("MA60斜率3","MA60斜率缺值率%")
    ]:
        _miss=float(t[_col].isna().mean()*100) if _col in t.columns and len(t) else np.nan
        _quality_rows.append({
            "樣本":"資料品質","診斷分類":"暖機完整度","分組":_label,
            "股票數":int(t["股票"].nunique()) if len(t) else 0,
            "交易數":len(t),
            "整體交易勝率":np.nan,
            "整體平均淨報酬":_miss,
            "整體PF":np.nan
        })
    summary=pd.concat([summary,pd.DataFrame(_quality_rows)],ignore_index=True)

    blocks=[]
    ts=_as_taipei_series(t["訊號時間"])
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
                    threads=False)
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
            threads=False,
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


@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
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
                threads=False,
                prepost=False,
                group_by="ticker",
            )
        except Exception:
            return pd.DataFrame()

    # 60m + 長期間時用較小批次，減少Yahoo整批失敗。
    batch_size=15 if interval=="60m" and period in ["6mo","1y","2y","5y","10y","max"] else 30

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
st.caption(f"{APP_VERSION}｜全市場動態雷達｜TOP1-100核心/觀察 + TOP101-150 S級擴充｜60m KD黃金交叉 + K<30")
st.info("正式股票池已不再使用人工母池：每天由官方上市/上櫃全市場，以前20個完成交易日流動性動態建立TOP150，再依驗證後架構顯示訊號。")

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
                ["長期集中度健診","長期穩健度驗證","延遲TimeStop驗證","早期路徑健診","固定停損驗證","獲利保護風險效益","獲利保護敏感度","獲利保護驗證","持有天數驗證","持有路徑健診","市場廣度轉折健診","環境×訊號交互驗證","環境Gate驗證","失效環境健診","雷達架構驗證","股票池分層驗證","股票池覆蓋健診","TOP50訊號等級驗證","TOP50Gate拆解驗證","TOP50候選Gate驗證","TOP50訊號品質健診","TOP50暖機修正驗證","市場環境健診_TOP50","核心池規模WalkForward","全市場WalkForward驗證","全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診","單一股票","跨股票批次","多週期當沖/隔日驗證","60m五日OOS驗證"], index=0)
            if research_mode == "長期集中度健診":
                st.caption("沿用1年長期樣本，檢查正期望是否被少數月份、少數股票或極端大賺單支撐；不改任何策略規則。")
            elif research_mode == "長期穩健度驗證":
                st.caption("停止微調出場參數；用1y 60m＋12mo歷史流動性，正式檢查核心5日策略在最後9mo是否跨時間穩健。")
            elif research_mode == "延遲TimeStop驗證":
                st.caption("固定同一批進場，比較第2/3日收盤弱勢才提早出場；避免盤中固定停損把會反彈的交易洗掉。")
            elif research_mode == "早期路徑健診":
                st.caption("固定5日基準，檢查第1/2/3日收盤的早期報酬是否能辨識最後會失敗的交易；只診斷、不直接加time-stop。")
            elif research_mode == "固定停損驗證":
                st.caption("獲利保護無法處理從未反彈的虧損單；固定同一批進場比較-5/-7.5/-10%停損，直接檢查左尾風險。")
            elif research_mode == "獲利保護風險效益":
                st.caption("用逐筆paired比較檢查獲利保護降低多少左尾風險、又犧牲多少大波段；不再只看PF。")
            elif research_mode == "獲利保護敏感度":
                st.caption("不追單一最佳值；用同一批進場比較觸發+4/+5/+6 × 保護+1/+2/+3，檢查獲利保護是否具鄰近參數穩健性。")
            elif research_mode == "獲利保護驗證":
                st.caption("固定同一批正式架構C進場，比較5日基準、4日固定出場，以及曾達+5%後的保本/+2/+3獲利保護。")
            elif research_mode == "持有天數驗證":
                st.caption("固定同一批正式架構C進場，只比較2/3/4/5日出場，避免短持有因多出新訊號造成不公平比較。")
            elif research_mode == "持有路徑健診":
                st.caption("固定5日持有規則，檢查MFE/MAE與回吐型態，判斷第1段問題較像進場失效還是持有過久。")
            elif research_mode == "市場廣度轉折健診":
                st.caption("不只看MA60廣度高低，進一步檢查MA15/30/60市場廣度5日變化，區分高檔擴散與高檔收斂。")
            elif research_mode == "環境×訊號交互驗證":
                st.caption("先用1y日K完整暖機TOP200市場MA60廣度，再檢查高檔風險下S/A/B差異；避免早期MA60缺值被誤判為正常環境。")
            elif research_mode == "環境Gate驗證":
                st.caption("固定正式架構C，比較幾個事先鎖定的環境排除條件；重點看第1段改善、樣本外與後3段代價。")
            elif research_mode == "失效環境健診":
                st.caption("專門檢查第1段共同失效：TOP200站上MA15/30/60比例、5日市場廣度、60m訊號擁擠度；只診斷、不先加Gate。")
            elif research_mode == "TOP50暖機修正驗證":
                st.caption("比較舊3mo直接計算 vs 6mo指標暖機後只評估最後3mo；同時修正訊號時間誤當UTC的問題。")
            elif research_mode == "TOP50訊號品質健診":
                st.caption("固定真正Walk-Forward TOP50與核心策略；使用6mo暖機、只評估最後3mo，診斷K深度、量比20、MA30/60方向與台北時間60m時段。")
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
            if research_mode not in ["長期集中度健診","長期穩健度驗證","延遲TimeStop驗證","早期路徑健診","固定停損驗證","獲利保護風險效益","獲利保護敏感度","獲利保護驗證","持有天數驗證","持有路徑健診","市場廣度轉折健診","環境×訊號交互驗證","環境Gate驗證","失效環境健診","雷達架構驗證","股票池分層驗證","股票池覆蓋健診","TOP50訊號等級驗證","TOP50Gate拆解驗證","TOP50候選Gate驗證","TOP50暖機修正驗證","TOP50訊號品質健診","市場環境健診_TOP50","核心池規模WalkForward驗證","全市場WalkForward驗證","全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診"]:
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
            "長期集中度健診":"🧮 執行長期集中度健診",
            "長期穩健度驗證":"🧱 執行1年長期穩健度驗證",
            "延遲TimeStop驗證":"⏳ 執行延遲Time-Stop A/B",
            "早期路徑健診":"🩺 執行早期路徑健診",
            "固定停損驗證":"🧯 執行固定停損A/B",
            "獲利保護風險效益":"⚖️ 執行獲利保護風險效益",
            "獲利保護敏感度":"🧪 執行獲利保護敏感度",
            "獲利保護驗證":"🛡️ 執行獲利保護A/B",
            "持有天數驗證":"⏱️ 執行2/3/4/5日持有A/B",
            "持有路徑健診":"🧭 執行持有路徑健診",
            "市場廣度轉折健診":"📉 執行市場廣度轉折健診",
            "環境×訊號交互驗證":"🧩 執行環境×訊號交互驗證",
            "環境Gate驗證":"🧪 執行環境Gate A/B",
            "失效環境健診":"🌦️ 執行第一段失效環境健診",
            "雷達架構驗證":"🛰️ 驗證正式雷達架構",
            "股票池分層驗證":"🧱 執行股票池分層驗證",
            "股票池覆蓋健診":"🌐 執行股票池覆蓋健診",
            "TOP50訊號等級驗證":"🏷️ 驗證TOP50訊號S/A/B等級",
            "TOP50Gate拆解驗證":"🧬 執行TOP50 Gate拆解",
            "TOP50候選Gate驗證":"🧪 執行TOP50候選Gate A/B",
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
    st.warning("V1.16.6 已從完整 V1.16.0 基底重建，修復後續版本編修時誤刪的核心函式。保留台北時區、6mo暖機、Yahoo分批重試與Streamlit Cloud資源限制；這版先驗證資料完整性，不新增策略條件。")

if run and simple_mode=="進階研究" and research_mode=="TOP50暖機修正驗證":
    st.subheader("🧰 TOP50暖機修正驗證")
    with st.spinner("同時跑舊3mo版本與6mo暖機版本，比較最後3個月結果…"):
        _wu_sum,_wu_blocks,_wu_trades,_wu_errs,_wu_check=validate_top50_warmup_correction(cost)
    st.session_state["st_v1166_warmup"]={
        "summary":_wu_sum,"blocks":_wu_blocks,"trades":_wu_trades,
        "check":_wu_check,"errors":_wu_errs
    }

_wu=st.session_state.get("st_v1166_warmup")
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

if simple_mode=="進階研究" and research_mode=="長期集中度健診" and not run:
    st.info("V1.16.32 的9個月正式架構C共有1510筆交易，樣本內/外都維持約+2%平均報酬，6個時間區塊也全部為正。這版進一步檢查報酬是不是其實集中在少數月份或少數股票。")

if run and simple_mode=="進階研究" and research_mode=="長期集中度健診":
    st.subheader("🧮 長期報酬分布 / 集中度健診")
    with st.spinner("沿用1y長期正式架構C，計算月度穩定度、股票貢獻與Top貢獻移除測試…"):
        _lc_month,_lc_stock,_lc_infl,_lc_dist,_lc_errs=validate_long_horizon_concentration(cost)
    st.session_state["st_v11633_long_concentration"]={
        "monthly":_lc_month,"stocks":_lc_stock,"influence":_lc_infl,
        "distribution":_lc_dist,"errors":_lc_errs
    }

_lc=st.session_state.get("st_v11633_long_concentration")
if simple_mode=="進階研究" and research_mode=="長期集中度健診" and _lc:
    _lm=_lc.get("monthly",pd.DataFrame()); _ls=_lc.get("stocks",pd.DataFrame())
    _li=_lc.get("influence",pd.DataFrame()); _ld=_lc.get("distribution",pd.DataFrame())
    _le=_lc.get("errors",[])
    st.subheader("🧮 長期集中度結果")
    if _le:
        st.warning("資料來源異常："+"；".join(_le))
    st.info("這版不找新參數；重點是確認長期+2%平均報酬是否具有廣度，而不是被少數大賺股票或少數月份拉高。")
    if not _ld.empty:
        st.markdown("#### 全體分布摘要")
        st.dataframe(_ld.round(3),use_container_width=True,hide_index=True)
    if not _li.empty:
        st.markdown("#### 移除Top正貢獻股票後")
        st.dataframe(_li.round(3),use_container_width=True,hide_index=True)
    if not _lm.empty:
        st.markdown("#### 月度穩定度")
        st.dataframe(_lm.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看各股票長期貢獻"):
        st.dataframe(_ls.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【長期分布摘要】",_ld.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_長期分布摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【長期月度穩定度】",_lm.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_長期月度穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【長期股票貢獻】",_ls.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_長期股票貢獻.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【長期Top貢獻移除測試】",_li.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_長期Top貢獻移除測試.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="長期穩健度驗證" and not run:
    st.info("V1.16.31 顯示延遲Time-Stop仍沒有真正解決第1段，而且大多數方案整體報酬與PF都低於原5日基準。繼續微調-3/-5或第2/3日容易過度擬合，因此這版先停止出場調參，改用更長歷史重新驗證核心策略。")

if run and simple_mode=="進階研究" and research_mode=="長期穩健度驗證":
    st.subheader("🧱 1年長期穩健度驗證")
    st.warning("這個模式會下載較多1y 60m資料，第一次執行可能比前面研究久；建議不要同時開多個分頁重跑。")
    with st.spinner("建立12mo歷史流動性資格＋下載1y 60m，評估最後9mo正式架構C…"):
        _lr_sum,_lr_blocks,_lr_detail,_lr_errs=validate_long_horizon_robustness(cost)
    st.session_state["st_v11632_long_robust"]={
        "summary":_lr_sum,"blocks":_lr_blocks,"detail":_lr_detail,"errors":_lr_errs
    }

_lr=st.session_state.get("st_v11632_long_robust")
if simple_mode=="進階研究" and research_mode=="長期穩健度驗證" and _lr:
    _ls=_lr.get("summary",pd.DataFrame()); _lb=_lr.get("blocks",pd.DataFrame())
    _ld=_lr.get("detail",pd.DataFrame()); _le=_lr.get("errors",[])
    st.subheader("🧱 長期穩健度結果")
    if _le:
        st.warning("資料來源異常："+"；".join(_le))
    st.info("這一版不比較新Gate與新停損，只驗證固定正式架構C＋5日持有在更長時間是否仍成立。")
    if not _ls.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ls.round(3),use_container_width=True,hide_index=True)
    if not _lb.empty:
        st.markdown("#### 9個月評估區｜6段時間穩定度")
        st.dataframe(_lb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看長期逐筆交易"):
        st.dataframe(_ld.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【長期穩健度摘要】",_ls.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_長期穩健度摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【長期六段穩定度】",_lb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_長期六段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【長期逐筆交易】",_ld.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_長期逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="延遲TimeStop驗證" and not run:
    st.info("V1.16.30 顯示第2日<=-3%後最終翻正率僅23.6%，第3日<=-3%更降到14.7%；第1段更差。這版直接驗證『等到第2/3日收盤仍弱才退出』是否比盤中固定停損更有效。")

if run and simple_mode=="進階研究" and research_mode=="延遲TimeStop驗證":
    st.subheader("⏳ 延遲 Time-Stop A/B")
    with st.spinner("沿用同一批正式架構C進場，重算第2/3日弱勢出場方案…"):
        _ts_sum,_ts_blocks,_ts_pair,_ts_detail,_ts_errs=validate_delayed_timestop_candidates(cost)
    st.session_state["st_v11631_timestop"]={
        "summary":_ts_sum,"blocks":_ts_blocks,"paired":_ts_pair,
        "detail":_ts_detail,"errors":_ts_errs
    }

_ts=st.session_state.get("st_v11631_timestop")
if simple_mode=="進階研究" and research_mode=="延遲TimeStop驗證" and _ts:
    _ss=_ts.get("summary",pd.DataFrame()); _sb=_ts.get("blocks",pd.DataFrame())
    _sp=_ts.get("paired",pd.DataFrame()); _sd=_ts.get("detail",pd.DataFrame())
    _se=_ts.get("errors",[])
    st.subheader("⏳ 延遲 Time-Stop 驗證結果")
    if _se:
        st.warning("資料來源異常："+"；".join(_se))
    st.info("所有提早出場都只在第2/3個交易日完成收盤後判斷；不使用盤中Low。這版仍只研究，不會自動改今日正式雷達。")
    if not _sp.empty:
        st.markdown("#### 逐筆 paired｜相對5日基準")
        st.dataframe(_sp.round(3),use_container_width=True,hide_index=True)
    if not _ss.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ss.round(3),use_container_width=True,hide_index=True)
    if not _sb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_sb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆Time-Stop模擬"):
        st.dataframe(_sd.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【延遲TimeStop驗證摘要】",_ss.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_延遲TimeStop驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【延遲TimeStop四段穩定度】",_sb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_延遲TimeStop四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【延遲TimeStopPaired比較】",_sp.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_延遲TimeStopPaired比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【延遲TimeStop逐筆比較】",_sd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_延遲TimeStop逐筆比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="早期路徑健診" and not run:
    st.info("V1.16.29 顯示固定停損雖能大幅壓低P5/P10/CVaR，但會砍掉太多後來能反彈的交易：-5%停損觸發約55%，整體平均從+2.12%降到+0.11%。這版先找『幾天後仍弱』是否比盤中直接停損更有辨識力。")

if run and simple_mode=="進階研究" and research_mode=="早期路徑健診":
    st.subheader("🩺 早期路徑健診")
    with st.spinner("固定同一批正式架構C進場，計算第1/2/3日收盤與最終5日結果…"):
        _ep_bucket,_ep_compare,_ep_detail,_ep_errs=validate_early_path_diagnostics(cost)
    st.session_state["st_v11630_early_path"]={
        "bucket":_ep_bucket,"compare":_ep_compare,"detail":_ep_detail,"errors":_ep_errs
    }

_ep=st.session_state.get("st_v11630_early_path")
if simple_mode=="進階研究" and research_mode=="早期路徑健診" and _ep:
    _eb=_ep.get("bucket",pd.DataFrame()); _ec=_ep.get("compare",pd.DataFrame())
    _ed=_ep.get("detail",pd.DataFrame()); _ee=_ep.get("errors",[])
    st.subheader("🩺 早期路徑健診結果")
    if _ee:
        st.warning("資料來源異常："+"；".join(_ee))
    st.info("這版只診斷：若第2或第3日仍明顯低於進場價，最後翻正率是否已經很低。確認後下一版才會測延遲time-stop。")
    if not _eb.empty:
        st.markdown("#### 第1/2/3日收盤分桶 → 最終5日結果")
        st.dataframe(_eb.round(3),use_container_width=True,hide_index=True)
    if not _ec.empty:
        st.markdown("#### 第1段 vs 第2~4段｜早期弱勢條件")
        st.dataframe(_ec.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆第1/2/3/5日報酬"):
        st.dataframe(_ed.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【早期路徑分桶】",_eb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_早期路徑分桶.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【早期弱勢區段比較】",_ec.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_早期弱勢區段比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【早期路徑逐筆明細】",_ed.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_早期路徑逐筆明細.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="固定停損驗證" and not run:
    st.info("V1.16.28 確認獲利保護主要作用在『曾經先漲』的交易，P5/P10/CVaR雖有改善，但最差單筆完全沒變；真正一路偏弱的交易根本不會啟動獲利保護。這版改直接測-5/-7.5/-10%固定停損。")

if run and simple_mode=="進階研究" and research_mode=="固定停損驗證":
    st.subheader("🧯 固定停損A/B")
    with st.spinner("固定同一批正式架構C進場，重算-5/-7.5/-10%停損與5日基準…"):
        _sl_sum,_sl_blocks,_sl_pair,_sl_detail,_sl_errs=validate_stoploss_candidates(cost)
    st.session_state["st_v11629_stoploss"]={
        "summary":_sl_sum,"blocks":_sl_blocks,"paired":_sl_pair,
        "detail":_sl_detail,"errors":_sl_errs
    }

_sl=st.session_state.get("st_v11629_stoploss")
if simple_mode=="進階研究" and research_mode=="固定停損驗證" and _sl:
    _ss=_sl.get("summary",pd.DataFrame()); _sb=_sl.get("blocks",pd.DataFrame())
    _sp=_sl.get("paired",pd.DataFrame()); _sd=_sl.get("detail",pd.DataFrame())
    _se=_sl.get("errors",[])
    st.subheader("🧯 固定停損驗證結果")
    if _se:
        st.warning("資料來源異常："+"；".join(_se))
    st.info("停損從進場後立即有效；後續若跳空跌破停損價，以實際開盤價出場。這版只測虧損保護，不與獲利保護混在一起。")
    if not _sp.empty:
        st.markdown("#### 逐筆 paired｜相對5日基準")
        st.dataframe(_sp.round(3),use_container_width=True,hide_index=True)
    if not _ss.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ss.round(3),use_container_width=True,hide_index=True)
    if not _sb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_sb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆停損模擬"):
        st.dataframe(_sd.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【固定停損驗證摘要】",_ss.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_固定停損驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【固定停損四段穩定度】",_sb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_固定停損四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【固定停損逐筆比較】",_sd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_固定停損逐筆比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【固定停損Paired比較】",_sp.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_固定停損Paired比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="獲利保護風險效益" and not run:
    st.info("V1.16.27 顯示9組附近參數幾乎都能提高PF與勝率，但樣本外平均報酬多數略低於5日基準。這代表獲利保護更像『降低左尾、犧牲部分大波段』，這版用逐筆paired差異與尾端風險正式量化。")

if run and simple_mode=="進階研究" and research_mode=="獲利保護風險效益":
    st.subheader("⚖️ 獲利保護風險效益")
    with st.spinner("重建同一批敏感度結果，計算逐筆Δ報酬與左尾風險…"):
        _rb_sum,_rb_blocks,_rb_focus,_rb_errs=validate_profit_protection_risk_benefit(cost)
    st.session_state["st_v11628_risk_benefit"]={
        "summary":_rb_sum,"blocks":_rb_blocks,"focus":_rb_focus,"errors":_rb_errs
    }

_rb=st.session_state.get("st_v11628_risk_benefit")
if simple_mode=="進階研究" and research_mode=="獲利保護風險效益" and _rb:
    _rs=_rb.get("summary",pd.DataFrame()); _rblk=_rb.get("blocks",pd.DataFrame())
    _rf=_rb.get("focus",pd.DataFrame()); _re=_rb.get("errors",[])
    st.subheader("⚖️ 獲利保護風險效益結果")
    if _re:
        st.warning("資料來源異常："+"；".join(_re))
    st.info("這版不以最高PF定參數；重點看paired平均Δ、左尾P5/P10/CVaR10、單筆大幅改善與大幅惡化比例。")
    if not _rf.empty:
        st.markdown("#### 五個代表候選｜逐筆相對5日基準")
        st.dataframe(_rf.round(3),use_container_width=True,hide_index=True)
    if not _rs.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外｜風險效益")
        st.dataframe(_rs.round(3),use_container_width=True,hide_index=True)
    if not _rblk.empty:
        st.markdown("#### 四段paired穩定度")
        st.dataframe(_rblk.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【獲利保護風險效益摘要】",_rs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護風險效益摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【獲利保護風險效益四段】",_rblk.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護風險效益四段.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【獲利保護代表候選比較】",_rf.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護代表候選比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="獲利保護敏感度" and not run:
    st.info("V1.16.26 顯示+5%後保+2/+3都比5日基準穩健，且樣本外以保+2稍佳。這版不直接定案，而是檢查附近參數：觸發+4/+5/+6 × 保護+1/+2/+3，確認優勢是不是一整片而不是單點。")

if run and simple_mode=="進階研究" and research_mode=="獲利保護敏感度":
    st.subheader("🧪 獲利保護敏感度")
    with st.spinner("固定同一批正式架構C進場，計算9組附近參數＋5日基準…"):
        _ps_sum,_ps_blocks,_ps_detail,_ps_errs=validate_profit_protection_sensitivity(cost)
    st.session_state["st_v11627_profit_sensitivity"]={
        "summary":_ps_sum,"blocks":_ps_blocks,"detail":_ps_detail,"errors":_ps_errs
    }

_ps=st.session_state.get("st_v11627_profit_sensitivity")
if simple_mode=="進階研究" and research_mode=="獲利保護敏感度" and _ps:
    _ss=_ps.get("summary",pd.DataFrame()); _sb=_ps.get("blocks",pd.DataFrame())
    _sd=_ps.get("detail",pd.DataFrame()); _se=_ps.get("errors",[])
    st.subheader("🧪 獲利保護敏感度結果")
    if _se:
        st.warning("資料來源異常："+"；".join(_se))
    st.info("這版看『鄰近參數是否一起有效』，不是挑最高PF。若只有單一格漂亮、周邊不穩，就視為可能過度擬合。")
    if not _ss.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ss.round(3),use_container_width=True,hide_index=True)
    if not _sb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_sb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆9組參數比較"):
        st.dataframe(_sd.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【獲利保護敏感度摘要】",_ss.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護敏感度摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【獲利保護敏感度四段】",_sb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護敏感度四段.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【獲利保護敏感度逐筆】",_sd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護敏感度逐筆.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="獲利保護驗證" and not run:
    st.info("V1.16.25 顯示4日整體略優於5日、且第1段明顯改善，但樣本外仍是5日最好，因此不直接把持有期縮成4日。這版測試『5日保留趨勢空間＋曾達+5%後才啟動獲利保護』。")

if run and simple_mode=="進階研究" and research_mode=="獲利保護驗證":
    st.subheader("🛡️ 獲利保護A/B")
    with st.spinner("固定同一批正式架構C進場，重算5日基準、4日與+5%後獲利保護…"):
        _pp_sum,_pp_blocks,_pp_detail,_pp_errs=validate_profit_protection_candidates(cost)
    st.session_state["st_v11626_profit_protect"]={
        "summary":_pp_sum,"blocks":_pp_blocks,"detail":_pp_detail,"errors":_pp_errs
    }

_pp=st.session_state.get("st_v11626_profit_protect")
if simple_mode=="進階研究" and research_mode=="獲利保護驗證" and _pp:
    _ps=_pp.get("summary",pd.DataFrame()); _pb=_pp.get("blocks",pd.DataFrame())
    _pd2=_pp.get("detail",pd.DataFrame()); _pe2=_pp.get("errors",[])
    st.subheader("🛡️ 獲利保護驗證結果")
    if _pe2:
        st.warning("資料來源異常："+"；".join(_pe2))
    st.info("保護規則採保守執行：某根完成60m K曾碰到+5%，保護從下一根K才啟動；若跳空跌破保護價，以實際開盤價出場。")
    if not _ps.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ps.round(3),use_container_width=True,hide_index=True)
    if not _pb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_pb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆獲利保護模擬"):
        st.dataframe(_pd2.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【獲利保護驗證摘要】",_ps.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【獲利保護四段穩定度】",_pb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【獲利保護逐筆比較】",_pd2.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_獲利保護逐筆比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="持有天數驗證" and not run:
    st.info("V1.16.24 顯示第1段有兩種問題同時存在：48.3%一路偏弱/未達+5%，但也有20.8%曾到+5%最後變虧。這版先做最乾淨的持有天數A/B：同一批進場只改成2/3/4/5日出場。")

if run and simple_mode=="進階研究" and research_mode=="持有天數驗證":
    st.subheader("⏱️ 2/3/4/5日持有A/B")
    with st.spinner("固定同一批正式架構C進場，重新計算不同持有天數出場…"):
        _hd_sum,_hd_blocks,_hd_detail,_hd_errs=validate_holding_period_candidates(cost)
    st.session_state["st_v11625_holding_days"]={
        "summary":_hd_sum,"blocks":_hd_blocks,"detail":_hd_detail,"errors":_hd_errs
    }

_hd=st.session_state.get("st_v11625_holding_days")
if simple_mode=="進階研究" and research_mode=="持有天數驗證" and _hd:
    _ds=_hd.get("summary",pd.DataFrame()); _db=_hd.get("blocks",pd.DataFrame())
    _dd=_hd.get("detail",pd.DataFrame()); _de=_hd.get("errors",[])
    st.subheader("⏱️ 持有天數驗證結果")
    if _de:
        st.warning("資料來源異常："+"；".join(_de))
    st.info("四個方案使用完全相同的股票、訊號與進場時間，只改出場天數。這比直接各自重跑短持有策略更能回答『5日是不是太久』。")
    if not _ds.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ds.round(3),use_container_width=True,hide_index=True)
    if not _db.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_db.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看同一批交易的2/3/4/5日重算明細"):
        st.dataframe(_dd.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【持有天數驗證摘要】",_ds.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_持有天數驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【持有天數四段穩定度】",_db.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_持有天數四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【持有天數逐筆比較】",_dd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_持有天數逐筆比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="持有路徑健診" and not run:
    st.info("V1.16.23 顯示『高檔收斂』雖然持續偏弱，但第1段其實不論高檔擴散/收斂都差，代表單靠市場狀態仍無法解決。這版轉而檢查5日持有路徑：訊號有沒有先反彈再把獲利吐回去。")

if run and simple_mode=="進階研究" and research_mode=="持有路徑健診":
    st.subheader("🧭 持有路徑健診")
    with st.spinner("沿用正式架構C逐筆交易，分析MFE/MAE與5日持有回吐…"):
        _hp_sum,_hp_paths,_hp_detail,_hp_errs=validate_holding_path_diagnostics(cost)
    st.session_state["st_v11624_holding_path"]={
        "summary":_hp_sum,"paths":_hp_paths,"detail":_hp_detail,"errors":_hp_errs
    }

_hp=st.session_state.get("st_v11624_holding_path")
if simple_mode=="進階研究" and research_mode=="持有路徑健診" and _hp:
    _hs=_hp.get("summary",pd.DataFrame()); _hpaths=_hp.get("paths",pd.DataFrame())
    _hd=_hp.get("detail",pd.DataFrame()); _he=_hp.get("errors",[])
    st.subheader("🧭 持有路徑健診結果")
    if _he:
        st.warning("資料來源異常："+"；".join(_he))
    st.info("這版只判斷『進場錯』還是『持有過久』；+3/+5/+10與-3/-5/-10都是固定描述門檻，不會直接當成新版停利停損。")
    if not _hs.empty:
        st.markdown("#### 四段 MFE / MAE / 回吐比較")
        st.dataframe(_hs.round(3),use_container_width=True,hide_index=True)
    if not _hpaths.empty:
        st.markdown("#### 四段持有路徑類型")
        st.dataframe(_hpaths.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆MFE/MAE與路徑標記"):
        st.dataframe(_hd.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【持有路徑健診摘要】",_hs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_持有路徑健診摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【持有路徑類型分析】",_hpaths.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_持有路徑類型分析.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【持有路徑逐筆明細】",_hd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_持有路徑逐筆明細.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="市場廣度轉折健診" and not run:
    st.info("今天的MA60廣度77%被標成高檔風險，但同時5日上漲家數89%、5日報酬中位數+5.14%，其實是強勢擴散而不是明顯轉弱。這版改檢查『廣度方向』，避免只靠MA60高低把強勢市場和高檔轉弱混在一起。")

if run and simple_mode=="進階研究" and research_mode=="市場廣度轉折健診":
    st.subheader("📉 市場廣度轉折健診")
    with st.spinner("重建TOP200市場廣度時間序列並計算3/5日變化…"):
        _bt_sum,_bt_buckets,_bt_blocks,_bt_detail,_bt_errs=validate_breadth_transition(cost)
    st.session_state["st_v11623_breadth_transition"]={
        "summary":_bt_sum,"buckets":_bt_buckets,"blocks":_bt_blocks,
        "detail":_bt_detail,"errors":_bt_errs
    }

_bt=st.session_state.get("st_v11623_breadth_transition")
if simple_mode=="進階研究" and research_mode=="市場廣度轉折健診" and _bt:
    _bs=_bt.get("summary",pd.DataFrame()); _bb=_bt.get("buckets",pd.DataFrame())
    _bk=_bt.get("blocks",pd.DataFrame()); _bd=_bt.get("detail",pd.DataFrame())
    _be=_bt.get("errors",[])
    st.subheader("📉 市場廣度轉折健診結果")
    if _be:
        st.warning("資料來源異常："+"；".join(_be))
    st.info("四種狀態：高檔擴散／高檔收斂／非高檔擴散／非高檔收斂。此版只診斷，不直接修改今日雷達。")
    if not _bs.empty:
        st.markdown("#### 轉折狀態｜全部 / 樣本內 / 樣本外")
        st.dataframe(_bs.round(3),use_container_width=True,hide_index=True)
    if not _bb.empty:
        st.markdown("#### MA15/30/60與5日上漲家數｜5日變化固定分桶")
        st.dataframe(_bb.round(3),use_container_width=True,hide_index=True)
    if not _bk.empty:
        st.markdown("#### 四段時間 × 轉折狀態")
        st.dataframe(_bk.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆交易＋市場廣度轉折"):
        st.dataframe(_bd.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【市場廣度轉折摘要】",_bs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_市場廣度轉折摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【市場廣度變化分桶】",_bb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_市場廣度變化分桶.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【市場廣度轉折四段】",_bk.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_市場廣度轉折四段.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【市場廣度轉折逐筆】",_bd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_市場廣度轉折逐筆.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="環境×訊號交互驗證" and not run:
    st.info("V1.16.20 發現評估最前端仍有少量市場MA60廣度缺值，且舊邏輯會把缺值誤歸類成正常環境。V1.16.21 改用1y日K暖機市場廣度，並將缺值明確標成資料不足後再重跑交互驗證。")

if run and simple_mode=="進階研究" and research_mode=="環境×訊號交互驗證":
    st.subheader("🧩 環境 × 訊號等級交互驗證")
    with st.spinner("沿用point-in-time正式架構C交易，拆解高檔風險環境中的S/A/B訊號…"):
        _ix_sum,_ix_blocks,_ix_policies,_ix_detail,_ix_errs=validate_environment_signal_interaction(cost)
    st.session_state["st_v11620_env_signal"]={
        "summary":_ix_sum,"blocks":_ix_blocks,"policies":_ix_policies,
        "detail":_ix_detail,"errors":_ix_errs
    }

_ix=st.session_state.get("st_v11620_env_signal")
if simple_mode=="進階研究" and research_mode=="環境×訊號交互驗證" and _ix:
    _is=_ix.get("summary",pd.DataFrame()); _ib=_ix.get("blocks",pd.DataFrame())
    _ip=_ix.get("policies",pd.DataFrame()); _id=_ix.get("detail",pd.DataFrame())
    _ie=_ix.get("errors",[])
    st.subheader("🧩 環境 × 訊號交互驗證結果")
    if _ie:
        st.warning("資料來源異常："+"；".join(_ie))
    st.info("重點不是找最高PF，而是判斷高檔風險是否應全部封鎖，或只需要保留S級、降權A/B級。")
    if not _is.empty:
        st.markdown("#### 市場風險 × S/A/B｜全部 / 樣本內 / 樣本外")
        st.dataframe(_is.round(3),use_container_width=True,hide_index=True)
    if not _ip.empty:
        st.markdown("#### 三種可落地政策比較")
        st.dataframe(_ip.round(3),use_container_width=True,hide_index=True)
        if "市場廣度缺值筆數" in _ip.columns:
            _miss=int(pd.to_numeric(_ip["市場廣度缺值筆數"],errors="coerce").fillna(0).max())
            if _miss==0:
                st.success("✅ 市場MA60廣度缺值：0筆")
            else:
                st.warning(f"⚠️ 市場MA60廣度仍有缺值：{_miss}筆；這些交易不納入政策比較。")
    if not _ib.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_ib.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆交易＋市場風險與訊號等級"):
        st.dataframe(_id.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【環境訊號交互摘要】",_is.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_環境訊號交互摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【環境訊號政策比較】",_ip.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_環境訊號政策比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【環境訊號四段穩定度】",_ib.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_環境訊號四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【環境訊號逐筆明細】",_id.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_環境訊號逐筆明細.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="環境Gate驗證" and not run:
    st.info("V1.16.18 顯示第1段不是單純弱勢市場：最明顯的是『MA60長期廣度仍高，但短期5日已轉弱』，以及6~10檔同時KD反彈的中度擁擠區。這版只做候選A/B，不直接改今日雷達。")

if run and simple_mode=="進階研究" and research_mode=="環境Gate驗證":
    st.subheader("🧪 環境Gate候選A/B")
    with st.spinner("沿用V1.16.18 point-in-time資料，驗證環境排除條件…"):
        _eg_sum,_eg_blocks,_eg_detail,_eg_errs=validate_environment_gate_candidates(cost)
    st.session_state["st_v11619_env_gate"]={
        "summary":_eg_sum,"blocks":_eg_blocks,"detail":_eg_detail,"errors":_eg_errs
    }

_eg=st.session_state.get("st_v11619_env_gate")
if simple_mode=="進階研究" and research_mode=="環境Gate驗證" and _eg:
    _es=_eg.get("summary",pd.DataFrame()); _eb=_eg.get("blocks",pd.DataFrame())
    _ed=_eg.get("detail",pd.DataFrame()); _ee=_eg.get("errors",[])
    st.subheader("🧪 環境Gate驗證結果")
    if _ee:
        st.warning("資料來源異常："+"；".join(_ee))
    st.info("評估重點：第1段是否改善、後3段是否被錯殺、樣本外是否仍保有優勢。單一最高PF不等於可直接採用。")
    if not _es.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_es.round(3),use_container_width=True,hide_index=True)
    if not _eb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_eb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆交易與各Gate排除標記"):
        st.dataframe(_ed.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【環境Gate驗證摘要】",_es.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_環境Gate驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【環境Gate四段穩定度】",_eb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_環境Gate四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【環境Gate逐筆明細】",_ed.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_環境Gate逐筆明細.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="失效環境健診" and not run:
    st.info("前面已排除暖機不足、股票池寬窄與量比/MA60本身。這次改查第1段是否是『整體市場廣度不利＋KD反彈訊號同時擁擠』造成。")

if run and simple_mode=="進階研究" and research_mode=="失效環境健診":
    st.subheader("🌦️ 第一段失效環境健診")
    with st.spinner("建立歷史TOP200市場廣度並重建正式架構C交易…"):
        _fe_blocks,_fe_diag,_fe_trades,_fe_errs=validate_failure_environment(cost)
    st.session_state["st_v11618_failure_env"]={
        "blocks":_fe_blocks,"diagnostics":_fe_diag,"trades":_fe_trades,"errors":_fe_errs
    }

_fe=st.session_state.get("st_v11618_failure_env")
if simple_mode=="進階研究" and research_mode=="失效環境健診" and _fe:
    _fb=_fe.get("blocks",pd.DataFrame()); _fd=_fe.get("diagnostics",pd.DataFrame())
    _ft=_fe.get("trades",pd.DataFrame()); _ferr=_fe.get("errors",[])
    st.subheader("🌦️ 第一段失效環境結果")
    if _ferr:
        st.warning("資料來源異常："+"；".join(_ferr))
    st.info("這一版只做原因診斷；不會因某一環境分組績效漂亮就直接加進正式Gate。")
    if not _fb.empty:
        st.markdown("#### 四段市場環境 vs 策略績效")
        st.dataframe(_fb.round(3),use_container_width=True,hide_index=True)
    if not _fd.empty:
        st.markdown("#### 固定環境分桶")
        st.dataframe(_fd.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看逐筆交易與訊號當下市場環境"):
        st.dataframe(_ft.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【失效環境四段比較】",_fb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_失效環境四段比較.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【失效環境因子分桶】",_fd.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_失效環境因子分桶.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【失效環境逐筆交易】",_ft.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_失效環境逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="雷達架構驗證" and not run:
    st.info("V1.16.14 顯示TOP51-100仍有穩定正向價值、TOP101-150品質較弱但S級仍強、TOP151-200整體已接近無優勢。這版直接驗證幾種可落地的正式雷達架構。")

if run and simple_mode=="進階研究" and research_mode=="雷達架構驗證":
    st.subheader("🛰️ 正式雷達架構驗證")
    with st.spinner("重建TOP200 Walk-Forward交易並比較核心/觀察/S級擴充架構…"):
        _ra_sum,_ra_blocks,_ra_detail,_ra_errs=validate_radar_architectures(cost)
    st.session_state["st_v11615_radar_arch"]={
        "summary":_ra_sum,"blocks":_ra_blocks,"detail":_ra_detail,"errors":_ra_errs
    }

_ra=st.session_state.get("st_v11615_radar_arch")
if simple_mode=="進階研究" and research_mode=="雷達架構驗證" and _ra:
    _as=_ra.get("summary",pd.DataFrame()); _ab=_ra.get("blocks",pd.DataFrame())
    _ad=_ra.get("detail",pd.DataFrame()); _ae=_ra.get("errors",[])
    st.subheader("🛰️ 雷達架構驗證結果")
    if _ae:
        st.warning("資料來源異常："+"；".join(_ae))
    st.info("比較重點不是只看最高報酬，而是同時看樣本外PF、四段穩定度與相對TOP200的訊號涵蓋率。")
    if not _as.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_as.round(3),use_container_width=True,hide_index=True)
    if not _ab.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_ab.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看各架構逐筆納入狀態"):
        st.dataframe(_ad.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【雷達架構驗證摘要】",_as.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_雷達架構驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【雷達架構四段穩定度】",_ab.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_雷達架構四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【雷達架構逐筆明細】",_ad.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_雷達架構逐筆明細.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="股票池分層驗證" and not run:
    st.info("V1.16.13 已確認母池夠廣，但TOP50只涵蓋約四分之一的TOP200核心策略交易。這一版不直接把池擴大，而是把TOP200拆成四個互斥層級，確認哪些層值得進正式雷達。")

if run and simple_mode=="進階研究" and research_mode=="股票池分層驗證":
    st.subheader("🧱 股票池分層驗證")
    with st.spinner("重建TOP200 Walk-Forward交易並拆解四層股票池品質…"):
        _ly_sum,_ly_blocks,_ly_grades,_ly_trades,_ly_errs=validate_pool_band_layers(cost)
    st.session_state["st_v11614_layers"]={
        "summary":_ly_sum,"blocks":_ly_blocks,"grades":_ly_grades,
        "trades":_ly_trades,"errors":_ly_errs
    }

_ly=st.session_state.get("st_v11614_layers")
if simple_mode=="進階研究" and research_mode=="股票池分層驗證" and _ly:
    _ls=_ly.get("summary",pd.DataFrame()); _lb=_ly.get("blocks",pd.DataFrame())
    _lg=_ly.get("grades",pd.DataFrame()); _lt=_ly.get("trades",pd.DataFrame())
    _le=_ly.get("errors",[])
    st.subheader("🧱 股票池分層驗證結果")
    if _le:
        st.warning("資料來源異常："+"；".join(_le))
    st.info("四層互斥比較：TOP1-50核心、51-100觀察、101-150擴充、151-200外圍。目標是同時兼顧訊號品質與市場涵蓋，不會只看哪一層平均報酬最高。")
    if not _ls.empty:
        st.markdown("#### 各層｜全部 / 樣本內 / 樣本外")
        st.dataframe(_ls.round(3),use_container_width=True,hide_index=True)
    if not _lg.empty:
        st.markdown("#### 各層 S/A/B 訊號結構")
        st.dataframe(_lg.round(3),use_container_width=True,hide_index=True)
    if not _lb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_lb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看分層逐筆交易"):
        st.dataframe(_lt.round(3),use_container_width=True,hide_index=True)

    st.download_button("⬇️ 下載【股票池分層驗證摘要】",_ls.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池分層驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池分層四段穩定度】",_lb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池分層四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池分層訊號等級】",_lg.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池分層訊號等級.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池分層逐筆交易】",_lt.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_股票池分層逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="股票池覆蓋健診" and not run:
    st.info("這個模式專門回答『股票池撈得夠不夠全面』：檢查母池、資料覆蓋、TOP50歷史輪動，以及TOP50相對TOP200能抓到多少核心策略交易。")

if run and simple_mode=="進階研究" and research_mode=="股票池覆蓋健診":
    st.subheader("🌐 股票池覆蓋健診")
    with st.spinner("建立全市場6mo歷史資格並計算TOP50/100/150/200覆蓋…"):
        _cv_sum,_cv_union,_cv_market,_cv_capture,_cv_errs=diagnose_stock_pool_coverage(cost)
    st.session_state["st_v11613_pool_coverage"]={"summary":_cv_sum,"union":_cv_union,"market":_cv_market,"capture":_cv_capture,"errors":_cv_errs}

_cv=st.session_state.get("st_v11613_pool_coverage")
if simple_mode=="進階研究" and research_mode=="股票池覆蓋健診" and _cv:
    _cs=_cv.get("summary",pd.DataFrame()); _cu=_cv.get("union",pd.DataFrame())
    _cm=_cv.get("market",pd.DataFrame()); _cc=_cv.get("capture",pd.DataFrame()); _ce=_cv.get("errors",[])
    st.subheader("🌐 股票池覆蓋健診結果")
    if _ce: st.warning("資料來源異常："+"；".join(_ce))
    st.info("目前母池涵蓋官方上市＋上櫃公司；興櫃不在母池。TOP50是每日動態核心池，不是固定50家公司。")
    if not _cs.empty: st.dataframe(_cs.round(3),use_container_width=True,hide_index=True)
    if not _cu.empty: st.dataframe(_cu.round(3),use_container_width=True,hide_index=True)
    if not _cm.empty: st.dataframe(_cm.round(3),use_container_width=True,hide_index=True)
    if not _cc.empty: st.dataframe(_cc.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【股票池覆蓋健診摘要】",_cs.to_csv(index=False).encode("utf-8-sig"),file_name=f"{APP_VERSION}_股票池覆蓋健診摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池歷史輪動範圍】",_cu.to_csv(index=False).encode("utf-8-sig"),file_name=f"{APP_VERSION}_股票池歷史輪動範圍.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池市場別分布】",_cm.to_csv(index=False).encode("utf-8-sig"),file_name=f"{APP_VERSION}_股票池市場別分布.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【股票池訊號涵蓋率】",_cc.to_csv(index=False).encode("utf-8-sig"),file_name=f"{APP_VERSION}_股票池訊號涵蓋率.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("四份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="TOP50訊號等級驗證" and not run:
    st.info("V1.16.11 顯示：兩者同時符合的D組品質最高，但樣本較少；只有MA60與只有量比的效果不同且量比單獨樣本很小。這版不再做硬Gate，而是驗證S/A/B訊號等級是否適合實際雷達。")

if run and simple_mode=="進階研究" and research_mode=="TOP50訊號等級驗證":
    st.subheader("🏷️ TOP50訊號S/A/B等級驗證")
    with st.spinner("重建6mo暖機、末3mo評估的TOP50交易並驗證訊號等級…"):
        _gr_sum,_gr_blocks,_gr_trades,_gr_errs=validate_top50_signal_grades(cost)
    st.session_state["st_v11612_grades"]={
        "summary":_gr_sum,"blocks":_gr_blocks,"trades":_gr_trades,"errors":_gr_errs
    }

_gr=st.session_state.get("st_v11612_grades")
if simple_mode=="進階研究" and research_mode=="TOP50訊號等級驗證" and _gr:
    _rs=_gr.get("summary",pd.DataFrame()); _rb=_gr.get("blocks",pd.DataFrame())
    _rt=_gr.get("trades",pd.DataFrame()); _re=_gr.get("errors",[])
    st.subheader("🏷️ TOP50訊號等級驗證結果")
    if _re:
        st.warning("資料來源異常："+"；".join(_re))
    st.info("S級=量比>=1.5且MA60未向上；A級=只符合其中一項；B級=兩者皆否。這版只驗證等級，不改進出場規則。")
    if not _rs.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_rs.round(3),use_container_width=True,hide_index=True)
    if not _rb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_rb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看等級逐筆交易"):
        st.dataframe(_rt.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【TOP50訊號等級驗證摘要】",_rs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50訊號等級驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50訊號等級四段穩定度】",_rb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50訊號等級四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50訊號等級逐筆交易】",_rt.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50訊號等級逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="TOP50Gate拆解驗證" and not run:
    st.info("V1.16.10 顯示量比>=1.5與MA60未向上都可能有增益，但B/C/D彼此重疊。這一版改成四個互斥組別，確認量比與MA60各自帶來多少獨立資訊。")

if run and simple_mode=="進階研究" and research_mode=="TOP50Gate拆解驗證":
    st.subheader("🧬 TOP50 Gate拆解驗證")
    with st.spinner("重建6mo暖機、末3mo評估的TOP50 Walk-Forward交易，拆解量比與MA60的獨立效果…"):
        _gd_sum,_gd_blocks,_gd_trades,_gd_errs=validate_top50_gate_decomposition(cost)
    st.session_state["st_v11611_gate_decomp"]={
        "summary":_gd_sum,"blocks":_gd_blocks,"trades":_gd_trades,"errors":_gd_errs
    }

_gd=st.session_state.get("st_v11611_gate_decomp")
if simple_mode=="進階研究" and research_mode=="TOP50Gate拆解驗證" and _gd:
    _ds=_gd.get("summary",pd.DataFrame()); _db=_gd.get("blocks",pd.DataFrame())
    _dt=_gd.get("trades",pd.DataFrame()); _de=_gd.get("errors",[])
    st.subheader("🧬 TOP50 Gate拆解結果")
    if _de:
        st.warning("資料來源異常："+"；".join(_de))
    st.info("四組互斥：兩者皆否／只有量比／只有MA60／兩者同時符合。這一步只看獨立資訊價值，不直接改正式雷達。")
    if not _ds.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_ds.round(3),use_container_width=True,hide_index=True)
    if not _db.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_db.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看拆解逐筆交易"):
        st.dataframe(_dt.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【TOP50Gate拆解摘要】",_ds.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50Gate拆解摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50Gate拆解四段穩定度】",_db.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50Gate拆解四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50Gate拆解逐筆交易】",_dt.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50Gate拆解逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="TOP50候選Gate驗證" and not run:
    st.info("V1.16.10 修正獨立研究模式與舊版共用流程的衝突：TOP50候選Gate驗證不再掉進舊流程引用未定義的 summary。Gate邏輯本身不變，仍只比較基準、量比>=1.5、MA60未向上與兩者組合。")

if run and simple_mode=="進階研究" and research_mode=="TOP50候選Gate驗證":
    st.subheader("🧪 TOP50候選Gate A/B")
    with st.spinner("重建6mo暖機、末3mo評估的TOP50 Walk-Forward交易，驗證候選Gate…"):
        _ga_sum,_ga_blocks,_ga_trades,_ga_errs=validate_top50_candidate_gates(cost)
    st.session_state["st_v11610_gate"]={
        "summary":_ga_sum,"blocks":_ga_blocks,"trades":_ga_trades,"errors":_ga_errs
    }

_ga=st.session_state.get("st_v11610_gate")
if simple_mode=="進階研究" and research_mode=="TOP50候選Gate驗證" and _ga:
    _gs=_ga.get("summary",pd.DataFrame()); _gb=_ga.get("blocks",pd.DataFrame())
    _gt=_ga.get("trades",pd.DataFrame()); _ge=_ga.get("errors",[])
    st.subheader("🧪 TOP50候選Gate驗證結果")
    if _ge:
        st.warning("資料來源異常："+"；".join(_ge))
    st.info("候選Gate若只改善整體平均、但樣本外或四段不穩定，就不會納入正式雷達。")
    if not _gs.empty:
        st.markdown("#### 全部 / 樣本內 / 樣本外")
        st.dataframe(_gs.round(3),use_container_width=True,hide_index=True)
    if not _gb.empty:
        st.markdown("#### 四段時間穩定度")
        st.dataframe(_gb.round(3),use_container_width=True,hide_index=True)
    with st.expander("查看Gate逐筆交易"):
        st.dataframe(_gt.round(3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ 下載【TOP50候選Gate驗證摘要】",_gs.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50候選Gate驗證摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50候選Gate四段穩定度】",_gb.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50候選Gate四段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.download_button("⬇️ 下載【TOP50候選Gate逐筆交易】",_gt.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{APP_VERSION}_TOP50候選Gate逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")
    st.success("三份檔案可連續下載，不需重跑。")

if simple_mode=="進階研究" and research_mode=="TOP50訊號品質健診" and not run:
    st.info("V1.16.7 的結果不能採用：函式雖然預設6mo暖機，但UI仍明確傳入3mo，把暖機設定覆蓋掉，因此量比20與MA60又出現缺值。V1.16.8 已移除3mo覆蓋，固定6mo暖機＋只評估最後3mo，並在輸出中直接標示資料窗與缺值率。")

if run and simple_mode=="進階研究" and research_mode=="TOP50訊號品質健診":
    st.subheader("🔬 TOP50訊號品質健診")
    with st.spinner("重建真正Walk-Forward TOP50交易，檢查K深度、量比20、MA30/60方向與60m時段…"):
        _sq_sum,_sq_blocks,_sq_trades,_sq_errs=validate_top50_signal_quality(cost)
    st.session_state["st_v1168_signal_quality"]={
        "summary":_sq_sum,"blocks":_sq_blocks,"trades":_sq_trades,"errors":_sq_errs
    }

_sq=st.session_state.get("st_v1168_signal_quality")
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

# V1.16.10：所有「上方已有獨立執行流程」的研究模式集中管理。
# 後續新增研究模式時只要加入此集合，就不會再掉進舊版共用流程而引用未定義的 summary。
INDEPENDENT_RESEARCH_MODES = {
    "長期集中度健診",
    "長期穩健度驗證",
    "延遲TimeStop驗證",
    "早期路徑健診",
    "固定停損驗證",
    "獲利保護風險效益",
    "獲利保護敏感度",
    "獲利保護驗證",
    "持有天數驗證",
    "持有路徑健診",
    "市場廣度轉折健診",
    "環境×訊號交互驗證",
    "環境Gate驗證",
    "失效環境健診",
    "雷達架構驗證",
    "股票池分層驗證",
    "股票池覆蓋健診",
    "TOP50訊號等級驗證",
    "TOP50Gate拆解驗證",
    "TOP50候選Gate驗證",
    "TOP50訊號品質健診",
    "TOP50暖機修正驗證",
    "市場環境健診_TOP50",
    "核心池規模WalkForward",
    "全市場WalkForward驗證",
    "全市場股票池研究",
    "全市場候選策略驗證",
    "股票池2.0研究",
    "股票池2.0歷史驗證",
    "股票池健診",
}

if run:
    # 舊版共用區一律先初始化，避免任何獨立研究模式觸發 NameError。
    summary, data_map, trade_map = pd.DataFrame(), {}, {}
    if "symbol" not in locals():
        symbol = ""

    _legacy_modes_without_rule_check = INDEPENDENT_RESEARCH_MODES | {"多週期當沖/隔日驗證","60m五日OOS驗證"}
    if research_mode not in _legacy_modes_without_rule_check and (not selected_intervals or not selected_rules or not selected_modes):
        st.error("請至少選擇一個K棒週期、進場規則與持有方式。")
        st.stop()

    if simple_mode=="今日雷達":
        with st.spinner("建立官方上市/上櫃動態股票池並掃描最新60m訊號…"):
            _dr,_dd,_pd,_pp,_env,_pe=build_formal_daily_radar(cost)
        st.session_state["st_v11622_daily"]={
            "radar":_dr,"scan_diag":_dd,"pool_diag":_pd,"pool":_pp,
            "environment":_env,"errors":_pe
        }
        # 今日雷達使用獨立正式流程，不再進入舊版人工母池/OOS流程。
        summary, data_map, trade_map = pd.DataFrame(), {}, {}
    elif research_mode in INDEPENDENT_RESEARCH_MODES:
        # 獨立研究模式已在上方完成；此處不再進入舊版策略流程。
        pass
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
        "summary": summary if isinstance(summary,pd.DataFrame) else pd.DataFrame(),
        "data_map": data_map if isinstance(data_map,dict) else {},
        "trade_map": trade_map if isinstance(trade_map,dict) else {},
    }


_daily=st.session_state.get("st_v11622_daily")
if simple_mode=="今日雷達":
    st.markdown("## 📡 今日60m正式雷達")
    st.caption("正式架構：TOP1-100保留全部核心KD訊號；TOP101-150只有S級進雷達。另以TOP200計算今日市場廣度風險，但目前只警示、不作硬Gate。")
    if _daily:
        _dr=_daily.get("radar",pd.DataFrame())
        _dd=_daily.get("scan_diag",pd.DataFrame())
        _pd=_daily.get("pool_diag",pd.DataFrame())
        _pp=_daily.get("pool",pd.DataFrame())
        _pe=_daily.get("errors",[])
        if _pe:
            st.warning("資料來源異常："+"；".join(_pe))

        _env=_daily.get("environment",pd.DataFrame())
        if not _env.empty:
            e0=_env.iloc[0]
            st.markdown("### 🌦️ 今日市場環境")
            e1,e2,e3,e4,e5=st.columns(5)
            e1.metric("市場狀態",str(e0.get("市場狀態","")))
            e2.metric("MA15廣度",f"{float(e0.get('站上MA15比例%',np.nan)):.1f}%")
            e3.metric("MA30廣度",f"{float(e0.get('站上MA30比例%',np.nan)):.1f}%")
            e4.metric("MA60廣度",f"{float(e0.get('站上MA60比例%',np.nan)):.1f}%")
            e5.metric("5日上漲家數",f"{float(e0.get('5日上漲家數比例%',np.nan)):.1f}%")
            if str(e0.get("市場狀態","")).startswith("🔴"):
                st.warning("目前屬於MA60高廣度風險環境。研究顯示這類環境歷史表現較差，但樣本外仍有例外，因此目前只警示、不自動過濾S/A/B訊號。")
            else:
                st.caption("環境風險目前只做提示，不改變今日雷達S/A/B訊號納入規則。")
            st.download_button("⬇️ 下載【今日市場環境】",
                               _env.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{APP_VERSION}_今日市場環境.csv",
                               mime="text/csv",use_container_width=True,on_click="ignore")

        if not _pd.empty:
            d0=_pd.iloc[0]
            c1,c2,c3,c4=st.columns(4)
            c1.metric("官方母池",int(d0.get("官方股票數",0)))
            c2.metric("可排名",int(d0.get("可排名股票數",0)))
            c3.metric("60m掃描",int(d0.get("正式掃描股票數",0)))
            c4.metric("正式訊號",len(_dr))
            st.caption(f"日K排名基準完成日：{d0.get('最新排名基準日','')}｜掃描範圍：核心50＋觀察50＋擴充50")
        if _dr.empty:
            st.info("本次TOP150最新60m掃描沒有仍在5交易日觀察窗內、且符合正式架構的訊號。可查看診斷確認150檔是否正常完成掃描。")
        else:
            _new=int((_dr["目前狀態"]=="🟢 新訊號").sum())
            _watch=int((_dr["目前狀態"]=="🟡 觀察中").sum())
            _s=int((_dr["訊號等級"]=="S級").sum())
            _a=int((_dr["訊號等級"]=="A級").sum())
            _b=int((_dr["訊號等級"]=="B級").sum())
            x1,x2,x3,x4,x5=st.columns(5)
            x1.metric("今日新訊號",_new)
            x2.metric("觀察中",_watch)
            x3.metric("S級",_s)
            x4.metric("A級",_a)
            x5.metric("B級",_b)

            _cols=[c for c in [
                "股票","公司","市場","流動性排名","股票池層級","訊號等級",
                "市場狀態","市場MA60廣度%","目前狀態","訊號時間","訊號K","量比20","MA60斜率3",
                "目前K","目前D","預計觀察至"
            ] if c in _dr.columns]
            st.dataframe(_dr[_cols],use_container_width=True,hide_index=True)
            st.download_button("⬇️ 下載【今日60m正式雷達】",
                               _dr.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{APP_VERSION}_今日60m正式雷達.csv",
                               mime="text/csv",use_container_width=True,on_click="ignore")
        if not _pp.empty:
            st.download_button("⬇️ 下載【今日正式股票池】",
                               _pp.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{APP_VERSION}_今日正式股票池.csv",
                               mime="text/csv",use_container_width=True,on_click="ignore")
        with st.expander("🔧 今日雷達掃描診斷"):
            if not _pd.empty:
                st.dataframe(_pd,use_container_width=True,hide_index=True)
            if not _dd.empty:
                _ok=int((_dd["狀態"]=="掃描完成").sum()) if "狀態" in _dd.columns else 0
                st.caption(f"60m成功掃描：{_ok}/{len(_dd)}")
                st.dataframe(_dd,use_container_width=True,hide_index=True)
                st.download_button("⬇️ 下載【今日60m掃描診斷】",
                                   _dd.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{APP_VERSION}_今日60m掃描診斷.csv",
                                   mime="text/csv",use_container_width=True,on_click="ignore")
    else:
        st.info("按左側「🔄 更新今日雷達」後，會先從官方上市/上櫃母池建立當日動態排名，再掃描最新60m訊號。")

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
    "ST V1.16.33 僅供策略研究與程式驗證，不送出證券委託。"
    "下一階段將根據實際回測結果，再判斷是否增加 VWAP、成交量/量比、MACD、ATR 或其他參數。"
)
