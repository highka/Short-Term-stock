# -*- coding: utf-8 -*-
"""
黑嚕嚕－短線交易雷達 ST V1.13.3
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

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

APP_VERSION = "ST V1.13.3"
APP_NAME = "黑嚕嚕－短線交易雷達"
MA_LIST = [5, 15, 30, 60, 200]
INTERVALS = ["5m", "15m", "60m"]

APP_VERSION = "ST_V1.13.3"
EXPORT_PREFIX = "ST_V1.13.3"

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
                      group_by="ticker", auto_adjust=False, progress=False, threads=True)
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


def run_oos_60m_5d(symbols: List[str], cost: CostConfig, period: str, allow_overlap: bool = False,
                     train_ratio: float = 0.60):
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
                x["_signal_dt"]=pd.to_datetime(x["訊號時間"], utc=True, errors="coerce")
                raw_trades.append(x)
        p.progress(n/max(1,len(symbols)), text=f"建立交易 {symbol}｜{n}/{len(symbols)}")
    p.empty()

    if not raw_trades:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_t=pd.concat(raw_trades,ignore_index=True).dropna(subset=["_signal_dt"]).sort_values("_signal_dt")
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
            progress=False, threads=True, group_by="ticker"
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
        rows.append({
            "股票":symbol,
            "研究主題":research_theme(symbol),
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
    radar=radar.sort_values(["_o","訊號時間"],ascending=[True,False]).drop(columns="_o").reset_index(drop=True)
    return radar, diagnostics



def fetch_official_tw_stock_universe():
    """
    V1.13.0：官方上市/上櫃公司基本資料。
    只保留4位數字公司代號；上市加.TW、上櫃加.TWO。
    官方來源失敗時回傳空表，由UI明確提示，不靜默冒充全市場。
    """
    endpoints=[
        ("上市","https://openapi.twse.com.tw/v1/opendata/t187ap03_L",".TW"),
        ("上櫃","https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",".TWO"),
    ]
    rows=[]
    errors=[]
    for market,url,suffix in endpoints:
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=20) as resp:
                data=json.loads(resp.read().decode("utf-8"))
            for item in data:
                code=str(item.get("公司代號",item.get("SecuritiesCompanyCode",""))).strip()
                name=str(item.get("公司簡稱",item.get("CompanyName",""))).strip()
                industry=str(item.get("產業別",item.get("SecuritiesIndustryCode",""))).strip()
                if len(code)==4 and code.isdigit():
                    rows.append({"股票":code+suffix,"代號":code,"公司":name,
                                 "市場":market,"官方產業別":industry,"官方來源":url})
        except Exception as e:
            errors.append(f"{market}: {e}")
    df=pd.DataFrame(rows).drop_duplicates("股票") if rows else pd.DataFrame()
    return df,errors


def download_daily_batches(symbols: List[str], period: str="2mo", batch_size: int=80):
    """分批下載日K，避免全市場一次向Yahoo請求過大。"""
    out={}
    syms=list(dict.fromkeys(symbols))
    for i in range(0,len(syms),batch_size):
        batch=syms[i:i+batch_size]
        try:
            raw=yf.download(tickers=batch,period=period,interval="1d",
                            group_by="ticker",auto_adjust=False,progress=False,threads=True)
            for s in batch:
                try:
                    if isinstance(raw.columns,pd.MultiIndex):
                        if s not in raw.columns.get_level_values(0):
                            continue
                        d=raw[s].copy()
                    else:
                        if len(batch)!=1:
                            continue
                        d=raw.copy()
                    d=d.dropna(subset=["Close","Volume"])
                    if len(d):
                        out[s]=d
                except Exception:
                    continue
        except Exception:
            continue
    return out




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
    """每個週期一次抓整批股票，再切回單檔；大幅減少 Yahoo 請求數。"""
    out = {s: pd.DataFrame() for s in symbols}
    if not symbols:
        return out
    try:
        raw = yf.download(
            tickers=" ".join(symbols),
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
            group_by="ticker",
        )
    except Exception:
        return out

    for symbol in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(0):
                    continue
                d = raw[symbol].copy()
            else:
                d = raw.copy()
            need = ["Open","High","Low","Close","Volume"]
            if not all(c in d.columns for c in need):
                continue
            d = d[need].copy()
            for c in need:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=["Open","High","Low","Close"])
            if d.empty:
                continue
            idx = pd.to_datetime(d.index)
            if getattr(idx, "tz", None) is not None:
                try:
                    idx = idx.tz_convert("Asia/Taipei").tz_localize(None)
                except Exception:
                    idx = idx.tz_localize(None)
            d.index = idx
            out[symbol] = d.sort_index()
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
                ["全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診","單一股票","跨股票批次","多週期當沖/隔日驗證","60m五日OOS驗證"], index=1)
            if research_mode == "全市場候選策略驗證":
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
            if research_mode not in ["全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診"]:
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
            "全市場股票池研究":"🌐 建立全市場研究股票池",
            "全市場候選策略驗證":"🧪 驗證核心TOP100 vs 熱門增補",
            "股票池2.0研究":"🧭 開始股票池2.0即時診斷",
            "股票池2.0歷史驗證":"🧪 開始股票池2.0歷史驗證",
            "股票池健診":"🧭 開始股票池健診",
        }.get(research_mode,"🚀 開始策略健診")
    run = st.button(_btn_label, type="primary", use_container_width=True)


if simple_mode == "今日雷達":
    st.markdown("""<style>div[data-baseweb="tab-list"]{display:none!important;}</style>""", unsafe_allow_html=True)

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

    if research_mode in ["全市場股票池研究","全市場候選策略驗證","股票池2.0研究","股票池2.0歷史驗證","股票池健診"]:
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
    "ST V1.13.3 僅供策略研究與程式驗證，不送出證券委託。"
    "下一階段將根據實際回測結果，再判斷是否增加 VWAP、成交量/量比、MACD、ATR 或其他參數。"
)
