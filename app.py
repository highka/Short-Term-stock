"""
黑嚕嚕－短線交易雷達 ST V1.16.51

正式核心策略已凍結：
- 官方 TWSE + TPEx 普通股母池
- Point-in-Time 20日成交金額中位數流動性排名
- TOP1-100 全部 + TOP101-150 僅 S級
- 60m KD黃金交叉 + K<30
- 訊號完成後下一根60m Open進場
- 持有5個交易日、non-overlap

本版清理：
- 移除已否決或只用於過往探索的研究策略與舊進階頁面
- 進階研究僅保留：正式凍結規格、長期穩健度、長期集中度
- 新增 Shioaji Stage 1 本機即時行情 Worker：登入測試 + 單股 Tick 訂閱 + 狀態輸出；不下單
"""


import math
import warnings
import os
import time
from dataclasses import dataclass
from pathlib import Path
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
import urllib.parse
import json

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

APP_VERSION = "ST V1.16.51"
APP_NAME = "黑嚕嚕－短線交易雷達"
MA_LIST = [5, 15, 30, 60, 200]
INTERVALS = ["5m", "15m", "60m"]

APP_VERSION = "ST_V1.16.51"
EXPORT_PREFIX = "ST_V1.16.51"

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




@st.cache_data(ttl=900, show_spinner=False)


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

def get_frozen_strategy_config():
    """
    V1.16.34 正式核心策略凍結設定。
    此函式作為未來 Shioaji 即時引擎與 Streamlit 雷達共用的單一規格來源。
    """
    return {
        "strategy_status":"FROZEN_BASELINE",
        "strategy_version":"ST V1.16.51",
        "universe_source":"官方TWSE+TPEx普通股母池",
        "liquidity_ranking":"前一完成交易日，20日成交金額中位數，Point-in-Time",
        "formal_pool_rule":"TOP1-100全部 + TOP101-150僅S級",
        "scan_pool_size":150,
        "market_breadth_pool_size":200,
        "signal_timeframe":"60m",
        "signal_rule":"KD黃金交叉 + K<30",
        "kd_rule":"KD(9,3,3)，RSV9，K/D遞迴2/3前值+1/3當值，初始50",
        "entry_rule":"完成訊號後下一根60m K Open",
        "holding_rule":"持有5個交易日",
        "overlap_rule":"non-overlap",
        "grade_S":"量比20>=1.5 且 MA60斜率3<=0",
        "grade_A":"量比20>=1.5 或 MA60斜率3<=0，僅符合一項",
        "grade_B":"兩項皆不符合",
        "market_risk_display":"TOP200 MA15/30/60廣度；目前僅警示，不作硬Gate",
        "profit_protection":"研究保留，不納入正式基準",
        "fixed_stoploss":"不採用",
        "delayed_timestop":"不採用",
        "realtime_plan":"Shioaji常駐行情引擎；Streamlit只顯示/研究",
    }


def get_shioaji_realtime_spec():
    """
    V1.16.34 即時串接規格（不需要安裝shioaji即可顯示）。
    真正連線/下單將放在本機或VPS常駐worker，不放Streamlit Cloud。
    """
    return pd.DataFrame([
        {"階段":"08:30~08:59","模組":"盤前建池","動作":"用前一完成日K重建官方全市場流動性排名，取TOP200；正式掃描TOP150"},
        {"階段":"09:00~13:30","模組":"即時行情","動作":"Shioaji訂閱TOP150股票即時行情/即時KBar"},
        {"階段":"盤中","模組":"K棒聚合","動作":"維護60m K；未完成K只標預備訊號，不當正式訊號"},
        {"階段":"60m完成","模組":"正式訊號","動作":"依凍結規則計算KD黃金交叉、K<30、量比20、MA60斜率3與S/A/B"},
        {"階段":"訊號成立","模組":"狀態儲存","動作":"寫入SQLite/PostgreSQL或JSON狀態檔，供Streamlit讀取"},
        {"階段":"通知","模組":"Telegram","動作":"新訊號/異常/收盤摘要通知；先不自動下單"},
        {"階段":"交易層","模組":"Shioaji Order","動作":"第二階段再加入模擬單→人工確認→自動下單"},
    ])

def get_shioaji_stage1_checklist():
    """V1.16.42：即時策略＋模擬交易＋Telegram。"""
    return pd.DataFrame([
        {"順序":1,"項目":"每日TOP150","內容":"沿用自動建池；TOP1-100保留全部核心訊號，TOP101-150只接受S級"},
        {"順序":2,"項目":"60m暖機","內容":"Worker啟動用Yahoo 60m歷史資料暖機KD/MA60/量比20，不占Shioaji歷史查詢額度"},
        {"順序":3,"項目":"正式訊號","內容":"完成60m K：KD(9,3,3)黃金交叉且K<30；S/A/B依量比20與MA60斜率3"},
        {"順序":4,"項目":"模擬進場","內容":"訊號完成後的下一根60m K第一筆Tick模擬成交，符合原回測「下一根Open」語意"},
        {"順序":5,"項目":"非重疊","內容":"同一股票已有模擬持倉或待進場時，不重複建立新部位"},
        {"順序":6,"項目":"模擬出場","內容":"固定5日：進場日不算，於第5個後續交易日最後一根60m K收盤模擬出場"},
        {"順序":7,"項目":"Telegram","內容":"模擬建倉與出場成功後推送通知；Telegram本版只通知，不接受下單指令"},
        {"順序":8,"項目":"正式下單","內容":"本版仍0真實委託；待模擬狀態/重啟/出場驗證後，再做人工確認正式下單"},
    ])


def read_shioaji_worker_status(path="runtime/shioaji_status.json"):
    """
    讀取同一台主機上的 Shioaji Worker 狀態。
    Streamlit Cloud 無法直接讀取使用者家中電腦的 runtime 檔案；
    Stage 1 先用來驗證本機 Worker。
    """
    p=Path(path)
    if not p.exists():
        return {}, f"找不到 {path}"
    try:
        obj=json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj,dict):
            return {}, "狀態檔格式不是JSON物件"
        return obj, ""
    except Exception as e:
        return {}, f"狀態檔讀取失敗：{e}"



def read_worker_json(path):
    p=Path(path)
    if not p.exists():
        return {}
    try:
        x=json.loads(p.read_text(encoding="utf-8"))
        return x if isinstance(x,(dict,list)) else {}
    except Exception:
        return {}


def shioaji_stage1_status_table(status):
    keys=[
        ("phase","階段"),
        ("message","訊息"),
        ("shioaji_version","Shioaji版本"),
        ("simulation","Shioaji模擬環境"),
        ("strategy_mode","交易模式"),
        ("pool_source","股票池來源"),
        ("pool_basis_date","排名基準完成日"),
        ("pool_code_count","股票池檔數"),
        ("warmup_success","60m暖機成功檔數"),
        ("warmup_failed","60m暖機失敗檔數"),
        ("subscription_target","目標訂閱數"),
        ("subscription_count","目前訂閱數"),
        ("subscription_failed","訂閱失敗數"),
        ("tick_count","Tick數"),
        ("active_bar_count","進行中60m K數"),
        ("finalized_bar_count","已完成60m K數"),
        ("formal_signal_count","正式訊號數"),
        ("pending_entry_count","待模擬進場"),
        ("sim_position_count","模擬持倉數"),
        ("sim_closed_trade_count","已完成模擬交易"),
        ("telegram_enabled","Telegram通知"),
        ("last_trade_event","最後交易事件"),
        ("usage_pct","Usage使用率%"),
        ("usage_level","Usage警示"),
        ("reconnect_attempt","目前重連次數"),
        ("updated_at","狀態更新時間"),
    ]
    return pd.DataFrame([{"項目":label,"內容":status.get(key,"")} for key,label in keys])



def build_current_market_breadth(top200_pool: pd.DataFrame):
    """
    今日市場環境：
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

    _tmp={
        "站上MA15比例%":ma15_pct,
        "站上MA30比例%":ma30_pct,
        "站上MA60比例%":ma60_pct,
    }
    risk,meaning=market_breadth_interpretation(_tmp)

    out=pd.DataFrame([{
        "市場狀態":risk,
        "狀態解讀":meaning,
        "基準完成日":str(max(latest_dates)) if latest_dates else "",
        "TOP200有效股票數":len(z),
        "站上MA15比例%":ma15_pct,
        "站上MA30比例%":ma30_pct,
        "站上MA60比例%":ma60_pct,
        "5日上漲家數比例%":up5_pct,
        "5日報酬中位數%":med5,
        "使用方式":"市場廣度只做環境提示，不直接過濾正式策略訊號"
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




















@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)















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

























def get_supabase_read_config():
    """Streamlit Cloud讀取共享狀態：優先新版Publishable Key，並相容舊anon key。"""
    url=""
    key=""
    try:
        if "supabase" in st.secrets:
            sec=st.secrets["supabase"]
            url=str(sec.get("url","")).strip().rstrip("/")
            key=str(sec.get("publishable_key","") or sec.get("anon_key","")).strip()
    except Exception:
        pass
    url=url or str(os.getenv("SUPABASE_URL","")).strip().rstrip("/")
    key=key or str(os.getenv("SUPABASE_PUBLISHABLE_KEY","") or os.getenv("SUPABASE_ANON_KEY","")).strip()
    return url,key


def read_supabase_runtime():
    """
    從共享資料表讀 status / sim_positions / sim_pending。
    回傳 (data,error)；不會把金鑰寫入畫面。
    """
    url,key=get_supabase_read_config()
    if not url or not key:
        return {}, "SHARED_NOT_CONFIGURED"

    try:
        endpoint=(
            f"{url}/rest/v1/heylulu_runtime"
            "?select=key,payload,updated_at"
            "&key=in.(status,sim_positions,sim_pending)"
        )
        req=urllib.request.Request(
            endpoint,
            headers={
                "apikey":key,
                "Authorization":f"Bearer {key}",
                "Accept":"application/json",
            },
        )
        with urllib.request.urlopen(req,timeout=12) as resp:
            rows=json.loads(resp.read().decode("utf-8"))

        out={"_source":"supabase"}
        for row in rows if isinstance(rows,list) else []:
            k=str(row.get("key",""))
            out[k]=row.get("payload",{})
            out[f"{k}_updated_at"]=row.get("updated_at","")
        return out,None
    except Exception as e:
        return {}, f"SHARED_READ_ERROR: {type(e).__name__}: {e}"


def read_worker_runtime_unified():
    """
    優先讀本機runtime；若不存在則讀Supabase共享狀態。
    """
    local_status,local_err=read_shioaji_worker_status()
    if local_status:
        return {
            "status":local_status,
            "sim_positions":read_worker_json("runtime/sim_positions.json"),
            "sim_pending":read_worker_json("runtime/sim_pending_entries.json"),
            "_source":"local",
        },None

    shared,shared_err=read_supabase_runtime()
    if shared and shared.get("status"):
        return {
            "status":shared.get("status",{}),
            "sim_positions":shared.get("sim_positions",{}),
            "sim_pending":shared.get("sim_pending",{}),
            "_source":"supabase",
            "status_updated_at":shared.get("status_updated_at",""),
        },None

    return {}, shared_err or local_err or "NO_RUNTIME_SOURCE"


def diagnose_worker_runtime():
    """
    診斷Streamlit目前為何讀不到Worker。
    會區分：本機runtime、Cloud但未設定共享橋接、共享橋接已設定但Worker未上傳、共享橋接正常。
    """
    cwd = Path.cwd()
    runtime_dir = cwd / "runtime"
    status_path = runtime_dir / "shioaji_status.json"
    cwd_text = str(cwd).replace("\\", "/")
    likely_cloud = (
        cwd_text.startswith("/mount/src/")
        or cwd_text.startswith("/home/adminuser/")
        or bool(os.getenv("STREAMLIT_SHARING_MODE"))
    )

    supa_url,supa_key=get_supabase_read_config()
    shared_configured=bool(supa_url and supa_key)
    unified,unified_err=read_worker_runtime_unified()

    result = {
        "cwd": str(cwd),
        "runtime_dir": str(runtime_dir),
        "status_path": str(status_path),
        "runtime_exists": runtime_dir.exists(),
        "status_exists": status_path.exists(),
        "likely_cloud": likely_cloud,
        "shared_configured": shared_configured,
        "shared_source": unified.get("_source","") if unified else "",
        "diagnosis": "",
        "suggestion": "",
        "level": "info",
    }

    if unified and unified.get("status"):
        src=unified.get("_source","")
        if src=="local":
            result["diagnosis"]="OK：Streamlit正在直接讀取同一台機器的Worker狀態。"
        else:
            result["diagnosis"]="OK：Streamlit Cloud已透過Supabase共享橋接讀到Windows Worker狀態。"
        result["suggestion"]="共享狀態已接通，可由自動刷新持續更新。"
        result["level"]="success"
        return result

    if likely_cloud and not shared_configured:
        result["diagnosis"]="Streamlit Cloud與Windows Worker分屬不同機器，目前尚未設定共享資料橋接。"
        result["suggestion"]="不是檔案放錯。請在Streamlit Secrets加入Supabase URL與Publishable Key，並在Windows .env加入URL與Secret Key。"
        result["level"]="warning"
        return result

    if likely_cloud and shared_configured:
        result["diagnosis"]="Supabase共享橋接已設定，但目前尚未讀到Worker上傳的狀態。"
        result["suggestion"]="請確認Windows Worker已更新到V1.16.45並正在執行；也可檢查Supabase heylulu_runtime資料表是否已有status資料列。"
        result["level"]="warning"
        return result

    if not runtime_dir.exists():
        result["diagnosis"]="本機Streamlit目前找不到runtime資料夾。"
        result["suggestion"]="若Streamlit與Worker在同一台電腦，請確認app.py與Worker使用同一個專案根目錄。"
        result["level"]="warning"
        return result

    result["diagnosis"]="runtime資料夾存在，但shioaji_status.json不存在。"
    result["suggestion"]="請確認Worker是否已啟動並成功寫入狀態，或Worker與App是否指向不同runtime路徑。"
    result["level"]="warning"
    return result



def market_breadth_interpretation(row):
    """把MA15/30/60廣度翻成較直觀的市場狀態；只做提示，不過濾策略訊號。"""
    def _num(k):
        try:
            v=float(row.get(k))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    b15=_num("站上MA15比例%")
    b30=_num("站上MA30比例%")
    b60=_num("站上MA60比例%")

    if b15 is None or b30 is None or b60 is None:
        return "⚪ 資料不足", "廣度資料不足，暫不判讀。"

    if b15 >= 70 and b30 >= 65 and b60 >= 60:
        return "🟢 強勢擴散", "短中期多數股票都站在均線上，市場參與度廣。"

    if b15 < 45 and b30 >= 55 and b60 >= 55:
        return "🟡 短線轉弱", "MA15先下滑，但MA30/60仍高，常見於強勢市場短線降溫。"

    if b15 >= 60 and b30 < 50 and b60 < 50:
        return "🟠 反彈初期", "短線先回到MA15之上，但中期廣度尚未跟上。"

    if b15 < 40 and b30 < 40 and b60 < 40:
        return "🔴 全面偏弱", "短中期大多數股票都在均線下方。"

    if b60 >= 65:
        return "🟠 中期高檔", "MA60廣度偏高，代表中期多數股票仍在強勢區，需留意高檔震盪風險。"

    return "⚪ 中性", "多空廣度沒有明顯極端，視個股訊號為主。"



def strategy_lab_extra_mask(d: pd.DataFrame, mode: str) -> pd.Series:
    false = pd.Series(False, index=d.index)
    if mode == "無（正式baseline）":
        return pd.Series(True, index=d.index)
    if mode == "S級雙條件":
        return (
            pd.to_numeric(d.get("VOL_RATIO20"), errors="coerce").ge(1.5)
            & pd.to_numeric(d.get("MA60_SLOPE3"), errors="coerce").le(0)
        ).fillna(False)
    if mode == "量比20 >= 1.5":
        return pd.to_numeric(d.get("VOL_RATIO20"), errors="coerce").ge(1.5).fillna(False)
    if mode == "MA60斜率3 <= 0":
        return pd.to_numeric(d.get("MA60_SLOPE3"), errors="coerce").le(0).fillna(False)
    if mode == "站上VWAP":
        return d.get("PRICE_GT_VWAP", false).fillna(False)
    if mode == "站上MA15":
        return (pd.to_numeric(d.get("Close"), errors="coerce") > pd.to_numeric(d.get("MA15"), errors="coerce")).fillna(False)
    if mode == "站上MA30":
        return (pd.to_numeric(d.get("Close"), errors="coerce") > pd.to_numeric(d.get("MA30"), errors="coerce")).fillna(False)
    return pd.Series(True, index=d.index)


def strategy_lab_base_mask(d: pd.DataFrame, rank: int) -> pd.Series:
    base = (
        d.get("KD_GOLD", pd.Series(False, index=d.index)).fillna(False)
        & pd.to_numeric(d.get("K"), errors="coerce").lt(30).fillna(False)
    )
    if int(rank) <= 100:
        return base
    s_grade = (
        pd.to_numeric(d.get("VOL_RATIO20"), errors="coerce").ge(1.5)
        & pd.to_numeric(d.get("MA60_SLOPE3"), errors="coerce").le(0)
    ).fillna(False)
    return base & s_grade



def add_kd_death_cross_flags(d: pd.DataFrame) -> pd.DataFrame:
    """補上完成60分K的KD死亡交叉旗標。"""
    x=d.copy()
    if "K" not in x.columns or "D" not in x.columns:
        x["KD_DEAD"]=False
        return x
    k=pd.to_numeric(x["K"],errors="coerce")
    dd=pd.to_numeric(x["D"],errors="coerce")
    x["KD_DEAD"]=((k.shift(1)>=dd.shift(1)) & (k<dd)).fillna(False)
    return x


def find_dynamic_exit(
    d: pd.DataFrame,
    entry_i: int,
    entry_price: float,
    exit_mode: str,
    max_hold_days: int=5,
):
    """
    動態出場規則（完成60分K後判斷）：

    共通停利：
      Close >= 進場價 且 K>80 且當根KD死亡交叉
      -> 下一根60m Open出場

    停損研究版本：
      A. 連續2根60m Close < 進場價 且目前K<D
      B. Close <= 進場價 * 0.97 且目前K<D
      C. Close <= 進場價 * 0.95 且目前K<D

    兜底：
      第5個後續交易日最後一根60m Close出場

    為避免look-ahead：
      KD/價格條件需等完成K棒後確認，實際成交用下一根60m Open。
    """
    if d is None or d.empty or entry_i>=len(d):
        return None,None,None

    x=add_kd_death_cross_flags(d)
    entry_date=pd.Timestamp(x.index[entry_i]).date()

    # 第5個後續交易日最後一根bar
    trade_dates=[]
    for j in range(entry_i, len(x)):
        dt=pd.Timestamp(x.index[j]).date()
        if dt>entry_date and dt not in trade_dates:
            trade_dates.append(dt)
    fallback_date=trade_dates[max_hold_days-1] if len(trade_dates)>=max_hold_days else None
    fallback_i=None
    if fallback_date is not None:
        idxs=[j for j in range(entry_i, len(x)) if pd.Timestamp(x.index[j]).date()==fallback_date]
        if idxs:
            fallback_i=max(idxs)

    below_entry_streak=0

    for j in range(entry_i, len(x)):
        if fallback_i is not None and j>=fallback_i:
            return fallback_i, "第5日兜底", "close"

        close=float(pd.to_numeric(pd.Series([x["Close"].iloc[j]]),errors="coerce").iloc[0])
        k=float(pd.to_numeric(pd.Series([x["K"].iloc[j]]),errors="coerce").iloc[0]) if "K" in x else float("nan")
        dd=float(pd.to_numeric(pd.Series([x["D"].iloc[j]]),errors="coerce").iloc[0]) if "D" in x else float("nan")
        dead=bool(x["KD_DEAD"].iloc[j])

        if np.isfinite(close) and close < entry_price:
            below_entry_streak += 1
        else:
            below_entry_streak = 0

        # 停利：保留目前已驗證版本
        take_profit = (
            exit_mode in [
                "KD停利+2根跌破停損+5日兜底",
                "KD停利+-3%停損+5日兜底",
                "KD停利+-5%停損+5日兜底",
                "只測KD停利+5日兜底",
            ]
            and np.isfinite(close) and close>=entry_price
            and np.isfinite(k) and k>80
            and dead
        )

        # 停損：完成60m後確認，下一根Open成交
        stop_2bars = (
            exit_mode=="KD停利+2根跌破停損+5日兜底"
            and below_entry_streak>=2
            and np.isfinite(k) and np.isfinite(dd) and k<dd
        )
        stop_3pct = (
            exit_mode=="KD停利+-3%停損+5日兜底"
            and np.isfinite(close) and close <= entry_price*0.97
            and np.isfinite(k) and np.isfinite(dd) and k<dd
        )
        stop_5pct = (
            exit_mode=="KD停利+-5%停損+5日兜底"
            and np.isfinite(close) and close <= entry_price*0.95
            and np.isfinite(k) and np.isfinite(dd) and k<dd
        )

        if j+1>=len(x):
            break

        if take_profit:
            return j+1, "KD>80死叉停利", "open"
        if stop_2bars:
            return j+1, "連續2根跌破進場價+K<D停損", "open"
        if stop_3pct:
            return j+1, "-3%且K<D停損", "open"
        if stop_5pct:
            return j+1, "-5%且K<D停損", "open"

    if fallback_i is not None:
        return fallback_i, "第5日兜底", "close"
    return None,None,None



def backtest_with_mask(d: pd.DataFrame, sig: pd.Series, hold_days: int, cost: CostConfig) -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame()
    rows=[]
    last_exit_i=-1
    mode=f"{int(hold_days)}日"
    for i in np.flatnonzero(sig.fillna(False).to_numpy()):
        entry_i=int(i)+1
        if entry_i>=len(d) or entry_i<=last_exit_i:
            continue
        exit_i=find_exit_index(d, entry_i, mode, "60m")
        if exit_i is None or exit_i<=entry_i:
            continue
        entry=float(d["Open"].iloc[entry_i])
        exitp=float(d["Close"].iloc[exit_i])
        if not np.isfinite(entry) or entry<=0 or not np.isfinite(exitp):
            continue
        gross=(exitp/entry-1)*100
        cost_pct=cost.roundtrip_cost_pct(daytrade=False)
        path=d.iloc[entry_i:exit_i+1]
        rows.append({
            "訊號時間":d.index[i],
            "進場時間":d.index[entry_i],
            "出場時間":d.index[exit_i],
            "持有":mode,
            "進場價":entry,
            "出場價":exitp,
            "毛報酬%":gross,
            "成本%":cost_pct,
            "淨報酬%":gross-cost_pct,
            "MFE%":(float(path["High"].max())/entry-1)*100,
            "MAE%":(float(path["Low"].min())/entry-1)*100,
            "訊號K":float(d["K"].iloc[i]) if pd.notna(d["K"].iloc[i]) else np.nan,
            "訊號D":float(d["D"].iloc[i]) if pd.notna(d["D"].iloc[i]) else np.nan,
            "量比20":float(d["VOL_RATIO20"].iloc[i]) if "VOL_RATIO20" in d and pd.notna(d["VOL_RATIO20"].iloc[i]) else np.nan,
            "MA60斜率3":float(d["MA60_SLOPE3"].iloc[i]) if "MA60_SLOPE3" in d and pd.notna(d["MA60_SLOPE3"].iloc[i]) else np.nan,
        })
        last_exit_i=exit_i
    return pd.DataFrame(rows)

def backtest_dynamic_exit(
    d: pd.DataFrame,
    sig: pd.Series,
    cost: CostConfig,
    exit_mode: str,
) -> pd.DataFrame:
    """下一根60m Open進場；動態停利/停損 + 第5日兜底，並追蹤停損反事實結果。"""
    if d is None or d.empty:
        return pd.DataFrame()

    x=add_kd_death_cross_flags(d)
    rows=[]
    last_exit_i=-1

    for i in np.flatnonzero(sig.fillna(False).to_numpy()):
        entry_i=int(i)+1
        if entry_i>=len(x) or entry_i<=last_exit_i:
            continue

        entry=float(x["Open"].iloc[entry_i])
        if not np.isfinite(entry) or entry<=0:
            continue

        exit_i,reason,fill_at=find_dynamic_exit(x,entry_i,entry,exit_mode,max_hold_days=5)
        if exit_i is None or exit_i<=entry_i:
            continue

        exitp=float(x["Open"].iloc[exit_i]) if fill_at=="open" else float(x["Close"].iloc[exit_i])
        if not np.isfinite(exitp):
            continue

        gross=(exitp/entry-1)*100
        cost_pct=cost.roundtrip_cost_pct(daytrade=False)
        path=x.iloc[entry_i:exit_i+1]

        # 反事實：若不在停損點出場，而是照原baseline抱到第5日，結果會怎樣
        cf_exit_i=find_exit_index(x, entry_i, "5日", "60m")
        cf_exit_price=np.nan
        cf_net=np.nan
        cf_delta=np.nan
        cf_recovered=np.nan
        if cf_exit_i is not None and cf_exit_i>entry_i:
            cf_exit_price=float(x["Close"].iloc[cf_exit_i])
            if np.isfinite(cf_exit_price):
                cf_net=(cf_exit_price/entry-1)*100-cost_pct
                cf_delta=cf_net-(gross-cost_pct)
                if "停損" in str(reason):
                    cf_recovered=bool(cf_net>0)

        rows.append({
            "訊號時間":x.index[i],
            "進場時間":x.index[entry_i],
            "出場時間":x.index[exit_i],
            "出場原因":reason,
            "成交依據":"下一根60m Open" if fill_at=="open" else "第5日最後60m Close",
            "進場價":entry,
            "出場價":exitp,
            "毛報酬%":gross,
            "成本%":cost_pct,
            "淨報酬%":gross-cost_pct,
            "MFE%":(float(path["High"].max())/entry-1)*100,
            "MAE%":(float(path["Low"].min())/entry-1)*100,
            "若抱到第5日出場價":cf_exit_price,
            "若抱到第5日淨報酬%":cf_net,
            "停損相對第5日改善pp":-cf_delta if ("停損" in str(reason) and pd.notna(cf_delta)) else np.nan,
            "停損後若續抱會轉正":cf_recovered,
            "訊號K":float(x["K"].iloc[i]) if pd.notna(x["K"].iloc[i]) else np.nan,
            "訊號D":float(x["D"].iloc[i]) if pd.notna(x["D"].iloc[i]) else np.nan,
            "量比20":float(x["VOL_RATIO20"].iloc[i]) if "VOL_RATIO20" in x and pd.notna(x["VOL_RATIO20"].iloc[i]) else np.nan,
            "MA60斜率3":float(x["MA60_SLOPE3"].iloc[i]) if "MA60_SLOPE3" in x and pd.notna(x["MA60_SLOPE3"].iloc[i]) else np.nan,
        })
        last_exit_i=exit_i

    return pd.DataFrame(rows)



def strategy_lab_exit_reason_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """候選策略依出場原因拆解績效。"""
    if trades is None or trades.empty or "出場原因" not in trades.columns:
        return pd.DataFrame()
    rows=[]
    for (sample,reason),g in trades.dropna(subset=["出場原因"]).groupby(["樣本","出場原因"],dropna=False):
        m=strategy_lab_metrics(g)
        rows.append({
            "樣本":sample,
            "出場原因":reason,
            **m,
            "平均MAE%":float(pd.to_numeric(g["MAE%"],errors="coerce").mean()) if "MAE%" in g else np.nan,
            "平均MFE%":float(pd.to_numeric(g["MFE%"],errors="coerce").mean()) if "MFE%" in g else np.nan,
            "占候選交易比%":float(len(g)/max(1,len(trades[trades["樣本"]==sample]))*100),
        })
    return pd.DataFrame(rows)



def strategy_lab_stop_counterfactual_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """只看停損交易：比較實際停損 vs 如果繼續抱到第5日。"""
    if trades is None or trades.empty or "出場原因" not in trades.columns:
        return pd.DataFrame()
    x=trades[trades["出場原因"].astype(str).str.contains("停損",na=False)].copy()
    if x.empty:
        return pd.DataFrame()

    rows=[]
    for sample,g in x.groupby("樣本"):
        actual=pd.to_numeric(g["淨報酬%"],errors="coerce")
        cf=pd.to_numeric(g["若抱到第5日淨報酬%"],errors="coerce")
        improve=pd.to_numeric(g["停損相對第5日改善pp"],errors="coerce")
        recovered=g["停損後若續抱會轉正"].dropna().astype(bool)
        rows.append({
            "樣本":sample,
            "停損筆數":int(len(g)),
            "實際停損平均淨報酬%":float(actual.mean()) if actual.notna().any() else np.nan,
            "若抱到第5日平均淨報酬%":float(cf.mean()) if cf.notna().any() else np.nan,
            "停損平均改善pp":float(improve.mean()) if improve.notna().any() else np.nan,
            "停損有效比例%":float((improve>0).mean()*100) if improve.notna().any() else np.nan,
            "被停損後若續抱會轉正比例%":float(recovered.mean()*100) if len(recovered) else np.nan,
        })
    return pd.DataFrame(rows)



def strategy_lab_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"交易數":0,"勝率%":np.nan,"平均淨報酬%":np.nan,"PF":np.nan,"報酬中位數%":np.nan}
    r=pd.to_numeric(trades["淨報酬%"],errors="coerce").dropna()
    if r.empty:
        return {"交易數":0,"勝率%":np.nan,"平均淨報酬%":np.nan,"PF":np.nan,"報酬中位數%":np.nan}
    gp=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    pf=np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)
    return {
        "交易數":int(len(r)),
        "勝率%":float((r>0).mean()*100),
        "平均淨報酬%":float(r.mean()),
        "PF":float(pf) if np.isfinite(pf) else pf,
        "報酬中位數%":float(r.median()),
    }


def run_strategy_lab(ranked_pool: pd.DataFrame, cost: CostConfig, candidate_filter: str,
                     candidate_hold_days: int, candidate_exit_mode: str="固定持有N日", period: str="1y"):
    if ranked_pool is None or ranked_pool.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

    ranked_pool=ranked_pool.copy()
    symbols=ranked_pool["股票"].astype(str).tolist()
    rank_map=dict(zip(ranked_pool["股票"].astype(str),ranked_pool["流動性排名"].astype(int)))
    raw=download_intraday_batch(symbols,"60m",period)

    base_rows=[]; cand_rows=[]
    p=st.progress(0,text="策略實驗室：計算60m資料…")
    for n,symbol in enumerate(symbols,1):
        d=raw.get(symbol,pd.DataFrame())
        if d is None or d.empty:
            p.progress(n/max(1,len(symbols)),text=f"{symbol} 無資料｜{n}/{len(symbols)}")
            continue
        x=add_indicators(d)
        rank=int(rank_map.get(symbol,9999))
        base_mask=strategy_lab_base_mask(x,rank)
        cand_mask=base_mask & strategy_lab_extra_mask(x,candidate_filter)

        b=backtest_with_mask(x,base_mask,5,cost)
        if candidate_exit_mode=="固定持有N日":
            c=backtest_with_mask(x,cand_mask,int(candidate_hold_days),cost)
        else:
            c=backtest_dynamic_exit(x,cand_mask,cost,candidate_exit_mode)
        if not b.empty:
            b.insert(0,"股票",symbol); b["流動性排名"]=rank; b["方案"]="正式Baseline"; base_rows.append(b)
        if not c.empty:
            c.insert(0,"股票",symbol); c["流動性排名"]=rank; c["方案"]="候選策略"; cand_rows.append(c)
        p.progress(n/max(1,len(symbols)),text=f"{symbol}｜{n}/{len(symbols)}")
    p.empty()

    base=pd.concat(base_rows,ignore_index=True) if base_rows else pd.DataFrame()
    cand=pd.concat(cand_rows,ignore_index=True) if cand_rows else pd.DataFrame()
    if base.empty and cand.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

    all_t=pd.concat([base,cand],ignore_index=True)
    all_t["_signal_dt"]=_as_taipei_series(all_t["訊號時間"])

    # 重要：OOS切點固定由正式Baseline決定。
    # 候選出場若提早解除non-overlap，會產生更多後續交易；
    # 若把候選交易時間也拿來決定60/40切點，不同候選方案會得到不同OOS區間。
    _base_times=all_t.loc[all_t["方案"]=="正式Baseline","_signal_dt"].dropna()
    times=pd.Series(_base_times.drop_duplicates().sort_values().to_list())
    if len(times)<2:
        return pd.DataFrame(),pd.DataFrame(),all_t.drop(columns=["_signal_dt"],errors="ignore")

    cut_i=max(1,min(len(times)-1,int(len(times)*0.60)))
    cutoff=times.iloc[cut_i]
    all_t["樣本"]=np.where(all_t["_signal_dt"]<cutoff,"樣本內60%","樣本外40%")
    all_t["共同切割時間"]=cutoff

    summary=[]
    for scheme in ["正式Baseline","候選策略"]:
        for sample in ["樣本內60%","樣本外40%"]:
            part=all_t[(all_t["方案"]==scheme)&(all_t["樣本"]==sample)]
            summary.append({
                "方案":scheme,
                "樣本":sample,
                "額外進場條件":"無" if scheme=="正式Baseline" else candidate_filter,
                "持有天數/出場":"5日固定" if scheme=="正式Baseline" else (f"{int(candidate_hold_days)}日固定" if candidate_exit_mode=="固定持有N日" else candidate_exit_mode),
                "共同切割時間":cutoff,
                **strategy_lab_metrics(part),
            })
    summary_df=pd.DataFrame(summary)

    delta=[]
    for sample in ["樣本內60%","樣本外40%"]:
        b=summary_df[(summary_df["方案"]=="正式Baseline")&(summary_df["樣本"]==sample)]
        c=summary_df[(summary_df["方案"]=="候選策略")&(summary_df["樣本"]==sample)]
        if b.empty or c.empty:
            continue
        br=b.iloc[0]; cr=c.iloc[0]
        delta.append({
            "樣本":sample,
            "候選-基準 交易數":int(cr["交易數"])-int(br["交易數"]),
            "候選-基準 勝率pp":float(cr["勝率%"]-br["勝率%"]) if pd.notna(cr["勝率%"]) and pd.notna(br["勝率%"]) else np.nan,
            "候選-基準 平均淨報酬pp":float(cr["平均淨報酬%"]-br["平均淨報酬%"]) if pd.notna(cr["平均淨報酬%"]) and pd.notna(br["平均淨報酬%"]) else np.nan,
            "候選-基準 PF":float(cr["PF"]-br["PF"]) if pd.notna(cr["PF"]) and pd.notna(br["PF"]) and np.isfinite(cr["PF"]) and np.isfinite(br["PF"]) else np.nan,
        })

    return summary_df,pd.DataFrame(delta),all_t.drop(columns=["_signal_dt"],errors="ignore").reset_index(drop=True)


def smart_refresh_seconds(now_ts=None):
    """
    60分K策略的智慧刷新：
    - 平常 60 秒
    - 09:55~10:02 / 10:55~11:02 / 11:55~12:02 / 12:55~13:02：10 秒
    - 13:25~13:32：10 秒
    - 非交易時段：300 秒
    """
    now_ts = now_ts or pd.Timestamp.now(tz="Asia/Taipei")
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("Asia/Taipei")
    else:
        now_ts = now_ts.tz_convert("Asia/Taipei")

    if now_ts.weekday() >= 5:
        return 300, "非交易日｜300秒"

    hhmm = now_ts.hour * 100 + now_ts.minute
    windows = [
        (955, 1002),
        (1055, 1102),
        (1155, 1202),
        (1255, 1302),
        (1325, 1332),
    ]
    if any(a <= hhmm <= b for a, b in windows):
        return 10, "60分K收盤前後｜10秒"

    if 900 <= hhmm <= 1335:
        return 60, "盤中一般監控｜60秒"

    return 300, "非交易時段｜300秒"


def resolve_refresh_seconds(enabled, mode, fixed_seconds):
    if not enabled:
        return None, "自動刷新關閉"
    if mode == "智慧":
        return smart_refresh_seconds()
    sec = int(fixed_seconds)
    return sec, f"固定刷新｜{sec}秒"



st.title(f"⚡ {APP_NAME}")
st.caption(f"{APP_VERSION}｜🔒 正式核心策略｜全市場動態TOP150｜60m KD黃金交叉 + K<30｜持有5日")
st.info("正式策略baseline仍凍結。B線策略實驗室與正式雷達/Worker完全分離；研究通過後才考慮併入正式版。")

with st.sidebar:
    st.header("⚡ 黑嚕嚕短線雷達")
    simple_mode=st.radio("操作模式",["今日雷達","進階研究"],index=0)

    research_mode="策略凍結與即時規格"
    if simple_mode=="進階研究":
        research_mode=st.radio(
            "研究模式",
            ["Shioaji即時引擎","策略實驗室","策略凍結與即時規格","長期穩健度驗證","長期集中度健診"],
            index=0
        )
        captions={
            "Shioaji即時引擎":"Stage 4.4：Supabase共享狀態正式化；支援新版Publishable/Secret Key並保留舊Key相容。",
            "策略實驗室":"B線研究區：正式baseline鎖定；可測2根跌破、-3%、-5%三種延遲停損與停損反事實，不影響今日雷達與Worker。",
            "策略凍結與即時規格":"查看正式凍結參數與未來 Shioaji 即時行情架構。",
            "長期穩健度驗證":"固定正式策略，以1y 60m資料、最後9mo評估與6段時間檢查長期穩健度。",
            "長期集中度健診":"沿用長期樣本，檢查月度、股票貢獻與Top貢獻集中度。"
        }
        st.caption(captions[research_mode])

    with st.expander("💰 成本設定"):
        fee_discount=st.number_input("手續費折數",min_value=0.1,max_value=1.0,value=0.28,step=0.01)
        slip_bp=st.number_input("單邊滑價（bp）",min_value=0.0,max_value=30.0,value=5.0,step=1.0)
    cost=CostConfig(fee_discount=fee_discount,slippage_pct=slip_bp/10000)

    refresh_enabled=False
    refresh_mode="智慧"
    refresh_fixed=60
    if simple_mode=="進階研究" and research_mode=="Shioaji即時引擎":
        with st.expander("🔄 即時畫面刷新", expanded=True):
            refresh_enabled=st.toggle("自動刷新", value=True)
            refresh_mode=st.radio("刷新模式", ["智慧","固定"], horizontal=True, index=0)
            if refresh_mode=="固定":
                refresh_fixed=st.select_slider(
                    "固定刷新秒數",
                    options=[5,10,30,60,300],
                    value=60
                )
            _refresh_sec,_refresh_desc=resolve_refresh_seconds(
                refresh_enabled, refresh_mode, refresh_fixed
            )
            st.caption(f"目前：{_refresh_desc}")
            st.caption("建議60分K策略使用智慧模式；平常60秒即可，接近60分K收盤才加速到10秒。")

    lab_filter="無（正式baseline）"
    lab_hold_days=5
    lab_exit_mode="KD停利+2根跌破停損+5日兜底"
    lab_pool_n=50
    if simple_mode=="進階研究" and research_mode=="策略實驗室":
        with st.expander("🧪 B線策略實驗設定", expanded=True):
            lab_pool_n=st.select_slider("測試流動性前N名", options=[30,50,100,150], value=50)
            lab_filter=st.selectbox(
                "候選額外進場條件",
                ["無（正式baseline）","S級雙條件","量比20 >= 1.5","MA60斜率3 <= 0","站上VWAP","站上MA15","站上MA30"],
                index=0
            )
            lab_exit_mode=st.selectbox(
                "候選出場方式",
                [
                    "KD停利+2根跌破停損+5日兜底",
                    "KD停利+-3%停損+5日兜底",
                    "KD停利+-5%停損+5日兜底",
                    "只測KD停利+5日兜底",
                    "固定持有N日",
                ],
                index=0
            )
            lab_hold_days=5
            if lab_exit_mode=="固定持有N日":
                lab_hold_days=st.select_slider("候選固定持有天數", options=[3,4,5,6,7], value=5)
            else:
                st.caption("KD動態出場皆以第5個後續交易日最後一根60分K收盤作最晚兜底。")
            st.caption("KD停利/停損皆用完成60分K確認；觸發後在下一根60分K Open模擬出場，並追蹤『若不停損抱到第5日』的反事實結果。")
            st.caption("30/50檔適合快速篩選；要考慮納入正式策略，至少再跑TOP150與OOS/時間區塊/集中度驗證。")
            st.caption("V1.16.51起：60/40 OOS切點只由正式Baseline決定，所有候選出場方案共用同一切點，避免候選提早出場造成比較區間漂移。")

    if simple_mode=="今日雷達":
        btn_label="🔄 更新今日雷達"
    else:
        btn_label={
            "Shioaji即時引擎":"🔌 檢查本機Worker狀態",
            "策略實驗室":"🧪 執行Baseline vs 候選策略",
            "策略凍結與即時規格":"🔒 顯示正式凍結規格",
            "長期穩健度驗證":"🧱 執行1年長期穩健度驗證",
            "長期集中度健診":"🧮 執行長期集中度健診",
        }[research_mode]
    run=st.button(btn_label,type="primary",use_container_width=True)


# ============================================================
# 今日正式雷達
# ============================================================
if run and simple_mode=="今日雷達":
    with st.spinner("建立官方上市/上櫃動態股票池並掃描最新60m訊號…"):
        _dr,_dd,_pd,_pp,_env,_pe=build_formal_daily_radar(cost)
    st.session_state["st_v11635_daily"]={
        "radar":_dr,"scan_diag":_dd,"pool_diag":_pd,"pool":_pp,
        "environment":_env,"errors":_pe
    }

_daily=st.session_state.get("st_v11635_daily")
if simple_mode=="今日雷達":
    st.markdown("## 📡 今日60m正式雷達")
    st.caption("🔒 正式核心：TOP1-100全部；TOP101-150僅S級。TOP200市場廣度只做風險提示，不作硬Gate。")

    if not _daily:
        st.info("按左側「🔄 更新今日雷達」開始。")
    else:
        _dr=_daily.get("radar",pd.DataFrame())
        _dd=_daily.get("scan_diag",pd.DataFrame())
        _pd=_daily.get("pool_diag",pd.DataFrame())
        _pp=_daily.get("pool",pd.DataFrame())
        _env=_daily.get("environment",pd.DataFrame())
        _pe=_daily.get("errors",[])

        if _pe:
            st.warning("資料來源異常："+"；".join(_pe))

        if not _env.empty:
            e0=_env.iloc[0]
            st.markdown("### 🌦️ 今日市場環境")
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("市場狀態",str(e0.get("市場狀態","")))
            c2.metric("MA15廣度",f"{float(e0.get('站上MA15比例%',np.nan)):.1f}%")
            c3.metric("MA30廣度",f"{float(e0.get('站上MA30比例%',np.nan)):.1f}%")
            c4.metric("MA60廣度",f"{float(e0.get('站上MA60比例%',np.nan)):.1f}%")
            c5.metric("5日上漲家數",f"{float(e0.get('5日上漲家數比例%',np.nan)):.1f}%")
            if str(e0.get("市場狀態","")).startswith("🔴"):
                st.warning("MA60市場廣度偏高；目前只提示風險，不會自動刪除S/A/B訊號。")

            st.download_button(
                "⬇️ 下載【今日市場環境】",
                _env.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{APP_VERSION}_今日市場環境.csv",
                mime="text/csv",use_container_width=True,on_click="ignore"
            )

        if not _pd.empty:
            d0=_pd.iloc[0]
            c1,c2,c3,c4=st.columns(4)
            c1.metric("官方母池",int(d0.get("官方股票數",0)))
            c2.metric("可排名",int(d0.get("可排名股票數",0)))
            c3.metric("60m掃描",int(d0.get("正式掃描股票數",0)))
            c4.metric("正式訊號",len(_dr))
            st.caption(
                f"日K排名基準完成日：{d0.get('最新排名基準日','')}｜"
                "正式掃描：TOP150；市場廣度：TOP200"
            )

        if _dr.empty:
            st.info("本次掃描沒有仍在5交易日觀察窗內且符合正式架構的訊號。")
        else:
            new_n=int((_dr["目前狀態"]=="🟢 新訊號").sum())
            watch_n=int((_dr["目前狀態"]=="🟡 觀察中").sum())
            s_n=int((_dr["訊號等級"]=="S級").sum())
            a_n=int((_dr["訊號等級"]=="A級").sum())
            b_n=int((_dr["訊號等級"]=="B級").sum())
            x1,x2,x3,x4,x5=st.columns(5)
            x1.metric("今日新訊號",new_n)
            x2.metric("觀察中",watch_n)
            x3.metric("S級",s_n)
            x4.metric("A級",a_n)
            x5.metric("B級",b_n)

            cols=[c for c in [
                "股票","公司","市場","流動性排名","股票池層級","訊號等級",
                "市場狀態","市場MA60廣度%","目前狀態","訊號時間","訊號K",
                "量比20","MA60斜率3","目前K","目前D","預計觀察至"
            ] if c in _dr.columns]
            st.dataframe(_dr[cols],use_container_width=True,hide_index=True)
            st.download_button(
                "⬇️ 下載【今日60m正式雷達】",
                _dr.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{APP_VERSION}_今日60m正式雷達.csv",
                mime="text/csv",use_container_width=True,on_click="ignore"
            )

        if not _pp.empty:
            st.download_button(
                "⬇️ 下載【今日正式股票池】",
                _pp.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{APP_VERSION}_今日正式股票池.csv",
                mime="text/csv",use_container_width=True,on_click="ignore"
            )

            _sj_pool=_pp.copy()
            _sj_pool["代號"]=_sj_pool["股票"].astype(str).str.split(".").str[0]
            _sj_cols=[c for c in ["代號","股票","公司","市場","流動性排名","股票池層級","排名基準完成日"] if c in _sj_pool.columns]
            _sj_pool=_sj_pool[_sj_cols].sort_values("流動性排名").head(150)
            st.download_button(
                "⬇️ 下載【Shioaji_TOP150訂閱清單】",
                _sj_pool.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{APP_VERSION}_Shioaji_TOP150訂閱清單.csv",
                mime="text/csv",use_container_width=True,on_click="ignore"
            )
            st.caption("V1.16.40起這份CSV只作人工備援；本機Worker平常會自行重建每日TOP150，不必每天下載。")

        with st.expander("🔧 今日雷達掃描診斷"):
            if not _pd.empty:
                st.dataframe(_pd,use_container_width=True,hide_index=True)
            if not _dd.empty:
                ok_n=int((_dd["狀態"]=="掃描完成").sum()) if "狀態" in _dd.columns else 0
                st.caption(f"60m成功掃描：{ok_n}/{len(_dd)}")
                st.dataframe(_dd,use_container_width=True,hide_index=True)
                st.download_button(
                    "⬇️ 下載【今日60m掃描診斷】",
                    _dd.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{APP_VERSION}_今日60m掃描診斷.csv",
                    mime="text/csv",use_container_width=True,on_click="ignore"
                )


# ============================================================

# ============================================================
# B線：策略實驗室
# ============================================================
if run and simple_mode=="進階研究" and research_mode=="策略實驗室":
    with st.spinner("建立股票池並執行Baseline / 候選策略比較…"):
        _lab_pool,_lab_diag,_lab_err=build_current_formal_radar_pool(max_rank=int(lab_pool_n))
        _lab_summary,_lab_delta,_lab_detail=run_strategy_lab(
            _lab_pool,cost,lab_filter,int(lab_hold_days),lab_exit_mode,period="1y"
        )
    st.session_state["st_v11647_strategy_lab"]={
        "pool":_lab_pool,"pool_diag":_lab_diag,"errors":_lab_err,
        "summary":_lab_summary,"delta":_lab_delta,"detail":_lab_detail,
        "filter":lab_filter,"hold":lab_hold_days,"exit_mode":lab_exit_mode,"pool_n":lab_pool_n,
    }

if simple_mode=="進階研究" and research_mode=="策略實驗室":
    st.markdown("## 🧪 B線策略實驗室")
    st.caption("比較原則：同股票池、同歷史資料、同Baseline OOS切點；候選策略不得改變測試區間。")
    st.warning("這裡只做研究。正式今日雷達、Shioaji Worker與Telegram仍維持原凍結baseline。")
    st.caption("Baseline：TOP1–100＝KD黃金交叉+K<30；TOP101–150再要求S級；下一根60m Open進場；固定5個後續交易日出場。")

    _lab=st.session_state.get("st_v11647_strategy_lab")
    if not _lab:
        st.info("左側設定候選條件後，按「🧪 執行Baseline vs 候選策略」。")
    else:
        if _lab.get("errors"):
            st.warning("股票池資料來源提醒："+"；".join(_lab.get("errors",[])))

        c1,c2,c3=st.columns(3)
        c1.metric("測試股票數",_lab.get("pool_n"))
        c2.metric("候選進場條件",_lab.get("filter"))
        c3.metric("候選出場",_lab.get("exit_mode"))

        _sum=_lab.get("summary",pd.DataFrame())
        _delta=_lab.get("delta",pd.DataFrame())
        _detail=_lab.get("detail",pd.DataFrame())

        if _sum is None or _sum.empty:
            st.error("本次沒有足夠交易資料可比較。")
        else:
            st.markdown("### Baseline vs 候選策略")
            st.dataframe(_sum,use_container_width=True,hide_index=True)

            if _delta is not None and not _delta.empty:
                st.markdown("### 候選策略相對Baseline差異")
                st.dataframe(_delta,use_container_width=True,hide_index=True)

            _cand_detail=_detail[_detail["方案"]=="候選策略"].copy() if (_detail is not None and not _detail.empty and "方案" in _detail.columns) else pd.DataFrame()
            _exit_sum=strategy_lab_exit_reason_summary(_cand_detail)
            if not _exit_sum.empty:
                st.markdown("### 出場原因拆解")
                st.dataframe(_exit_sum,use_container_width=True,hide_index=True)
                st.caption("這張表用來判斷究竟是停利、停損還是第5日兜底在改善/拖累績效。")

            _cf=strategy_lab_stop_counterfactual_summary(_cand_detail)
            if not _cf.empty:
                st.markdown("### 停損反事實追蹤")
                st.dataframe(_cf,use_container_width=True,hide_index=True)
                st.caption("停損有效比例＝實際停損結果優於同一筆交易若繼續抱到第5日的比例；越高代表停損比較像真的有幫助，而不是單純提早認賠。")

            oos=_sum[_sum["樣本"]=="樣本外40%"]
            if len(oos)>=2:
                b=oos[oos["方案"]=="正式Baseline"]
                c=oos[oos["方案"]=="候選策略"]
                if not b.empty and not c.empty:
                    b=b.iloc[0]; c=c.iloc[0]
                    st.markdown("### OOS判讀")
                    st.caption(
                        f"Baseline：交易 {int(b['交易數'])}｜平均淨報酬 {b['平均淨報酬%']:.3f}%｜PF {b['PF']:.3f}；"
                        f"候選：交易 {int(c['交易數'])}｜平均淨報酬 {c['平均淨報酬%']:.3f}%｜PF {c['PF']:.3f}。"
                    )
                    st.info("不會自動宣布候選策略勝出；要納入正式版，還需TOP150、時間區塊、集中度與資料品質驗證。")

            if _detail is not None and not _detail.empty:
                st.download_button(
                    "⬇️ 下載策略實驗交易明細",
                    data=_detail.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{APP_VERSION}_策略實驗室交易明細.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# 進階研究：正式研究3項 + Shioaji Stage 1
# ============================================================

if simple_mode=="進階研究" and research_mode=="Shioaji即時引擎":
    st.markdown("## 🧪 Shioaji Stage 4.4｜模擬交易＋雲端共享狀態")
    st.warning("Stage 4 只做程式內模擬成交，不呼叫Shioaji place_order/update_order/cancel_order。正式訊號在下一根60m K第一筆Tick模擬建倉，建倉/出場後可推送Telegram。")

    st.markdown("### 操作檢查表")
    st.dataframe(get_shioaji_stage1_checklist(),use_container_width=True,hide_index=True)

    st.markdown("### 🔎 Worker連線診斷")
    _diag=diagnose_worker_runtime()
    if _diag["level"]=="success":
        st.success(_diag["diagnosis"])
    elif _diag["level"]=="warning":
        st.warning(_diag["diagnosis"])
    else:
        st.info(_diag["diagnosis"])

    st.caption(_diag["suggestion"])
    with st.expander("查看診斷路徑"):
        st.code(
            f"Streamlit工作目錄：{_diag['cwd']}\n"
            f"預期runtime：{_diag['runtime_dir']}\n"
            f"預期狀態檔：{_diag['status_path']}\n"
            f"runtime存在：{_diag['runtime_exists']}\n"
            f"status存在：{_diag['status_exists']}\n"
            f"推測Cloud環境：{_diag['likely_cloud']}\n"
            f"共享橋接已設定：{_diag['shared_configured']}\n"
            f"目前共享來源：{_diag['shared_source']}"
        )

    st.markdown("### Worker 即時狀態")

    _refresh_sec,_refresh_desc=resolve_refresh_seconds(
        refresh_enabled, refresh_mode, refresh_fixed
    )
    st.caption(f"刷新策略：{_refresh_desc}")

    @st.fragment(run_every=_refresh_sec)
    def render_worker_live_panel():
        runtime,err=read_worker_runtime_unified()
        status=runtime.get("status",{}) if runtime else {}
        _source=runtime.get("_source","") if runtime else ""

        if err or not status:
            st.info("目前尚未取得Worker共享狀態。請查看上方【Worker連線診斷】。")
            if err:
                st.caption(f"偵測結果：{err}")
            return

        source_label="本機runtime" if _source=="local" else "Supabase共享狀態"
        phase=str(status.get("phase",""))
        if phase in ["running","subscribed"]:
            st.success(f"Worker 狀態：{phase}｜來源：{source_label}")
        elif phase=="error":
            st.error(str(status.get("message","Worker發生錯誤")))
        else:
            st.info(f"Worker 狀態：{phase}｜來源：{source_label}")

        st.dataframe(
            shioaji_stage1_status_table(status),
            use_container_width=True,
            hide_index=True
        )

        _sim_positions=runtime.get("sim_positions",{}) if runtime else {}
        _sim_pending=runtime.get("sim_pending",{}) if runtime else {}

        c1,c2,c3,c4=st.columns(4)
        c1.metric("模擬持倉", len(_sim_positions) if isinstance(_sim_positions,dict) else 0)
        c2.metric("待進場", len(_sim_pending) if isinstance(_sim_pending,dict) else 0)
        c3.metric("正式訊號", int(status.get("formal_signal_count",0) or 0))
        c4.metric("已完成模擬交易", int(status.get("sim_closed_trade_count",0) or 0))

        if isinstance(_sim_positions,dict) and _sim_positions:
            st.markdown("#### 🧪 模擬持倉")
            st.dataframe(pd.DataFrame(list(_sim_positions.values())),use_container_width=True,hide_index=True)

        if isinstance(_sim_pending,dict) and _sim_pending:
            st.markdown("#### ⏳ 待模擬進場")
            st.dataframe(pd.DataFrame(list(_sim_pending.values())),use_container_width=True,hide_index=True)

        _now_tw=pd.Timestamp.now(tz="Asia/Taipei")
        st.caption(f"畫面更新時間：{_now_tw.strftime('%Y-%m-%d %H:%M:%S')}｜資料來源：{source_label}")

    render_worker_live_panel()

    st.markdown("### Stage 4.1 驗證重點")
    st.caption(
        "確認：① TOP150自動建池正常、② 訂閱接近150/150、③ 60m暖機成功、④ Tick與60m K持續更新、"
        "⑤ 正式訊號→下一根60m第一筆Tick模擬建倉、⑥ Telegram建倉/出場通知、⑦ 第5個後續交易日出場正確。"
    )

if simple_mode=="進階研究" and research_mode=="策略凍結與即時規格":
    st.markdown("## 🔒 正式核心策略")
    if run:
        cfg=get_frozen_strategy_config()
        cfg_df=pd.DataFrame([{"項目":k,"設定":val} for k,val in cfg.items()])
        rt=get_shioaji_realtime_spec()

        st.dataframe(cfg_df,use_container_width=True,hide_index=True)
        st.markdown("### 🔌 Shioaji 即時串接規格")
        st.dataframe(rt,use_container_width=True,hide_index=True)
        st.info("第一階段只做即時行情、雷達與通知；自動下單仍保持關閉。")

        st.download_button(
            "⬇️ 下載【正式策略凍結設定】",
            cfg_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{APP_VERSION}_正式策略凍結設定.csv",
            mime="text/csv",use_container_width=True,on_click="ignore"
        )
        st.download_button(
            "⬇️ 下載【Shioaji即時串接規格】",
            rt.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{APP_VERSION}_Shioaji即時串接規格.csv",
            mime="text/csv",use_container_width=True,on_click="ignore"
        )
    else:
        st.info("核心參數已凍結；按左側按鈕可查看與下載正式設定。")


if run and simple_mode=="進階研究" and research_mode=="長期穩健度驗證":
    st.subheader("🧱 1年長期穩健度驗證")
    st.warning("此模式會下載較多1y 60m資料；請避免同時開多個分頁重跑。")
    with st.spinner("建立12mo歷史流動性資格＋1y 60m資料，評估最後9mo正式架構C…"):
        _lr_sum,_lr_blocks,_lr_detail,_lr_errs=validate_long_horizon_robustness(cost)
    st.session_state["st_v11635_long"]={
        "summary":_lr_sum,"blocks":_lr_blocks,"detail":_lr_detail,"errors":_lr_errs
    }

_lr=st.session_state.get("st_v11635_long")
if simple_mode=="進階研究" and research_mode=="長期穩健度驗證":
    if not _lr:
        st.info("固定正式策略不調參；按左側按鈕重新驗證長期時間穩定度。")
    else:
        ls=_lr.get("summary",pd.DataFrame())
        lb=_lr.get("blocks",pd.DataFrame())
        ld=_lr.get("detail",pd.DataFrame())
        le=_lr.get("errors",[])
        if le:
            st.warning("資料來源異常："+"；".join(le))
        if not ls.empty:
            st.markdown("#### 全部 / 樣本內 / 樣本外")
            st.dataframe(ls.round(3),use_container_width=True,hide_index=True)
        if not lb.empty:
            st.markdown("#### 最後9個月｜6段時間穩定度")
            st.dataframe(lb.round(3),use_container_width=True,hide_index=True)
        with st.expander("查看長期逐筆交易"):
            st.dataframe(ld.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【長期穩健度摘要】",ls.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_長期穩健度摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
        st.download_button("⬇️ 下載【長期六段穩定度】",lb.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_長期六段穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
        st.download_button("⬇️ 下載【長期逐筆交易】",ld.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_長期逐筆交易.csv",mime="text/csv",use_container_width=True,on_click="ignore")


if run and simple_mode=="進階研究" and research_mode=="長期集中度健診":
    st.subheader("🧮 長期報酬分布 / 集中度健診")
    with st.spinner("沿用長期正式架構C，計算月度穩定度、股票貢獻與Top貢獻移除測試…"):
        _lc_month,_lc_stock,_lc_infl,_lc_dist,_lc_errs=validate_long_horizon_concentration(cost)
    st.session_state["st_v11635_concentration"]={
        "monthly":_lc_month,"stocks":_lc_stock,"influence":_lc_infl,
        "distribution":_lc_dist,"errors":_lc_errs
    }

_lc=st.session_state.get("st_v11635_concentration")
if simple_mode=="進階研究" and research_mode=="長期集中度健診":
    if not _lc:
        st.info("此頁只驗證長期報酬分布與集中度，不新增策略條件。")
    else:
        lm=_lc.get("monthly",pd.DataFrame())
        ls=_lc.get("stocks",pd.DataFrame())
        li=_lc.get("influence",pd.DataFrame())
        ld=_lc.get("distribution",pd.DataFrame())
        le=_lc.get("errors",[])
        if le:
            st.warning("資料來源異常："+"；".join(le))
        if not ld.empty:
            st.markdown("#### 全體分布摘要")
            st.dataframe(ld.round(3),use_container_width=True,hide_index=True)
        if not li.empty:
            st.markdown("#### 移除Top正貢獻股票後")
            st.dataframe(li.round(3),use_container_width=True,hide_index=True)
        if not lm.empty:
            st.markdown("#### 月度穩定度")
            st.dataframe(lm.round(3),use_container_width=True,hide_index=True)
        with st.expander("查看各股票長期貢獻"):
            st.dataframe(ls.round(3),use_container_width=True,hide_index=True)

        st.download_button("⬇️ 下載【長期分布摘要】",ld.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_長期分布摘要.csv",mime="text/csv",use_container_width=True,on_click="ignore")
        st.download_button("⬇️ 下載【長期月度穩定度】",lm.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_長期月度穩定度.csv",mime="text/csv",use_container_width=True,on_click="ignore")
        st.download_button("⬇️ 下載【長期股票貢獻】",ls.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_長期股票貢獻.csv",mime="text/csv",use_container_width=True,on_click="ignore")
        st.download_button("⬇️ 下載【長期Top貢獻移除測試】",li.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_長期Top貢獻移除測試.csv",mime="text/csv",use_container_width=True,on_click="ignore")


with st.expander("🧹 V1.16.42 已移除項目"):
    st.caption(
        "已從程式與進階選單移除：多週期當沖/隔日、單股/跨股舊回測、股票池1.x/2.0探索、"
        "TOP50暖機/品質/Gate拆解、環境Gate/市場轉折、持有天數、獲利保護、固定停損、"
        "延遲Time-Stop等已完成且未納入正式基準的研究流程。"
    )
