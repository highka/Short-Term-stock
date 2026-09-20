# -*- coding: utf-8 -*-
"""
黑嚕嚕－短線交易雷達 ST V1.8.1
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

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

APP_VERSION = "ST V1.8.1"
APP_NAME = "黑嚕嚕－短線交易雷達"
MA_LIST = [5, 15, 30, 60, 200]
INTERVALS = ["5m", "15m", "60m"]

APP_VERSION = "ST_V1.8.1"
EXPORT_PREFIX = "ST_V1.8.1"

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
st.caption(f"{APP_VERSION}｜獨立短線研究版｜MA5 / 15 / 30 / 60 / 200 + KD｜5m / 15m / 60m")

st.info(
    "本版先做『基準模型』：刻意不加入 VWAP、MACD、ATR、量比等其他指標。"
    "先驗證 MA + KD 在不同K棒與持有方式的表現，再依結果決定新增什麼，避免一開始過度最佳化。"
)

with st.sidebar:
    st.header("研究設定")
    st.info("V1.5.0 預設進入【60m五日OOS驗證】；多週期當沖線已完成初步驗證，暫不繼續加參數。")
    research_mode = st.radio("研究模式", ["單一股票", "跨股票批次", "多週期當沖/隔日驗證", "60m五日OOS驗證"], index=3, horizontal=True)
    code = st.text_input("股票代號", value="2330")
    pool_mode = st.radio(
        "批次股票池",
        ["手動輸入", "族群代表池", "動態短線TOP池"],
        horizontal=True,
        disabled=(research_mode != "跨股票批次"),
    )
    batch_text = st.text_area(
        "批次股票代號",
        value="2330,2357,3711,2317,2454,2382",
        help="用逗號分隔。",
        disabled=(research_mode != "跨股票批次" or pool_mode != "手動輸入"),
    )
    selected_sectors = st.multiselect(
        "選擇族群",
        list(SECTOR_POOLS.keys()),
        default=["半導體/晶圓", "IC設計", "AI伺服器/ODM", "電腦品牌/板卡", "封測", "航運"],
        disabled=(research_mode != "跨股票批次" or pool_mode != "族群代表池"),
    )
    per_sector = st.slider(
        "每族群代表檔數",
        1, 4, 3,
        disabled=(research_mode != "跨股票批次" or pool_mode != "族群代表池"),
    )
    top_n = st.slider(
        "動態短線池檔數",
        10, 100, 50, step=10,
        disabled=not (research_mode in ["多週期當沖/隔日驗證","60m五日OOS驗證"] or (research_mode == "跨股票批次" and pool_mode == "動態短線TOP池")),
        help="先由候選母池用近期成交金額、成交量與振幅排序，再對入選股票執行分K策略健診。",
    )
    market = st.radio("市場", ["上市", "上櫃"], horizontal=True)
    symbol = normalize_symbol(code, market)

    selected_intervals = st.multiselect(
        "K棒週期", INTERVALS, default=["60m"],
        help="V1.2.7 預設只驗證目前最穩定的60m；需要對照時仍可自行加入15m/5m。"
    )
    period = st.selectbox("研究資料長度", ["60d", "1mo"], index=0)

    all_rules = [
        "MA5上穿MA15",
        "MA15上穿MA30",
        "KD黃金交叉",
        "MA5>15 + KD黃金交叉",
        "MA5>15>30 + KD黃金交叉",
        "MA5>15>30>60 + KD黃金交叉",
        "完整多頭排列",
        "完整多頭排列 + KD黃金交叉",
        "站上MA200 + KD黃金交叉",
        "KD黃金交叉 + K<30",
        "KD黃金交叉 + K30-50",
        "KD黃金交叉 + K50-80",
        "KD黃金交叉 + K>80",
        "KD黃金交叉 + MA30向上",
        "KD黃金交叉 + MA60向上",
        "KD黃金交叉 + 量比>1.2",
        "KD黃金交叉 + 量比>1.5",
        "KD黃金交叉 + 站上VWAP",
        "MA5>15 + KD + 站上VWAP",
    ]
    selected_rules = st.multiselect(
        "進場規則",
        all_rules,
        default=["KD黃金交叉 + K<30"],
        help="V1.2.7 先鎖定已通過30檔驗證的核心規則，避免同時測太多條件造成多重比較偏誤。",
    )
    overlap_mode = st.radio(
        "持倉期間新訊號處理",
        ["禁止重疊（較接近實際單一持倉）", "允許重疊（訊號事件研究）"],
        horizontal=True,
        help="禁止重疊：同一股票持倉尚未結束時忽略新訊號；允許重疊：保留舊版訊號事件統計。",
    )
    allow_overlap = overlap_mode.startswith("允許")

    selected_modes = st.multiselect(
        "持有方式",
        ["當沖", "隔日", "2日", "3日", "4日", "5日", "6日", "7日"],
        default=["4日", "5日", "6日", "7日"],
    )

    if research_mode == "跨股票批次":
        st.caption("V1.2.7 驗證門檻：優先觀察 60m｜KD黃金交叉+K<30｜4~7日；若TOP50仍維持正期望股票比例≥70%、期望中位數>0、PF中位數>1.2，再進入下一階段。")

    if research_mode == "多週期當沖/隔日驗證":
        st.markdown("#### V1.3.0 多週期進場")
        st.caption("固定60m K<30＋15m KD黃金交叉；V1.3.3加入「15m確認後首根5m」對照組，檢查等待5m KD是否反而延遲進場。")
        mtf_entry_rules = st.multiselect(
            "5分鐘進場觸發",
            ["15m確認後首根5m", "5m KD黃金交叉", "5m MA5上穿MA15", "5m KD黃金交叉+站上VWAP"],
            default=["15m確認後首根5m", "5m KD黃金交叉"],
        )
        mtf_modes = st.multiselect(
            "短線出場方式",
            ["當沖", "隔日", "2日"],
            default=["當沖", "隔日", "2日"],
        )
    else:
        mtf_entry_rules = []
        mtf_modes = []

    st.divider()
    st.subheader("交易成本")
    fee_discount = st.number_input("手續費折數", min_value=0.1, max_value=1.0, value=0.28, step=0.01)
    slip_bp = st.number_input("單邊滑價（bp）", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
    cost = CostConfig(fee_discount=fee_discount, slippage_pct=slip_bp / 10000)

    combo_est = max(1, len(selected_intervals)) * max(1, len(selected_rules)) * max(1, len(selected_modes))
    if research_mode == "跨股票批次" and pool_mode == "動態短線TOP池":
        st.info(f"本次預計：TOP {top_n} × {len(selected_intervals)}週期 × {len(selected_rules)}規則 × {len(selected_modes)}持有方式。V1.2.7 建議 TOP50，用來確認30檔結果的樣本穩健度。")
        if top_n > 50:
            st.warning("目前仍使用 yfinance。若超過50檔遇到下載限制或執行時間過長，先維持50檔即可。")
    run = st.button("🚀 開始策略健診", type="primary", use_container_width=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📊 單股總表", "🌐 跨股穩定度", "🔥 動態短線池", "🏭 族群比較", "🧬 狀態分類", "📈 K線/KD", "🔬 KD分區", "📦 量價/斜率", "🧾 交易明細"])

if run:
    if research_mode not in ["多週期當沖/隔日驗證","60m五日OOS驗證"] and (not selected_intervals or not selected_rules or not selected_modes):
        st.error("請至少選擇一個K棒週期、進場規則與持有方式。")
        st.stop()

    if research_mode == "60m五日OOS驗證":
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
        st.session_state["st_v181_oos"] = {
            "detail": oos_detail, "summary": oos_summary, "trades": oos_trades,
            "blocks": block_summary, "state_diag": state_diag,
            "filter_robust": filter_robust, "market_diag": market_diag,
            "market_blocks": market_blocks,
            "wf_summary": wf_summary, "wf_detail": wf_detail, "wf_status": wf_status, "wf_time": wf_time,
            "radar_candidates": radar_candidates, "radar_validation": radar_validation,
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

oos_state = st.session_state.get("st_v181_oos")
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
    if not osum.empty:
        st.dataframe(osum.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【OOS驗證總表】",osum.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_OOS驗證總表.csv",mime="text/csv",use_container_width=True)
    if not oblocks.empty:
        st.markdown("### 四段時間穩定度")
        st.dataframe(oblocks.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【四段時間穩定度】",oblocks.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_四段時間穩定度.csv",mime="text/csv",use_container_width=True)
    if not ostate.empty:
        st.markdown("### 訊號當下狀態診斷")
        st.caption("這張表只找失效環境，不會自動把表現最好的分組變成新策略，避免事後挑條件。")
        st.dataframe(ostate.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【訊號狀態診斷】",ostate.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_訊號狀態診斷.csv",mime="text/csv",use_container_width=True)
    if not ofilter.empty:
        st.markdown("### 候選市場狀態跨時間驗證")
        st.caption("候選A~D只是V1.5.1產生的假說。這裡檢查各候選在四段時間的交易數、涵蓋率、PF與報酬，不會自動選冠軍或直接變成正式策略。")
        st.dataframe(ofilter.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【候選市場狀態驗證】",ofilter.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_候選市場狀態驗證.csv",mime="text/csv",use_container_width=True)
    if not omarket.empty:
        st.markdown("### 大盤環境診斷")
        st.caption("使用 ^TWII 前一個已完成交易日的日線，只做市場regime診斷，不直接最佳化成濾網。")
        st.dataframe(omarket.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【大盤環境診斷】",omarket.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_大盤環境診斷.csv",mime="text/csv",use_container_width=True)
    if not omarket_blocks.empty:
        st.markdown("### 大盤環境 × 四段時間")
        st.caption("直接檢查每一時間區段內的大盤regime與策略績效，避免把全期間相關性誤認成失效原因。")
        st.dataframe(omarket_blocks.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【大盤環境四段交叉驗證】",omarket_blocks.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_大盤環境四段交叉驗證.csv",mime="text/csv",use_container_width=True)
    st.markdown("### Walk-Forward 歷史可交易性驗證")
    if owf.empty:
        st.error(f"Walk-Forward未產生結果：{owf_status}")
    else:
        st.success(owf_status)
    if not owf.empty:
        st.caption("每筆訊號只使用訊號日前一交易日以前的20日日線資訊，重建當時的成交金額/成交量/振幅相對排名。這是在目前候選universe內的歷史重建，不宣稱等同完整歷史上市櫃成分。")
        st.dataframe(owf.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【WalkForward股票池驗證】",owf.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_WalkForward股票池驗證.csv",mime="text/csv",use_container_width=True)
    if not owfd.empty:
        st.download_button("⬇️ 下載【WalkForward逐筆明細】",owfd.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_WalkForward逐筆明細.csv",mime="text/csv",use_container_width=True)
    if not owf_time.empty:
        st.markdown("### Walk-Forward 分層 × 四段時間")
        st.caption("檢查低/中/高可交易性分層是否跨時間一致；不因全期間某一層績效漂亮就直接改股票池。")
        st.dataframe(owf_time.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【WalkForward分層四段驗證】",owf_time.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_WalkForward分層四段驗證.csv",mime="text/csv",use_container_width=True)
    if not oradar_val.empty:
        st.markdown("### V1.8.0 雷達排序驗證")
        st.caption("檢查舊版『新鮮度70%＋K深度30%』是否真的能把較佳結果排前面；本版不再把此分數當正式排序。")
        st.dataframe(oradar_val.round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【雷達排序驗證】",oradar_val.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_雷達排序驗證.csv",mime="text/csv",use_container_width=True)
    if not oradar.empty:
        st.markdown("### 研究雷達候選")
        st.caption("核心規則仍固定60m KD黃金交叉＋K<30。V1.8.1取消預測性綜合分數，候選只按訊號時間由新到舊；WF、K深度與新鮮度保留為描述欄位。")
        st.dataframe(oradar.head(100).round(3),use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下載【研究雷達候選】",oradar.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_研究雷達候選.csv",mime="text/csv",use_container_width=True)
    if not odet.empty:
        st.download_button("⬇️ 下載【OOS個股策略明細】",odet.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_OOS個股策略明細.csv",mime="text/csv",use_container_width=True)
    if not otr.empty:
        st.download_button("⬇️ 下載【OOS逐筆交易明細】",otr.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{APP_VERSION}_OOS逐筆交易明細.csv",mime="text/csv",use_container_width=True)
    st.info("下一輪請提供：①【雷達排序驗證】②【研究雷達候選】③【WalkForward逐筆明細】。本版先驗證並取消未被證明的綜合排名，下一步再設計即時雷達的訊號狀態與風險欄位。")

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
    "ST V1.8.1 僅供策略研究與程式驗證，不送出證券委託。"
    "下一階段將根據實際回測結果，再判斷是否增加 VWAP、成交量/量比、MACD、ATR 或其他參數。"
)
