# -*- coding: utf-8 -*-
"""
黑嚕嚕－短線交易雷達 ST V1.2.0
獨立短線研究版：V1.2 跨股票策略健診；不沿用原黑嚕嚕 V3.x 策略/分數/帳本。

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

APP_VERSION = "ST V1.2.0"
APP_NAME = "黑嚕嚕－短線交易雷達"
MA_LIST = [5, 15, 30, 60, 200]
INTERVALS = ["5m", "15m", "60m"]

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
    }
    return rules.get(rule, false).fillna(False)


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

    day_map = {"隔日": 1, "2日": 2, "3日": 3, "5日": 5}
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
    return g


def run_batch_matrix(symbols: List[str], intervals: List[str], rules: List[str], modes: List[str],
                     cost: CostConfig, period: str):
    all_rows = []
    states = []
    total = max(1, len(symbols))
    p = st.progress(0, text="跨股票驗證中…")

    for idx, symbol in enumerate(symbols, 1):
        local_data = {}
        for interval in intervals:
            raw = download_intraday(symbol, interval, period)
            local_data[interval] = add_indicators(raw) if not raw.empty else pd.DataFrame()

        state = classify_stock_state(local_data)
        states.append({"股票": symbol, **state})

        for interval in intervals:
            d = local_data.get(interval, pd.DataFrame())
            for rule in rules:
                for mode in modes:
                    t = backtest(d, interval, rule, mode, cost)
                    m = metrics(t)
                    all_rows.append({
                        "股票": symbol,
                        "市場狀態": state["市場狀態"],
                        "週期": interval,
                        "規則": rule,
                        "持有": mode,
                        **m,
                    })
        p.progress(idx / total, text=f"{symbol}｜{idx}/{total}")
    p.empty()

    detail = pd.DataFrame(all_rows)
    states_df = pd.DataFrame(states)
    cross = cross_stock_summary(detail[detail["交易數"] > 0].copy()) if not detail.empty else pd.DataFrame()
    return detail, cross, states_df


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
    research_mode = st.radio("研究模式", ["單一股票", "跨股票批次"], horizontal=True)
    code = st.text_input("股票代號", value="2330")
    batch_text = st.text_area(
        "批次股票代號",
        value="2330,2357,3711,2317,2454,2382",
        help="用逗號分隔。V1.2 建議先跑 5～10 檔確認速度，再擴大到 20～50 檔。",
        disabled=(research_mode != "跨股票批次"),
    )
    market = st.radio("市場", ["上市", "上櫃"], horizontal=True)
    symbol = normalize_symbol(code, market)

    selected_intervals = st.multiselect("K棒週期", INTERVALS, default=INTERVALS)
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
        default=[
            "MA5上穿MA15",
            "KD黃金交叉",
            "MA5>15 + KD黃金交叉",
            "MA5>15>30 + KD黃金交叉",
            "完整多頭排列 + KD黃金交叉",
            "KD黃金交叉 + K<30",
            "KD黃金交叉 + K50-80",
            "KD黃金交叉 + MA30向上",
            "KD黃金交叉 + MA60向上",
            "KD黃金交叉 + 量比>1.2",
            "KD黃金交叉 + 站上VWAP",
        ],
    )
    selected_modes = st.multiselect(
        "持有方式",
        ["當沖", "隔日", "2日", "3日", "5日"],
        default=["當沖", "隔日", "3日", "5日"],
    )

    st.divider()
    st.subheader("交易成本")
    fee_discount = st.number_input("手續費折數", min_value=0.1, max_value=1.0, value=0.28, step=0.01)
    slip_bp = st.number_input("單邊滑價（bp）", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
    cost = CostConfig(fee_discount=fee_discount, slippage_pct=slip_bp / 10000)

    run = st.button("🚀 開始策略健診", type="primary", use_container_width=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 單股總表", "🌐 跨股穩定度", "🧬 狀態分類", "📈 K線/KD", "🔬 KD分區", "📦 量價/斜率", "🧾 交易明細"])

if run:
    if not selected_intervals or not selected_rules or not selected_modes:
        st.error("請至少選擇一個K棒週期、進場規則與持有方式。")
        st.stop()

    if research_mode == "跨股票批次":
        symbols = parse_batch_codes(batch_text, market)
        if not symbols:
            st.error("請輸入至少一檔股票。")
            st.stop()
        with st.spinner(f"跨股票策略健診：{len(symbols)} 檔…"):
            batch_detail, cross, states_df = run_batch_matrix(
                symbols, selected_intervals, selected_rules, selected_modes, cost, period
            )
        st.session_state["st_v120_batch"] = {
            "detail": batch_detail, "cross": cross, "states": states_df, "symbols": symbols
        }
        # 同時載入第一檔供圖表/單股頁查看
        symbol = symbols[0]
        summary, data_map, trade_map = run_matrix(
            symbol, selected_intervals, selected_rules, selected_modes, cost, period
        )
    else:
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

state = st.session_state.get("st_v120")
batch_state = st.session_state.get("st_v120_batch")
if state:
    symbol = state["symbol"]
    summary = state["summary"]
    data_map = state["data_map"]
    trade_map = state["trade_map"]

    with tab1:
        st.subheader(f"{symbol}｜多週期基準比較")
        valid = summary[summary["交易數"] > 0].copy()
        if valid.empty:
            st.warning("目前條件沒有產生足夠交易。可換股票、延長資料或放寬進場規則。")
        else:
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
                "⬇️ 下載跨股票穩定度 CSV",
                cx.to_csv(index=False).encode("utf-8-sig"),
                file_name="ST_V1.2_cross_stock_stability.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("股票目前狀態分類")
        if not batch_state or batch_state["states"].empty:
            local_state = classify_stock_state(data_map)
            st.dataframe(pd.DataFrame([{"股票": symbol, **local_state}]), use_container_width=True, hide_index=True)
        else:
            st.dataframe(batch_state["states"], use_container_width=True, hide_index=True)
            st.caption("這是研究用『當前狀態』，不是把股票永久分類；同一股票日後可能從趨勢轉為震盪或高波動。")

    with tab4:
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

    with tab5:
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

    with tab6:
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

    with tab7:
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
        st.write("完成驗證後顯示股票目前狀態分類。")
    with tab4:
        st.write("完成第一次驗證後顯示 MA5/15/30/60/200、VWAP 與 KD。")
    with tab5:
        st.write("完成第一次驗證後分析 KD 區間。")
    with tab6:
        st.write("完成第一次驗證後分析量比與均線斜率。")
    with tab7:
        st.write("完成第一次驗證後顯示逐筆交易。")

st.divider()
st.caption(
    "ST V1.2.0 僅供策略研究與程式驗證，不送出證券委託。"
    "下一階段將根據實際回測結果，再判斷是否增加 VWAP、成交量/量比、MACD、ATR 或其他參數。"
)
