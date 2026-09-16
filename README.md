# 黑嚕嚕－短線交易雷達 ST V1.0.0

這是與原本 V3.x「黑嚕嚕－台股盤中雷達」完全分開的短線研究專案。

## V1.0 目的
- MA：5 / 15 / 30 / 60 / 200
- KD：9,3,3
- K棒：5m / 15m / 60m
- 持有：當沖 / 隔日 / 2日 / 3日 / 5日
- 比較：交易數、勝率、平均報酬、盈虧比、Profit Factor、期望值、最大回撤、累積報酬
- 暫時不加入 VWAP / MACD / ATR / 量比，先建立基準線

## 啟動
pip install -r requirements_ST_V1.0.txt
streamlit run app_ST_V1.0.0.py

## 重要
V1.0 用 yfinance intraday 做研究原型，資料歷史有限。後續接 Shioaji 後，應自行累積分鐘K資料庫，再做更長期 walk-forward / out-of-sample 驗證。
