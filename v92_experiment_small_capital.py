# v92_experiment_small_capital.py

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from itertools import product


# =========================
# 設定
# =========================
PARAM_GRID = {
    "initial_capital": [20000, 30000, 50000, 80000, 100000],
    "hold_days": [4],
    "pullback_pct": [0.002,0.003, 0.005,0.006,0.007],
}

STOCK_UNIVERSE = ["9432.T"]

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 65
MIN_VALUE = 300_000_000
LOT_SIZE = 100


# =========================
# データ
# =========================
def load_data(t):
    df = yf.download(t, period="10y", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


# =========================
# シグナル
# =========================
def entry_signal(df, i, pullback_pct):
    c = df["Close"].iloc[i]
    prev = df["Close"].iloc[i - 1]
    ma = df["MA"].iloc[i]
    rsi = df["RSI"].iloc[i]
    v = df["VALUE20"].iloc[i]

    if c < ma:
        return False
    if c > prev * (1 - pullback_pct):
        return False
    if rsi > RSI_MAX:
        return False
    if v < MIN_VALUE:
        return False

    return True


# =========================
# バックテスト（少額対応）
# =========================
def backtest(params):

    capital = params["initial_capital"]
    hold_days = params["hold_days"]
    pullback_pct = params["pullback_pct"]

    trades = []
    skipped = 0

    for t in STOCK_UNIVERSE:

        df = load_data(t)

        for i in range(MA_DAYS + 2, len(df) - hold_days - 1):

            if not entry_signal(df, i, pullback_pct):
                continue

            price = df["Close"].iloc[i]

            # 必要資金
            cost = price * LOT_SIZE

            if cost > capital:
                skipped += 1
                continue

            exit_price = df["Close"].iloc[i + hold_days]

            ret = (exit_price - price) / price

            capital *= (1 + ret)
            trades.append(ret)

    if not trades:
        return {
            "trades": 0,
            "skipped": skipped,
            "final_capital": capital,
            "total_return": 0,
        }

    trades = np.array(trades)

    return {
        "trades": len(trades),
        "skipped": skipped,
        "win_rate": (trades > 0).mean(),
        "avg_return": trades.mean(),
        "final_capital": capital,
        "total_return": capital / params["initial_capital"] - 1,
    }


# =========================
# 実行
# =========================
def main():

    results = []

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    for combo in product(*values):

        params = dict(zip(keys, combo))

        print("RUN:", params)

        res = backtest(params)

        res.update(params)
        results.append(res)

    df = pd.DataFrame(results)

    df = df.sort_values("final_capital", ascending=False)

    print("\n=== RESULT ===")
    print(df.to_string(index=False))

    df.to_csv("experiment_small_capital.csv", index=False)

    print("\nSaved: experiment_small_capital.csv")


if __name__ == "__main__":
    main()