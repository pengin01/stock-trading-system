# v92_experiment.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from itertools import product


# =========================
# PARAM GRID（ここだけ触る）
# =========================
PARAM_GRID = {
    "hold_days": [3, 4, 5],
    "pullback_pct": [0.003, 0.005, 0.007],
}


STOCK_UNIVERSE = ["9432.T", "6758.T", "9984.T"]

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 65.0
MIN_VALUE = 300_000_000
INITIAL_CAPITAL = 80000


# =========================
# DATA
# =========================
def load_data(t):
    df = yf.download(t, period="10y", progress=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


# =========================
# SIGNAL
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
# BACKTEST
# =========================
def backtest(params):

    hold_days = params["hold_days"]
    pullback_pct = params["pullback_pct"]

    trades = []

    for t in STOCK_UNIVERSE:

        df = load_data(t)
        if df.empty:
            continue

        for i in range(MA_DAYS + 2, len(df) - hold_days - 1):

            if not entry_signal(df, i, pullback_pct):
                continue

            entry = df["Close"].iloc[i]

            exit_price = df["Close"].iloc[i + hold_days]

            ret = (exit_price - entry) / entry

            trades.append(ret)

    if not trades:
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "total_return": 0,
        }

    trades = np.array(trades)

    return {
        "trades": len(trades),
        "win_rate": (trades > 0).mean(),
        "avg_return": trades.mean(),
        "total_return": np.prod(1 + trades) - 1,
    }


# =========================
# MAIN
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

    df = df.sort_values("total_return", ascending=False)

    print("\n=== RESULT ===")
    print(df.to_string(index=False))

    df.to_csv("experiment_result.csv", index=False)

    print("\nSaved: experiment_result.csv")


if __name__ == "__main__":
    main()