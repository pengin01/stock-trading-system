# v97_risk_control_with_equity.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from itertools import product


PARAM_GRID = {
    "initial_capital": [20000, 30000, 50000, 80000, 100000],
    "hold_days": [4],
    "pullback_pct": [0.002,0.004,0.007],
}

STOCK_UNIVERSE = ["9432.T", "6758.T", "9984.T"]

PERIOD = "10y"
MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 65
MIN_VALUE = 300_000_000

MAX_POSITIONS = 1
RISK_RATIO = 0.7


def load_data(ticker):
    df = yf.download(ticker, period=PERIOD, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def load_all():
    data = {}
    for ticker in STOCK_UNIVERSE:
        df = load_data(ticker)
        if not df.empty:
            data[ticker] = df
    return data


def entry_signal(df, i, pullback):
    c = float(df["Close"].iloc[i])
    prev = float(df["Close"].iloc[i - 1])

    if c < float(df["MA"].iloc[i]):
        return False
    if c > prev * (1 - pullback):
        return False
    if float(df["RSI"].iloc[i]) > RSI_MAX:
        return False
    if float(df["VALUE20"].iloc[i]) < MIN_VALUE:
        return False

    return True


def calc_dd(eq):
    peak = eq.cummax()
    return (eq / peak - 1).min()


def backtest(params):

    capital = float(params["initial_capital"])
    hold_days = int(params["hold_days"])
    pullback = float(params["pullback_pct"])

    data = load_all()
    if not data:
        return {
            "final_capital": capital,
            "total_return": 0.0,
            "max_drawdown": 0.0,
        }

    all_dates = sorted(set().union(*[df.index for df in data.values()]))

    cash = capital
    positions = []
    trades = []
    equity_curve = []

    for date in all_dates:

        # EXIT
        new_pos = []
        for p in positions:
            if date < p["exit"]:
                new_pos.append(p)
                continue

            df = data[p["ticker"]]
            if date not in df.index:
                new_pos.append(p)
                continue

            price = float(df.loc[date, "Close"])
            proceeds = price * p["qty"]
            cash += proceeds

            ret = (price - p["entry_price"]) / p["entry_price"]
            trades.append(ret)

        positions = new_pos

        # ENTRY
        if len(positions) < MAX_POSITIONS:

            candidates = []

            for ticker, df in data.items():
                if date not in df.index:
                    continue

                i = df.index.get_loc(date)

                if i < MA_DAYS + 2:
                    continue
                if i + hold_days >= len(df):
                    continue

                if not entry_signal(df, i, pullback):
                    continue

                score = -float(df["RSI"].iloc[i])
                candidates.append((score, ticker, i))

            if candidates:
                candidates.sort()

                _, ticker, i = candidates[0]
                df = data[ticker]

                price = float(df["Close"].iloc[i])

                usable_cash = cash * RISK_RATIO
                qty = int(usable_cash // price)

                if qty > 0:
                    cost = price * qty
                    cash -= cost

                    positions.append({
                        "ticker": ticker,
                        "entry_price": price,
                        "qty": qty,
                        "exit": df.index[i + hold_days],
                    })

        # EQUITY（←ここは元コードそのまま + CSV用に保存）
        mtm = 0.0
        for p in positions:
            df = data[p["ticker"]]

            if date in df.index:
                px = float(df.loc[date, "Close"])
            else:
                hist = df.loc[:date]
                if hist.empty:
                    px = p["entry_price"]
                else:
                    px = float(hist["Close"].iloc[-1])

            mtm += px * p["qty"]

        total_equity = cash + mtm

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "cash": cash,
            "position_value": mtm,
            "total_equity": total_equity
        })

    eq = pd.DataFrame(equity_curve)

    return {
        "final_capital": float(eq["total_equity"].iloc[-1]),
        "total_return": float(eq["total_equity"].iloc[-1] / capital - 1),
        "max_drawdown": float(calc_dd(eq["total_equity"])),
        "equity_df": eq   # ← 追加
    }


def main():

    results = []

    for combo in product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID.keys(), combo))

        print("RUN:", params)

        res = backtest(params)

        # equity 保存
        eq = res.pop("equity_df")
        filename = f"equity_curve_v97_cap{params['initial_capital']}_pb{params['pullback_pct']}.csv"
        eq.to_csv(filename, index=False)
        print("Saved:", filename)

        res.update(params)
        results.append(res)

    df = pd.DataFrame(results)

    print("\n=== RESULT ===")
    print(df.to_string(index=False))

    df.to_csv("v97_result.csv", index=False)
    print("\nSaved: v97_result.csv")


if __name__ == "__main__":
    main()