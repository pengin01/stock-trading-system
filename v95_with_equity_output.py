# v95_with_equity_output.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from itertools import product


# =========================
# PARAM
# =========================
PARAM_GRID = {
    "initial_capital": [20000, 30000, 50000, 80000, 100000],
    "hold_days": [4],
    "pullback_pct": [0.002,0.004,0.007],
}

STOCK_UNIVERSE = ["9432.T", "6758.T", "9984.T"]

PERIOD = "5y"
MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 65
MIN_VALUE = 300_000_000
MAX_POSITIONS = 1


# =========================
# DATA
# =========================
def load_data(t):
    df = yf.download(t, period=PERIOD, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def load_all():
    return {t: load_data(t) for t in STOCK_UNIVERSE}


# =========================
# SIGNAL
# =========================
def entry_signal(df, i, pullback):
    c = df["Close"].iloc[i]
    prev = df["Close"].iloc[i - 1]

    if c < df["MA"].iloc[i]:
        return False
    if c > prev * (1 - pullback):
        return False
    if df["RSI"].iloc[i] > RSI_MAX:
        return False
    if df["VALUE20"].iloc[i] < MIN_VALUE:
        return False

    return True


# =========================
# DD
# =========================
def calc_dd(eq):
    peak = eq.cummax()
    dd = eq / peak - 1
    return dd.min()


# =========================
# BACKTEST
# =========================
def backtest(params):

    capital = params["initial_capital"]
    hold_days = params["hold_days"]
    pullback = params["pullback_pct"]

    data = load_all()
    all_dates = sorted(set().union(*[df.index for df in data.values()]))

    cash = capital
    positions = []

    trades = []
    skipped = 0

    equity_curve = []

    for date in all_dates:

        # ===== EXIT =====
        new_pos = []
        for p in positions:
            if date < p["exit"]:
                new_pos.append(p)
                continue

            df = data[p["ticker"]]
            if date not in df.index:
                new_pos.append(p)
                continue

            price = df.loc[date, "Close"]

            proceeds = price * p["qty"]
            cash += proceeds

            ret = (price - p["entry_price"]) / p["entry_price"]
            trades.append(ret)

        positions = new_pos

        # ===== ENTRY =====
        if len(positions) < MAX_POSITIONS:

            candidates = []

            for t, df in data.items():
                if date not in df.index:
                    continue

                i = df.index.get_loc(date)

                if i < MA_DAYS + 2:
                    continue
                if i + hold_days >= len(df):
                    continue

                if not entry_signal(df, i, pullback):
                    continue

                score = -df["RSI"].iloc[i]
                candidates.append((score, t, i))

            if candidates:
                candidates.sort()

                _, t, i = candidates[0]
                df = data[t]

                price = df["Close"].iloc[i]

                qty = int(cash // price)

                if qty > 0:
                    cost = price * qty
                    cash -= cost

                    positions.append({
                        "ticker": t,
                        "entry_price": price,
                        "qty": qty,
                        "exit": df.index[i + hold_days],
                    })
                else:
                    skipped += 1

        # ===== EQUITY =====
        mtm = 0
        for p in positions:
            df = data[p["ticker"]]

            if date in df.index:
                px = df.loc[date, "Close"]
            else:
                px = df.loc[:date]["Close"].iloc[-1]

            mtm += px * p["qty"]

        equity = cash + mtm

        equity_curve.append({
            "date": date,
            "equity": equity,
            "initial_capital": params["initial_capital"],
            "pullback_pct": pullback,
        })

    eq = pd.DataFrame(equity_curve)

    result = {
        "trades": len(trades),
        "skipped": skipped,
        "win_rate": np.mean(np.array(trades) > 0) if trades else 0,
        "avg_return": np.mean(trades) if trades else 0,
        "final_capital": eq["equity"].iloc[-1],
        "total_return": eq["equity"].iloc[-1] / capital - 1,
        "max_drawdown": calc_dd(eq["equity"]),
    }

    return result, eq


# =========================
# MAIN
# =========================
def main():

    results = []
    all_curves = []

    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()

    for combo in product(*values):
        params = dict(zip(keys, combo))

        print("RUN:", params)

        res, eq = backtest(params)
        res.update(params)

        results.append(res)
        all_curves.append(eq)

    result_df = pd.DataFrame(results)
    curve_df = pd.concat(all_curves)

    print("\n=== RESULT ===")
    print(result_df.to_string(index=False))

    result_df.to_csv("v95_result.csv", index=False)
    curve_df.to_csv("v95_equity_curve.csv", index=False)

    print("\nSaved: v95_result.csv, v95_equity_curve.csv")


if __name__ == "__main__":
    main()