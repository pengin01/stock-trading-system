# v500_walkforward_test.py
# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf

# ===== 採用パラメータ =====
INITIAL_CAPITAL = 20000

MA_SHORT = 25
MA_LONG = 75
BREAKOUT = 40

VOL_MULT = 1.5
MA_SLOPE_PCT = 0.02
BREAKOUT_BUFFER = 1.01

HOLD_DAYS = 7
MAX_POSITIONS = 2
RISK_RATIO = 0.5

MIN_VALUE = 100_000_000
YEARS = 5

TICKERS = [
    "7203.T",
    "6758.T",
    "9984.T",
    "8306.T",
    "8035.T",
    "6861.T",
    "6098.T",
    "9432.T",
    "6954.T",
    "4519.T",
    "6501.T",
    "7267.T",
    "6902.T",
    "8031.T",
    "4568.T",
    "4063.T",
    "7751.T",
    "8591.T",
    "9020.T",
    "4502.T",
]


def load_data(t):
    df = yf.download(t, period=f"{YEARS}y", progress=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["HH"] = df["Close"].rolling(BREAKOUT).max()

    return df.dropna()


data = {t: load_data(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}


def run(start, end):
    cash = INITIAL_CAPITAL
    pos = []
    eq = []

    dates = sorted(set().union(*[df.index for df in data.values()]))

    for d in dates:
        if d < start or d > end:
            continue

        # EXIT
        new = []
        for p in pos:
            df = data[p["t"]]
            if d not in df.index:
                new.append(p)
                continue

            price = df.loc[d, "Close"]
            ma25 = df.loc[d, "MA25"]

            hold = (df.index > p["d"]).sum()

            if price >= ma25 and hold < HOLD_DAYS:
                new.append(p)
                continue

            cash += price * p["q"]
        pos = new

        # ENTRY
        if len(pos) < MAX_POSITIONS:
            for t, df in data.items():
                if any(p["t"] == t for p in pos):
                    continue
                if d not in df.index:
                    continue

                i = df.index.get_loc(d)
                if i < BREAKOUT + 5:
                    continue

                close = df["Close"].iloc[i]
                ma25 = df["MA25"].iloc[i]
                ma75 = df["MA75"].iloc[i]
                hh = df["HH"].iloc[i - 1]

                vol = df["Volume"].iloc[i]
                vol20 = df["VOL20"].iloc[i]
                val = df["VALUE20"].iloc[i]

                ma_now = df["MA25"].iloc[i]
                ma_past = df["MA25"].iloc[i - 5]

                if not (close > ma25 > ma75):
                    continue
                if close <= hh * BREAKOUT_BUFFER:
                    continue
                if vol <= vol20 * VOL_MULT:
                    continue
                if (ma_now / ma_past - 1) < MA_SLOPE_PCT:
                    continue
                if val < MIN_VALUE:
                    continue

                qty = int((cash * RISK_RATIO) // close)
                if qty <= 0:
                    continue

                cash -= close * qty
                pos.append({"t": t, "p": close, "d": d, "q": qty})

        # EQUITY
        pv = 0
        for p in pos:
            df = data[p["t"]]
            if d in df.index:
                pv += df.loc[d, "Close"] * p["q"]

        eq.append(cash + pv)

    if not eq:
        return 0

    return eq[-1] / INITIAL_CAPITAL - 1


# ===== 分割 =====
print("\n=== WALK FORWARD ===")
print("2021-2023:", run(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")))
print("2024-2026:", run(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31")))
print("2024 only:", run(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")))
print("2025 only:", run(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")))
print("2026 only:", run(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")))
