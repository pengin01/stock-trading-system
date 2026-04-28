# v512_param_fine_grid.py
# -*- coding: utf-8 -*-

from itertools import product

import pandas as pd
import yfinance as yf

INITIAL_CAPITAL = 20000

MA_SHORT = 25
MA_LONG = 75

MAX_POSITIONS = 2
RISK_RATIO = 0.5

MIN_VALUE = 100_000_000
YEARS = 5

# ===== 微調整グリッド =====
HOLD_DAYS_LIST = [10, 11, 12, 13, 14]
PULLBACK_PCT_LIST = [0.028, 0.03, 0.032]
EXIT_MA_BUFFER_LIST = [0.98]

EXCLUDE_TICKERS = ["6758.T", "4568.T", "4063.T"]

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


def load_data(ticker):
    df = yf.download(ticker, period=f"{YEARS}y", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def is_bullish_engulfing(df, i):
    if i < 1:
        return False

    prev_open = df["Open"].iloc[i - 1]
    prev_close = df["Close"].iloc[i - 1]
    open_ = df["Open"].iloc[i]
    close = df["Close"].iloc[i]

    return (
        prev_close < prev_open
        and close > open_
        and open_ <= prev_close
        and close >= prev_open
    )


def has_pullback(df, i, pullback_pct):
    if i < 10:
        return False

    recent_high = df["Close"].iloc[i - 10 : i].max()
    current_close = df["Close"].iloc[i]

    return current_close <= recent_high * (1 - pullback_pct)


def run(params):
    HOLD_DAYS, PULLBACK_PCT, EXIT_MA_BUFFER = params

    cash = INITIAL_CAPITAL
    pos = []
    eq = []

    dates = sorted(set().union(*[df.index for df in data.values()]))

    for d in dates:
        # EXIT
        new_pos = []
        for p in pos:
            df = data[p["t"]]

            if d not in df.index:
                new_pos.append(p)
                continue

            price = df.loc[d, "Close"]
            ma25 = df.loc[d, "MA25"]

            entry_i = df.index.get_loc(p["d"])
            current_i = df.index.get_loc(d)
            hold = current_i - entry_i

            if price >= ma25 * EXIT_MA_BUFFER and hold < HOLD_DAYS:
                new_pos.append(p)
                continue

            cash += price * p["q"]

        pos = new_pos

        # ENTRY
        if len(pos) < MAX_POSITIONS:
            for t, df in data.items():
                if len(pos) >= MAX_POSITIONS:
                    break

                if t in EXCLUDE_TICKERS:
                    continue

                if any(p["t"] == t for p in pos):
                    continue

                if d not in df.index:
                    continue

                i = df.index.get_loc(d)

                if i < 75:
                    continue

                close = df["Close"].iloc[i]
                ma75 = df["MA75"].iloc[i]
                vol = df["Volume"].iloc[i]
                vol20 = df["VOL20"].iloc[i]
                value20 = df["VALUE20"].iloc[i]

                if value20 < MIN_VALUE:
                    continue

                if close <= ma75:
                    continue

                if not has_pullback(df, i, PULLBACK_PCT):
                    continue

                if not is_bullish_engulfing(df, i):
                    continue

                if vol < vol20:
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


# ===== DATA =====
data = {t: load_data(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}

# ===== GRID SEARCH =====
results = []

for params in product(HOLD_DAYS_LIST, PULLBACK_PCT_LIST, EXIT_MA_BUFFER_LIST):
    r = run(params)

    results.append(
        {
            "HOLD_DAYS": params[0],
            "PULLBACK_PCT": params[1],
            "EXIT_MA_BUFFER": params[2],
            "RETURN": r,
        }
    )

df = pd.DataFrame(results)
df = df.sort_values("RETURN", ascending=False)

print("\n=== FINE PARAM RESULT ===")
print(df.head(15))

df.to_csv("param_result_fine.csv", index=False)
print("Saved: param_result_fine.csv")
print("Saved: param_result_fine.csv")
