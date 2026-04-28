# v504_hammer_or_engulfing.py
# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf

INITIAL_CAPITAL = 20000

MA_SHORT = 25
MA_LONG = 75

HOLD_DAYS = 7
MAX_POSITIONS = 2
RISK_RATIO = 0.5

MIN_VALUE = 100_000_000
YEARS = 5

EXIT_MA_BUFFER = 0.98

PULLBACK_DAYS = 10
PULLBACK_PCT = 0.03

TICKERS = [
    "7203.T", "6758.T", "9984.T", "8306.T", "8035.T",
    "6861.T", "6098.T", "9432.T", "6954.T", "4519.T",
    "6501.T", "7267.T", "6902.T", "8031.T", "4568.T",
    "4063.T", "7751.T", "8591.T", "9020.T", "4502.T",
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


# ===== ハンマー =====
def is_hammer(df, i):
    o = df["Open"].iloc[i]
    c = df["Close"].iloc[i]
    h = df["High"].iloc[i]
    l = df["Low"].iloc[i]

    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)

    if body == 0:
        return False

    return (
        lower > body * 2
        and upper < body * 0.5
    )


# ===== 包み足 =====
def is_bullish_engulfing(df, i):
    if i < 1:
        return False

    prev_open = df["Open"].iloc[i - 1]
    prev_close = df["Close"].iloc[i - 1]

    open_ = df["Open"].iloc[i]
    close = df["Close"].iloc[i]

    return (
        prev_close < prev_open and
        close > open_ and
        open_ <= prev_close and
        close >= prev_open
    )


def has_pullback(df, i):
    if i < PULLBACK_DAYS:
        return False

    recent_high = df["Close"].iloc[i - PULLBACK_DAYS:i].max()
    current_close = df["Close"].iloc[i]

    return current_close <= recent_high * (1 - PULLBACK_PCT)


def run(start, end):
    cash = INITIAL_CAPITAL
    pos = []
    eq = []

    dates = sorted(set().union(*[df.index for df in data.values()]))

    for d in dates:
        if d < start or d > end:
            continue

        # ===== EXIT =====
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

        # ===== ENTRY =====
        if len(pos) < MAX_POSITIONS:
            for t, df in data.items():
                if len(pos) >= MAX_POSITIONS:
                    break

                if any(p["t"] == t for p in pos):
                    continue

                if d not in df.index:
                    continue

                i = df.index.get_loc(d)

                if i < max(MA_LONG, PULLBACK_DAYS) + 1:
                    continue

                close = df["Close"].iloc[i]
                ma75 = df["MA75"].iloc[i]
                vol = df["Volume"].iloc[i]
                vol20 = df["VOL20"].iloc[i]
                value20 = df["VALUE20"].iloc[i]

                # 流動性
                if value20 < MIN_VALUE:
                    continue

                # トレンド
                if close <= ma75:
                    continue

                # 押し目
                if not has_pullback(df, i):
                    continue

                # ★ 複合条件 ★
                if not (is_hammer(df, i) or is_bullish_engulfing(df, i)):
                    continue

                # 出来高
                if vol < vol20:
                    continue

                qty = int((cash * RISK_RATIO) // close)

                if qty <= 0:
                    continue

                cash -= close * qty

                pos.append({
                    "t": t,
                    "p": close,
                    "d": d,
                    "q": qty,
                })

        # ===== EQUITY =====
        position_value = 0

        for p in pos:
            df = data[p["t"]]
            if d in df.index:
                position_value += df.loc[d, "Close"] * p["q"]

        eq.append(cash + position_value)

    if not eq:
        return 0

    return eq[-1] / INITIAL_CAPITAL - 1


# ===== DATA LOAD =====
data = {t: load_data(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}

print("\n=== WALK FORWARD : HAMMER OR ENGULFING v504 ===")
print("HOLD_DAYS:", HOLD_DAYS)

print("2021-2023:", run(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")))
print("2024-2026:", run(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31")))
print("2024 only:", run(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")))
print("2025 only:", run(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")))
print("2026 only:", run(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")))