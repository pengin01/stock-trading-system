# v97_system.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import os

# =========================
# PARAMETERS（v97一致）
# =========================
INITIAL_CAPITAL = 20000
HOLD_DAYS = 4
PULLBACK = 0.004

RISK_RATIO = 0.7
MAX_POSITIONS = 1

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 65
MIN_VALUE = 300_000_000

STOCK_UNIVERSE = ["9432.T", "6758.T", "9984.T"]

POS_FILE = "positions.csv"
EQ_FILE = "equity.csv"

# =========================
# DATA
# =========================
def load_data(ticker):
    df = yf.download(ticker, period="1y", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()

# =========================
# SIGNAL
# =========================
def entry_signal(df, i):
    c = df["Close"].iloc[i]
    prev = df["Close"].iloc[i - 1]

    if c < df["MA"].iloc[i]:
        return False
    if c > prev * (1 - PULLBACK):
        return False
    if df["RSI"].iloc[i] > RSI_MAX:
        return False
    if df["VALUE20"].iloc[i] < MIN_VALUE:
        return False

    return True

# =========================
# FILE
# =========================
def load_positions():
    cols = ["ticker", "entry_date", "entry_price", "qty", "exit_date"]

    if not os.path.exists(POS_FILE):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(POS_FILE, parse_dates=["entry_date", "exit_date"])
    except Exception:
        return pd.DataFrame(columns=cols)

    if df.empty:
        return pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")

    return df[cols]

def save_positions(df):
    df.to_csv(POS_FILE, index=False)

def load_equity():
    if not os.path.exists(EQ_FILE):
        return INITIAL_CAPITAL

    try:
        df = pd.read_csv(EQ_FILE)
    except Exception:
        return INITIAL_CAPITAL

    if df.empty:
        return INITIAL_CAPITAL

    if "equity" not in df.columns:
        return INITIAL_CAPITAL

    s = pd.to_numeric(df["equity"], errors="coerce").dropna()
    if s.empty:
        return INITIAL_CAPITAL

    return float(s.iloc[-1])

def save_equity(val):
    df = pd.DataFrame([{
        "date": pd.Timestamp.now(),
        "equity": float(val)
    }])

    if os.path.exists(EQ_FILE):
        old = pd.read_csv(EQ_FILE)
        if old.empty:
            df.to_csv(EQ_FILE, index=False)
        else:
            df.to_csv(EQ_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(EQ_FILE, index=False)

# =========================
# MAIN
# =========================
def main():

    print("RUN:", pd.Timestamp.now())

    pos = load_positions()
    cash = load_equity()

    today = pd.Timestamp.now().normalize()

    entries = []
    exits = []

    data_cache = {t: load_data(t) for t in STOCK_UNIVERSE}

    # =====================
    # EXIT（時間決済）
    # =====================
    new_pos = []

    for _, p in pos.iterrows():

        if today < p["exit_date"]:
            new_pos.append(p)
            continue

        df = data_cache[p["ticker"]]
        if today not in df.index:
            new_pos.append(p)
            continue

        price = df.loc[today, "Close"]
        cash += price * p["qty"]

        exits.append({
            "ticker": p["ticker"],
            "reason": "time_exit"
        })

    pos = pd.DataFrame(new_pos, columns=["ticker", "entry_date", "entry_price", "qty", "exit_date"])
    # =====================
    # ENTRY
    # =====================
    if len(pos) < MAX_POSITIONS:

        candidates = []

        for t, df in data_cache.items():

            if t in pos["ticker"].values:
                continue
            if today not in df.index:
                continue

            i = df.index.get_loc(today)

            if i < MA_DAYS + 2:
                continue
            if i + HOLD_DAYS >= len(df):
                continue

            if not entry_signal(df, i):
                continue

            score = -df["RSI"].iloc[i]
            candidates.append((score, t, i))

        if candidates:
            candidates.sort()
            _, t, i = candidates[0]

            df = data_cache[t]
            price = df["Close"].iloc[i]

            usable_cash = cash * RISK_RATIO
            qty = int(usable_cash // price)

            if qty > 0:
                cost = price * qty
                cash -= cost

                new_pos_row = {
                    "ticker": t,
                    "entry_date": today,
                    "entry_price": price,
                    "qty": qty,
                    "exit_date": df.index[i + HOLD_DAYS]
                }

                pos = pd.concat([pos, pd.DataFrame([new_pos_row])], ignore_index=True)

                entries.append({
                    "ticker": t,
                    "signal_date": today,
                    "entry_price": price,
                    "qty": qty,
                    "rsi": df["RSI"].iloc[i],
                    "score": score
                })

    # =====================
    # EQUITY
    # =====================
    mtm = 0

    for _, p in pos.iterrows():
        df = data_cache[p["ticker"]]
        if today in df.index:
            px = df.loc[today, "Close"]
        else:
            px = p["entry_price"]
        mtm += px * p["qty"]

    equity = cash + mtm

    # =====================
    # SAVE FILES
    # =====================
    save_equity(equity)
    save_positions(pos)

    pd.DataFrame(entries).to_csv("today_entry.csv", index=False)
    pd.DataFrame(exits).to_csv("today_exit.csv", index=False)

    # =====================
    # LOG
    # =====================
    print("== ENTRY ==")
    print(entries if entries else "(none)")

    print("\n== EXIT ==")
    print(exits if exits else "(none)")

    print("\n== EQUITY ==")
    print(equity)

    print("\n== POSITIONS ==")
    print(pos if not pos.empty else "(empty)")


if __name__ == "__main__":
    main()