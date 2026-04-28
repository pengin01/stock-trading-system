# v530_daily_paper.py
# -*- coding: utf-8 -*-

from datetime import datetime

import pandas as pd
import yfinance as yf

# ===== 設定 =====
INITIAL_CAPITAL = 20000

MAX_POSITIONS = 1
RISK_RATIO = 1.0

HOLD_DAYS = 11
PULLBACK_PCT = 0.032
EXIT_MA_BUFFER = 0.98
SL_PCT = 0.07

MA_SHORT = 25
MA_LONG = 75
MIN_VALUE = 100_000_000
YEARS = 1  # ← 軽く

EXCLUDE_TICKERS = ["6758.T", "4568.T", "4063.T", "4519.T"]

TICKERS = [
    "7203.T","6758.T","9984.T","8306.T","8035.T",
    "6861.T","6098.T","9432.T","6954.T","4519.T",
    "6501.T","7267.T","6902.T","8031.T","4568.T",
    "4063.T","7751.T","8591.T","9020.T","4502.T",
]

POS_FILE = "positions.csv"
TRADE_FILE = "trades.csv"


# ===== データ =====
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

    return df.dropna()


# ===== シグナル =====
def is_engulf(df):
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    return (
        prev["Close"] < prev["Open"]
        and cur["Close"] > cur["Open"]
        and cur["Open"] <= prev["Close"]
        and cur["Close"] >= prev["Open"]
    )


def has_pullback(df):
    if len(df) < 10:
        return False

    recent_high = df["Close"].iloc[-10:].max()
    cur = df["Close"].iloc[-1]

    return cur <= recent_high * (1 - PULLBACK_PCT)


# ===== ポジション =====
def load_positions():
    try:
        df = pd.read_csv(POS_FILE, parse_dates=["entry_date"])
        return df.to_dict("records")
    except:
        return []


def save_positions(pos):
    pd.DataFrame(pos).to_csv(POS_FILE, index=False)


def log_trade(trade):
    try:
        df = pd.read_csv(TRADE_FILE)
        df = pd.concat([df, pd.DataFrame([trade])])
    except:
        df = pd.DataFrame([trade])

    df.to_csv(TRADE_FILE, index=False)


# ===== メイン =====
def run():

    today = pd.Timestamp.today().normalize()

    data = {t: load_data(t) for t in TICKERS}
    data = {k: v for k, v in data.items() if not v.empty}

    pos = load_positions()

    print("\n=== RUN INFO ===")
    print("today:", today.date())

    # ===== EXIT =====
    new_pos = []

    for p in pos:
        df = data.get(p["ticker"])

        if df is None or today not in df.index:
            new_pos.append(p)
            continue

        price = df.loc[today, "Close"]
        ma25 = df.loc[today, "MA25"]

        hold = (today - pd.to_datetime(p["entry_date"])).days

        if price <= p["entry_price"] * (1 - SL_PCT):
            log_trade({
                "ticker": p["ticker"],
                "entry": p["entry_price"],
                "exit": price,
                "date": today,
                "reason": "SL"
            })
            continue

        if price >= ma25 * EXIT_MA_BUFFER and hold < HOLD_DAYS:
            new_pos.append(p)
            continue

        log_trade({
            "ticker": p["ticker"],
            "entry": p["entry_price"],
            "exit": price,
            "date": today,
            "reason": "EXIT"
        })

    pos = new_pos

    # ===== ENTRY =====
    if len(pos) < MAX_POSITIONS:

        for t, df in data.items():

            if t in EXCLUDE_TICKERS:
                continue

            if any(p["ticker"] == t for p in pos):
                continue

            if len(df) < MA_LONG:
                continue

            last = df.iloc[-1]

            if last["Close"] <= last["MA75"]:
                continue

            if last["VALUE20"] < MIN_VALUE:
                continue

            if last["Volume"] < last["VOL20"]:
                continue

            if not has_pullback(df):
                continue

            if not is_engulf(df):
                continue

            pos.append({
                "ticker": t,
                "entry_price": last["Close"],
                "entry_date": today
            })

            print("\nBUY:", t, "price:", last["Close"])
            break

    save_positions(pos)

    print("\n=== POSITIONS ===")
    print(pos if pos else "(none)")


if __name__ == "__main__":
    run()    run()