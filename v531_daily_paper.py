# v531_daily_paper.py
# -*- coding: utf-8 -*-

import os

import pandas as pd
import requests
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
YEARS = 1

EXCLUDE_TICKERS = ["6758.T", "4568.T", "4063.T", "4519.T"]

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

POS_FILE = "positions.csv"
TRADE_FILE = "trades.csv"
CANDIDATE_FILE = "candidates.csv"

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


# ===== Discord =====
def send_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        print("no webhook")
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print(e)


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

    high = df["Close"].iloc[-10:].max()
    return df["Close"].iloc[-1] <= high * (1 - PULLBACK_PCT)


# ===== CSV =====
def load_positions():
    if not os.path.exists(POS_FILE) or os.path.getsize(POS_FILE) == 0:
        return []

    try:
        df = pd.read_csv(POS_FILE, parse_dates=["entry_date"])
        if df.empty:
            return []
        return df.to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def save_positions(pos):
    if not pos:
        pd.DataFrame(columns=["ticker", "entry_price", "entry_date", "qty"]).to_csv(
            POS_FILE, index=False
        )
    else:
        pd.DataFrame(pos).to_csv(POS_FILE, index=False)


def log_trade(t):
    if os.path.exists(TRADE_FILE):
        df = pd.read_csv(TRADE_FILE)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([t])])
    df.to_csv(TRADE_FILE, index=False)


def save_candidates(c):
    pd.DataFrame(c).to_csv(CANDIDATE_FILE, index=False)


# ===== メイン =====
def run():

    today = pd.Timestamp.today().normalize()
    print("run:", today.date())

    data = {t: load_data(t) for t in TICKERS}
    data = {k: v for k, v in data.items() if not v.empty}

    pos = load_positions()

    exit_msg = []
    buy_msg = []

    # ===== EXIT =====
    new_pos = []

    for p in pos:
        df = data.get(p["ticker"])
        if df is None:
            new_pos.append(p)
            continue

        last = df.iloc[-1]
        price = last["Close"]
        ma25 = last["MA25"]

        entry_price = p["entry_price"]
        entry_date = pd.to_datetime(p["entry_date"])

        hold = (df.index[-1] - entry_date).days

        if price <= entry_price * (1 - SL_PCT):
            reason = "SL"
        elif price >= ma25 * EXIT_MA_BUFFER and hold < HOLD_DAYS:
            new_pos.append(p)
            continue
        else:
            reason = "EXIT"

        log_trade(
            {
                "ticker": p["ticker"],
                "entry_price": entry_price,
                "exit_price": price,
                "entry_date": entry_date,
                "exit_date": df.index[-1],
                "reason": reason,
            }
        )

        exit_msg.append(f'SELL {p["ticker"]} {price:.1f}')

    pos = new_pos

    # ===== ENTRY =====
    candidates = []

    if len(pos) < MAX_POSITIONS:
        for t, df in data.items():

            if t in EXCLUDE_TICKERS:
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

            candidates.append({"ticker": t, "price": last["Close"]})

        if candidates:
            c = candidates[0]

            pos.append(
                {
                    "ticker": c["ticker"],
                    "entry_price": c["price"],
                    "entry_date": today,
                    "qty": 1,
                }
            )

            buy_msg.append(f'BUY {c["ticker"]} {c["price"]:.1f}')

    save_positions(pos)
    save_candidates(candidates)

    msg = f"""📊 v531

日付: {today.date()}

SELL:
{chr(10).join(exit_msg) if exit_msg else "(none)"}

BUY:
{chr(10).join(buy_msg) if buy_msg else "(none)"}

POS:
{pos}

CANDIDATES:
{candidates}
"""

    send_discord(msg)

    print(msg)


if __name__ == "__main__":
    run()
