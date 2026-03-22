# v88_daily_auto_position.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import os


# =========================
# PARAMETERS
# =========================
@dataclass
class Params:
    ma_days: int = 25
    rsi_days: int = 14
    rsi_max: float = 65.0
    pullback_pct: float = 0.005
    min_avg_value20: float = 300_000_000
    hold_days: int = 4
    years: int = 1
    pos_file: str = "positions.csv"


P = Params()

STOCK_UNIVERSE = ["9432.T", "6758.T", "9984.T"]


# =========================
# DATA
# =========================
def download(t):
    df = yf.download(t, period=f"{P.years}y", interval="1d", progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def add_features(df):
    close = df["Close"]
    df["MA"] = close.rolling(P.ma_days).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=P.rsi_days).rsi()
    df["VALUE20"] = (close * df["Volume"]).rolling(20).mean()
    return df


# =========================
# SIGNAL
# =========================
def entry_signal(df, i):
    c = df["Close"].iloc[i]
    prev = df["Close"].iloc[i - 1]
    ma = df["MA"].iloc[i]
    rsi = df["RSI"].iloc[i]
    v = df["VALUE20"].iloc[i]

    if not np.isfinite([c, prev, ma, rsi, v]).all():
        return False
    if c < ma:
        return False
    if c > prev * (1 - P.pullback_pct):
        return False
    if rsi > P.rsi_max:
        return False
    if v < P.min_avg_value20:
        return False

    return True


# =========================
# ENTRY
# =========================
def today_entry():

    today = pd.Timestamp.now().normalize()
    candidates = []

    for t in STOCK_UNIVERSE:

        df = download(t)
        if df.empty:
            continue

        df = add_features(df)

        hist = df[df.index.normalize() < today]
        if len(hist) < P.ma_days + 2:
            continue

        i = len(hist) - 1

        if not entry_signal(hist, i):
            continue

        candidates.append({
            "ticker": t,
            "date": hist.index[i],
            "rsi": hist["RSI"].iloc[i],
            "score": -hist["RSI"].iloc[i],
        })

    if not candidates:
        return pd.DataFrame()

    return pd.DataFrame(candidates).sort_values("score").head(1)


# =========================
# EXIT
# =========================
def today_exit(pos):

    today = pd.Timestamp.now().normalize()
    exits = []

    for _, row in pos.iterrows():

        t = row["ticker"]
        entry_date = row["entry_date"]

        df = download(t)
        if df.empty:
            continue

        df = add_features(df)

        hist = df[df.index.normalize() < today]
        if hist.empty:
            continue

        last = hist.iloc[-1]
        hold = (hist.index[-1] - entry_date).days

        if last["Close"] < last["MA"]:
            exits.append({"ticker": t, "reason": "MA_EXIT"})
        elif hold >= P.hold_days:
            exits.append({"ticker": t, "reason": "TIME"})

    return pd.DataFrame(exits)


# =========================
# POSITION LOAD/SAVE
# =========================
def load_positions():
    if not os.path.exists(P.pos_file):
        return pd.DataFrame(columns=["ticker", "entry_date"])

    df = pd.read_csv(P.pos_file, parse_dates=["entry_date"])
    return df


def save_positions(df):
    df.to_csv(P.pos_file, index=False)


# =========================
# MAIN
# =========================
def main():

    print("RUN:", pd.Timestamp.now())

    pos = load_positions()

    entry = today_entry()
    exit_df = today_exit(pos)

    # ===== EXIT反映 =====
    if not exit_df.empty:
        pos = pos[~pos["ticker"].isin(exit_df["ticker"])]

    # ===== ENTRY反映 =====
    if not entry.empty:
        new_rows = entry[["ticker", "date"]].rename(columns={"date": "entry_date"})
        pos = pd.concat([pos, new_rows], ignore_index=True)

    save_positions(pos)

    # ===== 表示 =====
    print("== TODAY ENTRY ==")
    print("(no entry)" if entry.empty else entry.to_string(index=False))

    print("\n== TODAY EXIT ==")
    print("(no exit)" if exit_df.empty else exit_df.to_string(index=False))

    print("\n== POSITIONS ==")
    print("(empty)" if pos.empty else pos.to_string(index=False))


if __name__ == "__main__":
    main()