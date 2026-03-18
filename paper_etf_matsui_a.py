# -*- coding: utf-8 -*-
# pip install pandas numpy yfinance ta

from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
import ta


# =========================
# PARAMETERS
# =========================
@dataclass
class Params:
    capital_yen: int = 80000

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
def download(ticker):
    df = yf.download(ticker, period=f"{P.years}y", interval="1d", progress=False)

    if df.empty:
        return df

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
# ENTRY
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
# TODAY ENTRY
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

    df = pd.DataFrame(candidates)
    return df.sort_values("score").head(1)


# =========================
# TODAY EXIT
# =========================
def today_exit():

    try:
        pos = pd.read_csv(P.pos_file, parse_dates=["entry_date"])
    except:
        return pd.DataFrame()

    if pos.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    exits = []

    for _, row in pos.iterrows():

        ticker = row["ticker"]
        entry_date = row["entry_date"]

        df = download(ticker)
        if df.empty:
            continue

        df = add_features(df)

        hist = df[df.index.normalize() < today]
        if hist.empty:
            continue

        last_i = len(hist) - 1
        last_date = hist.index[last_i]

        # 保有日数
        hold = (last_date - entry_date).days

        # MA_EXIT
        if hist["Close"].iloc[last_i] < hist["MA"].iloc[last_i]:
            exits.append({"ticker": ticker, "reason": "MA_EXIT"})
            continue

        # TIME EXIT
        if hold >= P.hold_days:
            exits.append({"ticker": ticker, "reason": "TIME"})
            continue

    return pd.DataFrame(exits)


# =========================
# MAIN
# =========================
def main():
    print("RUN:", pd.Timestamp.now())
    print("== TODAY ENTRY ==")
    entry = today_entry()

    if entry.empty:
        print("(no entry)")
    else:
        print(entry.to_string(index=False))

    print("\n== TODAY EXIT ==")
    exit_df = today_exit()

    if exit_df.empty:
        print("(no exit)")
    else:
        print(exit_df.to_string(index=False))


if __name__ == "__main__":
    main()