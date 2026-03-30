# daily_singnal.py
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
    initial_capital: int = 80000

    risk_per_trade: float = 0.02   # 1トレード2%
    max_positions: int = 3

    ma_days: int = 25
    rsi_days: int = 14

    rsi_max: float = 65.0
    pullback_pct: float = 0.005

    min_avg_value20: float = 300_000_000

    hold_days: int = 4

    years: int = 1

    pos_file: str = "positions.csv"
    equity_file: str = "equity.csv"


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
# POSITION MANAGEMENT
# =========================
def load_positions():
    if not os.path.exists(P.pos_file):
        return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "qty"])
    return pd.read_csv(P.pos_file, parse_dates=["entry_date"])


def save_positions(df):
    df.to_csv(P.pos_file, index=False)


def load_equity():
    if not os.path.exists(P.equity_file):
        return P.initial_capital
    df = pd.read_csv(P.equity_file)
    return df["equity"].iloc[-1]


def save_equity(value):
    df = pd.DataFrame([{"date": pd.Timestamp.now(), "equity": value}])
    if os.path.exists(P.equity_file):
        df.to_csv(P.equity_file, mode="a", header=False, index=False)
    else:
        df.to_csv(P.equity_file, index=False)


# =========================
# POSITION SIZE
# =========================
def calc_qty(capital, price):
    risk_amount = capital * P.risk_per_trade
    stop_width = price * 0.02  # 仮に2%損切り想定

    if stop_width <= 0:
        return 0

    qty = int(risk_amount / stop_width)
    return max(qty, 0)


# =========================
# MAIN
# =========================
def main():

    print("RUN:", pd.Timestamp.now())

    pos = load_positions()
    capital = load_equity()

    today = pd.Timestamp.now().normalize()

    entries = []
    exits = []

    # ===== EXIT =====
    for _, row in pos.iterrows():

        t = row["ticker"]
        df = download(t)
        if df.empty:
            continue

        df = add_features(df)
        hist = df[df.index.normalize() < today]

        if hist.empty:
            continue

        last = hist.iloc[-1]

        if last["Close"] < last["MA"]:
            exits.append(row)

    # EXIT処理
    if exits:
        for e in exits:
            capital += e["qty"] * e["entry_price"]  # 簡易
        pos = pos[~pos["ticker"].isin([e["ticker"] for e in exits])]

    # ===== ENTRY =====
    if len(pos) < P.max_positions:

        for t in STOCK_UNIVERSE:

            if t in pos["ticker"].values:
                continue

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

            price = hist["Close"].iloc[i]
            qty = calc_qty(capital, price)

            if qty <= 0:
                continue

            entries.append({
                "ticker": t,
                "entry_date": hist.index[i],
                "entry_price": price,
                "qty": qty
            })

            pos = pd.concat([pos, pd.DataFrame([entries[-1]])], ignore_index=True)

            if len(pos) >= P.max_positions:
                break

    save_positions(pos)
    save_equity(capital)

    # ===== OUTPUT =====
    print("== ENTRY ==")
    print(entries if entries else "(no entry)")

    print("\n== EXIT ==")
    print(exits if exits else "(no exit)")

    print("\n== CAPITAL ==")
    print(capital)

    print("\n== POSITIONS ==")
    print(pos if not pos.empty else "(empty)")


if __name__ == "__main__":
    main()