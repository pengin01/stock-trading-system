# v85_backtest.py
# -*- coding: utf-8 -*-
# pip install pandas numpy yfinance ta

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import ta


@dataclass
class Params:
    capital_yen: int = 80000
    max_positions_live: int = 1
    slippage_oneway: float = 0.001

    ma_days: int = 25
    rsi_days: int = 14
    rsi_max: float = 65.0
    pullback_pct: float = 0.005

    min_avg_value20: float = 300_000_000
    hold_days: int = 4

    years: int = 10

    out_trades: str = "trades_v85_stock.csv"
    out_summary: str = "summary_v85_stock.csv"


P = Params()
LOT_SIZE = 100

STOCK_UNIVERSE = ["8035.T","9984.T","7203.T","6758.T","6861.T","9432.T"]


def matsui_fee(total):
    if total <= 500_000:
        return 0
    if total <= 1_000_000:
        return 1100
    return 2200


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


def entry_signal(df, i):
    c = df["Close"].iloc[i]
    prev = df["Close"].iloc[i-1]
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


def backtest():
    trades = []

    for t in STOCK_UNIVERSE:
        df = download(t)
        if df.empty:
            continue
        df = add_features(df)

        for i in range(P.ma_days+2, len(df)-P.hold_days-2):

            if not entry_signal(df, i):
                continue

            entry = df["Open"].iloc[i+1]
            if entry <= 0:
                continue

            qty = int(P.capital_yen // (entry * LOT_SIZE)) * LOT_SIZE
            if qty <= 0:
                continue

            entry *= (1 + P.slippage_oneway)

            exit_price = None
            reason = "TIME"

            for j in range(i+1, i+1+P.hold_days):
                if df["Close"].iloc[j] < df["MA"].iloc[j]:
                    exit_price = df["Open"].iloc[j+1]
                    reason = "MA_EXIT"
                    break

            if exit_price is None:
                exit_price = df["Open"].iloc[i+1+P.hold_days]

            exit_price *= (1 - P.slippage_oneway)

            buy = entry * qty
            sell = exit_price * qty
            fee = matsui_fee(buy+sell)

            pnl = sell - buy - fee
            ret = pnl / buy

            trades.append({
                "ticker": t,
                "ret": ret,
                "pnl": pnl,
                "reason": reason
            })

    return pd.DataFrame(trades)


def main():
    trades = backtest()

    if trades.empty:
        print("No trades")
        return

    win = (trades["ret"] > 0).mean()
    avg = trades["ret"].mean()
    total = (1 + trades["ret"]).prod() - 1

    print("trades:", len(trades))
    print("win_rate:", win)
    print("avg_return:", avg)
    print("total_return:", total)

    trades.to_csv(P.out_trades, index=False)

    summary = {
        "trades": len(trades),
        "win_rate": win,
        "avg_return": avg,
        "total_return": total,
    }
    summary.update(asdict(P))

    pd.DataFrame([summary]).to_csv(P.out_summary, index=False)


if __name__ == "__main__":
    main()