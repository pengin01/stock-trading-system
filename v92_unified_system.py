# v92_unified_system.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, asdict
import os

import numpy as np
import pandas as pd
import yfinance as yf
import ta


# =========================
# PARAMETERS
# =========================
@dataclass
class Params:
    initial_capital: int = 80000

    # risk / portfolio
    risk_per_trade: float = 0.02
    max_positions: int = 3
    lot_size: int = 100

    # signal
    ma_days: int = 25
    rsi_days: int = 14
    rsi_max: float = 65.0
    pullback_pct: float = 0.005
    min_avg_value20: float = 300_000_000

    # exit
    hold_days: int = 4

    # data
    years: int = 2

    # files
    pos_file: str = "positions.csv"
    equity_file: str = "equity.csv"
    trade_log_file: str = "trades.csv"
    today_entry_file: str = "today_entry.csv"
    today_exit_file: str = "today_exit.csv"


P = Params()

STOCK_UNIVERSE = ["9432.T", "6758.T", "9984.T"]


# =========================
# DATA
# =========================
def download_ohlcv(ticker: str, years: int | None = None) -> pd.DataFrame:
    use_years = years if years is not None else P.years

    df = yf.download(
        ticker,
        period=f"{use_years}y",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    df["MA"] = close.rolling(P.ma_days).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=P.rsi_days).rsi()
    df["VALUE20"] = (close * volume).rolling(20).mean()

    return df


# =========================
# UNIFIED SIGNAL LOGIC
# =========================
def entry_signal(df: pd.DataFrame, i: int) -> bool:
    """
    v85/v87系の条件を統一
    """
    close = float(df["Close"].iloc[i])
    prev = float(df["Close"].iloc[i - 1])
    ma = float(df["MA"].iloc[i])
    rsi = float(df["RSI"].iloc[i])
    value20 = float(df["VALUE20"].iloc[i])

    arr = np.array([close, prev, ma, rsi, value20], dtype=float)
    if not np.isfinite(arr).all():
        return False

    # 上昇トレンド
    if close < ma:
        return False

    # 軽い押し目
    if close > prev * (1 - P.pullback_pct):
        return False

    # 過熱除外
    if rsi > P.rsi_max:
        return False

    # 流動性
    if value20 < P.min_avg_value20:
        return False

    return True


def exit_signal(df: pd.DataFrame, i: int, entry_date: pd.Timestamp) -> tuple[bool, str]:
    """
    実運用・検証共通のEXIT判定
    """
    close = float(df["Close"].iloc[i])
    ma = float(df["MA"].iloc[i])

    if np.isfinite(close) and np.isfinite(ma) and close < ma:
        return True, "MA_EXIT"

    hold_days = (pd.Timestamp(df.index[i]).normalize() - pd.Timestamp(entry_date).normalize()).days
    if hold_days >= P.hold_days:
        return True, "TIME"

    return False, ""


# =========================
# LOAD / SAVE
# =========================
def load_positions() -> pd.DataFrame:
    if not os.path.exists(P.pos_file):
        return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "qty"])

    df = pd.read_csv(P.pos_file, parse_dates=["entry_date"])
    return df


def save_positions(df: pd.DataFrame) -> None:
    df.to_csv(P.pos_file, index=False)


def load_equity() -> float:
    if not os.path.exists(P.equity_file):
        return float(P.initial_capital)

    df = pd.read_csv(P.equity_file)
    if df.empty:
        return float(P.initial_capital)

    return float(df["equity"].iloc[-1])


def save_equity(value: float) -> None:
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "equity": value,
    }])

    if os.path.exists(P.equity_file):
        row.to_csv(P.equity_file, mode="a", header=False, index=False)
    else:
        row.to_csv(P.equity_file, index=False)


def append_trade_log(row: dict) -> None:
    df = pd.DataFrame([row])

    if os.path.exists(P.trade_log_file):
        df.to_csv(P.trade_log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(P.trade_log_file, index=False)


# =========================
# POSITION SIZE
# =========================
def calc_qty(capital: float, entry_price: float) -> int:
    """
    2%リスク・100株単位
    """
    if not np.isfinite(capital) or capital <= 0:
        return 0
    if not np.isfinite(entry_price) or entry_price <= 0:
        return 0

    risk_amount = capital * P.risk_per_trade
    assumed_stop = entry_price * 0.02  # 仮の2%ストップ幅

    if assumed_stop <= 0:
        return 0

    shares = int(risk_amount // assumed_stop)
    lots = shares // P.lot_size

    return max(0, lots * P.lot_size)


# =========================
# TODAY ENTRY
# =========================
def build_today_entries() -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    candidates = []

    for ticker in STOCK_UNIVERSE:
        df = download_ohlcv(ticker)
        if df.empty:
            continue

        df = add_features(df)

        # 今日の足は除外（確定足のみ使う）
        hist = df[df.index.normalize() < today].copy()
        if len(hist) < P.ma_days + 2:
            continue

        i = len(hist) - 1

        if not entry_signal(hist, i):
            continue

        candidates.append({
            "ticker": ticker,
            "signal_date": hist.index[i],
            "close": float(hist["Close"].iloc[i]),
            "rsi": float(hist["RSI"].iloc[i]),
            "score": -float(hist["RSI"].iloc[i]),
        })

    if not candidates:
        return pd.DataFrame(columns=["ticker", "signal_date", "close", "rsi", "score"])

    df_candidates = pd.DataFrame(candidates)
    df_candidates = df_candidates.sort_values("score").reset_index(drop=True)
    return df_candidates


# =========================
# TODAY EXIT
# =========================
def build_today_exits(positions: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    rows = []

    if positions.empty:
        return pd.DataFrame(columns=["ticker", "reason"])

    for _, row in positions.iterrows():
        ticker = row["ticker"]
        entry_date = pd.Timestamp(row["entry_date"])

        df = download_ohlcv(ticker)
        if df.empty:
            continue

        df = add_features(df)
        hist = df[df.index.normalize() < today].copy()
        if hist.empty:
            continue

        i = len(hist) - 1
        should_exit, reason = exit_signal(hist, i, entry_date)

        if should_exit:
            rows.append({
                "ticker": ticker,
                "reason": reason,
            })

    if not rows:
        return pd.DataFrame(columns=["ticker", "reason"])

    return pd.DataFrame(rows)


# =========================
# REAL EXECUTION
# =========================
def execute_day() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    日次実行:
    1. EXIT判定 → 実損益反映
    2. ENTRY判定 → 空き枠があれば新規追加
    """
    positions = load_positions()
    capital = load_equity()
    today = pd.Timestamp.now().normalize()

    # ---------- EXIT ----------
    exit_df = build_today_exits(positions)
    exit_tickers = set(exit_df["ticker"].tolist()) if not exit_df.empty else set()

    if not exit_df.empty:
        remaining = []

        for _, pos in positions.iterrows():
            ticker = pos["ticker"]

            if ticker not in exit_tickers:
                remaining.append(pos.to_dict())
                continue

            df = download_ohlcv(ticker)
            if df.empty:
                remaining.append(pos.to_dict())
                continue

            # 今日の未確定足を除いた最新確定足で決済
            hist = df[df.index.normalize() < today].copy()
            if hist.empty:
                remaining.append(pos.to_dict())
                continue

            exit_price = float(hist["Close"].iloc[-1])
            if not np.isfinite(exit_price) or exit_price <= 0:
                remaining.append(pos.to_dict())
                continue

            qty = float(pos["qty"])
            entry_price = float(pos["entry_price"])
            pnl = (exit_price - entry_price) * qty
            ret = pnl / (entry_price * qty)

            capital += pnl

            reason = exit_df.loc[exit_df["ticker"] == ticker, "reason"].iloc[0]

            append_trade_log({
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": hist.index[-1],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "pnl": pnl,
                "ret": ret,
                "reason": reason,
            })

        positions = pd.DataFrame(remaining) if remaining else pd.DataFrame(
            columns=["ticker", "entry_date", "entry_price", "qty"]
        )

    # ---------- ENTRY ----------
    entry_candidates = build_today_entries()

    if not entry_candidates.empty:
        # すでに持ってる銘柄は除外
        holding = set(positions["ticker"].tolist()) if not positions.empty else set()
        entry_candidates = entry_candidates[~entry_candidates["ticker"].isin(holding)].copy()

    executed_entries = []

    if not entry_candidates.empty:
        slots_left = P.max_positions - len(positions)
        if slots_left > 0:
            for _, row in entry_candidates.head(slots_left).iterrows():
                ticker = row["ticker"]

                df = download_ohlcv(ticker)
                if df.empty:
                    continue

                hist = df[df.index.normalize() < today].copy()
                if hist.empty:
                    continue

                entry_price = float(hist["Close"].iloc[-1])
                if not np.isfinite(entry_price) or entry_price <= 0:
                    continue

                qty = calc_qty(capital, entry_price)
                if qty <= 0:
                    continue

                new_pos = {
                    "ticker": ticker,
                    "entry_date": hist.index[-1],
                    "entry_price": entry_price,
                    "qty": qty,
                }

                positions = pd.concat([positions, pd.DataFrame([new_pos])], ignore_index=True)
                executed_entries.append({
                    "ticker": ticker,
                    "signal_date": row["signal_date"],
                    "entry_price": entry_price,
                    "qty": qty,
                    "rsi": row["rsi"],
                    "score": row["score"],
                })

    entry_df = pd.DataFrame(executed_entries)

    save_positions(positions)
    save_equity(capital)

    # CSV出力
    if entry_df.empty:
        pd.DataFrame(columns=["ticker", "signal_date", "entry_price", "qty", "rsi", "score"]).to_csv(
            P.today_entry_file, index=False
        )
    else:
        entry_df.to_csv(P.today_entry_file, index=False)

    if exit_df.empty:
        pd.DataFrame(columns=["ticker", "reason"]).to_csv(P.today_exit_file, index=False)
    else:
        exit_df.to_csv(P.today_exit_file, index=False)

    return entry_df, exit_df, positions, capital


# =========================
# OPTIONAL BACKTEST USING SAME RULES
# =========================
def backtest_single_ticker(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    同じ entry_signal / exit_signal を使う簡易バックテスト
    """
    df = download_ohlcv(ticker, years=years)
    if df.empty:
        return pd.DataFrame()

    df = add_features(df)

    capital = float(P.initial_capital)
    position = None
    trades = []

    for i in range(P.ma_days + 2, len(df)):
        date_i = pd.Timestamp(df.index[i]).normalize()

        # EXIT
        if position is not None:
            should_exit, reason = exit_signal(df, i, position["entry_date"])
            if should_exit:
                exit_price = float(df["Close"].iloc[i])
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                ret = pnl / (position["entry_price"] * position["qty"])
                capital += pnl

                trades.append({
                    "ticker": ticker,
                    "entry_date": position["entry_date"],
                    "exit_date": date_i,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "qty": position["qty"],
                    "pnl": pnl,
                    "ret": ret,
                    "reason": reason,
                })
                position = None

        # ENTRY
        if position is None and entry_signal(df, i):
            entry_price = float(df["Close"].iloc[i])
            qty = calc_qty(capital, entry_price)

            if qty > 0:
                position = {
                    "entry_date": date_i,
                    "entry_price": entry_price,
                    "qty": qty,
                }

    return pd.DataFrame(trades)


# =========================
# MAIN
# =========================
def main():
    print("RUN:", pd.Timestamp.now())

    entry_df, exit_df, positions, capital = execute_day()

    print("== TODAY ENTRY ==")
    if entry_df.empty:
        print("(no entry)")
    else:
        print(entry_df.to_string(index=False))

    print("\n== TODAY EXIT ==")
    if exit_df.empty:
        print("(no exit)")
    else:
        print(exit_df.to_string(index=False))

    print("\n== CAPITAL ==")
    print(round(capital, 2))

    print("\n== POSITIONS ==")
    if positions.empty:
        print("(empty)")
    else:
        print(positions.to_string(index=False))

    print(f"\nSaved: {P.today_entry_file}, {P.today_exit_file}, {P.pos_file}, {P.equity_file}, {P.trade_log_file}")

if __name__ == "__main__":
    main()