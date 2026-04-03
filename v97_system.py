# v97_system.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import yfinance as yf
import ta

# =========================
# PARAMETERS
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

UNIVERSE_FILE = "nikkei225.csv"

POS_FILE = "positions.csv"
EQ_FILE = "equity.csv"
CASHFLOW_FILE = "cashflow.csv"
ENTRY_FILE = "today_entry.csv"
EXIT_FILE = "today_exit.csv"
CANDIDATE_FILE = "candidate_rank.csv"

POS_COLUMNS = ["ticker", "entry_date", "entry_price", "qty", "exit_date"]


# =========================
# UNIVERSE
# =========================
def load_universe():
    if not os.path.exists(UNIVERSE_FILE):
        return []

    df = pd.read_csv(UNIVERSE_FILE)
    return df["ticker"].dropna().astype(str).tolist()


# =========================
# DATA
# =========================
def load_data(ticker):
    df = yf.download(ticker, period="1y", progress=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def get_signal_date(data_cache):
    dates = [df.index.max() for df in data_cache.values() if not df.empty]
    return min(dates).normalize() if dates else None


# =========================
# FILE
# =========================
def ensure_files():
    (
        pd.DataFrame(columns=POS_COLUMNS).to_csv(POS_FILE, index=False)
        if not os.path.exists(POS_FILE)
        else None
    )
    (
        pd.DataFrame(columns=["date", "equity", "cash", "position_value"]).to_csv(
            EQ_FILE, index=False
        )
        if not os.path.exists(EQ_FILE)
        else None
    )
    (
        pd.DataFrame(columns=["date", "amount", "note"]).to_csv(
            CASHFLOW_FILE, index=False
        )
        if not os.path.exists(CASHFLOW_FILE)
        else None
    )
    (
        pd.DataFrame(
            columns=["ticker", "signal_date", "entry_price", "qty", "rsi", "score"]
        ).to_csv(ENTRY_FILE, index=False)
        if not os.path.exists(ENTRY_FILE)
        else None
    )
    (
        pd.DataFrame(columns=["ticker", "reason"]).to_csv(EXIT_FILE, index=False)
        if not os.path.exists(EXIT_FILE)
        else None
    )
    (
        pd.DataFrame(
            columns=[
                "date",
                "rank",
                "ticker",
                "close",
                "prev_close",
                "ma",
                "rsi",
                "value20",
                "pullback_ratio",
                "score",
            ]
        ).to_csv(CANDIDATE_FILE, index=False)
        if not os.path.exists(CANDIDATE_FILE)
        else None
    )


def load_positions():
    if not os.path.exists(POS_FILE):
        return pd.DataFrame(columns=POS_COLUMNS)
    return pd.read_csv(POS_FILE, parse_dates=["entry_date", "exit_date"])


def save_positions(df):
    df.to_csv(POS_FILE, index=False)


def load_equity():
    if not os.path.exists(EQ_FILE):
        return pd.DataFrame()
    return pd.read_csv(EQ_FILE)


def save_equity(equity, cash, pos_val):
    row = pd.DataFrame(
        [
            {
                "date": pd.Timestamp.now(),
                "equity": equity,
                "cash": cash,
                "position_value": pos_val,
            }
        ]
    )
    if not os.path.exists(EQ_FILE):
        row.to_csv(EQ_FILE, index=False)
    else:
        row.to_csv(EQ_FILE, mode="a", header=False, index=False)


# =========================
# CASHFLOW
# =========================
def load_cashflow():
    if not os.path.exists(CASHFLOW_FILE):
        return pd.DataFrame()
    df = pd.read_csv(CASHFLOW_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def ensure_initial_cashflow(today):
    cf = load_cashflow()
    if cf.empty:
        pd.DataFrame(
            [{"date": today, "amount": INITIAL_CAPITAL, "note": "initial"}]
        ).to_csv(CASHFLOW_FILE, index=False)


def get_cashflow_until(date):
    cf = load_cashflow()
    return cf[cf["date"] <= date]["amount"].sum() if not cf.empty else 0


# =========================
# CANDIDATES
# =========================
def build_candidates(signal_date, pos, data_cache):
    candidates = []
    stats = dict(
        total=0,
        has_data=0,
        not_held=0,
        tradable_today=0,
        enough_history=0,
        ma_fail=0,
        pullback_fail=0,
        rsi_fail=0,
        value20_fail=0,
        passed=0,
    )

    for t, df in data_cache.items():
        stats["total"] += 1
        if df.empty:
            continue
        stats["has_data"] += 1

        if t in pos["ticker"].values:
            continue
        stats["not_held"] += 1

        if signal_date not in df.index:
            continue
        stats["tradable_today"] += 1

        i = df.index.get_loc(signal_date)
        if i < MA_DAYS + 2:
            continue
        stats["enough_history"] += 1

        c, prev = df["Close"].iloc[i], df["Close"].iloc[i - 1]
        ma, rsi, val = df["MA"].iloc[i], df["RSI"].iloc[i], df["VALUE20"].iloc[i]

        if c < ma:
            stats["ma_fail"] += 1
            continue
        if c > prev * (1 - PULLBACK):
            stats["pullback_fail"] += 1
            continue
        if rsi > RSI_MAX:
            stats["rsi_fail"] += 1
            continue
        if val < MIN_VALUE:
            stats["value20_fail"] += 1
            continue

        # ★ RSI低い順
        score = rsi

        stats["passed"] += 1

        candidates.append(
            dict(
                ticker=t,
                close=c,
                prev_close=prev,
                ma=ma,
                rsi=rsi,
                value20=val,
                pullback_ratio=c / prev - 1,
                score=score,
                i=i,
            )
        )

    candidates.sort(key=lambda x: x["score"])
    return candidates, stats


def save_candidates(signal_date, cands):
    rows = []
    for i, c in enumerate(cands, 1):
        rows.append(dict(date=signal_date, rank=i, **c))
    pd.DataFrame(rows).to_csv(CANDIDATE_FILE, index=False)


# =========================
# HOLD DAYS
# =========================
def bars_passed(df, entry, signal):
    return ((df.index > entry) & (df.index <= signal)).sum()


# =========================
# MAIN
# =========================
def main():
    print("RUN:", pd.Timestamp.now())

    ensure_files()

    today = pd.Timestamp.now().normalize()
    ensure_initial_cashflow(today)

    universe = load_universe()
    print("Universe size:", len(universe))

    data_cache = {t: load_data(t) for t in universe}
    signal_date = get_signal_date(data_cache)

    print("Today:", today)
    print("Signal date:", signal_date)

    pos = load_positions()
    cash = get_cashflow_until(signal_date)

    entries, exits = [], []

    # EXIT
    new = []
    for _, p in pos.iterrows():
        df = data_cache.get(p["ticker"], pd.DataFrame())
        if df.empty or signal_date not in df.index:
            new.append(p)
            continue

        if bars_passed(df, p["entry_date"], signal_date) < HOLD_DAYS:
            new.append(p)
            continue

        price = df.loc[signal_date, "Close"]
        cash += price * p["qty"]
        exits.append(dict(ticker=p["ticker"], reason="time_exit"))

    pos = pd.DataFrame(new, columns=POS_COLUMNS)

    # ENTRY
    cands, stats = build_candidates(signal_date, pos, data_cache)
    save_candidates(signal_date, cands)

    if len(pos) < MAX_POSITIONS and cands:
        c = cands[0]
        price = c["close"]
        qty = int((cash * RISK_RATIO) // price)

        if qty > 0:
            cash -= price * qty
            pos = pd.concat(
                [
                    pos,
                    pd.DataFrame(
                        [
                            {
                                "ticker": c["ticker"],
                                "entry_date": signal_date,
                                "entry_price": price,
                                "qty": qty,
                                "exit_date": pd.NaT,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            entries.append(
                dict(
                    ticker=c["ticker"],
                    signal_date=signal_date,
                    entry_price=price,
                    qty=qty,
                    rsi=c["rsi"],
                    score=c["score"],
                )
            )

    # EQUITY
    pos_val = 0
    for _, p in pos.iterrows():
        df = data_cache.get(p["ticker"], pd.DataFrame())
        px = (
            df.loc[signal_date, "Close"]
            if signal_date in df.index
            else p["entry_price"]
        )
        pos_val += px * p["qty"]

    equity = cash + pos_val
    pnl = equity - get_cashflow_until(signal_date)

    save_positions(pos)
    save_equity(equity, cash, pos_val)

    pd.DataFrame(entries).to_csv(ENTRY_FILE, index=False)
    pd.DataFrame(exits).to_csv(EXIT_FILE, index=False)

    # LOG
    print("== ENTRY ==")
    print(entries or "(none)")
    print("\n== EXIT ==")
    print(exits or "(none)")
    print("\n== TOP CANDIDATES ==")
    print(
        pd.DataFrame(cands)[["ticker", "rsi", "pullback_ratio", "score"]].head(10)
        if cands
        else "(none)"
    )
    print("\n== FILTER SUMMARY ==")
    [print(f"{k}: {v}") for k, v in stats.items()]
    print("\n== CASHFLOW ==")
    print(get_cashflow_until(signal_date))
    print("\n== PNL ==")
    print(pnl)
    print("\n== EQUITY ==")
    print(equity)
    print("\n== POSITIONS ==")
    print(pos if not pos.empty else "(empty)")


if __name__ == "__main__":
    main()
