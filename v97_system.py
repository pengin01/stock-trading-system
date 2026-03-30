# v97_system.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import yfinance as yf
import ta

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
CASHFLOW_FILE = "cashflow.csv"
ENTRY_FILE = "today_entry.csv"
EXIT_FILE = "today_exit.csv"

POS_COLUMNS = ["ticker", "entry_date", "entry_price", "qty", "exit_date"]


# =========================
# DATA
# =========================
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1y", progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def entry_signal(df: pd.DataFrame, i: int) -> bool:
    c = float(df["Close"].iloc[i])
    prev = float(df["Close"].iloc[i - 1])

    if c < float(df["MA"].iloc[i]):
        return False
    if c > prev * (1 - PULLBACK):
        return False
    if float(df["RSI"].iloc[i]) > RSI_MAX:
        return False
    if float(df["VALUE20"].iloc[i]) < MIN_VALUE:
        return False

    return True


# =========================
# FILE HELPERS
# =========================
def ensure_files():
    if not os.path.exists(POS_FILE):
        pd.DataFrame(columns=POS_COLUMNS).to_csv(POS_FILE, index=False)

    if not os.path.exists(EQ_FILE):
        pd.DataFrame(columns=["date", "equity", "cash", "position_value"]).to_csv(EQ_FILE, index=False)

    if not os.path.exists(CASHFLOW_FILE):
        pd.DataFrame(columns=["date", "amount", "note"]).to_csv(CASHFLOW_FILE, index=False)

    if not os.path.exists(ENTRY_FILE):
        pd.DataFrame(columns=["ticker", "signal_date", "entry_price", "qty", "rsi", "score"]).to_csv(ENTRY_FILE, index=False)

    if not os.path.exists(EXIT_FILE):
        pd.DataFrame(columns=["ticker", "reason"]).to_csv(EXIT_FILE, index=False)


def load_positions() -> pd.DataFrame:
    if not os.path.exists(POS_FILE):
        return pd.DataFrame(columns=POS_COLUMNS)

    try:
        df = pd.read_csv(POS_FILE, parse_dates=["entry_date", "exit_date"])
    except Exception:
        return pd.DataFrame(columns=POS_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=POS_COLUMNS)

    for c in POS_COLUMNS:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")

    return df[POS_COLUMNS].copy()


def save_positions(df: pd.DataFrame):
    if df is None or df.empty:
        pd.DataFrame(columns=POS_COLUMNS).to_csv(POS_FILE, index=False)
        return

    out = df.copy()
    for c in POS_COLUMNS:
        if c not in out.columns:
            out[c] = pd.Series(dtype="object")

    out = out[POS_COLUMNS]
    out.to_csv(POS_FILE, index=False)


def load_equity_df() -> pd.DataFrame:
    if not os.path.exists(EQ_FILE):
        return pd.DataFrame(columns=["date", "equity", "cash", "position_value"])

    try:
        df = pd.read_csv(EQ_FILE)
    except Exception:
        return pd.DataFrame(columns=["date", "equity", "cash", "position_value"])

    if df.empty:
        return pd.DataFrame(columns=["date", "equity", "cash", "position_value"])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["equity", "cash", "position_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def save_equity(equity_value: float, cash_value: float, position_value: float):
    row = pd.DataFrame([{
        "date": pd.Timestamp.now(),
        "equity": float(equity_value),
        "cash": float(cash_value),
        "position_value": float(position_value),
    }])

    old = load_equity_df()
    if old.empty:
        row.to_csv(EQ_FILE, index=False)
    else:
        row.to_csv(EQ_FILE, mode="a", header=False, index=False)


def load_cashflow_df() -> pd.DataFrame:
    if not os.path.exists(CASHFLOW_FILE):
        return pd.DataFrame(columns=["date", "amount", "note"])

    try:
        df = pd.read_csv(CASHFLOW_FILE)
    except Exception:
        return pd.DataFrame(columns=["date", "amount", "note"])

    if df.empty:
        return pd.DataFrame(columns=["date", "amount", "note"])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "note" not in df.columns:
        df["note"] = ""

    df = df.dropna(subset=["date", "amount"]).sort_values("date").reset_index(drop=True)
    return df[["date", "amount", "note"]]


def ensure_initial_cashflow(today: pd.Timestamp):
    cf = load_cashflow_df()
    if cf.empty:
        init_row = pd.DataFrame([{
            "date": today,
            "amount": float(INITIAL_CAPITAL),
            "note": "initial"
        }])
        init_row.to_csv(CASHFLOW_FILE, index=False)


def get_total_cashflow_until(today: pd.Timestamp) -> float:
    cf = load_cashflow_df()
    if cf.empty:
        return 0.0

    x = cf[cf["date"] <= today]
    if x.empty:
        return 0.0

    return float(x["amount"].sum())


def get_latest_equity_value() -> float | None:
    eq = load_equity_df()
    if eq.empty or "equity" not in eq.columns:
        return None

    s = pd.to_numeric(eq["equity"], errors="coerce").dropna()
    if s.empty:
        return None

    return float(s.iloc[-1])


def calc_starting_cash_for_today(today: pd.Timestamp) -> float:
    """
    当日の運用開始cashを決める。
    基本は前回equityを引き継ぎつつ、当日までのcashflow差分を加える。
    """
    total_cashflow = get_total_cashflow_until(today)
    latest_equity = get_latest_equity_value()

    if latest_equity is None:
        return total_cashflow if total_cashflow != 0 else float(INITIAL_CAPITAL)

    eq = load_equity_df()
    prev_date = pd.to_datetime(eq["date"], errors="coerce").dropna()
    if prev_date.empty:
        return latest_equity

    last_eq_day = prev_date.iloc[-1].normalize()
    cashflow_until_prev = 0.0
    cf = load_cashflow_df()
    if not cf.empty:
        cashflow_until_prev = float(cf[cf["date"] <= last_eq_day]["amount"].sum())

    extra_flow = total_cashflow - cashflow_until_prev
    return latest_equity + extra_flow


# =========================
# MAIN
# =========================
def main():
    print("RUN:", pd.Timestamp.now())

    ensure_files()

    today = pd.Timestamp.now().normalize()
    ensure_initial_cashflow(today)

    pos = load_positions()
    cash = calc_starting_cash_for_today(today)

    entries = []
    exits = []

    data_cache = {t: load_data(t) for t in STOCK_UNIVERSE}

    # =====================
    # EXIT（時間決済）
    # =====================
    new_pos = []

    for _, p in pos.iterrows():
        exit_date = pd.to_datetime(p["exit_date"]).normalize()

        if today < exit_date:
            new_pos.append(p.to_dict())
            continue

        df = data_cache.get(p["ticker"], pd.DataFrame())
        if df.empty or today not in df.index:
            new_pos.append(p.to_dict())
            continue

        price = float(df.loc[today, "Close"])
        cash += price * float(p["qty"])

        exits.append({
            "ticker": p["ticker"],
            "reason": "time_exit",
        })

    pos = pd.DataFrame(new_pos, columns=POS_COLUMNS)

    # =====================
    # ENTRY
    # =====================
    if len(pos) < MAX_POSITIONS:
        candidates = []

        for t, df in data_cache.items():
            if df.empty:
                continue

            if "ticker" in pos.columns and t in pos["ticker"].values:
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

            score = -float(df["RSI"].iloc[i])
            candidates.append((score, t, i))

        if candidates:
            candidates.sort()
            score, t, i = candidates[0]

            df = data_cache[t]
            price = float(df["Close"].iloc[i])

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
                    "exit_date": df.index[i + HOLD_DAYS].normalize(),
                }

                pos = pd.concat([pos, pd.DataFrame([new_pos_row])], ignore_index=True)

                entries.append({
                    "ticker": t,
                    "signal_date": today.strftime("%Y-%m-%d"),
                    "entry_price": price,
                    "qty": qty,
                    "rsi": float(df["RSI"].iloc[i]),
                    "score": score,
                })

    # =====================
    # EQUITY
    # =====================
    position_value = 0.0

    for _, p in pos.iterrows():
        df = data_cache.get(p["ticker"], pd.DataFrame())
        if df.empty:
            px = float(p["entry_price"])
        elif today in df.index:
            px = float(df.loc[today, "Close"])
        else:
            px = float(p["entry_price"])

        position_value += px * float(p["qty"])

    equity = cash + position_value
    total_cashflow = get_total_cashflow_until(today)
    pnl = equity - total_cashflow

    # =====================
    # SAVE
    # =====================
    save_positions(pos)
    save_equity(equity, cash, position_value)

    pd.DataFrame(entries, columns=["ticker", "signal_date", "entry_price", "qty", "rsi", "score"]).to_csv(
        ENTRY_FILE, index=False
    )
    pd.DataFrame(exits, columns=["ticker", "reason"]).to_csv(
        EXIT_FILE, index=False
    )

    # =====================
    # LOG
    # =====================
    print("== ENTRY ==")
    print(pd.DataFrame(entries) if entries else "(none)")

    print("\n== EXIT ==")
    print(pd.DataFrame(exits) if exits else "(none)")

    print("\n== CASHFLOW ==")
    print(total_cashflow)

    print("\n== PNL ==")
    print(pnl)

    print("\n== EQUITY ==")
    print(equity)

    print("\n== POSITIONS ==")
    print(pos if not pos.empty else "(empty)")


if __name__ == "__main__":
    main()