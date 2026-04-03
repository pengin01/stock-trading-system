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
def load_universe() -> list[str]:
    if not os.path.exists(UNIVERSE_FILE):
        return []

    try:
        df = pd.read_csv(UNIVERSE_FILE)
    except Exception:
        return []

    if "ticker" not in df.columns:
        return []

    tickers = df["ticker"].dropna().astype(str).str.strip().tolist()
    return [t for t in tickers if t]


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


def get_signal_date(data_cache: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    dates = []
    for df in data_cache.values():
        if df is None or df.empty:
            continue
        dates.append(df.index.max())

    if not dates:
        return None

    # 全銘柄で共通に使える最新日
    return min(dates).normalize()


# =========================
# FILE HELPERS
# =========================
def ensure_files():
    if not os.path.exists(POS_FILE):
        pd.DataFrame(columns=POS_COLUMNS).to_csv(POS_FILE, index=False)

    if not os.path.exists(EQ_FILE):
        pd.DataFrame(columns=["date", "equity", "cash", "position_value"]).to_csv(
            EQ_FILE, index=False
        )

    if not os.path.exists(CASHFLOW_FILE):
        pd.DataFrame(columns=["date", "amount", "note"]).to_csv(
            CASHFLOW_FILE, index=False
        )

    if not os.path.exists(ENTRY_FILE):
        pd.DataFrame(
            columns=["ticker", "signal_date", "entry_price", "qty", "rsi", "score"]
        ).to_csv(ENTRY_FILE, index=False)

    if not os.path.exists(EXIT_FILE):
        pd.DataFrame(columns=["ticker", "reason"]).to_csv(EXIT_FILE, index=False)

    if not os.path.exists(CANDIDATE_FILE):
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
    row = pd.DataFrame(
        [
            {
                "date": pd.Timestamp.now(),
                "equity": float(equity_value),
                "cash": float(cash_value),
                "position_value": float(position_value),
            }
        ]
    )

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
        init_row = pd.DataFrame(
            [{"date": today, "amount": float(INITIAL_CAPITAL), "note": "initial"}]
        )
        init_row.to_csv(CASHFLOW_FILE, index=False)


def get_total_cashflow_until(base_date: pd.Timestamp) -> float:
    cf = load_cashflow_df()
    if cf.empty:
        return 0.0

    x = cf[cf["date"] <= base_date]
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


def calc_starting_cash_for_signal_date(signal_date: pd.Timestamp) -> float:
    total_cashflow = get_total_cashflow_until(signal_date)
    latest_equity = get_latest_equity_value()

    if latest_equity is None:
        return total_cashflow if total_cashflow != 0 else float(INITIAL_CAPITAL)

    eq = load_equity_df()
    prev_date = pd.to_datetime(eq["date"], errors="coerce").dropna()
    if prev_date.empty:
        return latest_equity

    last_eq_day = prev_date.iloc[-1].normalize()
    cf = load_cashflow_df()

    cashflow_until_prev = 0.0
    if not cf.empty:
        cashflow_until_prev = float(cf[cf["date"] <= last_eq_day]["amount"].sum())

    extra_flow = total_cashflow - cashflow_until_prev
    return latest_equity + extra_flow


# =========================
# CANDIDATES + DIAGNOSTICS
# =========================
def build_candidates_with_diagnostics(
    signal_date: pd.Timestamp, pos: pd.DataFrame, data_cache: dict[str, pd.DataFrame]
):
    candidates = []
    stats = {
        "total": 0,
        "has_data": 0,
        "not_held": 0,
        "tradable_today": 0,
        "enough_history": 0,
        "enough_exit_room": 0,
        "ma_fail": 0,
        "pullback_fail": 0,
        "rsi_fail": 0,
        "value20_fail": 0,
        "passed": 0,
    }

    for t, df in data_cache.items():
        stats["total"] += 1

        if df.empty:
            continue
        stats["has_data"] += 1

        if "ticker" in pos.columns and t in pos["ticker"].values:
            continue
        stats["not_held"] += 1

        if signal_date not in df.index:
            continue
        stats["tradable_today"] += 1

        i = df.index.get_loc(signal_date)

        if i < MA_DAYS + 2:
            continue
        stats["enough_history"] += 1

        if i + HOLD_DAYS >= len(df):
            continue
        stats["enough_exit_room"] += 1

        close = float(df["Close"].iloc[i])
        prev_close = float(df["Close"].iloc[i - 1])
        ma = float(df["MA"].iloc[i])
        rsi = float(df["RSI"].iloc[i])
        value20 = float(df["VALUE20"].iloc[i])
        pullback_ratio = close / prev_close - 1.0

        failed = False

        if close < ma:
            stats["ma_fail"] += 1
            failed = True

        if close > prev_close * (1 - PULLBACK):
            stats["pullback_fail"] += 1
            failed = True

        if rsi > RSI_MAX:
            stats["rsi_fail"] += 1
            failed = True

        if value20 < MIN_VALUE:
            stats["value20_fail"] += 1
            failed = True

        if failed:
            continue

        score = -rsi
        stats["passed"] += 1

        candidates.append(
            {
                "ticker": t,
                "close": close,
                "prev_close": prev_close,
                "ma": ma,
                "rsi": rsi,
                "value20": value20,
                "pullback_ratio": pullback_ratio,
                "score": score,
                "i": i,
            }
        )

    candidates.sort(key=lambda x: x["score"])
    return candidates, stats


def save_candidate_rank(signal_date: pd.Timestamp, candidates: list[dict]):
    rows = []
    for rank, c in enumerate(candidates, start=1):
        rows.append(
            {
                "date": signal_date.strftime("%Y-%m-%d"),
                "rank": rank,
                "ticker": c["ticker"],
                "close": c["close"],
                "prev_close": c["prev_close"],
                "ma": c["ma"],
                "rsi": c["rsi"],
                "value20": c["value20"],
                "pullback_ratio": c["pullback_ratio"],
                "score": c["score"],
            }
        )

    df = pd.DataFrame(
        rows,
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
        ],
    )
    df.to_csv(CANDIDATE_FILE, index=False)


# =========================
# MAIN
# =========================
def main():
    print("RUN:", pd.Timestamp.now())

    ensure_files()

    today = pd.Timestamp.now().normalize()
    ensure_initial_cashflow(today)

    universe = load_universe()
    if not universe:
        raise RuntimeError("nikkei225.csv is empty or missing ticker column")

    print("Universe size:", len(universe))
    data_cache = {t: load_data(t) for t in universe}

    signal_date = get_signal_date(data_cache)
    if signal_date is None:
        raise RuntimeError("No price data available")

    print("Today:", today)
    print("Signal date:", signal_date)

    pos = load_positions()
    cash = calc_starting_cash_for_signal_date(signal_date)

    entries = []
    exits = []

    # EXIT
    new_pos = []

    for _, p in pos.iterrows():
        exit_date = pd.to_datetime(p["exit_date"]).normalize()

        if signal_date < exit_date:
            new_pos.append(p.to_dict())
            continue

        df = data_cache.get(p["ticker"], pd.DataFrame())
        if df.empty or signal_date not in df.index:
            new_pos.append(p.to_dict())
            continue

        price = float(df.loc[signal_date, "Close"])
        cash += price * float(p["qty"])

        exits.append(
            {
                "ticker": p["ticker"],
                "reason": "time_exit",
            }
        )

    pos = pd.DataFrame(new_pos, columns=POS_COLUMNS)

    # ENTRY
    candidates, filter_stats = build_candidates_with_diagnostics(
        signal_date, pos, data_cache
    )
    save_candidate_rank(signal_date, candidates)

    if len(pos) < MAX_POSITIONS and candidates:
        c = candidates[0]
        t = c["ticker"]
        i = c["i"]

        df = data_cache[t]
        price = float(df["Close"].iloc[i])

        usable_cash = cash * RISK_RATIO
        qty = int(usable_cash // price)

        if qty > 0:
            cost = price * qty
            cash -= cost

            new_pos_row = {
                "ticker": t,
                "entry_date": signal_date,
                "entry_price": price,
                "qty": qty,
                "exit_date": df.index[i + HOLD_DAYS].normalize(),
            }

            pos = pd.concat([pos, pd.DataFrame([new_pos_row])], ignore_index=True)

            entries.append(
                {
                    "ticker": t,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "entry_price": price,
                    "qty": qty,
                    "rsi": c["rsi"],
                    "score": c["score"],
                }
            )

    # EQUITY
    position_value = 0.0

    for _, p in pos.iterrows():
        df = data_cache.get(p["ticker"], pd.DataFrame())
        if df.empty:
            px = float(p["entry_price"])
        elif signal_date in df.index:
            px = float(df.loc[signal_date, "Close"])
        else:
            px = float(p["entry_price"])

        position_value += px * float(p["qty"])

    equity = cash + position_value
    total_cashflow = get_total_cashflow_until(signal_date)
    pnl = equity - total_cashflow

    # SAVE
    save_positions(pos)
    save_equity(equity, cash, position_value)

    pd.DataFrame(
        entries, columns=["ticker", "signal_date", "entry_price", "qty", "rsi", "score"]
    ).to_csv(ENTRY_FILE, index=False)
    pd.DataFrame(exits, columns=["ticker", "reason"]).to_csv(EXIT_FILE, index=False)

    # LOG
    print("== ENTRY ==")
    print(pd.DataFrame(entries) if entries else "(none)")

    print("\n== EXIT ==")
    print(pd.DataFrame(exits) if exits else "(none)")

    print("\n== TOP CANDIDATES ==")
    if candidates:
        print(
            pd.DataFrame(candidates)[["ticker", "rsi", "pullback_ratio", "score"]]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("(none)")

    print("\n== FILTER SUMMARY ==")
    for k, v in filter_stats.items():
        print(f"{k}: {v}")

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
