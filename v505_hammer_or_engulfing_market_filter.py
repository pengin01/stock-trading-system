# v505_hammer_or_engulfing_market_filter.py
# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf

INITIAL_CAPITAL = 20000

MA_SHORT = 25
MA_LONG = 75

HOLD_DAYS = 7
MAX_POSITIONS = 2
RISK_RATIO = 0.5

MIN_VALUE = 100_000_000
YEARS = 5

EXIT_MA_BUFFER = 0.98

PULLBACK_DAYS = 10
PULLBACK_PCT = 0.03

MARKET_TICKER = "1321.T"
MARKET_MA = 25

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


def load_data(ticker):
    df = yf.download(ticker, period=f"{YEARS}y", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def load_market_data():
    df = yf.download(MARKET_TICKER, period=f"{YEARS}y", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MARKET_MA"] = df["Close"].rolling(MARKET_MA).mean()

    return df.dropna()


def is_market_ok(d):
    if market.empty:
        return True

    if d not in market.index:
        return False

    return market.loc[d, "Close"] > market.loc[d, "MARKET_MA"]


# ===== ハンマー =====
def is_hammer(df, i):
    o = df["Open"].iloc[i]
    c = df["Close"].iloc[i]
    h = df["High"].iloc[i]
    l = df["Low"].iloc[i]

    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)

    if body == 0:
        return False

    return lower > body * 2 and upper < body * 0.5


# ===== 陽の包み足 =====
def is_bullish_engulfing(df, i):
    if i < 1:
        return False

    prev_open = df["Open"].iloc[i - 1]
    prev_close = df["Close"].iloc[i - 1]

    open_ = df["Open"].iloc[i]
    close = df["Close"].iloc[i]

    return (
        prev_close < prev_open
        and close > open_
        and open_ <= prev_close
        and close >= prev_open
    )


def has_pullback(df, i):
    if i < PULLBACK_DAYS:
        return False

    recent_high = df["Close"].iloc[i - PULLBACK_DAYS : i].max()
    current_close = df["Close"].iloc[i]

    return current_close <= recent_high * (1 - PULLBACK_PCT)


def run(start, end):
    cash = INITIAL_CAPITAL
    pos = []
    eq = []

    trades = []

    dates = sorted(set().union(*[df.index for df in data.values()]))

    for d in dates:
        if d < start or d > end:
            continue

        # ===== EXIT =====
        new_pos = []

        for p in pos:
            df = data[p["t"]]

            if d not in df.index:
                new_pos.append(p)
                continue

            price = df.loc[d, "Close"]
            ma25 = df.loc[d, "MA25"]

            entry_i = df.index.get_loc(p["d"])
            current_i = df.index.get_loc(d)
            hold = current_i - entry_i

            if price >= ma25 * EXIT_MA_BUFFER and hold < HOLD_DAYS:
                new_pos.append(p)
                continue

            pnl = (price - p["p"]) * p["q"]
            ret = price / p["p"] - 1

            trades.append(
                {
                    "ticker": p["t"],
                    "entry_date": p["d"],
                    "exit_date": d,
                    "entry_price": p["p"],
                    "exit_price": price,
                    "qty": p["q"],
                    "pnl": pnl,
                    "return": ret,
                    "hold": hold,
                    "signal": p["signal"],
                }
            )

            cash += price * p["q"]

        pos = new_pos

        # ===== ENTRY =====
        if len(pos) < MAX_POSITIONS and is_market_ok(d):
            for t, df in data.items():
                if len(pos) >= MAX_POSITIONS:
                    break

                if any(p["t"] == t for p in pos):
                    continue

                if d not in df.index:
                    continue

                i = df.index.get_loc(d)

                if i < max(MA_LONG, PULLBACK_DAYS) + 1:
                    continue

                close = df["Close"].iloc[i]
                ma75 = df["MA75"].iloc[i]
                vol = df["Volume"].iloc[i]
                vol20 = df["VOL20"].iloc[i]
                value20 = df["VALUE20"].iloc[i]

                if value20 < MIN_VALUE:
                    continue

                if close <= ma75:
                    continue

                if not has_pullback(df, i):
                    continue

                hammer = is_hammer(df, i)
                engulfing = is_bullish_engulfing(df, i)

                if not (hammer or engulfing):
                    continue

                if vol < vol20:
                    continue

                qty = int((cash * RISK_RATIO) // close)

                if qty <= 0:
                    continue

                if hammer and engulfing:
                    signal = "hammer+engulfing"
                elif hammer:
                    signal = "hammer"
                else:
                    signal = "engulfing"

                cash -= close * qty

                pos.append(
                    {
                        "t": t,
                        "p": close,
                        "d": d,
                        "q": qty,
                        "signal": signal,
                    }
                )

        # ===== EQUITY =====
        position_value = 0

        for p in pos:
            df = data[p["t"]]

            if d in df.index:
                position_value += df.loc[d, "Close"] * p["q"]

        eq.append(cash + position_value)

    total_return = 0 if not eq else eq[-1] / INITIAL_CAPITAL - 1

    return total_return, trades, eq


def print_result(label, start, end):
    total_return, trades, eq = run(start, end)

    print(f"{label}: {total_return}")

    if trades:
        tdf = pd.DataFrame(trades)
        win_rate = (tdf["pnl"] > 0).mean()
        avg_return = tdf["return"].mean()
        trade_count = len(tdf)

        print(f"  trades    : {trade_count}")
        print(f"  win_rate  : {win_rate}")
        print(f"  avg_return: {avg_return}")

        print("  by_signal:")
        print(tdf.groupby("signal")["return"].agg(["count", "mean", "sum"]))
    else:
        print("  trades    : 0")


# ===== DATA LOAD =====
market = load_market_data()

data = {t: load_data(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}

print("\n=== WALK FORWARD : HAMMER OR ENGULFING + MARKET FILTER v505 ===")
print("HOLD_DAYS:", HOLD_DAYS)
print("EXIT_MA_BUFFER:", EXIT_MA_BUFFER)
print("PULLBACK_PCT:", PULLBACK_PCT)
print("MARKET_TICKER:", MARKET_TICKER)
print("MARKET_MA:", MARKET_MA)

print_result("2021-2023", pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"))
print_result("2024-2026", pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31"))
print_result("2024 only", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
print_result("2025 only", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
print_result("2026 only", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
