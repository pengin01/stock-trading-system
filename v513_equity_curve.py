# v513_equity_curve.py
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

INITIAL_CAPITAL = 20000

MA_SHORT = 25
MA_LONG = 75

MAX_POSITIONS = 2
RISK_RATIO = 0.5

MIN_VALUE = 100_000_000
YEARS = 3

# ===== 最適パラメータ =====
HOLD_DAYS = 11
PULLBACK_PCT = 0.032
EXIT_MA_BUFFER = 0.98

EXCLUDE_TICKERS = ["6758.T", "4568.T", "4063.T"]

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
    if i < 10:
        return False

    recent_high = df["Close"].iloc[i - 10 : i].max()
    current_close = df["Close"].iloc[i]

    return current_close <= recent_high * (1 - PULLBACK_PCT)


def run():
    cash = INITIAL_CAPITAL
    pos = []
    eq = []
    eq_dates = []

    dates = sorted(set().union(*[df.index for df in data.values()]))

    for d in dates:

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

            cash += price * p["q"]

        pos = new_pos

        # ===== ENTRY =====
        if len(pos) < MAX_POSITIONS:
            for t, df in data.items():
                if len(pos) >= MAX_POSITIONS:
                    break

                if t in EXCLUDE_TICKERS:
                    continue

                if any(p["t"] == t for p in pos):
                    continue

                if d not in df.index:
                    continue

                i = df.index.get_loc(d)

                if i < 75:
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

                if not is_bullish_engulfing(df, i):
                    continue

                if vol < vol20:
                    continue

                qty = int((cash * RISK_RATIO) // close)

                if qty <= 0:
                    continue

                cash -= close * qty
                pos.append({"t": t, "p": close, "d": d, "q": qty})

        # ===== EQUITY =====
        pv = 0
        for p in pos:
            df = data[p["t"]]
            if d in df.index:
                pv += df.loc[d, "Close"] * p["q"]

        eq.append(cash + pv)
        eq_dates.append(d)

    if not eq:
        return 0, pd.DataFrame()

    equity_df = pd.DataFrame({"date": eq_dates, "equity": eq})

    total_return = eq[-1] / INITIAL_CAPITAL - 1

    return total_return, equity_df


def plot_equity(equity_df):
    if equity_df.empty:
        print("No data")
        return

    equity_df = equity_df.copy()
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = equity_df["equity"] / equity_df["peak"] - 1

    equity_df.to_csv("equity_curve.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["date"], equity_df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.show()

    print("Saved: equity_curve.csv")
    print("Saved: equity_curve.png")
    print("Max Drawdown:", equity_df["drawdown"].min())


# ===== DATA LOAD =====
data = {t: load_data(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}

# ===== RUN =====
total_return, equity_df = run()

print("\n=== RESULT ===")
print("TOTAL RETURN:", total_return)

plot_equity(equity_df)
