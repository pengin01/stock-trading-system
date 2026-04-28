# v516_analysis.py

# -_- coding: utf-8 -_-

import pandas as pd
import yfinance as yf

INITIAL_CAPITAL = 20000

MA_SHORT = 25
MA_LONG = 75

MAX_POSITIONS = 1
RISK_RATIO = 1.0

MIN_VALUE = 100_000_000
YEARS = 5

HOLD_DAYS = 11
PULLBACK_PCT = 0.032
EXIT_MA_BUFFER = 0.98

SL_PCT = 0.07

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


def load_data(t):
    df = yf.download(t, period=f"{YEARS}y", progress=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def is_engulf(df, i):
    if i < 1:
        return False
    po, pc = df["Open"].iloc[i - 1], df["Close"].iloc[i - 1]
    o, c = df["Open"].iloc[i], df["Close"].iloc[i]
    return pc < po and c > o and o <= pc and c >= po


def has_pullback(df, i):
    if i < 10:
        return False
    high = df["Close"].iloc[i - 10 : i].max()
    return df["Close"].iloc[i] <= high * (1 - PULLBACK_PCT)


def run():
    cash = INITIAL_CAPITAL
    pos = []
    eq = []
    dates_eq = []
    trades = []

    dates = sorted(set().union(*[df.index for df in data.values()]))

    for d in dates:

        new_pos = []
        for p in pos:
            df = data[p["t"]]

            if d not in df.index:
                new_pos.append(p)
                continue

            price = df.loc[d, "Close"]
            ma25 = df.loc[d, "MA25"]

            entry_i = df.index.get_loc(p["d"])
            cur_i = df.index.get_loc(d)
            hold = cur_i - entry_i

            # SL
            if price <= p["p"] * (1 - SL_PCT):
                trades.append(
                    {
                        "ticker": p["t"],
                        "entry": p["p"],
                        "exit": price,
                        "return": price / p["p"] - 1,
                        "date": d,
                    }
                )
                cash += price * p["q"]
                continue

            if price >= ma25 * EXIT_MA_BUFFER and hold < HOLD_DAYS:
                new_pos.append(p)
                continue

            trades.append(
                {
                    "ticker": p["t"],
                    "entry": p["p"],
                    "exit": price,
                    "return": price / p["p"] - 1,
                    "date": d,
                }
            )
            cash += price * p["q"]

        pos = new_pos

        if len(pos) < MAX_POSITIONS:
            for t, df in data.items():

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

                if close <= df["MA75"].iloc[i]:
                    continue

                if df["VALUE20"].iloc[i] < MIN_VALUE:
                    continue

                if df["Volume"].iloc[i] < df["VOL20"].iloc[i]:
                    continue

                if not has_pullback(df, i):
                    continue

                if not is_engulf(df, i):
                    continue

                qty = int((cash * RISK_RATIO) // close)
                if qty <= 0:
                    continue

                cash -= qty * close
                pos.append({"t": t, "p": close, "d": d, "q": qty})

        pv = 0
        for p in pos:
            df = data[p["t"]]
            if d in df.index:
                pv += df.loc[d, "Close"] * p["q"]

        eq.append(cash + pv)
        dates_eq.append(d)

    df_eq = pd.DataFrame({"date": dates_eq, "equity": eq})
    df_eq["date"] = pd.to_datetime(df_eq["date"])
    df_eq["peak"] = df_eq["equity"].cummax()
    df_eq["dd"] = df_eq["equity"] / df_eq["peak"] - 1

    df_tr = pd.DataFrame(trades)

    return df_eq, df_tr


def analyze(df_eq, df_tr):
    print("\n=== SUMMARY ===")
    print("Final Equity:", df_eq["equity"].iloc[-1])
    print("Max DD:", df_eq["dd"].min())

    # 年別
    df_eq["year"] = df_eq["date"].dt.year
    yearly = df_eq.groupby("year")["equity"].last().pct_change()

    print("\n=== YEARLY ===")
    print(yearly)

    # 月別
    df_eq["month"] = df_eq["date"].dt.to_period("M")
    monthly = df_eq.groupby("month")["equity"].last().pct_change()

    print("\n=== MONTHLY ===")
    print(monthly.tail(12))

    # 連敗
    if not df_tr.empty:
        df_tr["win"] = df_tr["return"] > 0
        streak = 0
        max_streak = 0

        for w in df_tr["win"]:
            if not w:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        print("\nMax Losing Streak:", max_streak)

    df_tr.to_csv("trades.csv", index=False)
    df_eq.to_csv("equity.csv", index=False)
    print("\nSaved: trades.csv / equity.csv")


# ===== RUN =====

data = {t: load_data(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}

df_eq, df_tr = run()
analyze(df_eq, df_tr)
