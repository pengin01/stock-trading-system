# v500_grid_backtest_stock.py
# -*- coding: utf-8 -*-

import itertools
import pandas as pd
import yfinance as yf
import ta

# =========================
# FIXED
# =========================
INITIAL_CAPITAL = 20000

MA_SHORT = 25
MA_LONG = 75
BREAKOUT = 40

MA_SLOPE_PCT = 0.02
BREAKOUT_BUFFER = 1.01
MIN_VALUE = 100_000_000

RISK_RATIO = 0.5
YEARS = 3

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

# =========================
# GRID
# =========================
HOLD_DAYS_LIST = [5, 7, 10]
MAX_POS_LIST = [1, 2]
VOL_MULT_LIST = [1.5, 2.0, 2.5]

OUT = "v500_grid.csv"


# =========================
# DATA
# =========================
def load_data(t):
    df = yf.download(t, period=f"{YEARS}y", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Close"] = pd.to_numeric(df["Close"])
    df["Volume"] = pd.to_numeric(df["Volume"])

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


data_raw = {t: load_data(t) for t in TICKERS}
data_raw = {k: v for k, v in data_raw.items() if not v.empty}


def bars_passed(df, entry, now):
    return ((df.index > entry) & (df.index <= now)).sum()


def run(hold_days, max_pos, vol_mult):
    cash = INITIAL_CAPITAL
    pos = []
    eq = []
    trades = []

    data = {}
    for t, df in data_raw.items():
        x = df.copy()
        x["HH"] = x["Close"].rolling(BREAKOUT).max()
        data[t] = x.dropna()

    dates = sorted(set().union(*[df.index for df in data.values()]))

    for d in dates:

        # EXIT
        new = []
        for p in pos:
            df = data[p["t"]]
            if d not in df.index:
                new.append(p)
                continue

            price = df.loc[d, "Close"]
            ma25 = df.loc[d, "MA25"]
            hold = bars_passed(df, p["d"], d)

            if price >= ma25 and hold < hold_days:
                new.append(p)
                continue

            cash += price * p["q"]
            trades.append(price / p["p"] - 1)

        pos = new

        # ENTRY
        if len(pos) < max_pos:
            for t, df in data.items():
                if any(p["t"] == t for p in pos):
                    continue
                if d not in df.index:
                    continue

                i = df.index.get_loc(d)
                if i < BREAKOUT + 5:
                    continue

                close = df["Close"].iloc[i]
                ma25 = df["MA25"].iloc[i]
                ma75 = df["MA75"].iloc[i]
                hh = df["HH"].iloc[i - 1]

                vol = df["Volume"].iloc[i]
                vol20 = df["VOL20"].iloc[i]
                val = df["VALUE20"].iloc[i]

                ma_now = df["MA25"].iloc[i]
                ma_past = df["MA25"].iloc[i - 5]

                if not (close > ma25 > ma75):
                    continue
                if close <= hh * BREAKOUT_BUFFER:
                    continue
                if vol <= vol20 * vol_mult:
                    continue
                if (ma_now / ma_past - 1) < MA_SLOPE_PCT:
                    continue
                if val < MIN_VALUE:
                    continue

                price = close
                qty = int((cash * RISK_RATIO) // price)
                if qty <= 0:
                    continue

                cash -= price * qty
                pos.append({"t": t, "p": price, "d": d, "q": qty})

        # EQUITY
        pv = 0
        for p in pos:
            df = data[p["t"]]
            if d in df.index:
                pv += df.loc[d, "Close"] * p["q"]

        eq.append(cash + pv)

    eq = pd.Series(eq)
    peak = eq.cummax()
    dd = (eq / peak - 1).min()

    if trades:
        win = sum(r > 0 for r in trades) / len(trades)
        avg = sum(trades) / len(trades)
    else:
        win = 0
        avg = 0

    return {
        "hold": hold_days,
        "max_pos": max_pos,
        "vol_mult": vol_mult,
        "ret": eq.iloc[-1] / INITIAL_CAPITAL - 1,
        "dd": dd,
        "trades": len(trades),
        "win": win,
        "avg": avg,
    }


# =========================
# MAIN
# =========================
rows = []
for hold, pos, vol in itertools.product(HOLD_DAYS_LIST, MAX_POS_LIST, VOL_MULT_LIST):
    print(hold, pos, vol)
    rows.append(run(hold, pos, vol))

df = pd.DataFrame(rows)
df = df.sort_values(["ret", "dd"], ascending=[False, False])

print("\n=== TOP ===")
print(df.head(10))

df.to_csv(OUT, index=False)
print("Saved:", OUT)
