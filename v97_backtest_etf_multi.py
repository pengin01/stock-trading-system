# v97_backtest_etf_multi.py
# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf
import ta

# =========================
# PARAMETERS（安定版）
# =========================
INITIAL_CAPITAL = 20000

HOLD_DAYS = 6
PULLBACK = 0.007
RISK_RATIO = 0.7
MAX_POSITIONS = 2

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 55
MIN_VALUE = 100_000_000

TICKERS = ["1306.T", "1321.T", "1570.T", "1360.T", "1357.T"]
YEARS = 5


# =========================
# DATA
# =========================
def load_data(ticker):
    df = yf.download(ticker, period=f"{YEARS}y", progress=False)

    if df.empty:
        return df

    # MultiIndex対策
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


data = {t: load_data(t) for t in TICKERS}
data = {k: v for k, v in data.items() if not v.empty}

dates = sorted(set().union(*[df.index for df in data.values()]))

# =========================
# BACKTEST
# =========================
cash = INITIAL_CAPITAL
positions = []
equity_curve = []
trades = []


def bars_passed(df, entry_date, current_date):
    return ((df.index > entry_date) & (df.index <= current_date)).sum()


for d in dates:

    # EXIT
    new_pos = []
    for p in positions:
        df = data[p["ticker"]]

        if d not in df.index:
            new_pos.append(p)
            continue

        hold = bars_passed(df, p["entry_date"], d)

        if hold < HOLD_DAYS:
            new_pos.append(p)
            continue

        price = df.loc[d, "Close"]
        pnl = (price - p["entry_price"]) * p["qty"]

        cash += price * p["qty"]

        trades.append({
            "ticker": p["ticker"],
            "entry": p["entry_date"],
            "exit": d,
            "pnl": pnl,
            "return": price / p["entry_price"] - 1
        })

    positions = new_pos

    # ENTRY
    if len(positions) < MAX_POSITIONS:
        candidates = []

        for t, df in data.items():
            if d not in df.index:
                continue

            i = df.index.get_loc(d)
            if i < MA_DAYS + 2:
                continue

            close = df["Close"].iloc[i]
            prev_close = df["Close"].iloc[i - 1]

            if close < df["MA"].iloc[i]:
                continue
            if close > prev_close * (1 - PULLBACK):
                continue
            if df["RSI"].iloc[i] > RSI_MAX:
                continue
            if df["VALUE20"].iloc[i] < MIN_VALUE:
                continue

            candidates.append((t, df["RSI"].iloc[i]))

        candidates.sort(key=lambda x: x[1])

        for t, _ in candidates:
            if len(positions) >= MAX_POSITIONS:
                break

            price = data[t].loc[d, "Close"]
            qty = int((cash * RISK_RATIO) // price)

            if qty <= 0:
                continue

            cost = price * qty
            if cost > cash:
                continue

            cash -= cost

            positions.append({
                "ticker": t,
                "entry_date": d,
                "entry_price": price,
                "qty": qty
            })

    # EQUITY
    pos_val = 0
    for p in positions:
        df = data[p["ticker"]]
        if d in df.index:
            pos_val += df.loc[d, "Close"] * p["qty"]

    equity_curve.append(cash + pos_val)


# =========================
# RESULT
# =========================
eq = pd.Series(equity_curve, index=dates[:len(equity_curve)])
peak = eq.cummax()
dd = eq / peak - 1

trades_df = pd.DataFrame(trades)

print("\n=== RESULT ===")
print("final_equity:", eq.iloc[-1])
print("total_return:", eq.iloc[-1] / INITIAL_CAPITAL - 1)
print("max_drawdown:", dd.min())
print("trade_count:", len(trades_df))

if not trades_df.empty:
    print("win_rate:", (trades_df["return"] > 0).mean())
    print("avg_return:", trades_df["return"].mean())

eq.to_csv("v97_equity_curve.csv")
trades_df.to_csv("v97_trades.csv", index=False)

print("Saved: v97_equity_curve.csv")
print("Saved: v97_trades.csv")