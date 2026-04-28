import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# =====================
# パラメータ
# =====================
TICKERS = ["5020.T", "8306.T", "8604.T", "7186.T"]

START = "2022-01-01"
END = "2026-01-01"

INITIAL_CAPITAL = 100_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03

# ★ ここを比較
HOLD_LIST = [5, 7, 10]

MIN_SCORE = 0.015
MIN_VALUE20 = 100_000_000

PRICE_MIN = 300
PRICE_MAX = 3000

SLIPPAGE = 0.001
FEE_PCT = 0.0005


# =====================
# 指標
# =====================
def calc_indicators(df):
    df["ret3"] = df["Close"].pct_change(3)
    df["value"] = df["Close"] * df["Volume"]
    df["value20"] = df["value"].rolling(20).mean()
    return df


def is_hammer(row):
    o, h, l, c = (
        float(row["Open"]),
        float(row["High"]),
        float(row["Low"]),
        float(row["Close"]),
    )
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    body_safe = max(body, c * 0.001)
    return lower >= body_safe * 2 and upper <= body_safe and body <= rng * 0.4


# =====================
# データ
# =====================
def load_data():
    data = {}
    for t in TICKERS:
        print("Downloading:", t)
        df = yf.download(t, start=START, end=END)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = calc_indicators(df).dropna()
        data[t] = df

    return data


# =====================
# 実行
# =====================
def run(HOLD_DAYS):
    data = load_data()

    dates = sorted(set(d for df in data.values() for d in df.index))

    cash = INITIAL_CAPITAL
    positions = []
    trades = []

    for date in dates:

        # ===== EXIT =====
        new_pos = []
        for p in positions:
            df = data[p["ticker"]]

            if date not in df.index:
                new_pos.append(p)
                continue

            row = df.loc[date]
            high, low = row["High"], row["Low"]

            exit_flag = False

            # TP
            if high >= p["entry_price"] * (1 + TP_PCT):
                exit_price = p["entry_price"] * (1 + TP_PCT)
                reason = "TP"
                exit_flag = True

            # SL
            elif low <= p["entry_price"] * (1 + SL_PCT):
                exit_price = p["entry_price"] * (1 + SL_PCT)
                reason = "SL"
                exit_flag = True

            # TIME
            elif (date - p["entry_date"]).days >= HOLD_DAYS:
                exit_price = row["Close"]
                reason = "TIME"
                exit_flag = True

            if exit_flag:
                exit_price *= 1 - SLIPPAGE
                gross = exit_price * LOT_SIZE
                fee = gross * FEE_PCT
                pnl = gross - fee - p["cost"]

                cash += gross - fee

                trades.append({"pnl": pnl, "reason": reason})
            else:
                new_pos.append(p)

        positions = new_pos

        # ===== ENTRY =====
        if len(positions) < MAX_POSITIONS:
            candidates = []

            for t, df in data.items():
                if date not in df.index:
                    continue

                row = df.loc[date]
                price = row["Close"]

                if not (PRICE_MIN <= price <= PRICE_MAX):
                    continue

                if is_hammer(row) and row["ret3"] < 0 and row["value20"] >= MIN_VALUE20:
                    score = -row["ret3"]

                    if score >= MIN_SCORE:
                        candidates.append((score, t, row))

            candidates.sort(reverse=True)

            for score, t, row in candidates:
                price = row["Close"] * (1 + SLIPPAGE)
                cost = price * LOT_SIZE
                fee = cost * FEE_PCT
                total = cost + fee

                if cash >= total:
                    cash -= total
                    positions.append(
                        {
                            "ticker": t,
                            "entry_date": date,
                            "entry_price": price,
                            "cost": total,
                        }
                    )
                    break

    return pd.DataFrame(trades)


# =====================
# 実行
# =====================
for hold in HOLD_LIST:
    trades = run(hold)

    print(f"\n=== HOLD {hold} ===")
    print("trades:", len(trades))
    print("win_rate:", (trades["pnl"] > 0).mean())
    print("total_pnl:", trades["pnl"].sum())
    print(trades.groupby("reason")["pnl"].sum())
