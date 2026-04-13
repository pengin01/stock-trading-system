import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import yfinance as yf
import ta

# =========================
# PARAMETERS（Dベース + 強化）
# =========================
PERIOD = "10y"
INITIAL_CAPITAL = 300000

LOT_SIZE = 100
MAX_POSITIONS = 2

MA_DAYS = 25
MA_SLOPE_DAYS = 5

RSI_DAYS = 14
RSI_MAX = 50

PULLBACK_PCT = 0.02
MIN_VALUE20 = 500_000_000

TAKE_PROFIT_PCT = 0.09
STOP_LOSS_PCT = 0.02
HOLD_DAYS = 4

BUY_FEE_PCT = 0.0005
SELL_FEE_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

RESULT_DIR = "backtest_results_stock_v7"
UNIVERSE_FILE = "jp_universe.csv"

# 市場フィルタ
MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200

# ★戦略ON/OFF
EQUITY_MA_DAYS = 30


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int


def load_universe():
    df = pd.read_csv(UNIVERSE_FILE)
    return df.iloc[:, 0].tolist()


def normalize(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def load_data(t):
    df = yf.download(t, period=PERIOD, progress=False)
    if df.empty:
        return df

    df = normalize(df)

    close = df["Close"]
    vol = df["Volume"]

    df["MA"] = close.rolling(MA_DAYS).mean()
    df["MA_SLOPE"] = df["MA"] - df["MA"].shift(MA_SLOPE_DAYS)
    df["RSI"] = ta.momentum.RSIIndicator(close, RSI_DAYS).rsi()
    df["VALUE20"] = (close * vol).rolling(20).mean()

    return df.dropna()


def load_market():
    df = yf.download(MARKET_TICKER, period=PERIOD, progress=False)
    df = normalize(df)
    df["MA200"] = df["Close"].rolling(MARKET_MA_DAYS).mean()
    return df.dropna()


# =========================
def run():
    universe = load_universe()
    data = {t: load_data(t) for t in universe}
    data = {k: v for k, v in data.items() if not v.empty}

    market = load_market()

    dates = sorted(set().union(*[df.index for df in data.values()]))

    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity = []

    equity_series = []

    for date in dates:

        # ===== EXIT =====
        new_positions = []
        for p in positions:
            df = data[p.ticker]
            if date not in df.index:
                new_positions.append(p)
                continue

            row = df.loc[date]

            tp = p.entry_price * (1 + TAKE_PROFIT_PCT)
            sl = p.entry_price * (1 - STOP_LOSS_PCT)

            exit_price = None

            if row["Low"] <= sl:
                exit_price = sl
            elif row["High"] >= tp:
                exit_price = tp
            elif (date - p.entry_date).days >= HOLD_DAYS:
                exit_price = row["Close"]

            if exit_price is None:
                new_positions.append(p)
                continue

            exit_price *= 1 - SLIPPAGE_PCT
            cash += exit_price * p.qty

        positions = new_positions

        # ===== EQUITY =====
        pos_val = sum(
            data[p.ticker].loc[date]["Close"] * p.qty
            for p in positions
            if date in data[p.ticker].index
        )

        total_equity = cash + pos_val
        equity_series.append(total_equity)

        # ===== ON/OFF判定 =====
        trading_enabled = True
        if len(equity_series) > EQUITY_MA_DAYS:
            eq_ma = pd.Series(equity_series).rolling(EQUITY_MA_DAYS).mean().iloc[-1]
            if total_equity < eq_ma:
                trading_enabled = False

        # ===== ENTRY =====
        if not trading_enabled:
            equity.append({"date": date, "equity": total_equity})
            continue

        if date not in market.index:
            equity.append({"date": date, "equity": total_equity})
            continue

        if market.loc[date]["Close"] < market.loc[date]["MA200"]:
            equity.append({"date": date, "equity": total_equity})
            continue

        if len(positions) < MAX_POSITIONS:
            for t, df in data.items():
                if any(p.ticker == t for p in positions):
                    continue
                if date not in df.index:
                    continue

                i = df.index.get_loc(date)
                if i < 1:
                    continue

                row = df.iloc[i]
                prev = df.iloc[i - 1]

                if row["Close"] > prev["Close"] * (1 - PULLBACK_PCT):
                    continue
                if row["Close"] < row["MA"]:
                    continue
                if row["MA_SLOPE"] <= 0:
                    continue
                if row["RSI"] > RSI_MAX:
                    continue
                if row["VALUE20"] < MIN_VALUE20:
                    continue

                price = row["Close"] * (1 + SLIPPAGE_PCT)
                qty = int((cash / price) // LOT_SIZE) * LOT_SIZE

                if qty <= 0:
                    continue

                cash -= price * qty
                positions.append(Position(t, date, price, qty))

                if len(positions) >= MAX_POSITIONS:
                    break

        equity.append({"date": date, "equity": total_equity})

    return pd.DataFrame(equity)


# =========================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    equity = run()

    equity.to_csv(f"{RESULT_DIR}/equity_v7.csv", index=False)

    print("Saved equity_v7.csv")


if __name__ == "__main__":
    main()
