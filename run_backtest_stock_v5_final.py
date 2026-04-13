import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import yfinance as yf
import ta


# =========================
# PARAMETERS（Cベース + 市場フィルタ）
# =========================
PERIOD = "3y"
INITIAL_CAPITAL = 300000

LOT_SIZE = 100
MAX_POSITIONS = 2
RISK_RATIO = 1.0

MA_DAYS = 25
MA_SLOPE_DAYS = 5

RSI_DAYS = 14
RSI_MAX = 50

PULLBACK_PCT = 0.02

MIN_VALUE20 = 500_000_000

TAKE_PROFIT_PCT = 0.09
STOP_LOSS_PCT = 0.025
HOLD_DAYS = 4

BUY_FEE_PCT = 0.0005
SELL_FEE_PCT = 0.0005
SLIPPAGE_PCT = 0.0005

RESULT_DIR = "backtest_results_stock_v6"
UNIVERSE_FILE = "jp_universe.csv"

# ★追加
USE_MARKET_FILTER = True
MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200


# =========================
@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int


# =========================
def ensure_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_universe():
    df = pd.read_csv(UNIVERSE_FILE)
    col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    return sorted(set(df[col].astype(str).str.strip()))


def normalize(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]

    df.index = pd.to_datetime(df.index).tz_localize(None)
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


# =========================
# ★市場データ
# =========================
def load_market():
    df = yf.download(MARKET_TICKER, period=PERIOD, progress=False)
    df = normalize(df)
    df["MA200"] = df["Close"].rolling(MARKET_MA_DAYS).mean()
    return df.dropna()


# =========================
def run() -> Tuple[pd.DataFrame, pd.DataFrame]:

    universe = load_universe()
    data = {t: load_data(t) for t in universe}
    data = {k: v for k, v in data.items() if not v.empty}

    market = load_market()

    dates = sorted(set().union(*[df.index for df in data.values()]))

    cash = INITIAL_CAPITAL
    positions: List[Position] = []
    trades = []
    equity = []

    for date in dates:

        # ===== EXIT =====
        new_positions = []
        for p in positions:
            df = data[p.ticker]
            if date not in df.index:
                new_positions.append(p)
                continue

            row = df.loc[date]

            high = row["High"]
            low = row["Low"]
            close = row["Close"]

            tp = p.entry_price * (1 + TAKE_PROFIT_PCT)
            sl = p.entry_price * (1 - STOP_LOSS_PCT)

            days = (date - p.entry_date).days

            exit_price = None
            reason = None

            if low <= sl:
                exit_price = sl
                reason = "stop_loss"
            elif high >= tp:
                exit_price = tp
                reason = "take_profit"
            elif days >= HOLD_DAYS:
                exit_price = close
                reason = "time_exit"

            if exit_price is None:
                new_positions.append(p)
                continue

            exit_price *= 1 - SLIPPAGE_PCT
            proceeds = exit_price * p.qty
            fee = proceeds * SELL_FEE_PCT

            cash += proceeds - fee

            trades.append(
                {
                    "ticker": p.ticker,
                    "entry_date": p.entry_date,
                    "exit_date": date,
                    "return": exit_price / p.entry_price - 1,
                    "reason": reason,
                }
            )

        positions = new_positions

        # ===== ENTRY =====
        if len(positions) < MAX_POSITIONS:

            # ★市場フィルタ
            if USE_MARKET_FILTER:
                if date not in market.index:
                    continue

                m = market.loc[date]

                if m["Close"] < m["MA200"]:
                    continue

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

                close = row["Close"]
                prev_close = prev["Close"]

                if close > prev_close * (1 - PULLBACK_PCT):
                    continue

                if close < row["MA"]:
                    continue

                if row["MA_SLOPE"] <= 0:
                    continue

                if row["RSI"] > RSI_MAX:
                    continue

                if row["VALUE20"] < MIN_VALUE20:
                    continue

                price = close * (1 + SLIPPAGE_PCT)

                qty = int((cash / price) // LOT_SIZE) * LOT_SIZE
                if qty <= 0:
                    continue

                cost = price * qty
                fee = cost * BUY_FEE_PCT

                if cost + fee > cash:
                    continue

                cash -= cost + fee
                positions.append(Position(t, date, price, qty))

                if len(positions) >= MAX_POSITIONS:
                    break

        # ===== EQUITY =====
        pos_val = sum(
            data[p.ticker].loc[date]["Close"] * p.qty
            for p in positions
            if date in data[p.ticker].index
        )

        equity.append({"date": date, "equity": cash + pos_val})

    return pd.DataFrame(trades), pd.DataFrame(equity)


# =========================
def main():
    ensure_dir()

    trades, equity = run()

    final = equity["equity"].iloc[-1]
    ret = final / INITIAL_CAPITAL - 1

    peak = equity["equity"].cummax()
    dd = equity["equity"] / peak - 1

    print("\n=== RESULT ===")
    print("final:", final)
    print("return:", ret)
    print("max_dd:", dd.min())
    print("trade_count:", len(trades))

    trades.to_csv(f"{RESULT_DIR}/trades_v6.csv", index=False)
    equity.to_csv(f"{RESULT_DIR}/equity_v6.csv", index=False)


if __name__ == "__main__":
    main()
