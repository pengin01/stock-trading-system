# run_backtest_v99_adjusted.py

import pandas as pd
import yfinance as yf
import ta

# =========================
# PARAMETERS
# =========================
PERIOD = "5y"

INITIAL_CAPITAL = 20000
RISK_RATIO = 0.7

HOLD_DAYS = 4
PULLBACK = 0.006  # ← 緩和

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 58  # ← 緩和

TAKE_PROFIT = 0.014
STOP_LOSS = 0.008

MAX_POSITIONS = 2
MIN_VALUE = 100_000_000

MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200

# ETF拡張
BULL_ETF = ["1306.T", "1321.T", "1570.T"]
BEAR_ETF = ["1360.T"]
UNIVERSE = BULL_ETF + BEAR_ETF


# =========================
# NORMALIZE
# =========================
def normalize(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    return df.dropna()


# =========================
# LOAD
# =========================
def load_data(ticker):
    df = yf.download(ticker, period=PERIOD, progress=False)

    if df.empty:
        return df

    df = normalize(df)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def load_market():
    df = yf.download(MARKET_TICKER, period=PERIOD, progress=False)
    df = normalize(df)

    df["MA200"] = df["Close"].rolling(MARKET_MA_DAYS).mean()
    return df.dropna()


# =========================
# BACKTEST
# =========================
def run():
    data = {t: load_data(t) for t in UNIVERSE}
    market = load_market()

    dates = sorted(set().union(*[df.index for df in data.values()]))

    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity = []

    for date in dates:

        # ===== EXIT =====
        new_positions = []
        for p in positions:
            df = data[p["ticker"]]

            if date not in df.index:
                new_positions.append(p)
                continue

            row = df.loc[date]

            entry = p["price"]
            tp = entry * (1 + TAKE_PROFIT)
            sl = entry * (1 - STOP_LOSS)

            high = row["High"]
            low = row["Low"]
            close = row["Close"]

            exit_price = None
            reason = None

            if low <= sl:
                exit_price = sl
                reason = "stop_loss"
            elif high >= tp:
                exit_price = tp
                reason = "take_profit"
            elif (date - p["date"]).days >= HOLD_DAYS:
                exit_price = close
                reason = "time_exit"

            if exit_price:
                pnl = (exit_price - entry) * p["qty"]
                cash += exit_price * p["qty"]

                trades.append(
                    {
                        "ticker": p["ticker"],
                        "entry_date": p["date"],
                        "exit_date": date,
                        "return": exit_price / entry - 1,
                        "pnl": pnl,
                        "reason": reason,
                    }
                )
            else:
                new_positions.append(p)

        positions = new_positions

        # ===== ENTRY =====
        if len(positions) < MAX_POSITIONS:

            if date not in market.index:
                continue

            # ★ 少し緩めた市場フィルタ
            market_up = market.loc[date, "Close"] > market.loc[date, "MA200"] * 0.995

            target_universe = BULL_ETF if market_up else BEAR_ETF

            for ticker in target_universe:

                df = data[ticker]

                if ticker in [p["ticker"] for p in positions]:
                    continue

                if date not in df.index:
                    continue

                row = df.loc[date]
                prev = df.shift(1).loc[date]

                if row["Close"] < row["MA"]:
                    continue

                if row["Close"] > prev["Close"] * (1 - PULLBACK):
                    continue

                if row["RSI"] > RSI_MAX:
                    continue

                if row["VALUE20"] < MIN_VALUE:
                    continue

                price = row["Close"]
                qty = int((cash * RISK_RATIO) // price)

                if qty <= 0:
                    continue

                cash -= price * qty

                positions.append(
                    {"ticker": ticker, "date": date, "price": price, "qty": qty}
                )

                if len(positions) >= MAX_POSITIONS:
                    break

        # ===== EQUITY =====
        total = cash
        for p in positions:
            df = data[p["ticker"]]
            if date in df.index:
                total += df.loc[date]["Close"] * p["qty"]

        equity.append({"date": date, "equity": total})

    return pd.DataFrame(trades), pd.DataFrame(equity)


# =========================
# EXECUTE
# =========================
if __name__ == "__main__":
    trades, equity = run()

    equity["peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["peak"] - 1

    print("final:", equity.iloc[-1]["equity"])
    print("return:", equity.iloc[-1]["equity"] / INITIAL_CAPITAL - 1)
    print("max_dd:", equity["drawdown"].min())
    print("trade_count:", len(trades))

    trades.to_csv("trades_v99_adjusted.csv", index=False)
    equity.to_csv("equity_v99_adjusted.csv", index=False)
