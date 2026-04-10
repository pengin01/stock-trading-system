# run_backtest_v1_stock.py

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
PULLBACK = 0.01       # ETFより深め

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 60         # 少し緩める

TAKE_PROFIT = 0.02   # 大きく取る
STOP_LOSS = 0.01

MAX_POSITIONS = 3

# 流動性（重要）
MIN_VALUE = 1_000_000_000  # 10億

# 市場フィルタ
MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200

# ★ まずは大型株だけ（安全）
UNIVERSE = [
    "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
    "8035.T", "9432.T", "4063.T", "6098.T", "4519.T"
]

# =========================
# NORMALIZE
# =========================
def normalize(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open","High","Low","Close","Volume"]:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:,0]

    df = df.dropna()
    return df

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

                trades.append({
                    "ticker": p["ticker"],
                    "entry_date": p["date"],
                    "exit_date": date,
                    "return": exit_price/entry - 1,
                    "pnl": pnl,
                    "reason": reason
                })
            else:
                new_positions.append(p)

        positions = new_positions

        # ===== ENTRY =====
        if len(positions) < MAX_POSITIONS:

            if date not in market.index:
                continue

            # 市場フィルタ
            if market.loc[date,"Close"] <= market.loc[date,"MA200"]:
                continue

            for ticker, df in data.items():

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

                positions.append({
                    "ticker": ticker,
                    "date": date,
                    "price": price,
                    "qty": qty
                })

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
    print("return:", equity.iloc[-1]["equity"]/INITIAL_CAPITAL - 1)
    print("max_dd:", equity["drawdown"].min())

    print("trade_count:", len(trades))
    print("win_rate:", (trades["return"]>0).mean())
    print("avg_return:", trades["return"].mean())

    trades.to_csv("trades_stock_v1.csv", index=False)
    equity.to_csv("equity_stock_v1.csv", index=False)