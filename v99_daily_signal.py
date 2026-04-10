# v99_daily_signal.py

import pandas as pd
import yfinance as yf
import ta
import datetime
import os
import requests

# =========================
# PARAMETERS
# =========================
INITIAL_CAPITAL = 20000
RISK_RATIO = 0.7

HOLD_DAYS = 4
PULLBACK = 0.007

MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 56

TAKE_PROFIT = 0.014
STOP_LOSS = 0.008

MAX_POSITIONS = 2
MIN_VALUE = 100_000_000

MARKET_TICKER = "^N225"
MARKET_MA_DAYS = 200

BULL_ETF = ["1306.T", "1321.T", "1570.T"]
BEAR_ETF = ["1360.T"]

UNIVERSE = BULL_ETF + BEAR_ETF

POS_FILE = "positions.csv"


# =========================
# LOAD
# =========================
def load_data(ticker):
    df = yf.download(ticker, period="6mo", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def load_market():
    df = yf.download(MARKET_TICKER, period="2y", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA200"] = df["Close"].rolling(MARKET_MA_DAYS).mean()

    return df.dropna()


# =========================
# POSITION LOAD
# =========================
def load_positions():
    if os.path.exists(POS_FILE):
        return pd.read_csv(POS_FILE, parse_dates=["entry_date"])
    else:
        return pd.DataFrame(columns=["ticker", "entry_date", "price", "qty"])


def save_positions(df):
    df.to_csv(POS_FILE, index=False)


# =========================
# MAIN
# =========================
def run():

    today = datetime.datetime.now().date()

    data = {t: load_data(t) for t in UNIVERSE}
    market = load_market()

    positions = load_positions()

    entries = []
    exits = []

    # ===== EXIT =====
    new_positions = []

    for _, p in positions.iterrows():
        ticker = p["ticker"]
        df = data[ticker]

        if df.empty:
            new_positions.append(p)
            continue

        row = df.iloc[-1]

        entry_price = p["price"]
        tp = entry_price * (1 + TAKE_PROFIT)
        sl = entry_price * (1 - STOP_LOSS)

        exit_flag = False

        if row["Low"] <= sl:
            exits.append(f"{ticker} SL")
            exit_flag = True
        elif row["High"] >= tp:
            exits.append(f"{ticker} TP")
            exit_flag = True
        elif (today - p["entry_date"].date()).days >= HOLD_DAYS:
            exits.append(f"{ticker} TIME")
            exit_flag = True

        if not exit_flag:
            new_positions.append(p)

    positions = pd.DataFrame(new_positions)

    # ===== ENTRY =====
    if len(positions) < MAX_POSITIONS:

        market_up = market.iloc[-1]["Close"] > market.iloc[-1]["MA200"]
        target_universe = BULL_ETF if market_up else BEAR_ETF

        for ticker in target_universe:

            if ticker in positions["ticker"].values:
                continue

            df = data[ticker]

            if df.empty:
                continue

            row = df.iloc[-1]
            prev = df.iloc[-2]

            if row["Close"] < row["MA"]:
                continue

            if row["Close"] > prev["Close"] * (1 - PULLBACK):
                continue

            if row["RSI"] > RSI_MAX:
                continue

            if row["VALUE20"] < MIN_VALUE:
                continue

            price = row["Close"]
            qty = int((INITIAL_CAPITAL * RISK_RATIO) // price)

            if qty <= 0:
                continue

            entries.append(ticker)

            new_row = pd.DataFrame(
                [{"ticker": ticker, "entry_date": today, "price": price, "qty": qty}]
            )

            positions = pd.concat([positions, new_row], ignore_index=True)

            if len(positions) >= MAX_POSITIONS:
                break

    save_positions(positions)

    return entries, exits, positions


# =========================
# DISCORD
# =========================
def send_discord(msg):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return

    requests.post(url, json={"content": msg})


# =========================
# EXECUTE
# =========================
if __name__ == "__main__":
    entries, exits, positions = run()

    msg = "📊 Daily Signal v99\n\n"

    msg += "🟢 ENTRY\n"
    msg += "\n".join(entries) if entries else "none"
    msg += "\n\n"

    msg += "🔴 EXIT\n"
    msg += "\n".join(exits) if exits else "none"
    msg += "\n\n"

    msg += "📦 POSITIONS\n"
    msg += positions.to_string(index=False) if not positions.empty else "empty"

    print(msg)
    send_discord(msg)
