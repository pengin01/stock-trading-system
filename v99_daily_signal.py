# v99_daily_signal.py

import datetime
import os

import pandas as pd
import requests
import ta
import yfinance as yf
from pandas.errors import EmptyDataError

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
# HELPERS
# =========================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))
        price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if any(x in price_names for x in level0):
            df.columns = df.columns.get_level_values(0)
        elif any(x in price_names for x in level1):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join([str(a), str(b)]) for a, b in df.columns]

    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# =========================
# LOAD
# =========================
def load_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=False)
    except Exception as e:
        print(f"{ticker}: download error: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        df = normalize_ohlcv(df)
    except Exception as e:
        print(f"{ticker}: normalize error: {e}")
        return pd.DataFrame()

    close = df["Close"]
    volume = df["Volume"]

    df["MA"] = close.rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, RSI_DAYS).rsi()
    df["VALUE20"] = (close * volume).rolling(20).mean()

    return df.dropna(
        subset=["Open", "High", "Low", "Close", "Volume", "MA", "RSI", "VALUE20"]
    )


def load_market() -> pd.DataFrame:
    try:
        df = yf.download(MARKET_TICKER, period="2y", progress=False, auto_adjust=False)
    except Exception as e:
        print(f"{MARKET_TICKER}: download error: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        df = normalize_ohlcv(df)
    except Exception as e:
        print(f"{MARKET_TICKER}: normalize error: {e}")
        return pd.DataFrame()

    df["MA200"] = df["Close"].rolling(MARKET_MA_DAYS).mean()
    return df.dropna(subset=["Close", "MA200"])


# =========================
# POSITION LOAD
# =========================
def load_positions() -> pd.DataFrame:
    # if os.path.exists(POS_FILE):
    #     return pd.read_csv(POS_FILE, parse_dates=["entry_date"])
    # return pd.DataFrame(columns=["ticker", "entry_date", "price", "qty"])
    cols = ["ticker", "entry_date", "price", "qty"]

    if not os.path.exists(POS_FILE):
        return pd.DataFrame(columns=cols)

    # 0バイト対策
    if os.path.getsize(POS_FILE) == 0:
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(POS_FILE, parse_dates=["entry_date"])
    except EmptyDataError:
        return pd.DataFrame(columns=cols)

    # 列不足対策
    for c in cols:
        if c not in df.columns:
            return pd.DataFrame(columns=cols)

    return df[cols]


def save_positions(df: pd.DataFrame) -> None:
    df.to_csv(POS_FILE, index=False)


# =========================
# MAIN
# =========================
def run():
    today = datetime.datetime.now().date()

    data = {t: load_data(t) for t in UNIVERSE}
    market = load_market()

    if market.empty:
        raise RuntimeError("market data is empty")

    positions = load_positions()

    entries = []
    exits = []

    # ===== EXIT =====
    new_positions = []

    for _, p in positions.iterrows():
        ticker = p["ticker"]
        df = data.get(ticker, pd.DataFrame())

        if df.empty:
            new_positions.append(p)
            continue

        row = df.iloc[-1]

        entry_price = float(p["price"])
        tp = entry_price * (1 + TAKE_PROFIT)
        sl = entry_price * (1 - STOP_LOSS)

        exit_flag = False

        if float(row["Low"]) <= sl:
            exits.append(f"{ticker} SL")
            exit_flag = True
        elif float(row["High"]) >= tp:
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
        market_up = float(market.iloc[-1]["Close"]) > float(market.iloc[-1]["MA200"])
        target_universe = BULL_ETF if market_up else BEAR_ETF

        for ticker in target_universe:
            if not positions.empty and ticker in positions["ticker"].values:
                continue

            df = data.get(ticker, pd.DataFrame())
            if df.empty or len(df) < 2:
                continue

            row = df.iloc[-1]
            prev = df.iloc[-2]

            if float(row["Close"]) < float(row["MA"]):
                continue

            if float(row["Close"]) > float(prev["Close"]) * (1 - PULLBACK):
                continue

            if float(row["RSI"]) > RSI_MAX:
                continue

            if float(row["VALUE20"]) < MIN_VALUE:
                continue

            price = float(row["Close"])
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
def send_discord(msg: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return

    r = requests.post(url, json={"content": msg}, timeout=15)
    r.raise_for_status()


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
