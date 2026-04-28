# v531_daily_paper.py
# -*- coding: utf-8 -*-

import os

import pandas as pd
import requests
import yfinance as yf

# ===== 設定 =====
INITIAL_CAPITAL = 20000

MAX_POSITIONS = 1
RISK_RATIO = 1.0

HOLD_DAYS = 11
PULLBACK_PCT = 0.032
EXIT_MA_BUFFER = 0.98
SL_PCT = 0.07

MA_SHORT = 25
MA_LONG = 75
MIN_VALUE = 100_000_000
YEARS = 1

EXCLUDE_TICKERS = ["6758.T", "4568.T", "4063.T", "4519.T"]

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

POS_FILE = "positions.csv"
TRADE_FILE = "trades.csv"
CANDIDATE_FILE = "candidates.csv"

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord(message):
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL is not set")
        return

    try:
        res = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
        print("Discord status:", res.status_code)
        print(res.text)
    except Exception as e:
        print("Discord error:", e)


def load_data(ticker):
    df = yf.download(ticker, period=f"{YEARS}y", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def is_engulf(df):
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    return (
        prev["Close"] < prev["Open"]
        and cur["Close"] > cur["Open"]
        and cur["Open"] <= prev["Close"]
        and cur["Close"] >= prev["Open"]
    )


def has_pullback(df):
    if len(df) < 10:
        return False

    recent_high = df["Close"].iloc[-10:].max()
    current_close = df["Close"].iloc[-1]

    return current_close <= recent_high * (1 - PULLBACK_PCT)


def load_positions():
    if not os.path.exists(POS_FILE):
        return []

    try:
        df = pd.read_csv(POS_FILE, parse_dates=["entry_date"])
        if df.empty:
            return []
        return df.to_dict("records")
    except Exception as e:
        print("load_positions error:", e)
        return []


def save_positions(positions):
    columns = ["ticker", "entry_price", "entry_date"]

    if not positions:
        pd.DataFrame(columns=columns).to_csv(POS_FILE, index=False)
        return

    pd.DataFrame(positions)[columns].to_csv(POS_FILE, index=False)


def log_trade(trade):
    columns = [
        "ticker",
        "entry_price",
        "exit_price",
        "entry_date",
        "exit_date",
        "qty",
        "pnl",
        "return",
        "reason",
    ]

    if os.path.exists(TRADE_FILE):
        try:
            df = pd.read_csv(TRADE_FILE)
        except Exception:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
    df.to_csv(TRADE_FILE, index=False)


def save_candidates(candidates):
    columns = [
        "date",
        "ticker",
        "close",
        "ma75",
        "volume",
        "vol20",
        "value20",
        "pullback_rate",
        "signal",
    ]

    if not candidates:
        pd.DataFrame(columns=columns).to_csv(CANDIDATE_FILE, index=False)
        return

    pd.DataFrame(candidates)[columns].to_csv(CANDIDATE_FILE, index=False)


def format_candidates(candidates):
    if not candidates:
        return "(none)"

    lines = []

    for c in candidates:
        lines.append(
            f'{c["ticker"]} '
            f'close={c["close"]:.1f} '
            f'pullback={c["pullback_rate"]:.2%} '
            f'value20={c["value20"] / 100_000_000:.1f}億'
        )

    return "\n".join(lines)


def format_positions(positions):
    if not positions:
        return "(none)"

    lines = []

    for p in positions:
        lines.append(
            f'{p["ticker"]} '
            f'entry={float(p["entry_price"]):.1f} '
            f'date={pd.to_datetime(p["entry_date"]).date()}'
        )

    return "\n".join(lines)


def run():
    today = pd.Timestamp.today().normalize()

    print("\n=== RUN INFO ===")
    print("today:", today.date())

    data = {ticker: load_data(ticker) for ticker in TICKERS}
    data = {ticker: df for ticker, df in data.items() if not df.empty}

    positions = load_positions()

    exit_messages = []
    buy_messages = []

    # ===== EXIT =====
    new_positions = []

    for p in positions:
        ticker = p["ticker"]
        df = data.get(ticker)

        if df is None or df.empty:
            new_positions.append(p)
            continue

        latest_date = df.index[-1]
        latest = df.iloc[-1]

        price = float(latest["Close"])
        ma25 = float(latest["MA25"])

        entry_price = float(p["entry_price"])
        entry_date = pd.to_datetime(p["entry_date"])

        # 営業日ベースの保有日数
        try:
            entry_i = df.index.get_loc(entry_date)
            current_i = df.index.get_loc(latest_date)
            hold = current_i - entry_i
        except Exception:
            hold = (latest_date - entry_date).days

        exit_reason = None

        if price <= entry_price * (1 - SL_PCT):
            exit_reason = "SL"

        elif price >= ma25 * EXIT_MA_BUFFER and hold < HOLD_DAYS:
            new_positions.append(p)
            continue

        else:
            exit_reason = "MA_OR_HOLD"

        qty = int(p.get("qty", 1))
        pnl = (price - entry_price) * qty
        ret = price / entry_price - 1

        trade = {
            "ticker": ticker,
            "entry_price": entry_price,
            "exit_price": price,
            "entry_date": entry_date.date(),
            "exit_date": latest_date.date(),
            "qty": qty,
            "pnl": pnl,
            "return": ret,
            "reason": exit_reason,
        }

        log_trade(trade)

        exit_messages.append(
            f"SELL {ticker} price={price:.1f} " f"ret={ret:.2%} reason={exit_reason}"
        )

    positions = new_positions

    # ===== ENTRY候補一覧 =====
    candidates = []

    if len(positions) < MAX_POSITIONS:
        for ticker, df in data.items():
            if ticker in EXCLUDE_TICKERS:
                continue

            if any(p["ticker"] == ticker for p in positions):
                continue

            if len(df) < MA_LONG:
                continue

            latest_date = df.index[-1]
            latest = df.iloc[-1]

            close = float(latest["Close"])
            ma75 = float(latest["MA75"])
            volume = float(latest["Volume"])
            vol20 = float(latest["VOL20"])
            value20 = float(latest["VALUE20"])

            if close <= ma75:
                continue

            if value20 < MIN_VALUE:
                continue

            if volume < vol20:
                continue

            if not has_pullback(df):
                continue

            if not is_engulf(df):
                continue

            recent_high = df["Close"].iloc[-10:].max()
            pullback_rate = close / recent_high - 1

            candidates.append(
                {
                    "date": latest_date.date(),
                    "ticker": ticker,
                    "close": close,
                    "ma75": ma75,
                    "volume": volume,
                    "vol20": vol20,
                    "value20": value20,
                    "pullback_rate": pullback_rate,
                    "signal": "engulfing",
                }
            )

    # ===== ENTRY実行 =====
    if len(positions) < MAX_POSITIONS and candidates:
        c = candidates[0]

        entry_price = float(c["close"])
        qty = int((INITIAL_CAPITAL * RISK_RATIO) // entry_price)

        if qty > 0:
            position = {
                "ticker": c["ticker"],
                "entry_price": entry_price,
                "entry_date": c["date"],
                "qty": qty,
            }

            positions.append(position)

            buy_messages.append(f'BUY {c["ticker"]} price={entry_price:.1f} qty={qty}')

    save_candidates(candidates)
    save_positions(positions)

    print("\n=== EXIT ===")
    print("\n".join(exit_messages) if exit_messages else "(none)")

    print("\n=== BUY ===")
    print("\n".join(buy_messages) if buy_messages else "(none)")

    print("\n=== CANDIDATES ===")
    if candidates:
        print(pd.DataFrame(candidates))
    else:
        print("(none)")

    print("\n=== POSITIONS ===")
    print(positions if positions else "(none)")

    exit_text = "\n".join(exit_messages) if exit_messages else "(none)"
    buy_text = "\n".join(buy_messages) if buy_messages else "(none)"
    pos_text = format_positions(positions)
    cand_text = format_candidates(candidates)

    message = f"""📊 Daily Paper Trade v531

日付: {today.date()}

【SELL】
{exit_text}

【BUY】
{buy_text}

【保有】
{pos_text}

【本日の候補】
{cand_text}
"""

    if len(message) > 1900:
        message = message[:1900] + "\n...(truncated)"

    send_discord(message)


if __name__ == "__main__":
    run()
