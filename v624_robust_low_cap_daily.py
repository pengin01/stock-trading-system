import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from pandas.errors import EmptyDataError

# =====================
# v624 robust low-cap daily paper bot
#
# 2万円運用・100株単位
# v622/v623の頑健性確認済みルールをdaily運用化
#
# 目的:
# ・未来データ収集
# ・candidate / position / trade / equity の蓄積
# ・Discord通知
# ・equityグラフ送信
# =====================

TICKERS = [
    "9432.T",
    "9424.T",
]

HOLD_DAYS_MAP = {
    "9432.T": 5,
    "9424.T": 10,
}

START = "2022-01-01"

INITIAL_CAPITAL = 20_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03

MIN_SCORE = 0.012
MIN_VALUE20 = 50_000_000

PRICE_MIN = 50
PRICE_MAX = 200

SLIPPAGE = 0.001
FEE_PCT = 0.0005

POS_FILE = "positions_v624.csv"
TRADES_FILE = "trades_v624.csv"
EQUITY_FILE = "equity_v624.csv"
CANDIDATES_FILE = "candidates_v624.csv"
CHART_FILE = "equity_v624_chart.png"

POS_COLS = [
    "ticker", "entry_date", "entry_price", "shares", "cost", "score", "hold_days"
]

TRADES_COLS = [
    "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
    "shares", "cost", "proceeds", "pnl", "return", "reason", "score", "hold_days"
]

EQUITY_COLS = [
    "run_date", "signal_date", "cash", "position_value",
    "equity", "position_count"
]

CANDIDATE_COLS = [
    "run_date", "signal_date", "ticker", "close", "score", "ret3", "value20", "hold_days"
]


# =====================
# CSV安全処理
# =====================
def safe_read_csv(path, columns):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    return df[columns]


def safe_write_if_missing(path, columns):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def init_files():
    safe_write_if_missing(POS_FILE, POS_COLS)
    safe_write_if_missing(TRADES_FILE, TRADES_COLS)
    safe_write_if_missing(EQUITY_FILE, EQUITY_COLS)
    safe_write_if_missing(CANDIDATES_FILE, CANDIDATE_COLS)


def load_positions():
    df = safe_read_csv(POS_FILE, POS_COLS)

    if len(df) == 0:
        return []

    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df.to_dict("records")


def save_positions(positions):
    df = pd.DataFrame(positions)

    if len(df) == 0:
        df = pd.DataFrame(columns=POS_COLS)

    df.to_csv(POS_FILE, index=False)


def load_cash():
    trades = safe_read_csv(TRADES_FILE, TRADES_COLS)
    positions = safe_read_csv(POS_FILE, POS_COLS)

    cash = INITIAL_CAPITAL

    if len(trades) > 0:
        trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0)
        cash += trades["pnl"].sum()

    if len(positions) > 0:
        positions["cost"] = pd.to_numeric(positions["cost"], errors="coerce").fillna(0)
        cash -= positions["cost"].sum()

    return cash


# =====================
# Discord
# =====================
def send_discord(message):
    url = os.getenv("DISCORD_WEBHOOK_URL")

    if not url:
        print("DISCORD_WEBHOOK_URL not set")
        print(message)
        return

    try:
        res = requests.post(url, json={"content": message}, timeout=10)
        print("Discord status:", res.status_code)

        if res.status_code >= 300:
            print("Discord response:", res.text)

    except Exception as e:
        print("Discord error:", e)


def send_discord_image(path):
    url = os.getenv("DISCORD_WEBHOOK_URL")

    if not url:
        print("DISCORD_WEBHOOK_URL not set; skip image")
        return

    if not path or not os.path.exists(path):
        print("chart not found; skip image")
        return

    try:
        with open(path, "rb") as f:
            res = requests.post(
                url,
                files={"file": f},
                timeout=20,
            )

        print("Discord image status:", res.status_code)

        if res.status_code >= 300:
            print("Discord image response:", res.text)

    except Exception as e:
        print("Discord image error:", e)


def create_equity_chart():
    df = safe_read_csv(EQUITY_FILE, EQUITY_COLS)

    if len(df) == 0:
        return None

    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["equity"]).reset_index(drop=True)

    if len(df) == 0:
        return None

    plt.figure(figsize=(8, 4))
    plt.plot(df.index + 1, df["equity"])
    plt.title("v624 Equity Curve")
    plt.xlabel("Run")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    plt.close()

    return CHART_FILE


# =====================
# 指標
# =====================
def calc_indicators(df):
    df["ret3"] = df["Close"].pct_change(3)
    df["value"] = df["Close"] * df["Volume"]
    df["value20"] = df["value"].rolling(20).mean()
    return df


def is_hammer(row):
    o = float(row["Open"])
    h = float(row["High"])
    l = float(row["Low"])
    c = float(row["Close"])

    body = abs(c - o)
    rng = h - l

    if rng <= 0:
        return False

    lower = min(o, c) - l
    upper = h - max(o, c)
    body_safe = max(body, c * 0.001)

    return (
        lower >= body_safe * 2
        and upper <= body_safe
        and body <= rng * 0.4
    )


# =====================
# データ取得
# =====================
def download_data():
    data = {}

    for ticker in TICKERS:
        print("Downloading:", ticker)

        try:
            df = yf.download(
                ticker,
                start=START,
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            print("download error:", ticker, e)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 100:
            print("skip:", ticker)
            continue

        df = calc_indicators(df).dropna().copy()

        if len(df) == 0:
            continue

        data[ticker] = df

    return data


# =====================
# メイン処理
# =====================
def run():
    init_files()

    data = download_data()
    positions = load_positions()
    cash = load_cash()

    run_date = datetime.now().strftime("%Y-%m-%d")

    if len(data) == 0:
        print("No data")
        send_discord("⚠️ v624 ROBUST LOW-CAP DAILY\nNo data")
        return

    signal_date = max(df.index[-1] for df in data.values())

    sells = []
    buys = []
    candidates = []

    # =====================
    # SELL判定
    # =====================
    new_positions = []

    for pos in positions:
        ticker = pos["ticker"]

        if ticker not in data:
            new_positions.append(pos)
            continue

        df = data[ticker]

        if signal_date not in df.index:
            new_positions.append(pos)
            continue

        row = df.loc[signal_date]

        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        entry_price = float(pos["entry_price"])
        shares = int(pos["shares"])
        cost = float(pos["cost"])
        entry_date = pd.to_datetime(pos["entry_date"])
        hold_days = int(pos.get("hold_days", HOLD_DAYS_MAP.get(ticker, 10)))

        exit_flag = False
        exit_price = None
        reason = None

        if high >= entry_price * (1 + TP_PCT):
            exit_price = entry_price * (1 + TP_PCT)
            reason = "TP"
            exit_flag = True

        elif low <= entry_price * (1 + SL_PCT):
            exit_price = entry_price * (1 + SL_PCT)
            reason = "SL"
            exit_flag = True

        elif (signal_date - entry_date).days >= hold_days:
            exit_price = close
            reason = "TIME"
            exit_flag = True

        if exit_flag:
            exit_price *= (1 - SLIPPAGE)

            gross = exit_price * shares
            fee = gross * FEE_PCT
            proceeds = gross - fee

            pnl = proceeds - cost
            ret = pnl / cost

            cash += proceeds

            sell = {
                "ticker": ticker,
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": signal_date.strftime("%Y-%m-%d"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "cost": cost,
                "proceeds": proceeds,
                "pnl": pnl,
                "return": ret,
                "reason": reason,
                "score": pos.get("score", ""),
                "hold_days": hold_days,
            }

            sells.append(sell)

        else:
            new_positions.append(pos)

    positions = new_positions

    # =====================
    # BUY候補抽出
    # =====================
    if len(positions) < MAX_POSITIONS:
        for ticker, df in data.items():
            if signal_date not in df.index:
                continue

            row = df.loc[signal_date]
            close = float(row["Close"])

            if not (PRICE_MIN <= close <= PRICE_MAX):
                continue

            if (
                is_hammer(row)
                and float(row["ret3"]) < 0
                and float(row["value20"]) >= MIN_VALUE20
            ):
                score = -float(row["ret3"])

                if score >= MIN_SCORE:
                    candidates.append({
                        "run_date": run_date,
                        "signal_date": signal_date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "close": close,
                        "score": score,
                        "ret3": float(row["ret3"]),
                        "value20": float(row["value20"]),
                        "hold_days": HOLD_DAYS_MAP[ticker],
                    })

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        # =====================
        # BUY実行 paper
        # =====================
        for cand in candidates:
            ticker = cand["ticker"]
            close = float(cand["close"])

            entry_price = close * (1 + SLIPPAGE)
            shares = LOT_SIZE

            gross = entry_price * shares
            fee = gross * FEE_PCT
            total_cost = gross + fee

            if total_cost > INITIAL_CAPITAL:
                continue

            if cash < total_cost:
                continue

            cash -= total_cost

            pos = {
                "ticker": ticker,
                "entry_date": signal_date.strftime("%Y-%m-%d"),
                "entry_price": entry_price,
                "shares": shares,
                "cost": total_cost,
                "score": cand["score"],
                "hold_days": cand["hold_days"],
            }

            positions.append(pos)
            buys.append(pos)
            break

    # =====================
    # 評価額
    # =====================
    position_value = 0

    for pos in positions:
        ticker = pos["ticker"]

        if ticker in data and signal_date in data[ticker].index:
            close = float(data[ticker].loc[signal_date]["Close"])
            position_value += close * int(pos["shares"])

    equity = cash + position_value

    # =====================
    # 保存
    # =====================
    save_positions(positions)

    if sells:
        old = safe_read_csv(TRADES_FILE, TRADES_COLS)
        pd.concat([old, pd.DataFrame(sells)], ignore_index=True).to_csv(
            TRADES_FILE, index=False
        )

    if candidates:
        old = safe_read_csv(CANDIDATES_FILE, CANDIDATE_COLS)
        pd.concat([old, pd.DataFrame(candidates)], ignore_index=True).to_csv(
            CANDIDATES_FILE, index=False
        )

    old_eq = safe_read_csv(EQUITY_FILE, EQUITY_COLS)

    new_eq = pd.DataFrame([{
        "run_date": run_date,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "cash": cash,
        "position_value": position_value,
        "equity": equity,
        "position_count": len(positions),
    }])

    pd.concat([old_eq, new_eq], ignore_index=True).to_csv(EQUITY_FILE, index=False)

    # =====================
    # 表示
    # =====================
    print("\n==============================")
    print("v624 ROBUST LOW-CAP DAILY")
    print("==============================")
    print("run_date   :", run_date)
    print("signal_date:", signal_date.strftime("%Y-%m-%d"))
    print("cash       :", round(cash, 2))
    print("pos_value  :", round(position_value, 2))
    print("equity     :", round(equity, 2))
    print("positions  :", len(positions))

    print("\nSELL:")
    if sells:
        for s in sells:
            print(s)
    else:
        print("(none)")

    print("\nBUY:")
    if buys:
        for b in buys:
            print(b)
    else:
        print("(none)")

    print("\nCANDIDATES:")
    if candidates:
        for c in candidates:
            print(c)
    else:
        print("(none)")

    print("\nPOS:")
    print(positions)

    # =====================
    # Discord通知
    # =====================
    sell_text = "\n".join(
        [f'- {s["ticker"]} {s["reason"]} pnl={round(float(s["pnl"]), 1)} hold={s["hold_days"]}' for s in sells]
    ) if sells else "none"

    buy_text = "\n".join(
        [f'- {b["ticker"]} entry={round(float(b["entry_price"]), 2)} cost={round(float(b["cost"]), 1)} hold={b["hold_days"]}' for b in buys]
    ) if buys else "none"

    candidate_text = "\n".join(
        [
            f'{i + 1}. {c["ticker"]} score={round(float(c["score"]), 4)} close={round(float(c["close"]), 2)} hold={c["hold_days"]}'
            for i, c in enumerate(candidates[:5])
        ]
    ) if candidates else "none"

    pos_text = "\n".join(
        [f'- {p["ticker"]} entry={round(float(p["entry_price"]), 2)} cost={round(float(p["cost"]), 1)} hold={p["hold_days"]}' for p in positions]
    ) if positions else "none"

    msg = f"""📊 v624 ROBUST LOW-CAP DAILY

run_date: {run_date}
signal_date: {signal_date.strftime("%Y-%m-%d")}

💰 equity: {round(equity, 2)}
💵 cash: {round(cash, 2)}
📦 positions: {len(positions)}

SELL:
{sell_text}

BUY:
{buy_text}

🏆 TOP CANDIDATES:
{candidate_text}

POS:
{pos_text}
"""

    send_discord(msg)

    chart = create_equity_chart()
    if chart:
        send_discord_image(chart)

    print("\nSaved:")
    print(POS_FILE)
    print(TRADES_FILE)
    print(EQUITY_FILE)
    print(CANDIDATES_FILE)
    print(CHART_FILE)


if __name__ == "__main__":
    run()
