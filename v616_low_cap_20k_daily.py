import os
from datetime import datetime

import pandas as pd
import yfinance as yf

# =====================
# v616 low-cap 20k daily
# 2万円運用・100株単位・日次paper運用
# =====================

TICKERS = [
    "9432.T",  # NTT
    "4564.T",
    "8918.T",
    "5856.T",
    "9973.T",
    "2370.T",
    "3664.T",
]

START = "2022-01-01"

INITIAL_CAPITAL = 20_000
LOT_SIZE = 100
MAX_POSITIONS = 1

TP_PCT = 0.06
SL_PCT = -0.03
HOLD_DAYS = 10

MIN_SCORE = 0.012
MIN_VALUE20 = 50_000_000

PRICE_MIN = 50
PRICE_MAX = 200

SLIPPAGE = 0.001
FEE_PCT = 0.0005

POS_FILE = "positions_v616_20k.csv"
TRADES_FILE = "trades_v616_20k.csv"
EQUITY_FILE = "equity_v616_20k.csv"
CANDIDATES_FILE = "candidates_v616_20k.csv"


# =====================
# CSV初期化
# =====================
def init_files():
    if not os.path.exists(POS_FILE):
        pd.DataFrame(
            columns=["ticker", "entry_date", "entry_price", "shares", "cost", "score"]
        ).to_csv(POS_FILE, index=False)

    if not os.path.exists(TRADES_FILE):
        pd.DataFrame(
            columns=[
                "ticker",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "shares",
                "cost",
                "proceeds",
                "pnl",
                "return",
                "reason",
                "score",
            ]
        ).to_csv(TRADES_FILE, index=False)

    if not os.path.exists(EQUITY_FILE):
        pd.DataFrame(
            columns=[
                "run_date",
                "signal_date",
                "cash",
                "position_value",
                "equity",
                "position_count",
            ]
        ).to_csv(EQUITY_FILE, index=False)

    if not os.path.exists(CANDIDATES_FILE):
        pd.DataFrame(
            columns=[
                "run_date",
                "signal_date",
                "ticker",
                "close",
                "score",
                "ret3",
                "value20",
            ]
        ).to_csv(CANDIDATES_FILE, index=False)


def load_positions():
    df = pd.read_csv(POS_FILE)
    if len(df) == 0:
        return []
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df.to_dict("records")


def save_positions(positions):
    pd.DataFrame(positions).to_csv(POS_FILE, index=False)


def load_cash():
    trades = pd.read_csv(TRADES_FILE)
    positions = pd.read_csv(POS_FILE)

    cash = INITIAL_CAPITAL

    if len(trades) > 0:
        cash += trades["pnl"].sum()

    if len(positions) > 0:
        cash -= positions["cost"].sum()

    return cash


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

    return lower >= body_safe * 2 and upper <= body_safe and body <= rng * 0.4


# =====================
# データ取得
# =====================
def download_data():
    data = {}

    for ticker in TICKERS:
        print("Downloading:", ticker)

        df = yf.download(
            ticker,
            start=START,
            auto_adjust=False,
            progress=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 100:
            print("skip:", ticker)
            continue

        df = calc_indicators(df).dropna().copy()
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

        elif (signal_date - entry_date).days >= HOLD_DAYS:
            exit_price = close
            reason = "TIME"
            exit_flag = True

        if exit_flag:
            exit_price *= 1 - SLIPPAGE

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
                    candidates.append(
                        {
                            "run_date": run_date,
                            "signal_date": signal_date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "close": close,
                            "score": score,
                            "ret3": float(row["ret3"]),
                            "value20": float(row["value20"]),
                        }
                    )

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
        old = pd.read_csv(TRADES_FILE)
        pd.concat([old, pd.DataFrame(sells)], ignore_index=True).to_csv(
            TRADES_FILE, index=False
        )

    if candidates:
        old = pd.read_csv(CANDIDATES_FILE)
        pd.concat([old, pd.DataFrame(candidates)], ignore_index=True).to_csv(
            CANDIDATES_FILE, index=False
        )

    old_eq = pd.read_csv(EQUITY_FILE)
    new_eq = pd.DataFrame(
        [
            {
                "run_date": run_date,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "cash": cash,
                "position_value": position_value,
                "equity": equity,
                "position_count": len(positions),
            }
        ]
    )

    pd.concat([old_eq, new_eq], ignore_index=True).to_csv(EQUITY_FILE, index=False)

    # =====================
    # 表示
    # =====================
    print("\n==============================")
    print("v616 LOW-CAP 20K DAILY")
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

    print("\nSaved:")
    print(POS_FILE)
    print(TRADES_FILE)
    print(EQUITY_FILE)
    print(CANDIDATES_FILE)


if __name__ == "__main__":
    run()
