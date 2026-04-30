import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from pandas.errors import EmptyDataError

TRADES_FILE = "trades_v616_20k.csv"
EQUITY_FILE = "equity_v616_20k.csv"

TRADES_COLS = [
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

EQUITY_COLS = [
    "run_date",
    "signal_date",
    "cash",
    "position_value",
    "equity",
    "position_count",
]


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


def run():
    trades = safe_read_csv(TRADES_FILE, TRADES_COLS)
    equity = safe_read_csv(EQUITY_FILE, EQUITY_COLS)

    end = datetime.now()
    start = end - timedelta(days=7)

    if len(trades) == 0:
        last_equity = None

        if len(equity) > 0:
            equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
            last_equity = equity["equity"].dropna().iloc[-1]

        msg = f"""📊 v618 WEEKLY REPORT

期間: {start.date()} ～ {end.date()}

トレードはまだありません。

現在資産: {round(last_equity, 2) if last_equity is not None else "unknown"}
"""
        send_discord(msg)
        return

    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0)

    week = trades[
        (trades["exit_date"] >= pd.Timestamp(start))
        & (trades["exit_date"] <= pd.Timestamp(end))
    ]

    if len(week) == 0:
        last_equity = None
        max_dd = None

        if len(equity) > 0:
            equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
            eq = equity["equity"].dropna()

            if len(eq) > 0:
                last_equity = eq.iloc[-1]
                peak = eq.cummax()
                max_dd = (eq / peak - 1).min()

        msg = f"""📊 v618 WEEKLY REPORT

期間: {start.date()} ～ {end.date()}

今週の決済トレードはありません。

現在資産: {round(last_equity, 2) if last_equity is not None else "unknown"}
最大DD: {round(max_dd * 100, 2) if max_dd is not None else "unknown"}%
"""
        send_discord(msg)
        return

    total_pnl = week["pnl"].sum()
    trade_count = len(week)
    win_rate = (week["pnl"] > 0).mean()

    by_reason = week.groupby("reason")["pnl"].sum()
    by_ticker = week.groupby("ticker")["pnl"].sum()

    last_equity = None
    max_dd = None

    if len(equity) > 0:
        equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
        eq = equity["equity"].dropna()

        if len(eq) > 0:
            last_equity = eq.iloc[-1]
            peak = eq.cummax()
            max_dd = (eq / peak - 1).min()

    msg = f"""📊 v618 WEEKLY REPORT

期間: {start.date()} ～ {end.date()}

総損益: {round(total_pnl, 2)}
トレード数: {trade_count}
勝率: {round(win_rate * 100, 1)}%

理由別:
{by_reason.to_string()}

銘柄別:
{by_ticker.to_string()}

現在資産: {round(last_equity, 2) if last_equity is not None else "unknown"}
最大DD: {round(max_dd * 100, 2) if max_dd is not None else "unknown"}%
"""

    send_discord(msg)


if __name__ == "__main__":
    run()
