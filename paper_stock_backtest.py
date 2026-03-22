# -*- coding: utf-8 -*-
# pip install pandas numpy yfinance ta

from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import ta


# =========================
# PARAMETERS
# =========================
@dataclass
class Params:
    capital_yen: int = 80000

    ma_days: int = 25
    rsi_days: int = 14

    rsi_max: float = 65.0
    pullback_pct: float = 0.005
    min_avg_value20: float = 300_000_000

    hold_days: int = 4
    years: int = 1

    trades_file: str = "trades_backtest.csv"
    summary_file: str = "summary_backtest.csv"


P = Params()

STOCK_UNIVERSE = [
    "9432.T", "6758.T", "9984.T",
    "7203.T", "8306.T", "8035.T", "6501.T",
    "6861.T", "4063.T", "7267.T"
]


# =========================
# DATA
# =========================
def download(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{P.years}y",
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)

    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns: {missing}")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["MA"] = close.rolling(P.ma_days).mean()
    out["RSI"] = ta.momentum.RSIIndicator(close, window=P.rsi_days).rsi()
    out["VALUE20"] = (close * out["Volume"]).rolling(20).mean()

    return out


# =========================
# SIGNAL
# =========================
def entry_signal(df: pd.DataFrame, i: int) -> bool:
    if i <= 0:
        return False

    c = df["Close"].iloc[i]
    prev = df["Close"].iloc[i - 1]
    ma = df["MA"].iloc[i]
    rsi = df["RSI"].iloc[i]
    v = df["VALUE20"].iloc[i]

    vals = np.array([c, prev, ma, rsi, v], dtype=float)
    if not np.isfinite(vals).all():
        return False

    if c < ma:
        return False

    if c > prev * (1 - P.pullback_pct):
        return False

    if rsi > P.rsi_max:
        return False

    if v < P.min_avg_value20:
        return False

    return True


def exit_reason(df: pd.DataFrame, i: int, entry_i: int) -> str | None:
    c = df["Close"].iloc[i]
    ma = df["MA"].iloc[i]

    vals = np.array([c, ma], dtype=float)
    if not np.isfinite(vals).all():
        return None

    if c < ma:
        return "MA_EXIT"

    hold_days_bars = i - entry_i
    if hold_days_bars >= P.hold_days:
        return "TIME"

    return None


# =========================
# BACKTEST
# =========================
def backtest_one_ticker(ticker: str) -> pd.DataFrame:
    df = download(ticker)
    if df.empty:
        return pd.DataFrame()

    df = add_features(df)

    min_bars = max(P.ma_days, P.rsi_days, 20) + 1
    if len(df) < min_bars + 1:
        return pd.DataFrame()

    trades = []
    in_position = False
    entry_i = None
    entry_date = None
    entry_price = None
    shares = 0

    for i in range(min_bars, len(df)):
        if not in_position:
            if entry_signal(df, i):
                entry_price = float(df["Close"].iloc[i])
                if not np.isfinite(entry_price) or entry_price <= 0:
                    continue

                shares = int(P.capital_yen // entry_price)
                if shares <= 0:
                    continue

                entry_i = i
                entry_date = df.index[i]
                in_position = True
        else:
            reason = exit_reason(df, i, entry_i)
            if reason is None:
                continue

            exit_price = float(df["Close"].iloc[i])
            exit_date = df.index[i]

            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            pnl_yen = (exit_price - entry_price) * shares
            ret = (exit_price / entry_price) - 1
            hold_days_bars = i - entry_i

            trades.append({
                "ticker": ticker,
                "entry_date": entry_date.date(),
                "exit_date": exit_date.date(),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "shares": shares,
                "pnl_yen": round(pnl_yen, 2),
                "return_pct": round(ret, 5),
                "hold_days_bars": hold_days_bars,
                "exit_reason": reason,
            })

            in_position = False
            entry_i = None
            entry_date = None
            entry_price = None
            shares = 0

    return pd.DataFrame(trades)


def backtest_all() -> pd.DataFrame:
    all_trades = []

    for ticker in STOCK_UNIVERSE:
        try:
            print(f"[INFO] backtesting: {ticker}")
            tdf = backtest_one_ticker(ticker)
            if not tdf.empty:
                all_trades.append(tdf)
        except Exception as e:
            print(f"[WARN] backtest failed: {ticker}: {e}")

    if not all_trades:
        return pd.DataFrame(columns=[
            "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
            "shares", "pnl_yen", "return_pct", "hold_days_bars", "exit_reason"
        ])

    out = pd.concat(all_trades, ignore_index=True)
    out = out.sort_values(["entry_date", "ticker"]).reset_index(drop=True)
    return out


# =========================
# SUMMARY
# =========================
def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        summary = {
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "total_return": 0,
            "total_pnl_yen": 0,
            "avg_hold_days": 0,
            "capital_yen": P.capital_yen,
            "ma_days": P.ma_days,
            "rsi_days": P.rsi_days,
            "rsi_max": P.rsi_max,
            "pullback_pct": P.pullback_pct,
            "min_avg_value20": P.min_avg_value20,
            "hold_days": P.hold_days,
            "years": P.years,
            "universe_count": len(STOCK_UNIVERSE),
        }
        return pd.DataFrame([summary])

    wins = (trades["pnl_yen"] > 0).mean()
    avg_return = trades["return_pct"].mean()
    total_return = (1 + trades["return_pct"]).prod() - 1
    total_pnl_yen = trades["pnl_yen"].sum()
    avg_hold_days = trades["hold_days_bars"].mean()

    summary = {
        "trades": int(len(trades)),
        "win_rate": round(float(wins), 5),
        "avg_return": round(float(avg_return), 5),
        "total_return": round(float(total_return), 5),
        "total_pnl_yen": round(float(total_pnl_yen), 2),
        "avg_hold_days": round(float(avg_hold_days), 2),
        "capital_yen": P.capital_yen,
        "ma_days": P.ma_days,
        "rsi_days": P.rsi_days,
        "rsi_max": P.rsi_max,
        "pullback_pct": P.pullback_pct,
        "min_avg_value20": P.min_avg_value20,
        "hold_days": P.hold_days,
        "years": P.years,
        "universe_count": len(STOCK_UNIVERSE),
    }
    return pd.DataFrame([summary])


def summarize_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[
            "ticker", "trades", "win_rate", "avg_return",
            "total_return", "total_pnl_yen", "avg_hold_days"
        ])

    rows = []
    for ticker, g in trades.groupby("ticker"):
        rows.append({
            "ticker": ticker,
            "trades": int(len(g)),
            "win_rate": round(float((g["pnl_yen"] > 0).mean()), 5),
            "avg_return": round(float(g["return_pct"].mean()), 5),
            "total_return": round(float((1 + g["return_pct"]).prod() - 1), 5),
            "total_pnl_yen": round(float(g["pnl_yen"].sum()), 2),
            "avg_hold_days": round(float(g["hold_days_bars"].mean()), 2),
        })

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


# =========================
# SAVE
# =========================
def save_outputs(trades: pd.DataFrame, summary: pd.DataFrame, by_ticker: pd.DataFrame) -> None:
    trades.to_csv(P.trades_file, index=False, encoding="utf-8-sig")
    summary.to_csv(P.summary_file, index=False, encoding="utf-8-sig")

    by_ticker_file = Path(P.summary_file).with_name("summary_by_ticker_backtest.csv")
    by_ticker.to_csv(by_ticker_file, index=False, encoding="utf-8-sig")


# =========================
# DISPLAY
# =========================
def print_section(title: str) -> None:
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)


def print_params() -> None:
    print_section("PARAMETERS")
    for k, v in asdict(P).items():
        print(f"{k}: {v}")


# =========================
# MAIN
# =========================
def main() -> None:
    print("RUN:", pd.Timestamp.now())
    print_params()

    print_section("BACKTEST START")
    trades = backtest_all()

    print_section("TRADES")
    if trades.empty:
        print("(no trades)")
    else:
        print(trades.to_string(index=False))

    summary = summarize(trades)
    by_ticker = summarize_by_ticker(trades)

    print_section("SUMMARY")
    print(summary.to_string(index=False))

    print_section("SUMMARY BY TICKER")
    if by_ticker.empty:
        print("(no summary by ticker)")
    else:
        print(by_ticker.to_string(index=False))

    save_outputs(trades, summary, by_ticker)

    print_section("CSV OUTPUT")
    print(f"saved: {P.trades_file}")
    print(f"saved: {P.summary_file}")
    print("saved: summary_by_ticker_backtest.csv")


if __name__ == "__main__":
    main()