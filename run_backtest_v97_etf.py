# run_backtest_v97_etf.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
import ta

# =========================
# PARAMETERS
# =========================
PERIOD = "5y"
INITIAL_CAPITAL = 20000

# position sizing
RISK_RATIO = 0.7
MAX_POSITIONS = 1

# entry params
MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 55
PULLBACK = 0.007
MIN_VALUE = 100_000_000

# exit params
HOLD_DAYS = 4
TAKE_PROFIT = 0.012
STOP_LOSS = 0.010

UNIVERSE_FILE = "etf_universe.csv"

RESULT_SUMMARY_FILE = "backtest_summary_v97_etf.csv"
RESULT_TRADES_FILE = "backtest_trades_v97_etf.csv"
RESULT_EQUITY_FILE = "backtest_equity_curve_v97_etf.csv"


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int


def load_universe() -> list[str]:
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"{UNIVERSE_FILE} not found")

    df = pd.read_csv(UNIVERSE_FILE)
    if "ticker" not in df.columns:
        raise ValueError(f"{UNIVERSE_FILE} must contain 'ticker' column")

    tickers = df["ticker"].dropna().astype(str).str.strip().tolist()
    tickers = [t for t in tickers if t]

    if not tickers:
        raise ValueError(f"{UNIVERSE_FILE} has no tickers")

    return tickers


def load_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=PERIOD, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"{ticker}: download error: {e}")
        return pd.DataFrame()

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

    for col in ["Close", "Volume"]:
        if col not in df.columns:
            print(f"{ticker}: missing column {col}")
            return pd.DataFrame()

    close = df["Close"]
    volume = df["Volume"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce")

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["Close"] = close
    df["Volume"] = volume
    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df = df.dropna(subset=["Close", "Volume", "MA", "RSI", "VALUE20"]).copy()
    return df


def calc_bars_passed(
    df: pd.DataFrame, entry_date: pd.Timestamp, signal_date: pd.Timestamp
) -> int:
    mask = (df.index > entry_date) & (df.index <= signal_date)
    return int(mask.sum())


def build_candidates(
    signal_date: pd.Timestamp, held: set[str], data_cache: dict[str, pd.DataFrame]
) -> list[dict]:
    candidates = []

    for ticker, df in data_cache.items():
        if df.empty:
            continue
        if ticker in held:
            continue
        if signal_date not in df.index:
            continue

        i = df.index.get_loc(signal_date)
        if i < MA_DAYS + 2:
            continue

        close = float(df["Close"].iloc[i])
        prev_close = float(df["Close"].iloc[i - 1])
        ma = float(df["MA"].iloc[i])
        rsi = float(df["RSI"].iloc[i])
        value20 = float(df["VALUE20"].iloc[i])
        pullback_ratio = close / prev_close - 1.0

        if close < ma:
            continue
        if close > prev_close * (1 - PULLBACK):
            continue
        if rsi > RSI_MAX:
            continue
        if value20 < MIN_VALUE:
            continue

        candidates.append(
            {
                "ticker": ticker,
                "signal_date": signal_date,
                "close": close,
                "prev_close": prev_close,
                "ma": ma,
                "rsi": rsi,
                "value20": value20,
                "pullback_ratio": pullback_ratio,
                "score": rsi,  # RSI低い順
                "i": i,
            }
        )

    candidates.sort(key=lambda x: x["score"])
    return candidates


def calc_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def main():
    print("=== BACKTEST V97 ETF ===")
    print(f"initial_capital: {INITIAL_CAPITAL}")
    print(f"hold_days: {HOLD_DAYS}")
    print(f"pullback: {PULLBACK}")
    print(f"risk_ratio: {RISK_RATIO}")
    print(f"max_positions: {MAX_POSITIONS}")
    print(f"rsi_max: {RSI_MAX}")
    print(f"min_value: {MIN_VALUE}")

    universe = load_universe()
    print("universe:", universe)

    data_cache = {ticker: load_data(ticker) for ticker in universe}
    data_cache = {k: v for k, v in data_cache.items() if not v.empty}

    if not data_cache:
        raise RuntimeError("No data loaded")

    all_dates = sorted(set().union(*[df.index for df in data_cache.values()]))

    cash = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for signal_date in all_dates:
        # EXIT
        next_positions: list[Position] = []
        for p in positions:
            df = data_cache.get(p.ticker, pd.DataFrame())
            if df.empty or signal_date not in df.index:
                next_positions.append(p)
                continue

            bars = calc_bars_passed(df, p.entry_date, signal_date)

            if bars < HOLD_DAYS:
                next_positions.append(p)
                continue

            exit_price = float(df.loc[signal_date, "Close"])
            proceeds = exit_price * p.qty
            cash += proceeds

            ret = exit_price / p.entry_price - 1.0
            pnl = (exit_price - p.entry_price) * p.qty

            trades.append(
                {
                    "ticker": p.ticker,
                    "entry_date": p.entry_date.strftime("%Y-%m-%d"),
                    "exit_date": signal_date.strftime("%Y-%m-%d"),
                    "entry_price": p.entry_price,
                    "exit_price": exit_price,
                    "qty": p.qty,
                    "return": ret,
                    "pnl": pnl,
                    "reason": "time_exit",
                }
            )

        positions = next_positions

        # ENTRY
        if len(positions) < MAX_POSITIONS:
            held = {p.ticker for p in positions}
            candidates = build_candidates(signal_date, held, data_cache)

            if candidates:
                c = candidates[0]
                price = float(c["close"])
                usable_cash = cash * RISK_RATIO
                qty = int(usable_cash // price)

                if qty > 0:
                    cost = price * qty
                    if cost <= cash:
                        cash -= cost
                        positions.append(
                            Position(
                                ticker=c["ticker"],
                                entry_date=signal_date,
                                entry_price=price,
                                qty=qty,
                            )
                        )

        # EQUITY
        position_value = 0.0
        for p in positions:
            df = data_cache.get(p.ticker, pd.DataFrame())
            if df.empty:
                px = p.entry_price
            elif signal_date in df.index:
                px = float(df.loc[signal_date, "Close"])
            else:
                px = p.entry_price

            position_value += px * p.qty

        equity = cash + position_value

        equity_rows.append(
            {
                "date": signal_date.strftime("%Y-%m-%d"),
                "equity": equity,
                "cash": cash,
                "position_value": position_value,
                "positions": len(positions),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

    final_equity = float(equity_df["equity"].iloc[-1])
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    max_drawdown = calc_max_drawdown(equity_df["equity"])

    trade_count = len(trades_df)
    if trade_count > 0:
        win_rate = float((trades_df["return"] > 0).mean())
        avg_return = float(trades_df["return"].mean())
        total_pnl = float(trades_df["pnl"].sum())
    else:
        win_rate = 0.0
        avg_return = 0.0
        total_pnl = 0.0

    summary_df = pd.DataFrame(
        [
            {
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": final_equity,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "total_pnl": total_pnl,
                "hold_days": HOLD_DAYS,
                "pullback": PULLBACK,
                "risk_ratio": RISK_RATIO,
                "max_positions": MAX_POSITIONS,
                "rsi_max": RSI_MAX,
                "min_value": MIN_VALUE,
            }
        ]
    )

    equity_df.to_csv(RESULT_EQUITY_FILE, index=False)
    trades_df.to_csv(RESULT_TRADES_FILE, index=False)
    summary_df.to_csv(RESULT_SUMMARY_FILE, index=False)

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved: {RESULT_SUMMARY_FILE}")
    print(f"Saved: {RESULT_TRADES_FILE}")
    print(f"Saved: {RESULT_EQUITY_FILE}")


if __name__ == "__main__":
    main()
