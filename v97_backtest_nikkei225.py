# v97_backtest_nikkei225.py
# -*- coding: utf-8 -*-

import os
from itertools import product

import pandas as pd
import yfinance as yf
import ta

PARAM_GRID = {
    "initial_capital": [20000, 30000, 50000, 80000, 100000],
    "hold_days": [4],
    "pullback_pct": [0.002, 0.004, 0.007],
}

UNIVERSE_FILE = "nikkei225.csv"


PERID = "1y"
MA_DAYS = 25
RSI_DAYS = 14
RSI_MAX = 65
MIN_VALUE = 300_000_000

MAX_POSITIONS = 1
RISK_RATIO = 0.7

RESULT_CSV = "v97_backtest_result.csv"


# =========================
# UNIVERSE
# =========================
def load_universe() -> list[str]:
    if not os.path.exists(UNIVERSE_FILE):
        return []

    df = pd.read_csv(UNIVERSE_FILE)
    if "ticker" not in df.columns:
        return []

    return df["ticker"].dropna().astype(str).str.strip().tolist()


# =========================
# DATA
# =========================
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERID, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def load_all(universe: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in universe:
        df = load_data(ticker)
        if not df.empty:
            data[ticker] = df
    return data


def entry_signal(df: pd.DataFrame, i: int, pullback: float) -> bool:
    c = float(df["Close"].iloc[i])
    prev = float(df["Close"].iloc[i - 1])

    if c < float(df["MA"].iloc[i]):
        return False
    if c > prev * (1 - pullback):
        return False
    if float(df["RSI"].iloc[i]) > RSI_MAX:
        return False
    if float(df["VALUE20"].iloc[i]) < MIN_VALUE:
        return False

    return True


def calc_dd(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float((eq / peak - 1).min())


def build_candidates_for_date(date, pos, data, hold_days, pullback):
    candidates = []

    for ticker, df in data.items():
        if ticker in pos:
            continue
        if date not in df.index:
            continue

        i = df.index.get_loc(date)

        if i < MA_DAYS + 2:
            continue
        if i + hold_days >= len(df):
            continue
        if not entry_signal(df, i, pullback):
            continue

        rsi = float(df["RSI"].iloc[i])
        score = -rsi

        candidates.append(
            {
                "ticker": ticker,
                "i": i,
                "score": score,
                "rsi": rsi,
            }
        )

    candidates.sort(key=lambda x: x["score"])
    return candidates


def backtest(params, data):
    capital = float(params["initial_capital"])
    hold_days = int(params["hold_days"])
    pullback = float(params["pullback_pct"])

    if not data:
        return {
            "final_capital": capital,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "passed_candidates": 0,
        }

    all_dates = sorted(set().union(*[df.index for df in data.values()]))

    cash = capital
    positions = []
    trades = []
    equity_curve = []
    passed_candidates_total = 0

    for date in all_dates:
        # EXIT
        new_pos = []
        for p in positions:
            if date < p["exit"]:
                new_pos.append(p)
                continue

            df = data[p["ticker"]]
            if date not in df.index:
                new_pos.append(p)
                continue

            price = float(df.loc[date, "Close"])
            proceeds = price * p["qty"]
            cash += proceeds

            ret = (price - p["entry_price"]) / p["entry_price"]
            trades.append(ret)

        positions = new_pos

        # ENTRY
        if len(positions) < MAX_POSITIONS:
            held_tickers = {p["ticker"] for p in positions}
            candidates = build_candidates_for_date(
                date, held_tickers, data, hold_days, pullback
            )
            passed_candidates_total += len(candidates)

            if candidates:
                c = candidates[0]
                ticker = c["ticker"]
                i = c["i"]

                df = data[ticker]
                price = float(df["Close"].iloc[i])

                usable_cash = cash * RISK_RATIO
                qty = int(usable_cash // price)

                if qty > 0:
                    cost = price * qty
                    cash -= cost

                    positions.append(
                        {
                            "ticker": ticker,
                            "entry_price": price,
                            "qty": qty,
                            "exit": df.index[i + hold_days],
                        }
                    )

        # EQUITY
        mtm = 0.0
        for p in positions:
            df = data[p["ticker"]]

            if date in df.index:
                px = float(df.loc[date, "Close"])
            else:
                hist = df.loc[:date]
                if hist.empty:
                    px = p["entry_price"]
                else:
                    px = float(hist["Close"].iloc[-1])

            mtm += px * p["qty"]

        equity_curve.append(
            {
                "date": date,
                "equity": cash + mtm,
            }
        )

    eq = pd.DataFrame(equity_curve)

    if eq.empty:
        final_capital = capital
        total_return = 0.0
        max_drawdown = 0.0
    else:
        final_capital = float(eq["equity"].iloc[-1])
        total_return = float(eq["equity"].iloc[-1] / capital - 1)
        max_drawdown = calc_dd(eq["equity"])

    trades_n = len(trades)
    win_rate = float(sum(r > 0 for r in trades) / trades_n) if trades_n > 0 else 0.0
    avg_return = float(sum(trades) / trades_n) if trades_n > 0 else 0.0

    # === SAVE EQUITY CURVE ===
    eq.to_csv(
        f"equity_curve_v97_cap{params['initial_capital']}_pb{params['pullback_pct']}.csv",
        index=False,
    )

    return {
        "final_capital": final_capital,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "trades": trades_n,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "passed_candidates": int(passed_candidates_total),
    }


def main():
    universe = load_universe()
    if not universe:
        raise RuntimeError("nikkei225.csv is empty or missing ticker column")

    print("Universe size:", len(universe))
    print("Downloading data...")
    data = load_all(universe)

    results = []

    for combo in product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID.keys(), combo))
        print("RUN:", params)

        res = backtest(params, data)
        res.update(params)
        results.append(res)

    df = pd.DataFrame(results)

    print("\n=== RESULT ===")
    print(df.to_string(index=False))

    df.to_csv(RESULT_CSV, index=False)
    print(f"\nSaved: {RESULT_CSV}")


if __name__ == "__main__":
    main()
