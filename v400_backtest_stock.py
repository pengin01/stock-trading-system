# v401_grid_backtest_stock.py
# -*- coding: utf-8 -*-

import itertools
import pandas as pd
import yfinance as yf
import ta

# =========================
# FIXED PARAMETERS
# =========================
INITIAL_CAPITAL = 20000
RISK_RATIO = 0.5
MIN_VALUE = 100_000_000
YEARS = 3

MA_SHORT = 25
MA_LONG = 75

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

# =========================
# PARAM GRID
# =========================
HOLD_DAYS_LIST = [7, 10, 12]
BREAKOUT_LIST = [20, 40, 60]
RSI_MAX_LIST = [55, 60, 65]
MAX_POSITIONS_LIST = [2, 3]

OUT_SUMMARY = "v401_grid_summary.csv"


# =========================
# DATA
# =========================
def load_data(ticker: str, years: int) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=f"{years}y", progress=False, auto_adjust=False)
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

    df["Close"] = pd.to_numeric(close, errors="coerce")
    df["Volume"] = pd.to_numeric(volume, errors="coerce")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["MA25"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA75"] = df["Close"].rolling(MA_LONG).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    return df.dropna()


def bars_passed(
    df: pd.DataFrame, entry_date: pd.Timestamp, current_date: pd.Timestamp
) -> int:
    return int(((df.index > entry_date) & (df.index <= current_date)).sum())


def calc_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if not dd.empty else 0.0


# =========================
# LOAD ONCE
# =========================
data_raw = {t: load_data(t, YEARS) for t in TICKERS}
data_raw = {k: v for k, v in data_raw.items() if not v.empty}

if not data_raw:
    raise RuntimeError("No data loaded")

all_dates = sorted(set().union(*[df.index for df in data_raw.values()]))


# =========================
# BACKTEST FUNCTION
# =========================
def run_backtest(
    hold_days: int, breakout: int, rsi_max: int, max_positions: int
) -> dict:
    cash = float(INITIAL_CAPITAL)
    positions = []
    equity_curve = []
    trades = []

    # breakout列を都度作る
    data = {}
    for t, df in data_raw.items():
        x = df.copy()
        x["HH"] = x["Close"].rolling(breakout).max()
        x = x.dropna()
        if not x.empty:
            data[t] = x

    dates = sorted(set().union(*[df.index for df in data.values()]))
    if not dates:
        return {
            "hold_days": hold_days,
            "breakout": breakout,
            "rsi_max": rsi_max,
            "max_positions": max_positions,
            "final_equity": INITIAL_CAPITAL,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
        }

    for d in dates:
        # EXIT
        new_pos = []
        for p in positions:
            df = data[p["ticker"]]

            if d not in df.index:
                new_pos.append(p)
                continue

            price = float(df.loc[d, "Close"])
            ma25 = float(df.loc[d, "MA25"])
            hold = bars_passed(df, p["entry_date"], d)

            if price >= ma25 and hold < hold_days:
                new_pos.append(p)
                continue

            cash += price * p["qty"]
            trades.append(
                {
                    "ticker": p["ticker"],
                    "entry": p["entry_date"],
                    "exit": d,
                    "return": price / p["entry_price"] - 1.0,
                }
            )

        positions = new_pos

        # ENTRY
        if len(positions) < max_positions:
            candidates = []
            held = {p["ticker"] for p in positions}

            for t, df in data.items():
                if t in held:
                    continue
                if d not in df.index:
                    continue

                i = df.index.get_loc(d)
                if i < breakout:
                    continue

                close = float(df["Close"].iloc[i])
                ma25 = float(df["MA25"].iloc[i])
                ma75 = float(df["MA75"].iloc[i])
                hh = float(df["HH"].iloc[i - 1])
                value20 = float(df["VALUE20"].iloc[i])
                rsi = float(df["RSI"].iloc[i])

                if not (close > ma25 > ma75):
                    continue
                if close <= hh:
                    continue
                if value20 < MIN_VALUE:
                    continue
                if rsi > rsi_max:
                    continue

                candidates.append((t, rsi))

            candidates.sort(key=lambda x: x[1])

            for t, _ in candidates:
                if len(positions) >= max_positions:
                    break

                price = float(data[t].loc[d, "Close"])
                qty = int((cash * RISK_RATIO) // price)

                if qty <= 0:
                    continue

                cost = price * qty
                if cost > cash:
                    continue

                cash -= cost
                positions.append(
                    {
                        "ticker": t,
                        "entry_date": d,
                        "entry_price": price,
                        "qty": qty,
                    }
                )

        # EQUITY
        pos_val = 0.0
        for p in positions:
            df = data[p["ticker"]]
            if d in df.index:
                pos_val += float(df.loc[d, "Close"]) * p["qty"]

        equity_curve.append(cash + pos_val)

    eq = pd.Series(equity_curve, index=dates[: len(equity_curve)])
    trades_df = pd.DataFrame(trades)

    final_equity = float(eq.iloc[-1]) if not eq.empty else float(INITIAL_CAPITAL)
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    max_drawdown = calc_max_drawdown(eq)

    if trades_df.empty:
        trade_count = 0
        win_rate = 0.0
        avg_return = 0.0
    else:
        trade_count = len(trades_df)
        win_rate = float((trades_df["return"] > 0).mean())
        avg_return = float(trades_df["return"].mean())

    return {
        "hold_days": hold_days,
        "breakout": breakout,
        "rsi_max": rsi_max,
        "max_positions": max_positions,
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }


# =========================
# MAIN
# =========================
def main():
    results = []

    grid = itertools.product(
        HOLD_DAYS_LIST,
        BREAKOUT_LIST,
        RSI_MAX_LIST,
        MAX_POSITIONS_LIST,
    )

    run_id = 1
    for hold_days, breakout, rsi_max, max_positions in grid:
        print(
            f"run {run_id}: "
            f"hold_days={hold_days}, breakout={breakout}, "
            f"rsi_max={rsi_max}, max_positions={max_positions}"
        )
        result = run_backtest(
            hold_days=hold_days,
            breakout=breakout,
            rsi_max=rsi_max,
            max_positions=max_positions,
        )
        result["run_id"] = run_id
        results.append(result)
        run_id += 1

    df = pd.DataFrame(results)
    df = df[
        [
            "run_id",
            "final_equity",
            "total_return",
            "max_drawdown",
            "trade_count",
            "win_rate",
            "avg_return",
            "hold_days",
            "breakout",
            "rsi_max",
            "max_positions",
        ]
    ]

    df = df.sort_values(
        by=["total_return", "max_drawdown", "avg_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    print("\n=== TOP RESULTS ===")
    print(df.head(10).to_string(index=False))

    df.to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
