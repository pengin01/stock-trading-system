# run_backtest_v97_etf.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from itertools import product

import pandas as pd
import yfinance as yf
import ta

# =========================
# COMMON PARAMETERS
# =========================
PERIOD = "10y"
INITIAL_CAPITAL = 20000
RISK_RATIO = 0.7
MIN_VALUE = 100_000_000

MA_DAYS = 25
RSI_DAYS = 14

UNIVERSE_FILE = "etf_universe.csv"
RESULT_DIR = "backtest_results_v97_etf"

# =========================
# PARAMETER GRID
# =========================
# ここを増やせば複数実験できます
PARAM_GRID = {
    "hold_days": [4],
    "pullback": [0.007],
    "rsi_max": [56],
    "max_positions": [1],
    "take_profit": [0.013, 0.014],
    "stop_loss": [0.008],
}

# 例:
# PARAM_GRID = {
#     "hold_days": [3, 4],
#     "pullback": [0.007, 0.006],
#     "rsi_max": [55, 57],
#     "max_positions": [1, 2],
#     "take_profit": [0.010, 0.012],
#     "stop_loss": [0.008, 0.010],
# }

SUMMARY_ALL_FILE = os.path.join(RESULT_DIR, "backtest_summary_all_v97_etf.csv")


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int


def ensure_result_dir() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)


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

    for col in ["High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            print(f"{ticker}: missing column {col}")
            return pd.DataFrame()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce")

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["High"] = high
    df["Low"] = low
    df["Close"] = close
    df["Volume"] = volume

    df["MA"] = df["Close"].rolling(MA_DAYS).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], RSI_DAYS).rsi()
    df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df = df.dropna(
        subset=["High", "Low", "Close", "Volume", "MA", "RSI", "VALUE20"]
    ).copy()
    return df


def calc_bars_passed(
    df: pd.DataFrame, entry_date: pd.Timestamp, signal_date: pd.Timestamp
) -> int:
    mask = (df.index > entry_date) & (df.index <= signal_date)
    return int(mask.sum())


def build_candidates(
    signal_date: pd.Timestamp,
    held: set[str],
    data_cache: dict[str, pd.DataFrame],
    pullback: float,
    rsi_max: float,
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
        if close > prev_close * (1 - pullback):
            continue
        if rsi > rsi_max:
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
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def make_run_id(params: dict) -> str:
    return (
        f"hd{params['hold_days']}"
        f"_pb{str(params['pullback']).replace('.', '')}"
        f"_rsi{str(params['rsi_max']).replace('.', '')}"
        f"_mp{params['max_positions']}"
        f"_tp{str(params['take_profit']).replace('.', '')}"
        f"_sl{str(params['stop_loss']).replace('.', '')}"
    )


def run_backtest(
    params: dict,
    universe: list[str],
    data_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hold_days = int(params["hold_days"])
    pullback = float(params["pullback"])
    rsi_max = float(params["rsi_max"])
    max_positions = int(params["max_positions"])
    take_profit = float(params["take_profit"])
    stop_loss = float(params["stop_loss"])

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

            row = df.loc[signal_date]
            bars = calc_bars_passed(df, p.entry_date, signal_date)

            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            tp_price = p.entry_price * (1.0 + take_profit)
            sl_price = p.entry_price * (1.0 - stop_loss)

            exit_price = None
            exit_reason = None

            # 同日でTP/SL両方ヒットした場合は保守的にSL優先
            if low <= sl_price and high >= tp_price:
                exit_price = sl_price
                exit_reason = "sl_tp_same_day"
            elif low <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
            elif high >= tp_price:
                exit_price = tp_price
                exit_reason = "take_profit"
            elif bars >= hold_days:
                exit_price = close
                exit_reason = "time_exit"

            if exit_price is None:
                next_positions.append(p)
                continue

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
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "qty": p.qty,
                    "return": ret,
                    "pnl": pnl,
                    "bars_held": bars,
                    "reason": exit_reason,
                }
            )

        positions = next_positions

        # ENTRY
        if len(positions) < max_positions:
            held = {p.ticker for p in positions}
            candidates = build_candidates(
                signal_date=signal_date,
                held=held,
                data_cache=data_cache,
                pullback=pullback,
                rsi_max=rsi_max,
            )

            slots = max_positions - len(positions)
            if slots > 0 and candidates:
                for c in candidates[:slots]:
                    held_now = {p.ticker for p in positions}
                    if c["ticker"] in held_now:
                        continue

                    price = float(c["close"])
                    usable_cash = cash * RISK_RATIO
                    qty = int(usable_cash // price)

                    if qty <= 0:
                        continue

                    cost = price * qty
                    if cost > cash:
                        continue

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
                hist = df.loc[:signal_date]
                px = float(hist["Close"].iloc[-1]) if not hist.empty else p.entry_price

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

    final_equity = (
        float(equity_df["equity"].iloc[-1])
        if not equity_df.empty
        else float(INITIAL_CAPITAL)
    )
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    max_drawdown = (
        calc_max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0
    )

    trade_count = len(trades_df)
    if trade_count > 0:
        win_rate = float((trades_df["return"] > 0).mean())
        avg_return = float(trades_df["return"].mean())
        total_pnl = float(trades_df["pnl"].sum())
        tp_count = int((trades_df["reason"] == "take_profit").sum())
        sl_count = int((trades_df["reason"] == "stop_loss").sum())
        same_day_count = int((trades_df["reason"] == "sl_tp_same_day").sum())
        time_exit_count = int((trades_df["reason"] == "time_exit").sum())
        best_trade = float(trades_df["return"].max())
        worst_trade = float(trades_df["return"].min())
    else:
        win_rate = 0.0
        avg_return = 0.0
        total_pnl = 0.0
        tp_count = 0
        sl_count = 0
        same_day_count = 0
        time_exit_count = 0
        best_trade = 0.0
        worst_trade = 0.0

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
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "tp_count": tp_count,
                "sl_count": sl_count,
                "sl_tp_same_day_count": same_day_count,
                "time_exit_count": time_exit_count,
                "hold_days": hold_days,
                "pullback": pullback,
                "risk_ratio": RISK_RATIO,
                "max_positions": max_positions,
                "rsi_max": rsi_max,
                "min_value": MIN_VALUE,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "period": PERIOD,
                "universe": ",".join(universe),
            }
        ]
    )

    return summary_df, trades_df, equity_df


def main() -> None:
    ensure_result_dir()

    print("=== BACKTEST V97 ETF MULTI-PARAM ===")
    print(f"period: {PERIOD}")
    print(f"initial_capital: {INITIAL_CAPITAL}")
    print(f"risk_ratio: {RISK_RATIO}")
    print(f"min_value: {MIN_VALUE}")
    print(f"ma_days: {MA_DAYS}")
    print(f"rsi_days: {RSI_DAYS}")

    universe = load_universe()
    print("universe:", universe)

    data_cache = {ticker: load_data(ticker) for ticker in universe}
    data_cache = {k: v for k, v in data_cache.items() if not v.empty}

    if not data_cache:
        raise RuntimeError("No data loaded")

    keys = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))

    all_summaries = []

    print(f"\nTotal runs: {len(combos)}")

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        run_id = make_run_id(params)

        print("\n" + "=" * 60)
        print(f"RUN {idx}/{len(combos)}: {run_id}")
        print(params)

        summary_df, trades_df, equity_df = run_backtest(
            params=params,
            universe=universe,
            data_cache=data_cache,
        )

        summary_file = os.path.join(RESULT_DIR, f"summary_{run_id}.csv")
        trades_file = os.path.join(RESULT_DIR, f"trades_{run_id}.csv")
        equity_file = os.path.join(RESULT_DIR, f"equity_{run_id}.csv")

        summary_df.to_csv(summary_file, index=False)
        trades_df.to_csv(trades_file, index=False)
        equity_df.to_csv(equity_file, index=False)

        all_summaries.append(summary_df.iloc[0].to_dict())

        print(summary_df.to_string(index=False))
        print(f"Saved: {summary_file}")
        print(f"Saved: {trades_file}")
        print(f"Saved: {equity_file}")

    all_summary_df = pd.DataFrame(all_summaries)

    if not all_summary_df.empty:
        all_summary_df = all_summary_df.sort_values(
            by=["total_return", "max_drawdown", "win_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    all_summary_df.to_csv(SUMMARY_ALL_FILE, index=False)

    print("\n" + "=" * 60)
    print("=== ALL RUN SUMMARY ===")
    print(all_summary_df.to_string(index=False))
    print(f"\nSaved: {SUMMARY_ALL_FILE}")


if __name__ == "__main__":
    main()
